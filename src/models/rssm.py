import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        device: torch.device,
        *,
        category_size: int = 32,
        class_size: int = 32,
        deter_size: int = 600,
        hidden_size: int = 600,
        kl_balancing_alpha: float = 0.8,
        kl_beta: float = 1.0,
        min_std: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.device = device
        self.category_size = category_size
        self.class_size = class_size
        self.stoch_size = category_size * class_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.kl_balancing_alpha = kl_balancing_alpha
        self.kl_beta = kl_beta
        self.min_std = min_std

        self.cell = GRUCell(self.hidden_size, self.deter_size, norm=True)

        self.img_net = nn.Sequential(
            nn.Linear(self.stoch_size + self.action_dim, self.hidden_size),
            nn.ELU(),
        )
        self.obs_net = nn.Sequential(
            nn.Linear(self.deter_size + self.embed_dim, self.hidden_size),
            nn.ELU(),
        )

        self.ts_prior_logits = nn.Sequential(
            nn.Linear(self.deter_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.stoch_size),
        )
        self.ts_post_logits = nn.Sequential(
            nn.Linear(self.hidden_size, self.stoch_size),
        )

    def initial_state(self, batch_size: int):
        return (
            torch.zeros(batch_size, self.deter_size, device=self.device),
            torch.zeros(batch_size, self.stoch_size, device=self.device),
        )

    def forward(
        self,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
    ):
        h, z_post, kl_loss = self.obs_step(obs_embed, action, prev_h, prev_z)
        return h, z_post, kl_loss

    def imagine(
        self,
        action: torch.Tensor,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
    ):
        rnn_in = self.img_net(torch.cat([prev_z, action], dim=-1))
        h, _ = self.cell(rnn_in, [prev_h])
        
        prior_logits = self.ts_prior_logits(h).view(-1, self.category_size, self.class_size)
        prior_dist = OneHotCategorical(logits=prior_logits)
        z = prior_dist.sample().view(-1, self.stoch_size)
        return h, z

    def obs_step(self, obs_embed, prev_action, prev_h, prev_z):
        # Prior
        rnn_in = self.img_net(torch.cat([prev_z, prev_action], dim=-1))
        h, _ = self.cell(rnn_in, [prev_h])
        prior_logits = self.ts_prior_logits(h).view(-1, self.category_size, self.class_size)

        # Posterior
        post_in = self.obs_net(torch.cat([h, obs_embed], dim=-1))
        post_logits = self.ts_post_logits(post_in).view(-1, self.category_size, self.class_size)

        # Sample and calculate KL divergence
        z_post = self._straight_through(post_logits).view(-1, self.stoch_size)

        kl_loss = self.kl_divergence(post_logits, prior_logits) * self.kl_beta
        return h, z_post, kl_loss

    def kl_divergence(self, post_logits, prior_logits):
        post_dist = torch.distributions.Categorical(logits=post_logits)
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        
        # The original implementation was incorrect. .detach() is a tensor method.
        # The correct way is to detach the logits *before* creating the distribution.
        prior_dist_detached = torch.distributions.Categorical(logits=prior_logits.detach())
        post_dist_detached = torch.distributions.Categorical(logits=post_logits.detach())

        kl_post = torch.distributions.kl.kl_divergence(post_dist, prior_dist_detached).mean()
        kl_prior = torch.distributions.kl.kl_divergence(post_dist_detached, prior_dist).mean()

        # Balancing using stop-gradient
        kl_loss = (1 - self.kl_balancing_alpha) * kl_post + self.kl_balancing_alpha * kl_prior
        return kl_loss

    def _straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=logits)
        sample = dist.sample()
        hard = F.one_hot(sample, self.class_size).float()
        soft = dist.probs
        return (hard - soft).detach() + soft
