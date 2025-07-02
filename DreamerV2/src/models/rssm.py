import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        device: torch.device,
        *,
        category_size: int = 32,
        class_size: int = 32,
        deter_size: int = 1024,
        kl_balancing_alpha: float = 0.8,
        kl_beta: float = 1.0,
        latent_overshooting: bool = True,
        st_type: str = "straight_through",  # "straight_through" | "gumbel"
        gumbel_tau: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.action_dim        = action_dim
        self.embed_dim         = embed_dim
        self.device            = device
        self.category_size     = category_size
        self.class_size        = class_size
        self.stoch_size        = category_size * class_size
        self.deter_size        = deter_size
        self.kl_balancing_alpha= kl_balancing_alpha
        self.kl_beta           = kl_beta
        self.latent_overshooting = latent_overshooting
        self.st_type           = st_type
        self.gumbel_tau        = gumbel_tau

        # recurrent model h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.rnn = nn.GRUCell(self.stoch_size + action_dim, deter_size)

        # posterior q(z_t | h_t, o_t)
        self.repr_model = nn.Sequential(
            nn.Linear(deter_size + embed_dim, deter_size),
            nn.ELU(),
            nn.Linear(deter_size, self.stoch_size),
        )

        # prior p(z_t | h_t)
        self.trans_model = nn.Sequential(
            nn.Linear(deter_size, deter_size),
            nn.ELU(),
            nn.Linear(deter_size, self.stoch_size),
        )

    def initial_state(self, batch_size: int):
        """Return h0, z0 filled with zeros."""
        return (
            torch.zeros(batch_size, self.deter_size, device=self.device),
            torch.zeros(batch_size, self.stoch_size, device=self.device),
        )

    def _straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)                     
        hard  = F.one_hot(probs.argmax(-1), self.class_size)  
        hard  = hard.type_as(probs)                          
        return (hard - probs).detach() + probs                

    def _gumbel_softmax(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        y = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        return y                                              

    def forward(
        self,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        prev_state: torch.Tensor, # h_{t-1}
        prev_latent: torch.Tensor, # z_{t-1} 
    ):
        """
        One RSSM step conditioned on an observation.
        Returns h_t, z_t (one-hot flattened), and KL divergence loss.
        """
        # 1. deterministic update
        rnn_in = torch.cat([prev_latent, action], dim=-1)
        h_t = self.rnn(rnn_in, prev_state)

        # 2. posterior logits  q(z_t | h_t, o_t)
        post_logits = self.repr_model(
            torch.cat([h_t, obs_embed], dim=-1)
        ).view(-1, self.category_size, self.class_size)

        # 3. stochastic sample with gradient estimator
        if self.st_type == "gumbel":
            z_t_onehot = self._gumbel_softmax(post_logits, self.gumbel_tau)
        else:  # "straight_through"
            z_t_onehot = self._straight_through(post_logits)

        z_t = z_t_onehot.view(-1, self.stoch_size)  # flatten for concat

        # 4. prior logits  p(z_t | h_t)  (for KL only)
        prior_logits = self.trans_model(h_t).view(-1, self.category_size, self.class_size)

        post_dist  = OneHotCategorical(logits=post_logits)
        prior_dist = OneHotCategorical(logits=prior_logits)

        kl_loss = self.kl_divergence(post_dist, prior_dist) * self.kl_beta

        return h_t, z_t, kl_loss

    def imagine(
        self,
        action: torch.Tensor,
        prev_state: torch.Tensor,
        prev_latent: torch.Tensor,
    ):
        """
        One RSSM step *without* an observation (prior imagination).
        Returns h_t, z_t sampled from the prior (no gradient through sample).
        """
        rnn_in = torch.cat([prev_latent, action], dim=-1)
        h_t = self.rnn(rnn_in, prev_state)

        prior_logits = self.trans_model(h_t).view(-1, self.category_size, self.class_size)
        prior_dist   = OneHotCategorical(logits=prior_logits)
        z_t          = prior_dist.sample().view(-1, self.stoch_size)  # hard, no ST

        return h_t, z_t

    def kl_divergence(self, post_dist, prior_dist):
        sg_prior_dist = OneHotCategorical(logits=prior_dist.logits.detach())
        sg_post_dist  = OneHotCategorical(logits=post_dist.logits.detach())

        kl_post  = torch.distributions.kl.kl_divergence(post_dist, sg_prior_dist).mean()
        kl_prior = torch.distributions.kl.kl_divergence(sg_post_dist, prior_dist).mean()

        return (1 - self.kl_balancing_alpha) * kl_post + self.kl_balancing_alpha * kl_prior
