import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Categorical

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Critic(nn.Module):
    def __init__(self, stoch_size, deter_size, hidden_size=400, **kwargs):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2) # Output 2 for mean and std
        )
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        output = self.model(latent_state)
        mean, std = torch.chunk(output, 2, dim=-1)
        std = F.softplus(std) + 0.01 # Ensure std is positive and add a small epsilon
        return dist.Normal(mean, std)


class Actor(nn.Module):
    def __init__(self, stoch_size, deter_size, action_dim, hidden_size=400, **kwargs):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.action_dim = action_dim
        self.entropy_coef = kwargs.get('entropy_coef', 0.003) 
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.action_dim)
        )
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        logits = self.model(latent_state)
        return Categorical(logits=logits)
