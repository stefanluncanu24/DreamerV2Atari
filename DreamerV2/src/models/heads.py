import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Bernoulli, Categorical

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class RewardPredictor(nn.Module):
    def __init__(self, stoch_size, deter_size, hidden_size=400, reward_classes=1):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1) 
        )
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        mean = self.model(latent_state)
        return mean

class DiscountPredictor(nn.Module):
    def __init__(self, stoch_size, deter_size, hidden_size=400):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        logits = self.model(latent_state)
        return Bernoulli(logits=logits)
    
    # maybe neccesary, i will see
    def step(self, latent_state):
        dist = self.forward(latent_state)
        probs = dist.mean
        return (probs >= 0.5).float()
