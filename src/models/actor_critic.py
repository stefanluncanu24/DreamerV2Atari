import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def orthogonal_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Critic(nn.Module):
    def __init__(self, stoch_size, deter_size, hidden_size=400, layers=4, **kwargs):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        
        # Define the network layers
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(self.latent_dim, hidden_size))
        layers_list.append(nn.ELU())
        
        # Hidden layers
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden_size, hidden_size))
            layers_list.append(nn.ELU())
            
        # Output layer
        layers_list.append(nn.Linear(hidden_size, 2)) # Output 2 for mean and std
        
        self.model = nn.Sequential(*layers_list)
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        output = self.model(latent_state)
        mean, std = torch.chunk(output, 2, dim=-1)
        std = F.softplus(std) + 0.01 # Ensure std is positive
        return Normal(mean, std)


class Actor(nn.Module):    
    def __init__(self, state_dim, action_dim, hidden_size=400, num_layers=4, **kwargs):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        
        # Define the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_size))
        layers.append(nn.ELU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())
            
        # Output layer
        layers.append(nn.Linear(hidden_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.apply(orthogonal_init)

    def forward(self, state):
        logits = self.network(state)
        return Categorical(logits=logits)