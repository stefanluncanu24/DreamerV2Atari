import torch
import torch.nn as nn
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, embed_dim, in_channels):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.embed_dim)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, stoch_size, deter_size, out_channels=3):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 3136),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=8, stride=4, padding=0),
        )

# not sure if i should make it a distribution or just return the mean - TODO
    def forward(self, latent_state):
        mean = self.model(latent_state)
        return mean
