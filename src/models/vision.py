import torch
import torch.nn as nn
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, embed_dim, in_channels, cnn_depth=48):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, cnn_depth, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(cnn_depth, 2 * cnn_depth, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(2 * cnn_depth, 4 * cnn_depth, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(4 * cnn_depth, 8 * cnn_depth, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.model(x)
        # The original dreamer-torch calculates embed_dim inside the model,
        # but we'll assume it's passed correctly during initialization.
        return x

class Decoder(nn.Module):
    def __init__(self, stoch_size, deter_size, out_channels=1, cnn_depth=48):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * cnn_depth),
            nn.Unflatten(1, (32 * cnn_depth, 1, 1)),
            nn.ConvTranspose2d(32 * cnn_depth, 4 * cnn_depth, kernel_size=5, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(4 * cnn_depth, 2 * cnn_depth, kernel_size=5, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(2 * cnn_depth, cnn_depth, kernel_size=6, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(cnn_depth, out_channels, kernel_size=6, stride=2),
        )

    def forward(self, latent_state):
        mean = self.model(latent_state)
        return dist.Independent(dist.Normal(mean, 1), reinterpreted_batch_ndims=3)
