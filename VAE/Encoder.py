'''
Encoder implementation for VAE
'''

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim = 128):
        super(Encoder, self).__init__()

        # ConvNet encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        # Latent parameter heads
        self.mean_head = nn.Linear(256 * 4 * 4, latent_dim)
        self.mean_variance = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        mean, variance = self.mean_head(z), self.mean_variance(z)

        return mean, variance
