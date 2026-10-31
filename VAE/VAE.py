'''
Oren Gershony
Pytorch Implementation for Variational AutoEncoder
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

from .Decoder import Decoder
from .Encoder import Encoder

class VAE(nn.Module):
    def __init__(self, latent_dim = 128):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mean, variance = self.encoder(x)
        
        # Now to obtain latent space
        deviation = torch.exp(0.5 * variance)
        noise = torch.randn_like(deviation)

        z = mean + noise * deviation

        # Now feed latent space back into decoder
        out_x = self.decoder(z)
        return out_x, mean, variance

