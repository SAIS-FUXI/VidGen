#coding=utf-8
"""
Copy from https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
"""

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .modules_video import Encoder, Decoder


class DiagonalGaussianDistributionVideo(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters                                                                # [batch, embed_dim * 2, T/f, H/f, W/f]
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)   # [batch, embed_dim, T/f, H/f, W/f]
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4])

    def nll(self, sample, dims=[1,2,3,4]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    
class VAEVideo(nn.Module):
    def __init__(self,
                 ddconfig,      # config for encoder + decoder
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)                                     # [batch, z_channels * 2, T/f, H/f, W/f]
        moments = self.quant_conv(h)                            # [batch, embed_dim * 2 , T/f, H/f, W/f]
        posterior = DiagonalGaussianDistributionVideo(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()                              # [batch, embed_dim, T/f, H/f, W/f]
        else:
            z = posterior.mode()
        dec = self.decode(z)                                    # [batch, 3, T, H, W]
        return dec, posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight
