from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .ppo_modules import PPOActor


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="ELU"):
        super().__init__()
        act = getattr(nn, activation)()
        layers = [nn.Linear(input_dim, hidden_dims[0]), act]
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers += [nn.Linear(hidden_dims[l], hidden_dims[l + 1]), act]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HistoryEncoder(nn.Module):
    def __init__(self, history_dim, latent_dim, hidden_dims=(256, 128), activation="ELU"):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MLP(history_dim, 2 * latent_dim, list(hidden_dims), activation)

    def forward(self, history):
        out = self.net(history)
        mu = out[..., : self.latent_dim]
        logvar = out[..., self.latent_dim :]
        return mu, logvar

    def sample(self, history):
        mu, logvar = self.forward(history)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


class StatePredictor(nn.Module):
    def __init__(self, latent_dim, recon_dim, hidden_dims=(128, 256), activation="ELU"):
        super().__init__()
        self.net = MLP(latent_dim, recon_dim, list(hidden_dims), activation)

    def forward(self, z):
        return self.net(z)


def vae_kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def recon_loss_masked(pred, target, valid_mask):
    per_sample = ((pred - target) ** 2).mean(dim=-1)   # (B,)
    w = valid_mask.reshape(-1).to(per_sample.dtype)
    denom = torch.clamp(w.sum(), min=1.0)
    return (per_sample * w).sum() / denom
