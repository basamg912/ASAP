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


class PPOActorWithHistoryEncoder(PPOActor):
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std,
                 encoder_config, decoder_config, latent_dim):
        # base builds the actor trunk from module_config_dict; input_dim must already
        # include the numeric latent_dim, e.g. ["actor_obs", 16]
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std)
        self.latent_dim = latent_dim
        history_dim = obs_dim_dict[encoder_config["input_key"]]
        recon_dim = obs_dim_dict[decoder_config["target_key"]]
        self.history_encoder = HistoryEncoder(
            history_dim, latent_dim,
            tuple(encoder_config["hidden_dims"]), encoder_config["activation"])
        self.state_predictor = StatePredictor(
            latent_dim, recon_dim,
            tuple(decoder_config["hidden_dims"]), decoder_config["activation"])
        self._last_latent = {}

    def _encode(self, encoder_obs):
        z, mu, logvar = self.history_encoder.sample(encoder_obs)
        self._last_latent = {"z": z, "mu": mu, "logvar": logvar}
        return z

    def update_distribution(self, actor_obs, encoder_obs):
        z = self._encode(encoder_obs)
        mean = self.actor(torch.cat([actor_obs, z], dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs, encoder_obs, **kwargs):
        self.update_distribution(actor_obs, encoder_obs)
        return self.distribution.sample()

    def act_inference(self, actor_obs, encoder_obs):
        z, _, _ = self.history_encoder.sample(encoder_obs)  # eval mode -> z = mu
        return self.actor(torch.cat([actor_obs, z], dim=-1))

    def predict_next_state(self):
        return self.state_predictor(self._last_latent["z"])

    def get_latent_stats(self):
        return self._last_latent
