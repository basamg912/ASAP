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

class MLP_mixer(nn.Module):
    def __init__(self, feature_dim, channel_dim, output_dim, hidden_dims, activation="ELU",
                 channel_hidden_dims=None):
        # channel_hidden_dims: channel mixing MLP 전용 hidden (미지정 시 hidden_dims 공유 — 하위호환).
        # channel 차원(=history 길이, 보통 5)은 작아서 hidden 256 은 과설계 —
        # 연산 병목(행 75회 반복)이므로 [64] 수준으로 분리 지정 권장.
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)
        self.act = getattr(nn, activation)()
        # Feature mixing
        self.feature_layer = [nn.Linear(feature_dim, hidden_dims[0]), self.act]
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                self.feature_layer.append(nn.Linear(hidden_dims[l], feature_dim))
            else:
                self.feature_layer += [nn.Linear(hidden_dims[l], hidden_dims[l + 1]), self.act]
        self.feature_mixing = nn.Sequential(*self.feature_layer)

        # Channel mixing
        ch_hidden = list(channel_hidden_dims) if channel_hidden_dims else list(hidden_dims)
        self.channel_layer = [nn.Linear(channel_dim, ch_hidden[0]), self.act]
        for l in range(len(ch_hidden)):
            if l == len(ch_hidden) - 1:
                self.channel_layer.append(nn.Linear(ch_hidden[l], channel_dim))
            else:
                self.channel_layer += [nn.Linear(ch_hidden[l], ch_hidden[l + 1]), self.act]
        self.channel_mixing = nn.Sequential(*self.channel_layer)

        # Output
        self.output_layer = nn.Linear(channel_dim * feature_dim, output_dim)

    def forward(self, x):
        # input shape [B, History_length, Obs_dims(Feature)]
        y = self.feature_mixing(self.ln(x))
        x = x + y
        y = self.ln2(x).transpose(1,2)
        y = self.channel_mixing(y)
        x = x + y.transpose(1,2)
        x = x.flatten(1)
        return self.output_layer(x)

# (1) MLP 모듈 내부 init/forward 수정
# (2) forward 부분 이해
class HistoryEncoderMixer(nn.Module):
    def __init__(self, history_dim, latent_dim, hidden_dims=(256,), activation="ELU",
        key_dims=None, num_keys=None, history_length=None, channel_hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        assert history_dim == sum(key_dims) * history_length
        self.net = MLP_mixer(sum(key_dims), history_length, 2 * latent_dim, list(hidden_dims), activation,
                             channel_hidden_dims=channel_hidden_dims)
        self.key_dims = key_dims
        self.num_keys = num_keys
        self.history_length = history_length

    def forward(self, history):
        x = unflatten_history(history, self.key_dims, self.history_length)
        out = self.net(x)
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

class HistoryEncoder(nn.Module):
    def __init__(self, history_dim, latent_dim, hidden_dims=(256, 128), activation="ELU", encoder_struct=None):
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


def vae_kl_loss(mu, logvar, free_bits=0.0):
    # free bits (Kingma et al., IAF): 차원당 배치평균 KL 이 floor(λ) 이하면 clamp 되어
    # 그래디언트가 0 -> KL 압력이 posterior 를 prior 로 붕괴시키는 것을 구조적으로 차단
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, L]
    kl_dim = kl_per_dim.mean(dim=0)                              # [L]
    if free_bits > 0.0:
        kl_dim = torch.clamp(kl_dim, min=free_bits)
    return kl_dim.sum()


def recon_loss_masked(pred, target, valid_mask):
    per_sample = ((pred - target) ** 2).mean(dim=-1)   # (B,)
    w = valid_mask.reshape(-1).to(per_sample.dtype)
    denom = torch.clamp(w.sum(), min=1.0)
    return (per_sample * w).sum() / denom


class PPOActorWithHistoryEncoder(PPOActor):
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std,
                 encoder_config, decoder_config, latent_dim, encoder_struct):
        # base builds the actor trunk from module_config_dict; input_dim must already
        # include the numeric latent_dim, e.g. ["actor_obs", 16]
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std)
        self.latent_dim = latent_dim
        history_dim = obs_dim_dict[encoder_config["input_key"]] # history : pre_process_config 에서 해석, int
        recon_dim = obs_dim_dict[decoder_config["target_key"]]

        self.s = encoder_struct
        self.key_dims = self.s["key_dims"]
        self.num_keys = self.s["num_keys"]
        self.history_length = self.s["history_length"]

        self.history_encoder = HistoryEncoderMixer( # config 에 정의한 [history] 통해 접근
            history_dim, latent_dim,
            tuple(encoder_config["hidden_dims"]), encoder_config["activation"],
            self.key_dims, self.num_keys, self.history_length,
            channel_hidden_dims=encoder_config.get("channel_hidden_dims", None),
        )
        self.state_predictor = StatePredictor(
            latent_dim, recon_dim,
            tuple(decoder_config["hidden_dims"]), decoder_config["activation"])
        self._last_latent = {}

    def _encode(self, encoder_obs):
        z, mu, logvar = self.history_encoder.sample(encoder_obs)
        self._last_latent = {"z": z, "mu": mu, "logvar": logvar}
        return z

    # TODO : action sampling 부분
    def update_distribution(self, actor_obs, encoder_obs):
        z = self._encode(encoder_obs)
        mean = self.actor(torch.cat([actor_obs, z], dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs, encoder_obs, **kwargs):
        self.update_distribution(actor_obs, encoder_obs)
        return self.distribution.sample()

    def act_inference(self, actor_obs, encoder_obs):
        z, _, _ = self.history_encoder.sample(encoder_obs)  # eval mode -> z = mu
        return self.actor(torch.cat([actor_obs, z], dim=-1)) # module (MLP policy) forward 통과

    def predict_next_state(self):
        return self.state_predictor(self._last_latent["z"])

    def get_latent_stats(self):
        return self._last_latent

def unflatten_history(flat, key_dims, T, chronological=False):
    N = flat.shape[0] # Batch size
    chunks = torch.split(flat, [d * T for d in key_dims], dim=1)
    seq = torch.cat([c.view(N, T, d) for c,d in zip(chunks, key_dims)], dim=-1)
    if chronological:
        seq = seq.flip(1)
    return seq
