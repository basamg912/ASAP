'''
History Encoder V2 (student-teacher, DreamWaQ 계열 concurrent distillation):

  Student: [history] --MLP-mixer--> [v_base(3), latent z]   (결정론적, 배포에 사용)
  Teacher: [next obs] --MLP--> (mu, logvar) --z--> decoder --> next obs recon (VAE, 학습 전용)

  Policy 입력 = cat[actor_obs, v_hat, z]. detach_encoder_output=True 면 encoder 는
  PPO gradient 를 받지 않고 supervised loss (v_base MSE + latent match) 로만 학습.

기존 ppo_hist_modules.py 는 수정하지 않고 재사용한다 (MLP, MLP_mixer, StatePredictor,
unflatten_history, vae_kl_loss, recon_loss_masked).
'''
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .ppo_modules import PPOActor
from .ppo_hist_modules import MLP, MLP_mixer, StatePredictor, unflatten_history


class StudentEncoder(nn.Module):
    """[history] -> MLP-mixer -> [v_base(vel_dim), z(latent_dim)]. 결정론적."""

    def __init__(self, history_dim, latent_dim, vel_dim=3, hidden_dims=(256,),
                 activation="ELU", key_dims=None, history_length=None,
                 channel_hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.vel_dim = vel_dim
        assert history_dim == sum(key_dims) * history_length
        self.net = MLP_mixer(sum(key_dims), history_length, vel_dim + latent_dim,
                             list(hidden_dims), activation,
                             channel_hidden_dims=channel_hidden_dims)
        self.key_dims = key_dims
        self.history_length = history_length

    def forward(self, history):
        x = unflatten_history(history, self.key_dims, self.history_length)
        out = self.net(x)
        return out[..., : self.vel_dim], out[..., self.vel_dim:]


class TeacherVAE(nn.Module):
    """[next obs] -> encoder -> (mu, logvar) -> z -> decoder -> next obs recon."""

    def __init__(self, obs_dim, latent_dim, enc_hidden_dims=(128, 64),
                 dec_hidden_dims=(128, 256), activation="ELU"):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(obs_dim, 2 * latent_dim, list(enc_hidden_dims), activation)
        self.decoder = StatePredictor(latent_dim, obs_dim, tuple(dec_hidden_dims), activation)

    def encode(self, obs):
        out = self.encoder(obs)
        return out[..., : self.latent_dim], out[..., self.latent_dim:]

    def forward(self, obs):
        mu, logvar = self.encode(obs)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return self.decoder(z), mu, logvar


class PPOActorWithStudentEncoder(PPOActor):
    """actor trunk 입력 = cat[actor_obs, v_hat, z].

    module_config 의 input_dim 은 [actor_obs, vel_dim + latent_dim] 숫자 합산으로
    trunk 차원이 맞춰져 있어야 한다 (예: [actor_obs, 19]).
    detach_encoder_output=True: policy 경로에는 detach 된 v/z 를 넣어 PPO gradient 가
    student 로 흐르지 않게 한다. supervised loss 용 그래프는 get_student_outputs() 로 접근.
    """

    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std,
                 encoder_config, latent_dim, vel_dim, encoder_struct,
                 detach_encoder_output=True):
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std)
        history_dim = obs_dim_dict[encoder_config["input_key"]]
        s = encoder_struct
        self.student = StudentEncoder(
            history_dim, latent_dim, vel_dim,
            tuple(encoder_config["hidden_dims"]), encoder_config["activation"],
            s["key_dims"], s["history_length"],
            channel_hidden_dims=encoder_config.get("channel_hidden_dims", None),
        )
        self.detach_encoder_output = detach_encoder_output
        self._last_student = {}

    def _student_forward(self, encoder_obs):
        v, z = self.student(encoder_obs)
        self._last_student = {"v": v, "z": z}  # 그래프 유지 (supervised loss 용)
        return v, z

    def update_distribution(self, actor_obs, encoder_obs):
        v, z = self._student_forward(encoder_obs)
        if self.detach_encoder_output:
            v, z = v.detach(), z.detach()
        mean = self.actor(torch.cat([actor_obs, v, z], dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs, encoder_obs, **kwargs):
        self.update_distribution(actor_obs, encoder_obs)
        return self.distribution.sample()

    def act_inference(self, actor_obs, encoder_obs):
        v, z = self.student(encoder_obs)
        return self.actor(torch.cat([actor_obs, v, z], dim=-1))

    def get_student_outputs(self):
        """직전 forward 의 {v, z} (detach 되지 않은 그래프)."""
        return self._last_student
