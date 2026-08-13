'''
History Encoder V5 = V3 의 teacher 를 context 조건부 VAE(CVAE)로 교체.

  Student: [history_t] -> MLP-mixer -> [v_base, z_s]        (v2/v3 와 동일, 배포용)
  Teacher (학습 전용):
    1. 입력 = student 와 동일한 window 의 history_t + o_{t+1}
    2. context c = MLP-mixer(history_t)                      (history 압축)
    3. encoder(c, o_{t+1}) -> (t_mu, t_logvar)
    4. decoder(c, t_mu)    -> o_{t+1} recon
  + contrastive heads (v3 그대로: h(z_s) 는 actor 소유, g(z_t) 는 teacher 소유)

의미: decoder 가 context 를 함께 받으므로 recon 은 "history 로 예측 가능한 부분"을
c 가 담당하고, z_t 는 o_{t+1} 이 새로 가져오는 잔차 정보만 인코딩하게 된다.
student 의 z_s 는 이 잔차 표현에 (contrastive/MSE 로) 정렬된다.

주의: 스펙 4 에 따라 decoder 는 sampled z 가 아니라 t_mu 를 직접 받는다 —
reparam 샘플이 decode 경로에서 빠지므로 teacher forward 는 결정론적이고,
t_logvar 는 KL(vae_beta, free_bits) 로만 학습된다.

history 는 student 가 보는 것과 완전히 동일한 텐서(encoder_obs 그룹, rollout 에
이미 저장됨)를 쓰고 o_{t+1} 은 v2 부터 캡처하던 next_obs_target 을 쓰므로,
별도의 obs 그룹/스토리지 키 추가가 필요 없다.

v2/v3 모듈(ppo_hist_v2_modules.py, ppo_hist_v3_modules.py)은 수정하지 않는다 (additive).
'''
from __future__ import annotations

import torch
import torch.nn as nn

from .ppo_hist_modules import MLP, MLP_mixer, StatePredictor, unflatten_history


class TeacherContextCVAEContrastive(nn.Module):
    """context = mixer(history_t); encoder(c, o_{t+1}) -> (mu, logvar); decoder(c, mu) -> recon.

    v3 의 TeacherVAEContrastive 와 학습 루프 인터페이스 호환 (forward 가
    (recon, mu, logvar) 반환, project_teacher_latent) — 입력만 (history, next_obs) 2개.
    """

    def __init__(self, history_dim, obs_dim, latent_dim, encoder_struct, context_dim=32,
                 mixer_hidden_dims=(256, 128), mixer_channel_hidden_dims=None,
                 enc_hidden_dims=(128, 64), dec_hidden_dims=(128, 256), activation="ELU",
                 proj_dim=16, proj_hidden_dims=(32,), proj_activation="ELU"):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.key_dims = encoder_struct["key_dims"]
        self.history_length = encoder_struct["history_length"]
        assert history_dim == sum(self.key_dims) * self.history_length
        # 2. history -> context (student 인코더와 같은 mixer 구조, 가중치는 teacher 소유)
        self.context_mixer = MLP_mixer(sum(self.key_dims), self.history_length, context_dim,
                                       list(mixer_hidden_dims), activation,
                                       channel_hidden_dims=mixer_channel_hidden_dims)
        # 3. (context, o_{t+1}) -> (mu, logvar)
        self.encoder = MLP(context_dim + obs_dim, 2 * latent_dim,
                           list(enc_hidden_dims), activation)
        # 4. (context, t_mu) -> o_{t+1} recon
        self.decoder = StatePredictor(context_dim + latent_dim, obs_dim,
                                      tuple(dec_hidden_dims), activation)
        self.projection_head = MLP(latent_dim, proj_dim, list(proj_hidden_dims), proj_activation)

    def _context(self, history):
        x = unflatten_history(history, self.key_dims, self.history_length)
        return self.context_mixer(x)

    def encode(self, history, next_obs):
        c = self._context(history)
        out = self.encoder(torch.cat([c, next_obs], dim=-1))
        return out[..., : self.latent_dim], out[..., self.latent_dim:], c

    def forward(self, history, next_obs):
        mu, logvar, c = self.encode(history, next_obs)
        recon = self.decoder(torch.cat([c, mu], dim=-1))  # 스펙 4: t_mu 를 직접 decode
        return recon, mu, logvar

    def project_teacher_latent(self, z_t):
        return self.projection_head(z_t)
