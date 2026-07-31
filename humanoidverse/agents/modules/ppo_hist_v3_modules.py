'''
History Encoder V3 = V2 + contrastive projection heads.

  Student: [history] -> [v_base, z_s]     (변경 없음, z_s 가 policy 입력)
  Teacher: [next obs] -> VAE -> z_t -> recon (변경 없음, recon 은 z_t 사용)
  + h(z_s): student contrastive head
  + g(z_t): teacher contrastive head   (z_t 는 detach — teacher encoder 는 recon 으로만 학습)
  contrastive loss = 양방향 InfoNCE(h(z_s), g(z_t))

목적: VAE(KL) 압력이 만드는 teacher latent 의 노이즈 좌표를 MSE 로 그대로 쫓는 대신,
projection head 를 거친 공간에서 상대 유사도(배치 negative 대비)만 정렬한다.

v2 모듈(ppo_hist_v2_modules.py)은 수정하지 않는다 (additive 상속).
'''
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo_hist_modules import MLP
from .ppo_hist_v2_modules import PPOActorWithStudentEncoder, TeacherVAE


def info_nce_loss(p_a, p_b, temperature=0.1):
    """양방향 InfoNCE (CLIP-style).

    p_a[i] 의 positive 는 p_b[i], negatives 는 배치 내 나머지 전부.
    cosine 유사도 기반이라 projection 좌표의 절대 스케일/노이즈에 불변.
    """
    p_a = F.normalize(p_a, dim=-1)
    p_b = F.normalize(p_b, dim=-1)
    logits = p_a @ p_b.t() / temperature
    labels = torch.arange(p_a.shape[0], device=p_a.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))


class PPOActorWithStudentEncoderContrastive(PPOActorWithStudentEncoder):
    """v2 actor + h(z_s) contrastive head. policy 경로는 v2 와 동일 (z_s 사용)."""

    def __init__(self, *args, proj_dim=16, proj_hidden_dims=(32,), proj_activation="ELU",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_head = MLP(
            self.student.latent_dim, proj_dim, list(proj_hidden_dims), proj_activation)

    def project_student_latent(self, z_s):
        return self.contrastive_head(z_s)


class TeacherVAEContrastive(TeacherVAE):
    """v2 teacher + g(z_t) contrastive head. recon 경로는 v2 와 동일 (z_t 사용)."""

    def __init__(self, *args, proj_dim=16, proj_hidden_dims=(32,), proj_activation="ELU",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_head = MLP(
            self.latent_dim, proj_dim, list(proj_hidden_dims), proj_activation)

    def project_teacher_latent(self, z_t):
        return self.projection_head(z_t)
