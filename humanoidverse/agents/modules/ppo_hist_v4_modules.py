'''
History Encoder V4: teacher 를 "privileged obs 의 latent encoding" 만 담당하도록 축소한 구조.

  Student (배포):      [history]      --MLP-mixer--> [v_base(3), z_s]  (v2 StudentEncoder 재사용)
  Teacher phi(학습전용):[teacher_obs] --MLP-->        z                 (결정론적 encoding only)
                       decoder / logvar / recon / KL 전부 없음 — v2·v3 의 TeacherVAE 대체.

  z 는 생성(recon) 목적이 아니라 정렬 타깃으로만 쓰인다:
      student : latent_coef * MSE(z_s, phi(teacher_obs_{t+1}).detach())

  teacher 를 무엇으로 학습시키는가가 유일한 설계 선택 (붕괴 방지 때문):
    teacher_mode="critic" (기본) phi(teacher_obs_t) 를 main critic 입력에 붙이고, critic 의
        value loss gradient 로 phi 를 학습한다. z 가 value 예측에 기여하지 못하면 critic
        loss 가 커지므로 붕괴가 최적해가 아니다 -> var/cov 같은 인위적 항이 불필요하고,
        latent 이 task 에 유용한 축으로 정렬된다 (PBL, Guo et al. 2020 계열).
        - phi 입력은 t 시점 privileged obs 이므로 a_t 에 의존하지 않는다 -> baseline 무편향.
        - rollout 시점에 존재하는 obs 그룹이라 GAE 용 value 계산 타이밍도 문제없다.
        - teacher_obs 는 critic_obs 에 없는 항목(base_pos_z, feet_contact_force 등)을
          포함해야 한다. 부분집합이면 critic 이 z 를 무시해도 손해가 없어 압력이 0 이 된다.
    teacher_mode="critic_aux" 학습 전용 auxiliary value head V_aux(critic_obs, phi(o_{t+1})) 로
        return 을 회귀. main critic / GAE 는 무수정이라 편향이 원천적으로 없고, 미래 정보의
        강한 신호를 얻는다. V_aux 출력은 advantage 에 쓰이지 않는다.
        (o_{t+1} 을 main critic 에 직접 넣으면 baseline 이 a_t 에 의존해 advantage 가
         눌리므로 반드시 분리한다.)
    teacher_mode="frozen"  학습하지 않음. z = 고정 랜덤 비선형 사영.
        타깃이 처음부터 stationary 이고 상수 붕괴가 구조적으로 불가능하다
        (RND 의 fixed random target 과 같은 논리).
    teacher_mode="vicreg" align(z, z_s.detach()) + variance/covariance 정규화로 학습.
        (VICReg, Bardes et al. 2022). 정규화 없이 align 만 주면 z, z_s 가 같은 상수로
        붕괴해 latent 이 무의미해진다 (v2 가 teacher latent 을 detach 한 이유와 동일).

기존 v1/v2/v3 모듈은 수정하지 않는다 (additive).
'''
from __future__ import annotations

import torch
import torch.nn as nn

from .ppo_hist_modules import MLP


class TeacherEncoder(nn.Module):
    """phi: [teacher_obs] -> MLP -> z. 결정론적이며 decoder/분포 파라미터가 없다.

    같은 phi 를 두 시점에 쓴다: phi(teacher_obs_t) 는 critic 입력, phi(teacher_obs_{t+1}) 은
    student 타깃. 그래서 입력 그룹 하나로 "value 에 유용한 privileged 상태 인코더" 가 된다.

    output_norm=True 면 latent 차원에 대해 (affine 없는) LayerNorm 을 걸어 z 의
    per-sample 평균/스케일을 고정한다. frozen 모드에서 랜덤 초기화된 net 의 출력
    스케일이 임의로 커지거나 작아져 MSE 크기가 latent_coef 와 어긋나는 것을 막고,
    critic 모드에서는 critic 입력 채널의 스케일을 안정화한다.
    """

    def __init__(self, obs_dim, latent_dim, hidden_dims=(128, 64), activation="ELU",
                 output_norm=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MLP(obs_dim, latent_dim, list(hidden_dims), activation)
        self.norm = (nn.LayerNorm(latent_dim, elementwise_affine=False)
                     if output_norm else nn.Identity())

    def forward(self, obs):
        return self.norm(self.net(obs))

    def encode(self, obs):
        """v2 TeacherVAE.encode 와 이름을 맞춘 별칭 (여기서는 mu/logvar 가 없다)."""
        return self.forward(obs)


class TeacherValueHead(nn.Module):
    """teacher_mode="critic_aux" 전용 auxiliary value head: cat[critic_obs, z_t] -> V_aux.

    main critic (PPOCritic) 은 손대지 않는다 — advantage/GAE 는 critic_obs 만 보는
    기존 value 로 계산되고, 이 head 의 출력은 어디에도 쓰이지 않는다.
    존재 이유는 단 하나: return 회귀 오차를 통해 teacher encoder 에 gradient 를 주는 것.
    critic_obs 를 함께 받기 때문에 z_t 는 "critic_obs 로 설명되지 않는 o_{t+1} 의 정보"
    (전이 결과 = 지형/외력 반응) 만 담으면 되고, 그 부분이 곧 student 가 history 로
    추정해야 하는 양이다.
    """

    def __init__(self, critic_obs_dim, latent_dim, hidden_dims=(256, 128), activation="ELU"):
        super().__init__()
        self.net = MLP(critic_obs_dim + latent_dim, 1, list(hidden_dims), activation)

    def forward(self, critic_obs, z):
        return self.net(torch.cat([critic_obs, z], dim=-1))


def vicreg_var_loss(z, target_std=1.0, eps=1e-4):
    """차원별 표준편차가 target_std 미만이면 hinge 페널티 -> 상수 붕괴 차단."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def vicreg_cov_loss(z):
    """off-diagonal 공분산 제곱합 / D -> 차원 간 중복(정보 축 붕괴) 억제."""
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / (n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d
