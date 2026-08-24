'''
History Encoder V4 = privileged teacher를 task critic으로 학습하고 history student를
policy + velocity + contrastive loss로 학습시키는 버전.

  - Student (배포 사용): [history] -> MLP-mixer -> [v_base(3), latent z_s]   (v2 와 동일)
  - Teacher phi (학습 전용): [teacher_obs] -> MLP -> z                       (encoding only)
      decoder / logvar / recon / KL 없음. privileged obs 를 latent 로 encoding 하는 일만 한다.
      phi(teacher_obs_t) -> main critic 입력에 concat (teacher_mode="critic")
  - Loss:
      critic  : value_loss_coef * PPO value loss   (gradient 가 phi 까지 전파 -> phi 학습)
      student : PPO policy loss + vel_coef * MSE(v_hat, GT base_lin_vel)
              + contrastive_coef * InfoNCE(h(z_s), g(phi(teacher_obs_t).detach()))
      teacher : encoder는 value loss만, projection head는 contrastive loss만 받는다.

teacher_mode (teacher 학습 신호 선택; student/latent 구조는 4개 모드 모두 동일):
  critic     (기본) phi(teacher_obs_t) 를 main critic 입력에 붙여 value loss 로 학습.
  critic_aux 학습 전용 aux value head V_aux(critic_obs, phi(o_{t+1})) 로 returns 회귀.
             main critic 무수정 (미래 정보를 쓰지만 advantage 에 영향 없음).
  frozen     teacher 미학습 (고정 랜덤 사영 = stationary target).
  vicreg     align(z, z_s.detach()) + var/cov 정규화로 학습.

critic 모드가 무편향인 이유:
  phi 입력이 t 시점 obs 이므로 baseline 이 a_t 에 의존하지 않는다. (o_{t+1} 을 main critic
  에 넣으면 baseline 이 행동 의존이 되어 delta_t = r_t + gamma*V(s_{t+1}) - V(s_t) 가
  정책 그래디언트 신호를 스스로 설명해 advantage 가 눌린다 -> 그 변형은 critic_aux 로 분리.)
  또 teacher_obs 는 매 스텝 obs 그룹이라 rollout 시점(env.step 이전)에 이미 존재한다.

teacher_obs 는 critic_obs 에 없는 privileged 항목(base_pos_z, feet_contact_force 등)을
포함해야 한다. 부분집합이면 critic 이 z 채널을 무시해도 손해가 없어 phi 로 가는 gradient
압력이 0 이 된다. -> obs/loco/leggedloco_obs_history_encoder_v4.yaml 사용.

v1/v2/v3 (agents/ppo_hist*, modules/ppo_hist*_modules.py) 는 수정하지 않는다 (additive).
타이밍 정렬/스토리지/rollout 은 v2 를 그대로 상속한다:
  base_vel_target : env.step "이전" base_lin_vel (s_t)
  next_obs_target : v2 상속 storage가 보존하는 o_{t+1}. 현재 critic+contrastive 경로에서는
                    사용하지 않고, vicreg/critic_aux legacy 모드에서만 teacher 입력으로 쓴다.

Launch: +exp=locomotion_hist_v4 +obs=loco/leggedloco_obs_history_encoder_v4
'''
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from omegaconf import open_dict

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.ppo_modules import PPOCritic
from humanoidverse.agents.modules.ppo_hist_modules import recon_loss_masked
from humanoidverse.agents.modules.ppo_hist_v2_modules import PPOActorWithStudentEncoder
from humanoidverse.agents.modules.ppo_hist_v3_modules import (
    PPOActorWithStudentEncoderContrastive, info_nce_loss,
)
from humanoidverse.agents.modules.ppo_hist_v4_modules import (
    TeacherEncoder, TeacherEncoderContrastive, TeacherValueHead,
    vicreg_var_loss, vicreg_cov_loss,
)
from humanoidverse.agents.ppo_hist_v2.ppo_hist_v2 import PPOHistV2


class PPOHistV4(PPOHistV2):
    """PPOHistV2 - (teacher VAE recon/KL) + deterministic next-obs latent encoder."""

    # ---- config passthrough ----
    def _init_config(self):
        # v2._init_config 는 teacher VAE 용 키(recon_coef/vae_beta/vae_free_bits)를 읽는다.
        # v4 는 recon/KL 자체가 없으므로 yaml 에서 빼고, 상속 호환용 값만 주입한다.
        with open_dict(self.config):
            for unused_key in ('recon_coef', 'vae_beta', 'vae_free_bits'):
                if unused_key not in self.config:
                    self.config[unused_key] = 0.0
        super()._init_config()

        self.teacher_mode = str(self.config.get('teacher_mode', 'critic'))
        assert self.teacher_mode in ('frozen', 'vicreg', 'critic', 'critic_aux'), \
            ("teacher_mode must be one of frozen/vicreg/critic/critic_aux, "
             f"got {self.teacher_mode}")
        self.teacher_trainable = self.teacher_mode != 'frozen'
        # critic 모드만 main critic 입력에 latent 를 붙인다 (phi 는 critic loss 로 학습)
        self.teacher_in_critic = (self.teacher_mode == 'critic')

        # Saved V4 configs before this variant do not have this key. They keep
        # the original actor/teacher module shapes and remain loadable.
        self.use_contrastive = bool(self.config.get('use_contrastive', False))
        self.contrastive_coef = float(self.config.get('contrastive_coef', 0.0))
        self.contrastive_temperature = float(
            self.config.get('contrastive_temperature', 0.1))
        self.contrastive_batch_size = int(
            self.config.get('contrastive_batch_size', 1024))
        if self.use_contrastive and not self.teacher_in_critic:
            raise ValueError(
                "V4 contrastive path requires teacher_mode=critic so the teacher "
                "encoder is anchored only by the task value loss")

        # phi 입력 그룹. 미지정이면 v2 상속 이름(recon_target_key)을 그대로 쓴다.
        self.teacher_obs_key = str(self.config.get('teacher_obs_key', self.recon_target_key))
        assert self.teacher_obs_key in self.algo_obs_dim_dict, (
            f"obs group '{self.teacher_obs_key}' 가 없다 — "
            "+obs=loco/leggedloco_obs_history_encoder_v4 로 실행해야 한다 "
            f"(available: {sorted(self.algo_obs_dim_dict.keys())})")
        assert self.teacher_obs_key == self.recon_target_key, (
            "v2 상속 rollout 은 recon_target_key 그룹의 t+1 값을 next_obs_target 으로 "
            "캡처한다. critic 모드에서는 이 슬롯을 사용하지 않지만 legacy teacher 모드와 "
            "공통 observation 계약을 유지하려면 두 키가 같아야 한다 "
            f"(teacher_obs_key={self.teacher_obs_key}, recon_target_key={self.recon_target_key})")

        tcfg = self.config.teacher_config
        self.teacher_align_coef = float(tcfg.get('align_coef', 1.0))
        self.teacher_var_coef = float(tcfg.get('var_coef', 1.0))
        self.teacher_cov_coef = float(tcfg.get('cov_coef', 0.04))
        self.teacher_value_coef = float(tcfg.get('value_coef', 1.0))

    # ---- loss keys: VAE 전용 키 제거 + v4 진단 키 추가 ----
    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict.pop('teacher_recon', None)   # v4 에는 recon 이 없다
        loss_dict.pop('teacher_kl', None)      # v4 에는 KL 이 없다
        # teacher align 항의 값은 latent_match 와 항상 동일(detach 방향만 반대)하므로 따로 찍지 않는다.
        loss_dict['teacher_var'] = 0
        loss_dict['teacher_cov'] = 0
        loss_dict['teacher_value'] = 0   # critic_aux 모드: V_aux 의 return 회귀 MSE
        loss_dict['contrastive'] = 0
        return loss_dict

    # ---- models: actor 는 v2 student, teacher 는 결정론적 encoder ----
    def _setup_models_and_optimizer(self):
        actor_cls = (PPOActorWithStudentEncoderContrastive
                     if self.use_contrastive else PPOActorWithStudentEncoder)
        actor_kwargs = {}
        if self.use_contrastive:
            proj = self.config.projection_config
            actor_kwargs = {
                'proj_dim': int(proj.proj_dim),
                'proj_hidden_dims': tuple(proj.hidden_dims),
                'proj_activation': proj.activation,
            }
        self.actor = actor_cls(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            encoder_config=self.config.encoder_config,
            latent_dim=self.latent_dim,
            vel_dim=self.vel_dim,
            encoder_struct=self.encoder_struct,
            detach_encoder_output=bool(self.config.get('detach_encoder_for_policy', True)),
            **actor_kwargs,
        ).to(self.device)
        # critic 모드: main critic 입력 = cat[critic_obs, z]. yaml 에 숫자를 중복 적어
        # latent_dim 과 어긋나는 footgun 을 피하려고 여기서 append 한다.
        critic_module_config = self.config.module_dict.critic
        if self.teacher_in_critic:
            critic_module_config = copy.deepcopy(critic_module_config)
            with open_dict(critic_module_config):
                critic_module_config.input_dim = (list(critic_module_config.input_dim)
                                                  + [self.latent_dim])
            logger.info(f"[hist_v4] critic input_dim -> {list(critic_module_config.input_dim)} "
                        "(teacher latent concat)")
        self.critic = PPOCritic(self.algo_obs_dim_dict, critic_module_config).to(self.device)

        tcfg = self.config.teacher_config
        teacher_cls = (TeacherEncoderContrastive
                       if self.use_contrastive else TeacherEncoder)
        teacher_kwargs = {}
        if self.use_contrastive:
            proj = self.config.projection_config
            teacher_kwargs = {
                'proj_dim': int(proj.proj_dim),
                'proj_hidden_dims': tuple(proj.hidden_dims),
                'proj_activation': proj.activation,
            }
        self.teacher = teacher_cls(
            obs_dim=self.algo_obs_dim_dict[self.teacher_obs_key],
            latent_dim=self.latent_dim,
            hidden_dims=tuple(tcfg.enc_hidden_dims),
            activation=tcfg.activation,
            output_norm=bool(tcfg.get('output_norm', True)),
            **teacher_kwargs,
        ).to(self.device)

        # critic_aux 모드 전용: 학습 전용 aux value head (main critic 은 무수정 -> GAE 불변)
        self.teacher_value_head = None
        if self.teacher_mode == 'critic_aux':
            self.teacher_value_head = TeacherValueHead(
                critic_obs_dim=self.algo_obs_dim_dict['critic_obs'],
                latent_dim=self.latent_dim,
                hidden_dims=tuple(tcfg.get('value_hidden_dims', [256, 128])),
                activation=tcfg.activation,
            ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        if self.teacher_trainable:
            teacher_params = list(self.teacher.parameters())
            if self.teacher_value_head is not None:
                teacher_params += list(self.teacher_value_head.parameters())
            self.teacher_optimizer = optim.Adam(
                teacher_params, lr=float(tcfg.learning_rate))
        else:
            # frozen: 랜덤 초기값을 그대로 고정 사영으로 쓴다 -> optimizer/grad 불필요
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            self.teacher_optimizer = None
        logger.info(f"[hist_v4] teacher_mode={self.teacher_mode} "
                    f"(trainable={self.teacher_trainable}, "
                    f"teacher_obs={self.teacher_obs_key}, "
                    f"contrastive={self.use_contrastive})")

    # ---- critic: critic 모드에서만 입력에 phi(teacher_obs_t) 를 붙인다 ----
    # rollout 에서는 inference_mode 라 그래프가 없고, update 에서 재평가될 때 그래프가 생겨
    # critic value loss 의 gradient 가 phi 로 흐른다 (= teacher 학습 경로).
    # 입력이 t 시점 obs 이므로 baseline 은 a_t 에 의존하지 않는다 (무편향).
    def _critic_eval_step(self, obs_dict):
        if not self.teacher_in_critic:
            return super()._critic_eval_step(obs_dict)
        z_now = self.teacher(obs_dict[self.teacher_obs_key])
        self._last_critic_latent = z_now
        return self.critic.evaluate(torch.cat([obs_dict["critic_obs"], z_now], dim=-1))

    # PPO._compute_returns 는 bootstrap value 를 critic.evaluate(last_obs_dict["critic_obs"]) 로
    # 직접 계산한다 (_critic_eval_step 우회). critic 모드에서는 입력 차원이 안 맞으므로
    # 그 한 곳만 latent 를 붙인 텐서로 바꿔 넘긴다 (GAE 본체는 base 구현 그대로 사용).
    def _compute_returns(self, last_obs_dict, policy_state_dict):
        if not self.teacher_in_critic:
            return super()._compute_returns(last_obs_dict, policy_state_dict)
        patched = dict(last_obs_dict)
        with torch.no_grad():
            z_last = self.teacher(last_obs_dict[self.teacher_obs_key])
        patched["critic_obs"] = torch.cat([last_obs_dict["critic_obs"], z_last], dim=-1)
        return super()._compute_returns(patched, policy_state_dict)

    def _contrastive_loss(self, z_student, z_teacher):
        """Align the student to a stop-gradient, same-step teacher target."""
        if not self.use_contrastive:
            return torch.zeros((), device=z_student.device)
        batch_size = z_student.shape[0]
        if batch_size < 2:
            return torch.zeros((), device=z_student.device)
        if batch_size > self.contrastive_batch_size:
            idx = torch.randperm(batch_size, device=z_student.device)[
                :self.contrastive_batch_size]
            z_student = z_student[idx]
            z_teacher = z_teacher[idx]
        p_student = self.actor.project_student_latent(z_student)
        p_teacher = self.teacher.project_teacher_latent(z_teacher.detach())
        return info_nce_loss(
            p_student, p_teacher, self.contrastive_temperature)

    # ---- update: v2 의 _update_ppo 복제 + teacher VAE 블록을 latent encoder 로 교체 ----
    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)  # distribution 재계산 + student {v,z} 스태시
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)
                for pg in self.actor_optimizer.param_groups:
                    pg['lr'] = self.actor_learning_rate
                for pg in self.critic_optimizer.param_groups:
                    pg['lr'] = self.critic_learning_rate

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()

        # Non-critic legacy modes still use t+1 targets. In critic mode the
        # student target is the same-step task latent already computed above.
        target_next = policy_state_dict['next_obs_target']
        valid_mask = (~policy_state_dict['dones'].bool()).float()
        student = self.actor.get_student_outputs()
        # student 타깃은 항상 detach 되므로, 이 forward 에 grad 가 필요한 모드는
        # z_next 자체를 teacher loss 에 쓰는 vicreg / critic_aux 뿐이다.
        need_grad_on_next = self.teacher_mode in ('vicreg', 'critic_aux')
        if need_grad_on_next:
            z_next = self.teacher(target_next)
        elif not self.teacher_in_critic:
            with torch.no_grad():
                z_next = self.teacher(target_next)
        else:
            z_next = None

        # ---- teacher 학습 (critic 모드는 critic_loss.backward() 로 학습되므로 여기서 제외) ----
        t_var = torch.zeros((), device=self.device)
        t_cov = torch.zeros((), device=self.device)
        t_value = torch.zeros((), device=self.device)
        if need_grad_on_next:
            if self.teacher_mode == 'vicreg':
                valid_idx = torch.nonzero(valid_mask.reshape(-1) > 0).squeeze(-1)
                z_valid = z_next[valid_idx]
                # align 은 student 쪽 latent_loss 와 같은 항을 teacher 파라미터로 미분하는 것
                t_align = recon_loss_masked(z_next, student['z'].detach(), valid_mask)
                t_var = vicreg_var_loss(z_valid)
                t_cov = vicreg_cov_loss(z_valid)
                teacher_loss = (self.teacher_align_coef * t_align
                                + self.teacher_var_coef * t_var
                                + self.teacher_cov_coef * t_cov)
            else:  # 'critic_aux': aux value head 의 return 회귀만 (GAE/advantage 와 무관)
                v_aux = self.teacher_value_head(policy_state_dict['critic_obs'], z_next)
                # o_{t+1} 이 리셋 후 obs 인 스텝은 제외 (latent loss 와 동일 마스크)
                t_value = recon_loss_masked(v_aux, returns_batch, valid_mask)
                teacher_loss = self.teacher_value_coef * t_value
            self.teacher_optimizer.zero_grad()
            teacher_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for g in self.teacher_optimizer.param_groups for p in g['params']],
                self.max_grad_norm)
            self.teacher_optimizer.step()

        # critic mode uses phi(teacher_obs_t): both critic and student targets
        # are aligned to the current transition state, and terminal samples are
        # valid. Other legacy modes retain their t+1 target and done mask.
        if self.teacher_in_critic:
            student_target = self._last_critic_latent
            student_mask = torch.ones_like(valid_mask)
        else:
            student_target = z_next
            student_mask = valid_mask

        vel_loss = (student['v'] - policy_state_dict['base_vel_target']).pow(2).mean()
        latent_loss = recon_loss_masked(
            student['z'], student_target.detach(), student_mask)
        contrastive_loss = self._contrastive_loss(
            student['z'], student_target)
        latent_coef = self.latent_coef
        contrastive_coef = self.contrastive_coef
        if self.latent_coef_warmup_iters > 0:
            warm = min(1.0, self.current_learning_iteration
                       / self.latent_coef_warmup_iters)
            latent_coef *= warm
            contrastive_coef *= warm

        actor_loss = (surrogate_loss - self.entropy_coef * entropy_loss
                      + self.vel_coef * vel_loss + latent_coef * latent_loss
                      + contrastive_coef * contrastive_loss)
        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        if self.teacher_in_critic:
            self.teacher_optimizer.zero_grad()   # phi 는 critic_loss 로 학습된다
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        if self.teacher_in_critic:
            nn.utils.clip_grad_norm_(self.teacher.parameters(), self.max_grad_norm)
            self.teacher_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['vel_est'] += vel_loss.item()
        loss_dict['latent_match'] += latent_loss.item()
        loss_dict['contrastive'] += contrastive_loss.item()
        loss_dict['teacher_var'] += t_var.item()
        loss_dict['teacher_cov'] += t_cov.item()
        loss_dict['teacher_value'] += t_value.item()
        # 붕괴 진단: 차원별 std 의 평균 (0 에 붙으면 latent 이 상수로 붕괴)
        loss_dict['teacher_latent_std'] += student_target.std(dim=0).mean().item()
        return loss_dict

    # ---- checkpoint: frozen 모드도 teacher 가중치 저장 필수 (랜덤 초기값이 곧 타깃 함수) ----
    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        state = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'teacher_model_state_dict': self.teacher.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.teacher_optimizer is not None:
            state['teacher_optimizer_state_dict'] = self.teacher_optimizer.state_dict()
        if self.teacher_value_head is not None:
            state['teacher_value_head_state_dict'] = self.teacher_value_head.state_dict()
        torch.save(state, path)

    def load(self, ckpt_path):
        # v2.load 는 teacher_optimizer 를 무조건 참조하므로 frozen 모드에서 터진다 -> PPO.load 사용
        infos = PPO.load(self, ckpt_path)
        if ckpt_path is not None:
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            if 'teacher_model_state_dict' in loaded_dict:
                self.teacher.load_state_dict(loaded_dict['teacher_model_state_dict'])
                if (self.teacher_value_head is not None
                        and 'teacher_value_head_state_dict' in loaded_dict):
                    self.teacher_value_head.load_state_dict(
                        loaded_dict['teacher_value_head_state_dict'])
                if (self.teacher_optimizer is not None and self.load_optimizer
                        and 'teacher_optimizer_state_dict' in loaded_dict):
                    self.teacher_optimizer.load_state_dict(
                        loaded_dict['teacher_optimizer_state_dict'])
            else:
                logger.warning("checkpoint has no teacher weights — frozen 모드라면 "
                               "타깃 함수가 재초기화되어 latent_match 가 재학습된다")
        return infos
