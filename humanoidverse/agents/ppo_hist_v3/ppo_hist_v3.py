'''
History Encoder V3 = V2(student-teacher) + contrastive projection heads.

  1. h(z_s): student latent 용 헤드 (actor 소유 -> actor_optimizer)
  2. g(z_t): teacher latent 용 헤드 (teacher 소유 -> teacher_optimizer)
  3. contrastive loss = 양방향 InfoNCE(h(z_s), g(t_mu.detach()))
     - teacher encoder 는 여전히 recon+KL 로만 학습 (detach)
     - h, g, student 가 contrastive gradient 를 받음
     - 대형 미니배치에서는 유효(non-done) 샘플 중 contrastive_batch_size 개만 서브샘플
  4. v2 는 무수정 유지 — 이 클래스는 PPOHistV2 상속 (additive)
  5. reconstruction 은 z_t 사용 유지 (TeacherVAE.forward 그대로)
  6. policy 입력은 z_s 유지 (v2 actor 경로 그대로)

기본값: latent_coef=0 (v2 의 MSE latent match 를 contrastive 로 대체; yaml 에서 병행 가능)

Launch: +exp=locomotion_hist_v3 +obs=loco/leggedloco_obs_history_encoder
'''
import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOCritic
from humanoidverse.agents.modules.ppo_hist_modules import vae_kl_loss, recon_loss_masked
from humanoidverse.agents.modules.ppo_hist_v3_modules import (
    PPOActorWithStudentEncoderContrastive, TeacherVAEContrastive, info_nce_loss,
)
from humanoidverse.agents.ppo_hist_v2.ppo_hist_v2 import PPOHistV2


class PPOHistV3(PPOHistV2):
    """PPOHistV2 + contrastive latent alignment (InfoNCE via projection heads)."""

    def _init_config(self):
        super()._init_config()
        self.contrastive_coef = self.config.contrastive_coef
        self.contrastive_temperature = float(self.config.get('contrastive_temperature', 0.1))
        self.contrastive_batch_size = int(self.config.get('contrastive_batch_size', 1024))

    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['contrastive'] = 0
        return loss_dict

    def _setup_models_and_optimizer(self):
        proj = self.config.projection_config
        self.actor = PPOActorWithStudentEncoderContrastive(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            encoder_config=self.config.encoder_config,
            latent_dim=self.latent_dim,
            vel_dim=self.vel_dim,
            encoder_struct=self.encoder_struct,
            detach_encoder_output=bool(self.config.get('detach_encoder_for_policy', True)),
            proj_dim=int(proj.proj_dim),
            proj_hidden_dims=tuple(proj.hidden_dims),
            proj_activation=proj.activation,
        ).to(self.device)
        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)
        self.teacher = TeacherVAEContrastive(
            obs_dim=self.algo_obs_dim_dict[self.recon_target_key],
            latent_dim=self.latent_dim,
            enc_hidden_dims=tuple(self.config.teacher_config.enc_hidden_dims),
            dec_hidden_dims=tuple(self.config.teacher_config.dec_hidden_dims),
            activation=self.config.teacher_config.activation,
            proj_dim=int(proj.proj_dim),
            proj_hidden_dims=tuple(proj.hidden_dims),
            proj_activation=proj.activation,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.teacher_optimizer = optim.Adam(
            self.teacher.parameters(), lr=float(self.config.teacher_config.learning_rate))

    def _contrastive_loss(self, z_s, t_mu, valid_mask):
        """유효(non-done) 샘플 서브샘플 위에서 InfoNCE(h(z_s), g(t_mu.detach()))."""
        valid_idx = torch.nonzero(valid_mask.reshape(-1) > 0).squeeze(-1)
        if valid_idx.numel() < 2:  # negative 가 없으면 계산 불가
            return torch.zeros((), device=z_s.device)
        if valid_idx.numel() > self.contrastive_batch_size:
            sel = torch.randperm(valid_idx.numel(), device=z_s.device)[: self.contrastive_batch_size]
            valid_idx = valid_idx[sel]
        p_s = self.actor.project_student_latent(z_s[valid_idx])
        p_t = self.teacher.project_teacher_latent(t_mu[valid_idx].detach())
        return info_nce_loss(p_s, p_t, self.contrastive_temperature)

    # ---- update: v2 의 _update_ppo 복제 + contrastive + optimizer 순서 변경 ----
    # (g 가 teacher_optimizer 소속이라 actor_loss.backward() 이후에 teacher step 필요)
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

        # ---- teacher (next-obs VAE): recon 은 z_t 사용 (스펙 5) ----
        target_next = policy_state_dict['next_obs_target']
        valid_mask = (~policy_state_dict['dones'].bool()).float()
        recon, t_mu, t_logvar = self.teacher(target_next)
        teacher_recon = recon_loss_masked(recon, target_next, valid_mask)
        teacher_kl = vae_kl_loss(t_mu, t_logvar, self.vae_free_bits)
        teacher_loss = self.recon_coef * teacher_recon + self.vae_beta * teacher_kl

        # ---- student supervised + contrastive ----
        student = self.actor.get_student_outputs()
        vel_loss = (student['v'] - policy_state_dict['base_vel_target']).pow(2).mean()
        latent_loss = recon_loss_masked(student['z'], t_mu.detach(), valid_mask)
        contrastive_loss = self._contrastive_loss(student['z'], t_mu, valid_mask)
        latent_coef = self.latent_coef
        contrastive_coef = self.contrastive_coef
        if self.latent_coef_warmup_iters > 0:
            warm = min(1.0, self.current_learning_iteration / self.latent_coef_warmup_iters)
            latent_coef *= warm
            contrastive_coef *= warm

        actor_loss = (surrogate_loss - self.entropy_coef * entropy_loss
                      + self.vel_coef * vel_loss + latent_coef * latent_loss
                      + contrastive_coef * contrastive_loss)
        critic_loss = self.value_loss_coef * value_loss

        # g(teacher 소유)가 actor_loss 에서 gradient 를 받으므로,
        # 두 backward 를 모두 끝낸 뒤 teacher step (v2 와 순서 다름)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.teacher.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.teacher_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['vel_est'] += vel_loss.item()
        loss_dict['latent_match'] += latent_loss.item()
        loss_dict['contrastive'] += contrastive_loss.item()
        loss_dict['teacher_recon'] += teacher_recon.item()
        loss_dict['teacher_kl'] += teacher_kl.item()
        loss_dict['teacher_latent_std'] += torch.exp(0.5 * t_logvar).mean().item()
        return loss_dict
