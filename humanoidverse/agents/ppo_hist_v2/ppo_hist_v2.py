'''
History Encoder V2: student-teacher concurrent distillation (DreamWaQ CENet 분해형).

  - Student (배포 사용): [History] -> MLP-mixer -> [v_base(3), latent z]
  - Teacher (학습 전용): [next obs] -> VAE -> latent -> decoder -> next obs recon
  - Loss:
      teacher : recon_coef * recon(next_obs) + vae_beta * KL   (자체 optimizer)
      student : vel_coef * MSE(v_hat, GT base_lin_vel)
              + latent_coef * MSE(z_student, teacher_mu.detach())   (collapse 방지 detach)
      policy  : PPO (detach_encoder_output=True 면 encoder 로 PPO gradient 차단)

타이밍 정렬 (rollout 캡처 위치가 서로 다름에 주의):
  base_vel_target : env.step "이전" self.env.base_lin_vel  (상태 s_t, student 입력 history<=t 와 정렬)
  next_obs_target : env.step "이후" obs[recon_target_key]  (o_{t+1}, teacher 입력)
  dones 마스크로 에피소드 경계(o_{t+1} 이 리셋 후 obs 인 스텝)를 latent/teacher loss 에서 제외.

Launch: +exp=locomotion_hist_v2 +obs=loco/leggedloco_obs_history_encoder
'''
import time

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.ppo_modules import PPOCritic
from humanoidverse.agents.modules.ppo_hist_modules import vae_kl_loss, recon_loss_masked
from humanoidverse.agents.modules.ppo_hist_v2_modules import (
    PPOActorWithStudentEncoder, TeacherVAE,
)


class PPOHistV2(PPO):
    """PPO + student(v, z) / teacher(next-obs VAE) concurrent distillation."""

    # ---- config passthrough ----
    def _init_config(self):
        super()._init_config()
        self.latent_dim = self.config.latent_dim
        self.vel_dim = int(self.config.get('vel_dim', 3))
        self.vel_coef = self.config.vel_coef
        self.latent_coef = self.config.latent_coef
        self.latent_coef_warmup_iters = int(self.config.get('latent_coef_warmup_iters', 0))
        self.recon_coef = self.config.recon_coef
        self.vae_beta = self.config.vae_beta
        self.vae_free_bits = self.config.get('vae_free_bits', 0.0)
        self.encoder_obs_key = self.config.encoder_obs_key
        self.recon_target_key = self.config.recon_target_key

        obs_cfg = self.env.config.obs
        aux = {}
        for comp in obs_cfg.obs_dict[self.encoder_obs_key]:
            if comp in obs_cfg.obs_auxiliary:
                aux.update(dict(obs_cfg.obs_auxiliary[comp]))
        keys = sorted(aux.keys())
        lengths = [aux[k] for k in keys]
        assert len(set(lengths)) == 1, "encoder history lengths differ"
        self.encoder_struct = {
            "keys": keys,
            "num_keys": len(keys),
            "history_length": lengths[0],
            "key_dims": [obs_cfg.obs_dims[k] for k in keys],
            "per_step_dim": sum(obs_cfg.obs_dims[k] for k in keys),
        }

    # ---- loss keys: _training_step 평균 + _logging_to_writer 출력용 ----
    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['vel_est'] = 0
        loss_dict['latent_match'] = 0
        loss_dict['teacher_recon'] = 0
        loss_dict['teacher_kl'] = 0
        loss_dict['teacher_latent_std'] = 0
        return loss_dict

    # ---- models: actor 는 student 만 보유, teacher 는 algo 소유 (배포에 미포함) ----
    def _setup_models_and_optimizer(self):
        self.actor = PPOActorWithStudentEncoder(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            encoder_config=self.config.encoder_config,
            latent_dim=self.latent_dim,
            vel_dim=self.vel_dim,
            encoder_struct=self.encoder_struct,
            detach_encoder_output=bool(self.config.get('detach_encoder_for_policy', True)),
        ).to(self.device)
        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)
        self.teacher = TeacherVAE(
            obs_dim=self.algo_obs_dim_dict[self.recon_target_key],
            latent_dim=self.latent_dim,
            enc_hidden_dims=tuple(self.config.teacher_config.enc_hidden_dims),
            dec_hidden_dims=tuple(self.config.teacher_config.dec_hidden_dims),
            activation=self.config.teacher_config.activation,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.teacher_optimizer = optim.Adam(
            self.teacher.parameters(), lr=float(self.config.teacher_config.learning_rate))

    # ---- storage: recon 타깃(o_{t+1}) + base velocity GT(s_t) ----
    def _setup_storage(self):
        super()._setup_storage()
        recon_dim = self.algo_obs_dim_dict[self.recon_target_key]
        self.storage.register_key('next_obs_target', shape=(recon_dim,), dtype=torch.float)
        self.storage.register_key('base_vel_target', shape=(self.vel_dim,), dtype=torch.float)

    # ---- act uses both actor_obs and encoder_obs ----
    def _actor_act_step(self, obs_dict):
        return self.actor.act(obs_dict["actor_obs"], obs_dict[self.encoder_obs_key])

    # ---- rollout: base PPO._rollout_step 복제 + 두 타깃 캡처 ----
    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])
                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                # ---- ADDED: v_base GT 는 env.step "이전"(상태 s_t) 에 캡처 ----
                self.storage.update_key('base_vel_target',
                                        self.env.base_lin_vel.detach().clone())

                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # ---- ADDED: teacher 입력(o_{t+1}) 은 env.step "이후" 캡처 ----
                self.storage.update_key('next_obs_target',
                                        obs_dict[self.recon_target_key])

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'),
                                       dones=self.storage.query_key('dones'),
                                       rewards=self.storage.query_key('rewards')))
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)
        return obs_dict

    # ---- update: base PPO._update_ppo 복제 + teacher/student loss ----
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

        # ---- ADDED: teacher (next-obs VAE) — 자체 optimizer 로 먼저 학습 ----
        target_next = policy_state_dict['next_obs_target']
        valid_mask = (~policy_state_dict['dones'].bool()).float()
        recon, t_mu, t_logvar = self.teacher(target_next)
        teacher_recon = recon_loss_masked(recon, target_next, valid_mask)
        teacher_kl = vae_kl_loss(t_mu, t_logvar, self.vae_free_bits)
        teacher_loss = self.recon_coef * teacher_recon + self.vae_beta * teacher_kl
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        nn.utils.clip_grad_norm_(self.teacher.parameters(), self.max_grad_norm)
        self.teacher_optimizer.step()

        # ---- ADDED: student supervised (teacher latent 은 detach — collapse 방지) ----
        student = self.actor.get_student_outputs()
        vel_loss = (student['v'] - policy_state_dict['base_vel_target']).pow(2).mean()
        latent_loss = recon_loss_masked(student['z'], t_mu.detach(), valid_mask)
        latent_coef = self.latent_coef
        if self.latent_coef_warmup_iters > 0:
            latent_coef *= min(1.0, self.current_learning_iteration
                               / self.latent_coef_warmup_iters)

        actor_loss = (surrogate_loss - self.entropy_coef * entropy_loss
                      + self.vel_coef * vel_loss + latent_coef * latent_loss)
        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['vel_est'] += vel_loss.item()
        loss_dict['latent_match'] += latent_loss.item()
        loss_dict['teacher_recon'] += teacher_recon.item()
        loss_dict['teacher_kl'] += teacher_kl.item()
        loss_dict['teacher_latent_std'] += torch.exp(0.5 * t_logvar).mean().item()
        return loss_dict

    # ---- checkpoint: teacher 포함 (resume 용; 배포/평가에는 불필요) ----
    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'teacher_model_state_dict': self.teacher.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'teacher_optimizer_state_dict': self.teacher_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, ckpt_path):
        infos = super().load(ckpt_path)
        if ckpt_path is not None:
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            if 'teacher_model_state_dict' in loaded_dict:
                self.teacher.load_state_dict(loaded_dict['teacher_model_state_dict'])
                if self.load_optimizer and 'teacher_optimizer_state_dict' in loaded_dict:
                    self.teacher_optimizer.load_state_dict(
                        loaded_dict['teacher_optimizer_state_dict'])
            else:
                logger.warning("checkpoint has no teacher weights (eval-only or v1 ckpt)")
        return infos

    # ---- eval/deploy: student 만 사용 (base 는 actor_obs 만 전달하므로 오버라이드) ----
    def _get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference

    def _pre_eval_env_step(self, actor_state):
        obs = actor_state["obs"]
        actions = self.eval_policy(obs['actor_obs'], obs[self.encoder_obs_key])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
