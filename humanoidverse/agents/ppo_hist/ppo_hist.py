import time
import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.ppo_modules import PPOCritic
from humanoidverse.agents.modules.ppo_hist_modules import (
    PPOActorWithHistoryEncoder, vae_kl_loss, recon_loss_masked,
)


class PPOHistoryEncoder(PPO):
    """PPO + concurrent VAE history encoder (RMA-style, end-to-end, joint grad)."""

    # ---- config passthrough ----
    def _init_config(self):
        super()._init_config()
        self.latent_dim = self.config.latent_dim
        self.vae_beta = self.config.vae_beta
        self.recon_coef = self.config.recon_coef
        self.encoder_obs_key = self.config.encoder_obs_key
        self.recon_target_key = self.config.recon_target_key

    # ---- add VAE keys so _training_step averages + _logging_to_writer emits them ----
    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['recon'] = 0
        loss_dict['vae_kl'] = 0
        loss_dict['latent_std'] = 0
        return loss_dict

    # ---- models: actor carries encoder+decoder ----
    def _setup_models_and_optimizer(self):
        self.actor = PPOActorWithHistoryEncoder(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            encoder_config=self.config.encoder_config,
            decoder_config=self.config.decoder_config,
            latent_dim=self.latent_dim,
        ).to(self.device)
        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    # ---- storage: add shifted next-obs reconstruction target ----
    def _setup_storage(self):
        super()._setup_storage()
        recon_dim = self.algo_obs_dim_dict[self.recon_target_key]
        self.storage.register_key('next_obs_target', shape=(recon_dim,), dtype=torch.float)

    # ---- act uses both actor_obs and encoder_obs ----
    def _actor_act_step(self, obs_dict):
        return self.actor.act(obs_dict["actor_obs"], obs_dict[self.encoder_obs_key])

    # ---- rollout: copy of base PPO._rollout_step + next_obs_target capture ----
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

                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # ---- ADDED: next-step reconstruction target for the step just stored ----
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

    # ---- update: copy of base PPO._update_ppo + recon + KL terms ----
    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)  # recomputes distribution + stashes latent
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

        # ---- ADDED: VAE reconstruction + KL (joint grad; latent not detached) ----
        latent = self.actor.get_latent_stats()
        pred_next = self.actor.predict_next_state()
        target_next = policy_state_dict['next_obs_target']
        valid_mask = (~policy_state_dict['dones'].bool()).float()
        recon_loss = recon_loss_masked(pred_next, target_next, valid_mask)
        kl_vae = vae_kl_loss(latent['mu'], latent['logvar'])

        actor_loss = (surrogate_loss - self.entropy_coef * entropy_loss
                      + self.recon_coef * recon_loss + self.vae_beta * kl_vae)
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
        loss_dict['recon'] = loss_dict.get('recon', 0.0) + recon_loss.item()
        loss_dict['vae_kl'] = loss_dict.get('vae_kl', 0.0) + kl_vae.item()
        loss_dict['latent_std'] = loss_dict.get('latent_std', 0.0) + \
            torch.exp(0.5 * latent['logvar']).mean().item()
        return loss_dict
