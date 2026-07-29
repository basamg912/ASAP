'''
PPOLowerBody: 하체+허리(lower_body_actions_dim)만 policy 가 제어하는 PPO.

상체(나머지 DoF)는 action=0 을 넣어 PD 제어가 default_dof_pos 를 유지하도록 한다
(env 의 PD target = action * action_scale + default_dof_pos 이므로 action=0 == 기본 자세 유지).

전제: robot.dof_names 순서에서 하체 DoF 가 앞쪽 연속 블록일 것
(KAPEX: 다리 14 + 허리 3 = 앞 17, 팔 14 = 뒤 14 — kapex_31dof.yaml 참조).

PPODecoupled(agents/decouple/ppo_decoupled.py) 와 같은 구조이지만,
상체를 env.ref_upper_dof_pos 가 아닌 0 으로 고정하고 upper action scale 커리큘럼이 없다.
기존 PPO/env/deploy 코드는 수정하지 않는다 (additive).

Launch: +exp=locomotion_lowerbody (config/exp/locomotion_lowerbody.yaml → /algo: ppo_lowerbody)
'''
import time

import torch
from loguru import logger

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask


class PPOLowerBody(PPO):
    def __init__(self, env: BaseTask, config, log_dir=None, device='cpu'):
        super().__init__(env, config, log_dir, device)

    def _init_config(self):
        super()._init_config()
        self.num_act = self.env.config.robot.lower_body_actions_dim
        self.num_upper_act = self.env.config.robot.actions_dim - self.num_act
        dof_names = list(self.env.config.robot.dof_names)
        logger.info(
            f"PPOLowerBody: policy controls {self.num_act} DoF {dof_names[:self.num_act]}"
        )
        logger.info(
            f"PPOLowerBody: PD holds default pose for {self.num_upper_act} DoF "
            f"{dof_names[self.num_act:]}"
        )

    def setup(self):
        logger.info("Setting up PPOLowerBody")
        super().setup()

    def _compose_whole_body_actions(self, actions_lower_body):
        """하체 action 뒤에 상체 0 action 을 붙여 전신 action 을 만든다."""
        actions_upper_body = torch.zeros(
            actions_lower_body.shape[0], self.num_upper_act,
            dtype=actions_lower_body.dtype, device=actions_lower_body.device,
        )
        return torch.cat([actions_lower_body, actions_upper_body], dim=1)

    def _rollout_step(self, obs_dict):
        # PPO._rollout_step 와 동일하되 env.step 직전에 상체 0 action 을 붙인다.
        # (storage 에는 하체 action(num_act)만 저장되어 PPO update 는 하체만 학습)
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])
                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                actions = self._compose_whole_body_actions(policy_state_dict["actions"])
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
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

            # prepare data for training
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'),
                dones=self.storage.query_key('dones'),
                rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def env_step(self, actor_state):
        # eval/deploy 경로: actor_state["actions"] 는 하체(num_act)만 유지해
        # deploy_agent 수집 데이터에도 하체 action 만 기록된다.
        actions = self._compose_whole_body_actions(actor_state["actions"])
        obs_dict, rewards, dones, extras = self.env.step({"actions": actions})
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state
