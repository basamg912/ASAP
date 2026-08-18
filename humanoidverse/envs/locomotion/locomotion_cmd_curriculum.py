'''
LeggedRobotLocomotionCmdCurriculum: locomotion_command_ranges 를 학습 성과에 따라
점진 확장하는 command curriculum (unitree_rl_lab 의 lin_vel_cmd_levels 대응).

판정 지표:
  bootstrap:  progress < bootstrap_until_progress 이고 EPL >= bootstrap_level_up_epl
              → progress 를 bootstrap 상한까지 확장
  level-up:   bootstrap 이후 XY velocity error < tracking_error_level_up_threshold
              → progress += degree
  level-down: average_episode_length < level_down_threshold → progress -= degree
  ([0, 1] 클립)
command_ranges = lerp(initial_ranges, locomotion_command_ranges, progress)

XY velocity error 는 매 스텝 body-frame command와 base linear velocity의 L2 오차를
누적하고 실제 episode 길이로 나눈 값이다. 지수 tracking reward를 gate로 재사용하지
않으므로 작은 속도 명령에서 정지 정책이 높은 점수를 받는 문제가 없고, vy 오차도
동일한 gate에 포함된다. 축별 MAE도 진단용으로 함께 기록한다.

- reset 마다 얻는 episode 평균을 EMA 로 한 번 더 평활한다.
- 첫 유효 episode 값으로 EMA 를 초기화해 0에서 천천히 상승하는 낙관 편향을 막는다.
- 초기에는 작은 command의 tracking reward 변별력이 약하므로 EPL로
  bootstrap_until_progress까지만 확장한다. 그 이후는 다시 실제 XY 추종
  오차로만 확장해 서 있기 정책이 전 범위를 통과하는 것을 막는다.
- average_episode_length가 level_down_threshold보다 낮으면 error보다 level-down을
  우선한다. 짧게 생존한 episode가 우연히 작은 속도 오차를 내고 level-up하는 것을
  막기 위한 붕괴 안전장치다.
- 같은 게이트를 reward penalty curriculum 의 level-up 에도 적용한다
  (_update_reward_penalty_curriculum 오버라이드): gait 를 배우기 전에 모션
  페널티가 최대에 도달해 서있기를 고착시키는 것을 막는다. level-down 은 그대로.

- 최종 범위는 기존 키 locomotion_command_ranges 그대로 사용
- initial_ranges 에 없는 키(예: heading)는 처음부터 최종 범위 사용
- resume 시 initial_progress 로 진행도를 이어서 시작 (checkpoint 에 저장되지 않음)
- eval(set_is_evaluating) 중에는 갱신하지 않음

기존 파이프라인 무수정 (additive): config/env/locomotion_cmd_curriculum.yaml 이
_target_ 만 이 클래스로 교체한다. 사용: train_agent.py ... env=locomotion_cmd_curriculum
'''
import torch
from loguru import logger

from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion


class LeggedRobotLocomotionCmdCurriculum(LeggedRobotLocomotion):
    def _init_buffers(self):
        super()._init_buffers()
        cc = self.config.get("command_curriculum", None)
        self.use_command_curriculum = bool(cc is not None and cc.get("enabled", False))
        if not self.use_command_curriculum:
            logger.info("Command curriculum disabled: using static locomotion_command_ranges")
            return
        self.command_curriculum_config = cc
        # config 객체는 건드리지 않고 plain dict 로 소유한다
        # (self.command_ranges 소비처: _resample_commands, heading→yaw clip)
        self.final_command_ranges = {
            k: [float(v[0]), float(v[1])]
            for k, v in self.config.locomotion_command_ranges.items()
        }
        self.initial_command_ranges = {
            k: ([float(cc.initial_ranges[k][0]), float(cc.initial_ranges[k][1])]
                if k in cc.initial_ranges else list(v))
            for k, v in self.final_command_ranges.items()
        }
        self.command_curriculum_progress = float(cc.get("initial_progress", 0.0))
        self.command_curriculum_bootstrap_epl = float(
            cc.get("bootstrap_level_up_epl", 600)
        )
        self.command_curriculum_bootstrap_until_progress = float(
            cc.get("bootstrap_until_progress", 0.2)
        )
        if not 0.0 <= self.command_curriculum_bootstrap_until_progress <= 1.0:
            raise ValueError("bootstrap_until_progress must be in [0, 1]")
        if self.command_curriculum_bootstrap_epl <= float(cc.level_down_threshold):
            raise ValueError(
                "bootstrap_level_up_epl must be greater than level_down_threshold"
            )
        self.command_tracking_error_threshold = float(
            cc.get("tracking_error_level_up_threshold", 0.06)
        )
        self.command_tracking_ema_alpha = float(
            cc.get("tracking_error_ema_alpha", cc.get("tracking_ema_alpha", 0.001))
        )
        self.command_tracking_error = 0.0
        self.command_tracking_error_x = 0.0
        self.command_tracking_error_y = 0.0
        self.command_tracking_error_initialized = False
        self.command_tracking_error_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.command_tracking_abs_error_sum = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        self._apply_command_curriculum()
        logger.info(
            f"Command curriculum enabled: progress={self.command_curriculum_progress:.4f}, "
            f"initial={self.initial_command_ranges}, final={self.final_command_ranges}, "
            f"degree={cc.degree}, level_down_epl<{cc.level_down_threshold}, "
            f"bootstrap_epl>={self.command_curriculum_bootstrap_epl} "
            f"until_progress={self.command_curriculum_bootstrap_until_progress}, "
            f"velocity_error_gate<{self.command_tracking_error_threshold} m/s"
        )

    def _apply_command_curriculum(self):
        p = min(max(self.command_curriculum_progress, 0.0), 1.0)
        self.command_curriculum_progress = p
        self.command_ranges = {
            k: [lo + (self.final_command_ranges[k][0] - lo) * p,
                hi + (self.final_command_ranges[k][1] - hi) * p]
            for k, (lo, hi) in self.initial_command_ranges.items()
        }

    def _command_tracking_error_metrics(self, env_ids):
        """Return mean episode XY L2 error and per-axis MAE in m/s."""
        episode_lengths = self.last_episode_length_buf[env_ids].float()
        valid = episode_lengths > 0
        if not torch.any(valid):
            return None

        valid_ids = env_ids[valid]
        lengths = episode_lengths[valid]
        l2_error = self.command_tracking_error_sum[valid_ids] / lengths
        axis_mae = self.command_tracking_abs_error_sum[valid_ids] / lengths.unsqueeze(1)
        return (
            torch.mean(l2_error).item(),
            torch.mean(axis_mae[:, 0]).item(),
            torch.mean(axis_mae[:, 1]).item(),
        )

    def _update_command_curriculum(self, env_ids):
        cc = self.command_curriculum_config
        metrics = self._command_tracking_error_metrics(env_ids)
        if metrics is not None:
            error, error_x, error_y = metrics
            if self.command_tracking_error_initialized:
                a = self.command_tracking_ema_alpha
                self.command_tracking_error += a * (error - self.command_tracking_error)
                self.command_tracking_error_x += a * (error_x - self.command_tracking_error_x)
                self.command_tracking_error_y += a * (error_y - self.command_tracking_error_y)
            else:
                self.command_tracking_error = error
                self.command_tracking_error_x = error_x
                self.command_tracking_error_y = error_y
                self.command_tracking_error_initialized = True

        # 붕괴 중인 짧은 episode는 다른 gate보다 level-down을 우선한다.
        if self.average_episode_length < cc.level_down_threshold:
            self.command_curriculum_progress -= cc.degree
        elif (self.command_curriculum_progress
              < self.command_curriculum_bootstrap_until_progress):
            if self.average_episode_length >= self.command_curriculum_bootstrap_epl:
                self.command_curriculum_progress = min(
                    self.command_curriculum_progress + cc.degree,
                    self.command_curriculum_bootstrap_until_progress,
                )
        elif (self.command_tracking_error_initialized
              and self.command_tracking_error < self.command_tracking_error_threshold):
            self.command_curriculum_progress += cc.degree
        self._apply_command_curriculum()

    def _update_reward_penalty_curriculum(self):
        # penalty curriculum 에도 같은 tracking 게이트: 서있기(생존)만으로 epl 이 높은
        # 동안 모션 페널티가 최대로 올라 걷기 시도를 고착 전에 차단하는 문제 방지
        # (20260730~31 G1 런들: gait 없이 penalty_scale 1.0 도달 → mean_reward 붕괴).
        # epl 낮음 → 완화(level-down)는 게이트와 무관하게 base 로직 그대로 동작.
        if (self.use_command_curriculum
                and self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold
                and (not self.command_tracking_error_initialized
                     or self.command_tracking_error >= self.command_tracking_error_threshold)):
            return  # 추종을 못 하는 동안 penalty 인상 보류
        super()._update_reward_penalty_curriculum()

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.use_command_curriculum:
            velocity_error = self.commands[:, :2] - self.base_lin_vel[:, :2]
            self.command_tracking_error_sum += torch.norm(velocity_error, dim=1)
            self.command_tracking_abs_error_sum += torch.abs(velocity_error)

    def _reset_tasks_callback(self, env_ids):
        # range 갱신을 먼저 하고 super 를 호출해야 이번 리샘플부터 새 range 가 적용된다.
        # super 안에서 _update_reward_penalty_curriculum 이 돌므로 error도 여기서 먼저 갱신된다.
        if self.use_command_curriculum and not self.is_evaluating and len(env_ids) > 0:
            self._update_command_curriculum(env_ids)
        if self.use_command_curriculum and len(env_ids) > 0:
            self.command_tracking_error_sum[env_ids] = 0.0
            self.command_tracking_abs_error_sum[env_ids] = 0.0
        super()._reset_tasks_callback(env_ids)

    def _post_physics_step(self):
        super()._post_physics_step()
        if self.use_command_curriculum:
            # error는 에피소드 단위 지표라 reset 시점(_update_command_curriculum)에만 갱신된다.
            self.log_dict["command_curriculum_progress"] = torch.tensor(
                self.command_curriculum_progress, dtype=torch.float)
            self.log_dict["command_curriculum_bootstrap_active"] = torch.tensor(
                self.command_curriculum_progress
                < self.command_curriculum_bootstrap_until_progress,
                dtype=torch.float,
            )
            self.log_dict["command_lin_vel_x_max"] = torch.tensor(
                self.command_ranges["lin_vel_x"][1], dtype=torch.float)
            self.log_dict["command_tracking_error"] = torch.tensor(
                self.command_tracking_error, dtype=torch.float)
            self.log_dict["command_tracking_error_x"] = torch.tensor(
                self.command_tracking_error_x, dtype=torch.float)
            self.log_dict["command_tracking_error_y"] = torch.tensor(
                self.command_tracking_error_y, dtype=torch.float)
