'''
LeggedRobotLocomotionCmdCurriculum: locomotion_command_ranges 를 학습 성과에 따라
점진 확장하는 command curriculum (unitree_rl_lab 의 lin_vel_cmd_levels 대응).

판정 지표:
  level-up:   average_episode_length > level_up_threshold
              AND tracking_lin_vel EMA > tracking_level_up_threshold  → progress += degree
  level-down: average_episode_length < level_down_threshold           → progress -= degree
  ([0, 1] 클립)
command_ranges = lerp(initial_ranges, locomotion_command_ranges, progress)

tracking 게이트: epl 만으로 판정하면 "서있기"로도 range 가 full 까지 확장됨
(20260730 G1 런 3회 연속 서있기 붕괴). walk 커맨드(norm>0.1) env 의
_reward_tracking_lin_vel 원값(exp, 가중치/dt 미적용, [0,1]) 평균을 EMA 로 유지해
정책이 현재 range 를 실제로 추종할 때만 확장한다 (unitree_rl_lab 이 tracking
reward 로 lin_vel_cmd_levels 를 올리는 것과 동일한 원리).
- EMA 는 1.0 (낙관) 초기화: 초기 range 에선 커맨드가 전부 0 으로 클립되어 walk
  env 가 없고, 이때 게이트는 열려 있어야 데드락이 없다. resume 시에도 ~수십 iter
  내 재수렴 (checkpoint 에 저장되지 않음).
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
        self.command_tracking_threshold = float(cc.get("tracking_level_up_threshold", 0.8))
        self.command_tracking_ema_alpha = float(cc.get("tracking_ema_alpha", 0.001))
        self.command_tracking_ema = 1.0  # 낙관 초기화 — walk env 가 없는 초기 range 에서 게이트 개방
        self._apply_command_curriculum()
        logger.info(
            f"Command curriculum enabled: progress={self.command_curriculum_progress:.4f}, "
            f"initial={self.initial_command_ranges}, final={self.final_command_ranges}, "
            f"degree={cc.degree}, thresholds=({cc.level_down_threshold}, {cc.level_up_threshold}), "
            f"tracking_gate>{self.command_tracking_threshold}"
        )

    def _apply_command_curriculum(self):
        p = min(max(self.command_curriculum_progress, 0.0), 1.0)
        self.command_curriculum_progress = p
        self.command_ranges = {
            k: [lo + (self.final_command_ranges[k][0] - lo) * p,
                hi + (self.final_command_ranges[k][1] - hi) * p]
            for k, (lo, hi) in self.initial_command_ranges.items()
        }

    def _update_command_curriculum(self):
        cc = self.command_curriculum_config
        if (self.average_episode_length > cc.level_up_threshold
                and self.command_tracking_ema > self.command_tracking_threshold):
            self.command_curriculum_progress += cc.degree
        elif self.average_episode_length < cc.level_down_threshold:
            self.command_curriculum_progress -= cc.degree
        self._apply_command_curriculum()

    def _update_reward_penalty_curriculum(self):
        # penalty curriculum 에도 같은 tracking 게이트: 서있기(생존)만으로 epl 이 높은
        # 동안 모션 페널티가 최대로 올라 걷기 시도를 고착 전에 차단하는 문제 방지
        # (20260730~31 G1 런들: gait 없이 penalty_scale 1.0 도달 → mean_reward 붕괴).
        # epl 낮음 → 완화(level-down)는 게이트와 무관하게 base 로직 그대로 동작.
        if (self.use_command_curriculum
                and self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold
                and self.command_tracking_ema <= self.command_tracking_threshold):
            return  # 추종을 못 하는 동안 penalty 인상 보류
        super()._update_reward_penalty_curriculum()

    def _reset_tasks_callback(self, env_ids):
        # range 갱신을 먼저 하고 super 를 호출해야 이번 리샘플부터 새 range 가 적용된다
        if self.use_command_curriculum and not self.is_evaluating and len(env_ids) > 0:
            self._update_command_curriculum()
        super()._reset_tasks_callback(env_ids)

    def _post_physics_step(self):
        super()._post_physics_step()
        if self.use_command_curriculum:
            # standstill 은 서있어도 tracking 만점이라 walk 커맨드 env 만으로 EMA 갱신
            if not self.is_evaluating:
                walking = torch.norm(self.commands[:, :2], dim=1) > 0.1
                if walking.any():
                    val = self._reward_tracking_lin_vel()[walking].mean().item()
                    a = self.command_tracking_ema_alpha
                    self.command_tracking_ema += a * (val - self.command_tracking_ema)
            self.log_dict["command_curriculum_progress"] = torch.tensor(
                self.command_curriculum_progress, dtype=torch.float)
            self.log_dict["command_lin_vel_x_max"] = torch.tensor(
                self.command_ranges["lin_vel_x"][1], dtype=torch.float)
            self.log_dict["command_tracking_lin_vel_ema"] = torch.tensor(
                self.command_tracking_ema, dtype=torch.float)
