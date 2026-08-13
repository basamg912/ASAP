'''
LeggedRobotLocomotionCmdCurriculum: locomotion_command_ranges 를 학습 성과에 따라
점진 확장하는 command curriculum (unitree_rl_lab 의 lin_vel_cmd_levels 대응).

판정 지표:
  level-up:   tracking score > tracking_level_up_threshold  → progress += degree
  level-down: average_episode_length < level_down_threshold → progress -= degree
  ([0, 1] 클립)
command_ranges = lerp(initial_ranges, locomotion_command_ranges, progress)

tracking score 는 unitree_rl_lab / Isaac Lab 의 lin_vel_cmd_levels 와 동일한
지표다: episode_sums["tracking_lin_vel"][env_ids].mean() / max_episode_length_s
(legged_robot_base.py:388 의 Episode/rew_* 로깅과 같은 식). weight 로 나눠 [0,1]
로 정규화하므로 threshold 는 "최대 추종 성능 대비 비율" 로 읽는다.

이 지표는 분모가 max_episode_length_s 로 고정이라 조기 종료한 env 는 분자만
줄어 점수가 자동으로 낮아진다 — 생존이 지표에 내장되어 있어 epl 조건을 AND 로
따로 걸 필요가 없다.
- 과거 설계는 epl 과 tracking EMA 를 AND 로 묶었는데, 두 값이 독립적으로 움직여
  교집합이 생기지 않는 데드락이 있었다 (20260813 런: 낙관 초기화한 EMA 가
  epl<400 인 초반에 소진되고, epl 이 400 을 넘었을 땐 EMA 가 이미 무너져 있어
  1000+ iter 동안 progress 가 정확히 0). 단일 조건은 이 실패가 원리적으로 불가능.
- 낙관 초기화도 불필요해졌다: 지표가 전체 env 기준이라 walk env 가 없는 초기
  range 에서도 항상 정의된다. 0.0 에서 시작해 ~수십 iter 내 실제값으로 수렴.
- standstill 커맨드 env 는 tracking 만점(~1.0)이라 평균을 조금 끌어올린다. 정지
  비율 f 에서 walk env 는 (threshold - f)/(1 - f) 를 넘겨야 하므로
  (f=0.2, threshold=0.8 → 0.75) 여전히 실제 보행 성능을 요구한다.
  epl 만으로 판정해 "서있기" 로 range 가 확장되던 문제(20260730 G1 런 3회 연속
  붕괴)는 tracking 기반 판정 자체로 막힌다.
- reset 마다 얻는 에피소드 평균을 EMA 로 한 번 더 평활한다. Isaac Lab 은 그 순간
  리셋되는 env 부분집합을 그대로 쓰는데, 표본 잡음이 커서 EMA 를 유지한다.
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
        # 지표가 전체 env 기준이라 항상 정의된다 → 낙관 초기화 불필요
        self.command_tracking_score = 0.0
        self._apply_command_curriculum()
        logger.info(
            f"Command curriculum enabled: progress={self.command_curriculum_progress:.4f}, "
            f"initial={self.initial_command_ranges}, final={self.final_command_ranges}, "
            f"degree={cc.degree}, level_down_epl<{cc.level_down_threshold}, "
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

    def _command_tracking_score(self, env_ids):
        """직전 에피소드의 tracking_lin_vel 시간평균을 [0,1] 로 정규화해 반환.

        episode_sums 는 raw × weight × dt 누적(legged_robot_base.py:146, :500)이라
        max_episode_length_s 로 나누면 weight 배율의 시간평균이 된다. weight 로 한 번
        더 나눠 정규화한다. 호출 시점(_reset_tasks_callback)은 episode_sums 가 0 으로
        초기화되는 reset_envs_idx:389 보다 앞이라 직전 에피소드 값이 살아 있다.
        """
        if "tracking_lin_vel" not in self.reward_scales:
            return None
        weight = float(self.reward_scales["tracking_lin_vel"]) / self.dt
        if weight <= 0.0:
            return None
        mean_sum = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]).item()
        return mean_sum / self.max_episode_length_s / weight

    def _update_command_curriculum(self, env_ids):
        cc = self.command_curriculum_config
        score = self._command_tracking_score(env_ids)
        if score is not None:
            a = self.command_tracking_ema_alpha
            self.command_tracking_score += a * (score - self.command_tracking_score)

        # 생존은 지표에 내장되어 있으므로 level-up 은 단일 조건.
        # level-down 은 붕괴 시 빠른 완화를 위해 epl 기준 유지 (Isaac Lab 엔 없는 안전장치).
        if self.command_tracking_score > self.command_tracking_threshold:
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
                and self.command_tracking_score <= self.command_tracking_threshold):
            return  # 추종을 못 하는 동안 penalty 인상 보류
        super()._update_reward_penalty_curriculum()

    def _reset_tasks_callback(self, env_ids):
        # range 갱신을 먼저 하고 super 를 호출해야 이번 리샘플부터 새 range 가 적용된다.
        # super 안에서 _update_reward_penalty_curriculum 이 돌므로 score 도 여기서 먼저 갱신된다.
        if self.use_command_curriculum and not self.is_evaluating and len(env_ids) > 0:
            self._update_command_curriculum(env_ids)
        super()._reset_tasks_callback(env_ids)

    def _post_physics_step(self):
        super()._post_physics_step()
        if self.use_command_curriculum:
            # score 는 에피소드 단위 지표라 reset 시점(_update_command_curriculum)에만 갱신된다
            self.log_dict["command_curriculum_progress"] = torch.tensor(
                self.command_curriculum_progress, dtype=torch.float)
            self.log_dict["command_lin_vel_x_max"] = torch.tensor(
                self.command_ranges["lin_vel_x"][1], dtype=torch.float)
            self.log_dict["command_tracking_score"] = torch.tensor(
                self.command_tracking_score, dtype=torch.float)
