'''Command-range curriculum for velocity locomotion.

This follows Unitree RL Lab's ``lin_vel_cmd_levels`` behavior:

- evaluate completed environments' normalized ``tracking_lin_vel`` episode
  reward once per horizon;
- expand only ``lin_vel_x`` and ``lin_vel_y`` by a fixed amount when the score
  exceeds the configured threshold;
- clamp both ranges to ``locomotion_command_ranges``;
- keep yaw at its initial range.

Dividing by the full episode duration also makes early termination lower the
score without a separate episode-length gate. Per-axis velocity errors remain
available as diagnostics, but do not control the curriculum.
'''
import torch
from loguru import logger

from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion


class LeggedRobotLocomotionCmdCurriculum(LeggedRobotLocomotion):
    _CURRICULUM_AXES = ("lin_vel_x", "lin_vel_y")
    _REWARD_THRESHOLD_EPS = 1e-6

    def _init_buffers(self):
        super()._init_buffers()
        cc = self.config.get("command_curriculum", None)
        self.use_command_curriculum = bool(cc is not None and cc.get("enabled", False))
        if not self.use_command_curriculum:
            logger.info("Command curriculum disabled: using static locomotion_command_ranges")
            return

        self.command_curriculum_config = cc
        self.final_command_ranges = {
            key: [float(value[0]), float(value[1])]
            for key, value in self.config.locomotion_command_ranges.items()
        }
        self.initial_command_ranges = {
            key: (
                [float(cc.initial_ranges[key][0]), float(cc.initial_ranges[key][1])]
                if key in cc.initial_ranges
                else list(value)
            )
            for key, value in self.final_command_ranges.items()
        }
        self.command_ranges = {
            key: list(value) for key, value in self.initial_command_ranges.items()
        }

        for axis in self._CURRICULUM_AXES:
            if axis not in self.command_ranges:
                raise ValueError(f"Missing command curriculum range: {axis}")
            initial_lo, initial_hi = self.initial_command_ranges[axis]
            final_lo, final_hi = self.final_command_ranges[axis]
            if not final_lo <= initial_lo <= initial_hi <= final_hi:
                raise ValueError(
                    f"Initial {axis} range must be inside its final range: "
                    f"initial={self.initial_command_ranges[axis]}, "
                    f"final={self.final_command_ranges[axis]}"
                )

        self.command_curriculum_reward_name = str(
            cc.get("reward_term_name", "tracking_lin_vel")
        )
        reward_scales = self.config.rewards.reward_scales
        if self.command_curriculum_reward_name not in reward_scales:
            raise ValueError(
                f"Command curriculum reward is disabled: "
                f"{self.command_curriculum_reward_name}"
            )
        self.command_curriculum_reward_weight = float(
            reward_scales[self.command_curriculum_reward_name]
        )
        if self.command_curriculum_reward_weight <= 0.0:
            raise ValueError("Command curriculum reward weight must be positive")

        self.command_curriculum_reward_threshold = float(
            cc.get("tracking_reward_threshold", 0.8)
        )
        if not 0.0 <= self.command_curriculum_reward_threshold <= 1.0:
            raise ValueError("tracking_reward_threshold must be in [0, 1]")

        self.command_curriculum_range_step = float(cc.get("range_step", 0.1))
        if self.command_curriculum_range_step <= 0.0:
            raise ValueError("range_step must be greater than zero")

        update_interval = cc.get("update_interval_steps", None)
        self.command_curriculum_update_interval = int(
            self.max_episode_length if update_interval is None else update_interval
        )
        if self.command_curriculum_update_interval <= 0:
            raise ValueError("update_interval_steps must be greater than zero")
        self.command_curriculum_last_update_step = -1
        self.command_tracking_reward_score = 0.0

        self.command_tracking_ema_alpha = float(
            cc.get("tracking_error_ema_alpha", 0.001)
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
        self._refresh_command_curriculum_progress()

        final_xy_ranges = {
            axis: self.final_command_ranges[axis]
            for axis in self._CURRICULUM_AXES
        }
        logger.info(
            f"Command curriculum enabled: initial={self.initial_command_ranges}, "
            f"final_xy={final_xy_ranges}, "
            f"reward={self.command_curriculum_reward_name}, "
            f"threshold={self.command_curriculum_reward_threshold}, "
            f"range_step={self.command_curriculum_range_step}, "
            f"update_interval={self.command_curriculum_update_interval} steps"
        )

    def _refresh_command_curriculum_progress(self):
        progress = []
        for axis in self._CURRICULUM_AXES:
            initial_lo, initial_hi = self.initial_command_ranges[axis]
            current_lo, current_hi = self.command_ranges[axis]
            final_lo, final_hi = self.final_command_ranges[axis]
            if initial_lo != final_lo:
                progress.append((initial_lo - current_lo) / (initial_lo - final_lo))
            if initial_hi != final_hi:
                progress.append((current_hi - initial_hi) / (final_hi - initial_hi))
        self.command_curriculum_progress = min(progress, default=1.0)
        self.command_curriculum_progress = min(
            max(self.command_curriculum_progress, 0.0), 1.0
        )

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

    def _update_command_tracking_error_diagnostics(self, env_ids):
        metrics = self._command_tracking_error_metrics(env_ids)
        if metrics is None:
            return
        error, error_x, error_y = metrics
        if self.command_tracking_error_initialized:
            alpha = self.command_tracking_ema_alpha
            self.command_tracking_error += alpha * (
                error - self.command_tracking_error
            )
            self.command_tracking_error_x += alpha * (
                error_x - self.command_tracking_error_x
            )
            self.command_tracking_error_y += alpha * (
                error_y - self.command_tracking_error_y
            )
        else:
            self.command_tracking_error = error
            self.command_tracking_error_x = error_x
            self.command_tracking_error_y = error_y
            self.command_tracking_error_initialized = True

    def _compute_command_tracking_reward_score(self, env_ids):
        """Return the Unitree-style normalized linear tracking score."""
        weighted_episode_reward = torch.mean(
            self.episode_sums[self.command_curriculum_reward_name][env_ids]
        )
        normalization = (
            self.max_episode_length_s * self.command_curriculum_reward_weight
        )
        return float((weighted_episode_reward / normalization).item())

    def _expand_linear_command_ranges(self):
        step = self.command_curriculum_range_step
        for axis in self._CURRICULUM_AXES:
            current_lo, current_hi = self.command_ranges[axis]
            final_lo, final_hi = self.final_command_ranges[axis]
            self.command_ranges[axis] = [
                max(current_lo - step, final_lo),
                min(current_hi + step, final_hi),
            ]
        self._refresh_command_curriculum_progress()

    def _update_command_curriculum(self, env_ids):
        step = int(self.common_step_counter)
        if step <= 0 or step % self.command_curriculum_update_interval != 0:
            return
        if step == self.command_curriculum_last_update_step:
            return
        self.command_curriculum_last_update_step = step

        self.command_tracking_reward_score = (
            self._compute_command_tracking_reward_score(env_ids)
        )
        if (
            self.command_tracking_reward_score
            > self.command_curriculum_reward_threshold + self._REWARD_THRESHOLD_EPS
        ):
            self._expand_linear_command_ranges()

    def _update_reward_penalty_curriculum(self):
        if (
            self.use_command_curriculum
            and self.average_episode_length
            > self.config.rewards.reward_penalty_level_up_threshold
            and self.command_tracking_reward_score
            <= self.command_curriculum_reward_threshold + self._REWARD_THRESHOLD_EPS
        ):
            return
        super()._update_reward_penalty_curriculum()

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.use_command_curriculum:
            velocity_error = self._get_lin_vel_tracking_error()
            self.command_tracking_error_sum += torch.norm(velocity_error, dim=1)
            self.command_tracking_abs_error_sum += torch.abs(velocity_error)

    def _reset_tasks_callback(self, env_ids):
        if self.use_command_curriculum and len(env_ids) > 0:
            if not self.is_evaluating:
                # episode_sums are still available here and are cleared after
                # this callback by reset_envs_idx().
                self._update_command_curriculum(env_ids)
            self._update_command_tracking_error_diagnostics(env_ids)
            self.command_tracking_error_sum[env_ids] = 0.0
            self.command_tracking_abs_error_sum[env_ids] = 0.0
        super()._reset_tasks_callback(env_ids)

    def _post_physics_step(self):
        super()._post_physics_step()
        if self.use_command_curriculum:
            self.log_dict["command_curriculum_progress"] = torch.tensor(
                self.command_curriculum_progress, dtype=torch.float
            )
            self.log_dict["command_tracking_reward_score"] = torch.tensor(
                self.command_tracking_reward_score, dtype=torch.float
            )
            for axis in self._CURRICULUM_AXES:
                self.log_dict[f"command_{axis}_min"] = torch.tensor(
                    self.command_ranges[axis][0], dtype=torch.float
                )
                self.log_dict[f"command_{axis}_max"] = torch.tensor(
                    self.command_ranges[axis][1], dtype=torch.float
                )
            self.log_dict["command_tracking_error"] = torch.tensor(
                self.command_tracking_error, dtype=torch.float
            )
            self.log_dict["command_tracking_error_x"] = torch.tensor(
                self.command_tracking_error_x, dtype=torch.float
            )
            self.log_dict["command_tracking_error_y"] = torch.tensor(
                self.command_tracking_error_y, dtype=torch.float
            )
