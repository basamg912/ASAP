from types import SimpleNamespace

import pytest
import torch

from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
from humanoidverse.envs.locomotion.locomotion_cmd_curriculum import (
    LeggedRobotLocomotionCmdCurriculum,
)


def test_command_tracking_error_metrics_include_both_xy_axes():
    env = object.__new__(LeggedRobotLocomotionCmdCurriculum)
    env.last_episode_length_buf = torch.tensor([4, 2, 0])
    env.command_tracking_error_sum = torch.tensor([0.4, 0.4, 10.0])
    env.command_tracking_abs_error_sum = torch.tensor(
        [[0.2, 0.3], [0.1, 0.3], [10.0, 10.0]]
    )

    error, error_x, error_y = env._command_tracking_error_metrics(torch.arange(3))

    assert error == pytest.approx(0.15)
    assert error_x == pytest.approx(0.05)
    assert error_y == pytest.approx(0.1125)


def test_command_curriculum_prioritizes_level_down_when_epl_collapses():
    env = object.__new__(LeggedRobotLocomotionCmdCurriculum)
    env.command_curriculum_config = SimpleNamespace(
        degree=0.1, level_down_threshold=200
    )
    env.command_curriculum_bootstrap_epl = 600
    env.command_curriculum_bootstrap_until_progress = 0.2
    env.command_tracking_ema_alpha = 0.001
    env.command_tracking_error = 0.01
    env.command_tracking_error_x = 0.01
    env.command_tracking_error_y = 0.0
    env.command_tracking_error_initialized = True
    env.command_tracking_error_threshold = 0.06
    env.command_curriculum_progress = 0.5
    env.average_episode_length = 100
    env._command_tracking_error_metrics = lambda _env_ids: (0.01, 0.01, 0.0)
    env._apply_command_curriculum = lambda: None

    env._update_command_curriculum(torch.tensor([0]))

    assert env.command_curriculum_progress == pytest.approx(0.4)


def test_command_curriculum_bootstraps_from_epl_only_up_to_cap():
    env = object.__new__(LeggedRobotLocomotionCmdCurriculum)
    env.command_curriculum_config = SimpleNamespace(
        degree=0.1, level_down_threshold=200
    )
    env.command_curriculum_bootstrap_epl = 600
    env.command_curriculum_bootstrap_until_progress = 0.2
    env.command_tracking_ema_alpha = 1.0
    env.command_tracking_error = 0.5
    env.command_tracking_error_x = 0.5
    env.command_tracking_error_y = 0.0
    env.command_tracking_error_initialized = True
    env.command_tracking_error_threshold = 0.06
    env.command_curriculum_progress = 0.15
    env.average_episode_length = 700
    env._command_tracking_error_metrics = lambda _env_ids: (0.5, 0.5, 0.0)
    env._apply_command_curriculum = lambda: None

    env._update_command_curriculum(torch.tensor([0]))

    assert env.command_curriculum_progress == pytest.approx(0.2)


def test_command_curriculum_uses_tracking_error_after_bootstrap():
    env = object.__new__(LeggedRobotLocomotionCmdCurriculum)
    env.command_curriculum_config = SimpleNamespace(
        degree=0.1, level_down_threshold=200
    )
    env.command_curriculum_bootstrap_epl = 600
    env.command_curriculum_bootstrap_until_progress = 0.2
    env.command_tracking_ema_alpha = 1.0
    env.command_tracking_error = 0.5
    env.command_tracking_error_x = 0.5
    env.command_tracking_error_y = 0.0
    env.command_tracking_error_initialized = True
    env.command_tracking_error_threshold = 0.06
    env.command_curriculum_progress = 0.2
    env.average_episode_length = 700
    env._command_tracking_error_metrics = lambda _env_ids: (0.5, 0.5, 0.0)
    env._apply_command_curriculum = lambda: None

    env._update_command_curriculum(torch.tensor([0]))

    assert env.command_curriculum_progress == pytest.approx(0.2)

    env._command_tracking_error_metrics = lambda _env_ids: (0.05, 0.05, 0.0)
    env._update_command_curriculum(torch.tensor([0]))

    assert env.command_curriculum_progress == pytest.approx(0.3)


def test_gait_motion_gate_caps_reward_by_commanded_motion():
    env = object.__new__(LeggedRobotLocomotion)
    env.config = SimpleNamespace(
        rewards={
            "gait_gate_lin_vel_threshold": 0.1,
            "gait_gate_ang_vel_threshold": 0.1,
        }
    )
    env.commands = torch.tensor(
        [
            [0.02, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0],
            [0.20, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0],
            [0.00, 0.0, 0.05, 0.0],
            [0.10, 0.0, 0.0, 0.0],
        ]
    )
    env.base_lin_vel = torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.00, 0.0, 0.0],
            [0.00, 0.0, 0.0],
            [0.10, 0.0, 0.0],
        ]
    )
    env.base_ang_vel = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.10],  # unintended yaw cannot satisfy a linear command
            [0.0, 0.0, 0.10],
            [0.0, 0.0, 0.0],
        ]
    )
    env.is_standing_env = torch.tensor(
        [False, False, False, False, False, False, True]
    )

    gate = env._gait_motion_gate()

    assert torch.allclose(
        gate,
        torch.tensor([0.0, 0.2, 1.0, 0.5, 0.0, 0.5, 0.0]),
    )


def test_contact_reward_rejects_double_contact_during_single_support():
    env = object.__new__(LeggedRobotLocomotion)
    env.num_envs = 3
    env.device = torch.device("cpu")
    env.feet_indices = torch.tensor([0, 1])
    env.gait_stance_threshold = 0.55
    env.leg_phase = torch.tensor(
        [
            [0.25, 0.75],  # left stance, right swing
            [0.75, 0.25],  # left swing, right stance
            [0.02, 0.52],  # intended double support
        ]
    )
    env._gait_motion_gate = lambda: torch.ones(3)
    env.simulator = SimpleNamespace(contact_forces=torch.zeros(3, 2, 3))
    env.simulator.contact_forces[:, :, 2] = 2.0

    reward = env._reward_contact()

    assert torch.equal(reward, torch.tensor([0.0, 0.0, 2.0]))


def test_contact_reward_preserves_full_score_for_expected_pattern():
    env = object.__new__(LeggedRobotLocomotion)
    env.num_envs = 3
    env.device = torch.device("cpu")
    env.feet_indices = torch.tensor([0, 1])
    env.gait_stance_threshold = 0.55
    env.leg_phase = torch.tensor(
        [[0.25, 0.75], [0.75, 0.25], [0.02, 0.52]]
    )
    env._gait_motion_gate = lambda: torch.ones(3)
    contact_forces = torch.zeros(3, 2, 3)
    contact_forces[:, :, 2] = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    )
    env.simulator = SimpleNamespace(contact_forces=contact_forces)

    reward = env._reward_contact()

    assert torch.equal(reward, torch.full((3,), 2.0))
