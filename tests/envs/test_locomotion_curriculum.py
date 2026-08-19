from types import SimpleNamespace

import pytest
import torch

from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
from humanoidverse.envs.locomotion.locomotion_cmd_curriculum import (
    LeggedRobotLocomotionCmdCurriculum,
)
from humanoidverse.utils.torch_utils import quat_from_euler_xyz


def test_linear_tracking_uses_gravity_aligned_yaw_frame():
    env = object.__new__(LeggedRobotLocomotion)
    env.base_quat = quat_from_euler_xyz(
        torch.tensor([torch.pi / 2, 0.0]),
        torch.zeros(2),
        torch.tensor([0.0, torch.pi / 2]),
    )
    root_states = torch.zeros(2, 13)
    root_states[:, 7:10] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    env.simulator = SimpleNamespace(robot_root_states=root_states)
    env.commands = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    env.config = SimpleNamespace(
        rewards=SimpleNamespace(
            reward_tracking_sigma=SimpleNamespace(lin_vel=0.25)
        )
    )

    yaw_frame_velocity = env._get_base_lin_vel_yaw_frame()
    reward = env._reward_tracking_lin_vel()

    assert torch.allclose(
        yaw_frame_velocity,
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        atol=1e-6,
    )
    assert torch.allclose(reward, torch.ones(2), atol=1e-6)


def test_command_resampling_assigns_yaw_rate_directly():
    env = object.__new__(LeggedRobotLocomotion)
    env.num_envs = 4
    env.device = torch.device("cpu")
    env.commands = torch.zeros(4, 3)
    env.command_ranges = {
        "lin_vel_x": [0.2, 0.2],
        "lin_vel_y": [-0.1, -0.1],
        "ang_vel_yaw": [0.15, 0.15],
    }
    env.config = {"locomotion_stand_still_prob": 0.0}
    env.is_standing_env = torch.zeros(4, dtype=torch.bool)

    torch.manual_seed(0)
    env._resample_commands(torch.arange(4))

    assert torch.allclose(
        env.commands,
        torch.tensor([[0.2, -0.1, 0.15]]).repeat(4, 1),
    )
    assert not torch.any(env.is_standing_env)


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


def _make_reward_curriculum_env(score, common_step_counter=1000):
    env = object.__new__(LeggedRobotLocomotionCmdCurriculum)
    env.command_curriculum_reward_name = "tracking_lin_vel"
    env.command_curriculum_reward_weight = 1.5
    env.command_curriculum_reward_threshold = 0.8
    env.command_curriculum_range_step = 0.1
    env.command_curriculum_update_interval = 1000
    env.command_curriculum_last_update_step = -1
    env.command_tracking_reward_score = 0.0
    env.common_step_counter = common_step_counter
    env.max_episode_length_s = 20.0
    env.episode_sums = {
        "tracking_lin_vel": torch.full((4,), score * 20.0 * 1.5)
    }
    env.initial_command_ranges = {
        "lin_vel_x": [-0.1, 0.1],
        "lin_vel_y": [-0.1, 0.1],
        "ang_vel_yaw": [-0.1, 0.1],
    }
    env.command_ranges = {
        key: list(value) for key, value in env.initial_command_ranges.items()
    }
    env.final_command_ranges = {
        "lin_vel_x": [-0.5, 1.0],
        "lin_vel_y": [-0.3, 0.3],
        "ang_vel_yaw": [-0.2, 0.2],
    }
    env._refresh_command_curriculum_progress()
    return env


def test_command_curriculum_expands_xy_from_normalized_tracking_reward():
    env = _make_reward_curriculum_env(score=0.81)

    env._update_command_curriculum(torch.arange(4))

    assert env.command_tracking_reward_score == pytest.approx(0.81)
    assert env.command_ranges["lin_vel_x"] == pytest.approx([-0.2, 0.2])
    assert env.command_ranges["lin_vel_y"] == pytest.approx([-0.2, 0.2])
    assert env.command_ranges["ang_vel_yaw"] == pytest.approx([-0.1, 0.1])


def test_command_curriculum_requires_threshold_and_update_interval():
    below_threshold = _make_reward_curriculum_env(score=0.8)
    before_interval = _make_reward_curriculum_env(
        score=1.0, common_step_counter=999
    )

    below_threshold._update_command_curriculum(torch.arange(4))
    before_interval._update_command_curriculum(torch.arange(4))

    assert below_threshold.command_ranges["lin_vel_x"] == pytest.approx([-0.1, 0.1])
    assert before_interval.command_ranges["lin_vel_x"] == pytest.approx([-0.1, 0.1])


def test_command_curriculum_clamps_xy_and_keeps_yaw_fixed():
    env = _make_reward_curriculum_env(score=1.0)
    env.command_ranges["lin_vel_x"] = [-0.45, 0.95]
    env.command_ranges["lin_vel_y"] = [-0.25, 0.25]
    env.command_ranges["ang_vel_yaw"] = [-0.1, 0.1]

    env._update_command_curriculum(torch.arange(4))

    assert env.command_ranges["lin_vel_x"] == pytest.approx([-0.5, 1.0])
    assert env.command_ranges["lin_vel_y"] == pytest.approx([-0.3, 0.3])
    assert env.command_ranges["ang_vel_yaw"] == pytest.approx([-0.1, 0.1])
    assert env.command_curriculum_progress == pytest.approx(1.0)


def test_command_curriculum_score_uses_only_completed_env_ids():
    env = _make_reward_curriculum_env(score=0.2)
    env.episode_sums["tracking_lin_vel"][[1, 3]] = 0.9 * 20.0 * 1.5

    score = env._compute_command_tracking_reward_score(torch.tensor([1, 3]))

    assert score == pytest.approx(0.9)


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
