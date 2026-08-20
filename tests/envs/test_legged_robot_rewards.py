from types import SimpleNamespace

import torch

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def test_energy_matches_absolute_joint_mechanical_power():
    env = object.__new__(LeggedRobotBase)
    env.simulator = SimpleNamespace(
        dof_vel=torch.tensor([[2.0, -3.0, 0.0], [-1.0, 4.0, -2.0]])
    )
    env.torques = torch.tensor([[-5.0, 2.0, 7.0], [3.0, -0.5, -4.0]])

    energy = env._reward_energy()

    assert torch.equal(energy, torch.tensor([16.0, 13.0]))


def test_dof_acceleration_reward_uses_simulator_physics_step_acceleration():
    env = object.__new__(LeggedRobotBase)
    env.simulator = SimpleNamespace(
        dof_vel=torch.tensor([[100.0, 100.0]]),
        dof_acc=torch.tensor([[2.0, -3.0]]),
    )
    env.last_dof_vel = torch.zeros(1, 2)
    env.dt = 0.02

    reward = env._reward_penalty_dof_acc()

    assert torch.equal(reward, torch.tensor([13.0]))


def test_slippage_uses_xy_speed_and_max_contact_over_three_frames():
    env = object.__new__(LeggedRobotBase)
    env.feet_indices = torch.tensor([0, 1])
    contact_history = torch.zeros(1, 3, 2, 3)
    contact_history[0, 1, 0, 2] = 2.0
    contact_history[0, 2, 1, 2] = 2.0
    env.simulator = SimpleNamespace(
        _rigid_body_vel=torch.tensor([[[3.0, 4.0, 100.0], [1.0, 0.0, 100.0]]]),
        contact_forces=torch.zeros(1, 2, 3),
        contact_forces_history=contact_history,
    )

    reward = env._reward_penalty_slippage()

    assert torch.equal(reward, torch.tensor([6.0]))


def test_undesired_contacts_counts_non_ankle_bodies_over_three_frames():
    env = object.__new__(LeggedRobotBase)
    env.num_envs = 1
    env.device = "cpu"
    # Body 1 represents an ankle and is intentionally absent from these indices.
    env.undesired_contact_indices = torch.tensor([0, 2, 3])
    env.config = SimpleNamespace(
        rewards=AttrDict(undesired_contact_force_threshold=1.0)
    )
    contact_history = torch.zeros(1, 3, 4, 3)
    contact_history[0, 1, 0, 2] = 2.0
    contact_history[0, 0, 1, 2] = 100.0
    contact_history[0, 2, 2, 2] = 1.0
    contact_history[0, 2, 3, 0] = 3.0
    env.simulator = SimpleNamespace(
        contact_forces=torch.zeros(1, 4, 3),
        contact_forces_history=contact_history,
    )

    reward = env._reward_undesired_contacts()

    assert torch.equal(reward, torch.tensor([2.0]))


def test_orientation_termination_uses_total_tilt_angle():
    env = object.__new__(LeggedRobotBase)
    env.reset_buf = torch.zeros(2, dtype=torch.bool)
    env.projected_gravity = torch.tensor(
        [[0.0, 0.0, -1.0], [0.6, 0.6, -(0.28**0.5)]]
    )
    env.config = SimpleNamespace(
        termination=SimpleNamespace(
            terminate_by_contact=False,
            terminate_by_gravity=True,
            terminate_by_low_height=False,
            terminate_when_close_to_dof_pos_limit=False,
            terminate_when_close_to_dof_vel_limit=False,
            terminate_when_close_to_torque_limit=False,
        ),
        termination_scales=AttrDict(termination_orientation_limit=0.8),
    )

    env._update_reset_buf()

    assert torch.equal(env.reset_buf, torch.tensor([False, True]))
