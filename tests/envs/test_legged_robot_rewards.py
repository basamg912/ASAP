from types import SimpleNamespace

import torch

from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


def test_energy_matches_absolute_joint_mechanical_power():
    env = object.__new__(LeggedRobotBase)
    env.simulator = SimpleNamespace(
        dof_vel=torch.tensor([[2.0, -3.0, 0.0], [-1.0, 4.0, -2.0]])
    )
    env.torques = torch.tensor([[-5.0, 2.0, 7.0], [3.0, -0.5, -4.0]])

    energy = env._reward_energy()

    assert torch.equal(energy, torch.tensor([16.0, 13.0]))
