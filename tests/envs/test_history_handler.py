from types import SimpleNamespace

import torch

from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase


def test_first_sample_fills_reset_history_like_circular_buffer():
    handler = HistoryHandler(
        num_envs=2,
        history_config={"short_history": {"signal": 3}},
        obs_dims={"signal": 1},
        device="cpu",
        fill_on_first_add=True,
    )

    handler.add("signal", torch.tensor([[1.0], [2.0]]))
    assert torch.equal(
        handler.query("signal"),
        torch.tensor([[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]]),
    )

    handler.add("signal", torch.tensor([[3.0], [4.0]]))
    handler.reset(torch.tensor([1]))
    handler.add("signal", torch.tensor([[5.0], [6.0]]))

    assert torch.equal(
        handler.query("signal"),
        torch.tensor([[[5.0], [3.0], [1.0]], [[6.0], [6.0], [6.0]]]),
    )


def test_observation_history_is_flattened_oldest_to_current():
    env = object.__new__(LeggedRobotBase)
    env.config = SimpleNamespace(
        obs=SimpleNamespace(obs_auxiliary={"short_history": {"signal": 3}})
    )
    env.history_oldest_first = True
    env._observation_history_handler = SimpleNamespace(
        query=lambda _key: torch.tensor([[[3.0], [2.0], [1.0]]])
    )

    history = env._get_obs_short_history()

    assert torch.equal(history, torch.tensor([[1.0, 2.0, 3.0]]))


def test_critic_history_stays_clean_when_actor_history_uses_noise():
    env = object.__new__(LeggedRobotBase)
    env.config = SimpleNamespace(
        obs=SimpleNamespace(
            obs_dict={
                "actor_obs": ["short_history"],
                "critic_obs": ["short_history"],
            },
            obs_auxiliary={"short_history": {"signal": 3}},
            obs_scales={"signal": 1.0, "short_history": 1.0},
            noise_scales={"signal": 1.0, "short_history": 0.0},
        )
    )
    env.add_noise_currculum = False
    env.history_include_current = True
    env.history_oldest_first = True
    env.history_handler = HistoryHandler(
        1,
        env.config.obs.obs_auxiliary,
        {"signal": 1},
        "cpu",
        fill_on_first_add=True,
    )
    env.critic_history_handler = HistoryHandler(
        1,
        env.config.obs.obs_auxiliary,
        {"signal": 1},
        "cpu",
        fill_on_first_add=True,
    )
    env._observation_history_handler = env.history_handler
    env._get_obs_signal = lambda: torch.zeros(1, 1)

    torch.manual_seed(0)
    env._compute_observations()

    assert not torch.equal(env.obs_buf_dict["actor_obs"], torch.zeros(1, 3))
    assert torch.equal(env.obs_buf_dict["critic_obs"], torch.zeros(1, 3))
