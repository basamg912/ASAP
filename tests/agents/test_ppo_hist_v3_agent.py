"""PPOHistV3 teacher-latent critic wiring tests."""
import copy

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.ppo_hist_v3.ppo_hist_v3 import PPOHistV3


ACTOR_OBS, CRITIC_OBS, RECON_OBS, ACT = 30, 45, 22, 8
HIST_KEYS = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
DIMS = {"actions": ACT, "base_ang_vel": 3, "dof_pos": 6,
        "dof_vel": 6, "projected_gravity": 3}
T = 4
ENCODER_OBS = sum(DIMS[k] for k in HIST_KEYS) * T

ENV_CFG = OmegaConf.create({
    "num_envs": 4,
    "robot": {
        "algo_obs_dim_dict": {
            "actor_obs": ACTOR_OBS,
            "critic_obs": CRITIC_OBS,
            "encoder_obs": ENCODER_OBS,
            "recon_target": RECON_OBS,
            "robot_action_dim": ACT,
        },
        "actions_dim": ACT,
    },
    "obs": {
        "obs_dict": {"encoder_obs": ["history"]},
        "obs_auxiliary": {"history": {k: T for k in HIST_KEYS}},
        "obs_dims": DIMS,
    },
})


class _StubEnv:
    config = ENV_CFG
    num_envs = 4


def make_algo(teacher_in_critic=True):
    cfg = copy.deepcopy(
        OmegaConf.load("humanoidverse/config/algo/ppo_hist_v3.yaml").algo.config)
    cfg.teacher_in_critic = teacher_in_critic
    cfg.module_dict.actor.input_dim = ["actor_obs", cfg.vel_dim + cfg.latent_dim]
    cfg.module_dict.actor.layer_config.hidden_dims = [32]
    cfg.module_dict.critic.layer_config.hidden_dims = [32]
    cfg.encoder_config.hidden_dims = [32]
    cfg.encoder_config.channel_hidden_dims = [8]
    cfg.teacher_config.enc_hidden_dims = [32]
    cfg.teacher_config.dec_hidden_dims = [32]
    cfg.projection_config.hidden_dims = [16]
    cfg.contrastive_batch_size = 32
    OmegaConf.set_struct(cfg, True)

    algo = PPOHistV3.__new__(PPOHistV3)
    algo.device = "cpu"
    algo.env = _StubEnv()
    algo.config = cfg
    algo.log_dir = None
    algo._init_config()
    algo.current_learning_iteration = 100
    algo._setup_models_and_optimizer()
    return algo


def fake_batch(batch_size=64):
    return {
        "actor_obs": torch.randn(batch_size, ACTOR_OBS),
        "critic_obs": torch.randn(batch_size, CRITIC_OBS),
        "encoder_obs": torch.randn(batch_size, ENCODER_OBS),
        "recon_target": torch.randn(batch_size, RECON_OBS),
        "next_obs_target": torch.randn(batch_size, RECON_OBS),
        "base_vel_target": torch.randn(batch_size, 3),
        "actions": torch.randn(batch_size, ACT),
        "values": torch.randn(batch_size, 1),
        "advantages": torch.randn(batch_size, 1),
        "returns": torch.randn(batch_size, 1),
        "actions_log_prob": torch.randn(batch_size, 1),
        "action_mean": torch.randn(batch_size, ACT),
        "action_sigma": torch.rand(batch_size, ACT) + 0.5,
        "dones": torch.rand(batch_size, 1) < 0.1,
    }


def first_linear_in_features(module):
    return next(m for m in module.modules()
                if isinstance(m, torch.nn.Linear)).in_features


def test_teacher_critic_appends_latent_without_mutating_yaml_config():
    algo = make_algo(True)
    assert first_linear_in_features(algo.critic) == CRITIC_OBS + algo.latent_dim
    assert list(algo.config.module_dict.critic.input_dim) == ["critic_obs"]


def test_archived_config_path_keeps_plain_critic_when_disabled():
    algo = make_algo(False)
    assert first_linear_in_features(algo.critic) == CRITIC_OBS


def test_value_gradient_reaches_teacher_encoder_only():
    algo = make_algo(True)
    batch = fake_batch()
    algo.teacher.zero_grad()
    algo.critic.zero_grad()
    algo._critic_eval_step(batch).square().mean().backward()

    encoder_grad = next(algo.teacher.encoder.parameters()).grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0
    assert all(p.grad is None for p in algo.teacher.decoder.parameters())
    assert all(p.grad is None for p in algo.teacher.projection_head.parameters())


def test_critic_uses_current_recon_target_not_next_target():
    algo = make_algo(True)
    batch = fake_batch()
    batch["next_obs_target"] = torch.full_like(
        batch["next_obs_target"], float("nan"))
    value = algo._critic_eval_step(batch)
    assert value.shape == (64, 1)
    assert torch.isfinite(value).all()


def test_teacher_critic_bootstrap_path_has_matching_input_shape():
    algo = make_algo(True)
    last_obs = {
        "critic_obs": torch.randn(4, CRITIC_OBS),
        "recon_target": torch.randn(4, RECON_OBS),
    }
    policy_state = {
        "values": torch.randn(3, 4, 1),
        "dones": torch.zeros(3, 4, 1).bool(),
        "rewards": torch.randn(3, 4, 1),
    }
    returns, advantages = algo._compute_returns(last_obs, policy_state)
    assert returns.shape == (3, 4, 1)
    assert advantages.shape == (3, 4, 1)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()


def test_v3_update_combines_teacher_and_critic_gradients():
    algo = make_algo(True)
    loss_dict = algo._update_ppo(
        fake_batch(), algo._init_loss_dict_at_training_step())
    assert all(torch.isfinite(torch.tensor(value))
               for value in loss_dict.values())
