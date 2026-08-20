"""PPOHistV4 agent 배선 테스트 (env stub).

핵심 검증 대상:
  - teacher_mode=critic 에서 main critic 입력 = critic_obs + latent_dim, 그리고
    critic value loss gradient 가 teacher(phi) 까지 도달하는지
  - critic 경로에 미래 정보(o_{t+1}) 가 섞이지 않는지 (baseline 무편향의 구조적 근거)
  - 다른 모드(frozen/vicreg/critic_aux)에서는 critic 이 critic_obs 만 보는지
  - save/load 왕복, 배포 경로(student only)
"""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from humanoidverse.agents.ppo_hist_v4.ppo_hist_v4 import PPOHistV4

ACTOR_OBS, CRITIC_OBS, TEACHER_OBS, ACT = 30, 45, 22, 8
HIST_KEYS = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
DIMS = {"actions": ACT, "base_ang_vel": 3, "dof_pos": 6, "dof_vel": 6, "projected_gravity": 3}
T = 4
ENCODER_OBS = sum(DIMS[k] for k in HIST_KEYS) * T
MODES = ["critic", "critic_aux", "frozen", "vicreg"]

ENV_CFG = OmegaConf.create({
    "num_envs": 4,
    "robot": {
        "algo_obs_dim_dict": {
            "actor_obs": ACTOR_OBS, "critic_obs": CRITIC_OBS,
            "encoder_obs": ENCODER_OBS, "teacher_obs": TEACHER_OBS,
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


def make_algo(teacher_mode):
    cfg = copy.deepcopy(
        OmegaConf.load("humanoidverse/config/algo/ppo_hist_v4.yaml").algo.config)
    cfg.teacher_mode = teacher_mode
    cfg.module_dict.actor.input_dim = ["actor_obs", cfg.vel_dim + cfg.latent_dim]
    cfg.module_dict.actor.layer_config.hidden_dims = [32]
    cfg.module_dict.critic.layer_config.hidden_dims = [32]
    cfg.encoder_config.hidden_dims = [32]
    cfg.encoder_config.channel_hidden_dims = [8]
    cfg.teacher_config.value_hidden_dims = [32]
    OmegaConf.set_struct(cfg, True)          # hydra 런타임과 동일한 struct 모드

    algo = PPOHistV4.__new__(PPOHistV4)      # env/writer 없이 배선만 세운다
    algo.device = "cpu"
    algo.env = _StubEnv()
    algo.config = cfg
    algo.log_dir = None
    algo._init_config()
    algo.current_learning_iteration = 0
    algo._setup_models_and_optimizer()
    return algo


def fake_batch(B=64):
    return {
        "actor_obs": torch.randn(B, ACTOR_OBS),
        "critic_obs": torch.randn(B, CRITIC_OBS),
        "encoder_obs": torch.randn(B, ENCODER_OBS),
        "teacher_obs": torch.randn(B, TEACHER_OBS),          # phi 입력 @ t
        "next_obs_target": torch.randn(B, TEACHER_OBS),      # phi 입력 @ t+1 (student 타깃)
        "base_vel_target": torch.randn(B, 3),
        "actions": torch.randn(B, ACT),
        "values": torch.randn(B, 1),
        "advantages": torch.randn(B, 1),
        "returns": torch.randn(B, 1),
        "actions_log_prob": torch.randn(B, 1),
        "action_mean": torch.randn(B, ACT),
        "action_sigma": torch.rand(B, ACT) + 0.5,
        "dones": (torch.rand(B, 1) < 0.1),
    }


def first_linear_in_features(module):
    return [m for m in module.modules() if isinstance(m, torch.nn.Linear)][0].in_features


def test_invalid_teacher_mode_rejected():
    with pytest.raises(AssertionError, match="teacher_mode"):
        make_algo("bogus")


def test_missing_teacher_obs_group_gives_actionable_error():
    algo = PPOHistV4.__new__(PPOHistV4)
    algo.device, algo.env, algo.log_dir = "cpu", _StubEnv(), None
    cfg = copy.deepcopy(
        OmegaConf.load("humanoidverse/config/algo/ppo_hist_v4.yaml").algo.config)
    cfg.teacher_obs_key = "recon_target"     # 구 obs 파일을 쓴 상황
    cfg.recon_target_key = "recon_target"
    algo.config = cfg
    with pytest.raises(AssertionError, match="leggedloco_obs_history_encoder_v4"):
        algo._init_config()


def test_critic_mode_appends_latent_to_critic_input():
    algo = make_algo("critic")
    assert first_linear_in_features(algo.critic) == CRITIC_OBS + algo.latent_dim
    # yaml 은 [critic_obs] 그대로 — append 는 코드에서만 일어난다 (사본 수정)
    assert list(algo.config.module_dict.critic.input_dim) == ["critic_obs"]


@pytest.mark.parametrize("mode", ["critic_aux", "frozen", "vicreg"])
def test_non_critic_modes_keep_plain_critic_input(mode):
    algo = make_algo(mode)
    assert first_linear_in_features(algo.critic) == CRITIC_OBS


def test_critic_mode_value_loss_trains_teacher():
    """critic value loss gradient 가 phi 파라미터까지 도달해야 한다."""
    algo = make_algo("critic")
    before = copy.deepcopy(algo.teacher.state_dict())
    algo._update_ppo(fake_batch(), algo._init_loss_dict_at_training_step())
    assert any(not torch.allclose(before[k], v)
               for k, v in algo.teacher.state_dict().items())


def test_critic_eval_step_ignores_future_obs():
    """critic 은 phi(teacher_obs_t) 만 본다 — next_obs_target 이 NaN 이어도 value 는 유한."""
    algo = make_algo("critic")
    batch = fake_batch()
    batch["next_obs_target"] = torch.full_like(batch["next_obs_target"], float("nan"))
    value = algo._critic_eval_step(batch)
    assert value.shape == (64, 1)
    assert torch.isfinite(value).all()


def test_critic_mode_bootstrap_value_path_matches_critic_input():
    """_compute_returns 의 bootstrap 도 latent 를 붙인 입력으로 계산돼야 한다."""
    algo = make_algo("critic")
    last_obs = {"critic_obs": torch.randn(4, CRITIC_OBS),
                "teacher_obs": torch.randn(4, TEACHER_OBS)}
    policy_state = {"values": torch.randn(3, 4, 1), "dones": torch.zeros(3, 4, 1).bool(),
                    "rewards": torch.randn(3, 4, 1)}
    returns, advantages = algo._compute_returns(last_obs, policy_state)
    assert returns.shape == (3, 4, 1) and advantages.shape == (3, 4, 1)
    assert torch.isfinite(returns).all() and torch.isfinite(advantages).all()


@pytest.mark.parametrize("mode", MODES)
def test_update_runs_and_trains_expected_parts(mode):
    algo = make_algo(mode)
    t_before = copy.deepcopy(algo.teacher.state_dict())
    s_before = copy.deepcopy(algo.actor.student.state_dict())
    loss_dict = algo._init_loss_dict_at_training_step()
    for _ in range(2):
        loss_dict = algo._update_ppo(fake_batch(), loss_dict)

    assert all(torch.isfinite(torch.tensor(v)) for v in loss_dict.values())
    t_changed = any(not torch.allclose(t_before[k], v)
                    for k, v in algo.teacher.state_dict().items())
    s_changed = any(not torch.allclose(s_before[k], v)
                    for k, v in algo.actor.student.state_dict().items())
    assert t_changed == (mode != "frozen")   # frozen 은 고정 랜덤 사영
    assert s_changed                          # student 는 모든 모드에서 학습


@pytest.mark.parametrize("mode", MODES)
def test_save_load_roundtrip(mode, tmp_path):
    algo = make_algo(mode)
    algo._update_ppo(fake_batch(), algo._init_loss_dict_at_training_step())
    path = str(tmp_path / f"v4_{mode}.ckpt")
    algo.save(path, infos={"note": mode})

    algo2 = make_algo(mode)
    assert algo2.load(path) == {"note": mode}
    assert all(torch.allclose(algo.teacher.state_dict()[k], v)
               for k, v in algo2.teacher.state_dict().items())
    if algo.teacher_value_head is not None:
        assert all(torch.allclose(algo.teacher_value_head.state_dict()[k], v)
                   for k, v in algo2.teacher_value_head.state_dict().items())


@pytest.mark.parametrize("mode", MODES)
def test_deploy_path_is_student_only(mode):
    """배포는 teacher/critic 없이 student + actor MLP 만으로 동작해야 한다."""
    from humanoidverse.agents.ppo_hist_v4.inference_wrapper import HistV4InferenceModule
    algo = make_algo(mode)
    algo.actor.eval()
    ao, eo = torch.randn(4, ACTOR_OBS), torch.randn(4, ENCODER_OBS)
    ref = algo.actor.act_inference(ao, eo)
    got = HistV4InferenceModule(algo.actor, ACTOR_OBS, ENCODER_OBS)(torch.cat([ao, eo], -1))
    assert ref.shape == (4, ACT)
    assert torch.allclose(ref, got, atol=1e-6)
