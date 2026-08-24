import numpy as np
import pytest

from humanoidverse.eval_obs_corruption import actor_obs_spans, measured_sigma


ACTOR_OBS_WPHASE = [
    "base_ang_vel",
    "projected_gravity",
    "command_lin_vel",
    "command_ang_vel",
    "command_stand",
    "dof_pos",
    "dof_vel",
    "cos_phase",
    "sin_phase",
    "actions",
]

OBS_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command_lin_vel": 2,
    "command_ang_vel": 1,
    "command_stand": 1,
    "dof_pos": 23,
    "dof_vel": 23,
    "cos_phase": 1,
    "sin_phase": 1,
    "actions": 23,
}


def test_actor_obs_spans_matches_81d_wphase_layout():
    spans, width = actor_obs_spans(ACTOR_OBS_WPHASE, OBS_DIMS)

    assert width == 81
    assert spans["cos_phase"] == (30, 31)
    assert spans["dof_pos"] == (31, 54)
    assert spans["dof_vel"] == (54, 77)
    assert spans["projected_gravity"] == (77, 80)


def test_actor_obs_spans_expands_397d_baseline_history():
    history = {
        key: 5 for key in (
            "base_ang_vel", "projected_gravity", "dof_pos", "dof_vel",
            "actions", "command_lin_vel", "command_ang_vel", "command_stand")
    }

    spans, width = actor_obs_spans(
        ["cos_phase", "sin_phase", "short_history"], OBS_DIMS,
        obs_auxiliary={"short_history": history})

    assert width == 397
    assert spans["base_ang_vel"] == (116, 131)
    assert spans["dof_pos"] == (151, 266)
    assert spans["dof_vel"] == (266, 381)
    assert spans["projected_gravity"] == (381, 396)


def test_measured_sigma_uses_dynamic_81d_spans(tmp_path):
    rng = np.random.default_rng(7)
    actor_obs = rng.normal(size=(3, 4, 81)).astype(np.float32)
    path = tmp_path / "obs.npz"
    np.savez(path, actor_obs=actor_obs, ep_len=np.full((3, 4), 30),
             done=np.zeros((3, 4), dtype=bool))
    scales = {
        "base_ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "projected_gravity": 1.0,
    }

    sigma = measured_sigma(
        tmp_path, scales, npz_path=path,
        actor_obs_keys=ACTOR_OBS_WPHASE, obs_dims=OBS_DIMS)
    spans, _ = actor_obs_spans(ACTOR_OBS_WPHASE, OBS_DIMS)
    flat = actor_obs.reshape(-1, actor_obs.shape[-1])

    for key, scale in scales.items():
        start, end = spans[key]
        expected = float(flat[:, start:end].std(0).mean()) / scale
        assert sigma[key] == pytest.approx(expected)
