"""관절 단위 센서 고장(dropout)에 대한 정책 강건성 평가.

eval_obs_corruption.py 가 관측 전체에 노이즈를 더하는 실험이라면, 이쪽은 **관절
하나의 관측만 망가뜨린다**. 실제 로봇에서 흔한 고장 양상 두 가지를 본다:

  zero   : 해당 관절의 관측이 0 으로 죽는다. dof_pos 관측은 default 자세 기준
           오프셋이라 0 = "중립 자세", dof_vel 0 = "안 움직임" 으로 읽힌다.
  freeze : 고장 시점의 값에 멈춘다 (stuck sensor). history 에는 같은 값이 계속
           쌓이므로 시간 평균으로도 걸러지지 않는다.

채널은 dof_pos / dof_vel 를 따로, 그리고 둘 다 동시에 끊는 경우를 본다.

주입 지점은 eval_obs_corruption.py 와 같다 (helpers.parse_observation 교체).
현재 obs 와 history 버퍼 양쪽에 동일하게 적용되고, 노이즈 실험과 달리 변환이
결정론적이라 두 경로가 항상 같은 값을 본다.

사용 예:
  python humanoidverse/eval_joint_dropout.py \
    +checkpoint=ckpt/hist_ablation/rew1/baseline/model_61000.pt +simulator=isaacsim \
    ++headless=True ++vx=0.5 ++envs_per_cond=24
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.logging import HydraLoggerBridge
from humanoidverse.eval_obs_corruption import run  # 동일한 롤아웃/집계 루프

DOF = ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
       "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
       "waist_yaw", "waist_roll", "waist_pitch",
       "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow",
       "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow"]
# 다리 12 + 허리 3 + 팔 1(대조군: 보행과 무관해야 함)
JOINTS = list(range(15))
# CHANNELS = {"pos+vel": ["dof_pos", "dof_vel"], "pos": ["dof_pos"], "vel": ["dof_vel"]}
CHANNELS = {"pos": ["dof_pos"], "vel": ["dof_vel"]}
MODES = ["zero", "freeze"]


def build_conditions():
    conds = [("clean", "none", "none")]
    for mode in MODES:
        for ch in CHANNELS:
            for j in JOINTS:
                conds.append((mode, ch, DOF[j]))
    return conds


def install_dropout(torch, num_envs, device, conds, env_cond):
    """parse_observation 을 per-env 관절 dropout 버전으로 교체."""
    import humanoidverse.utils.helpers as H

    n_dof = len(DOF)
    # (env, dof) 마스크: 해당 관절 관측을 망가뜨릴지. 채널마다 따로.
    hit = {k: torch.zeros(num_envs, n_dof, dtype=torch.bool, device=device)
           for k in ("dof_pos", "dof_vel")}
    is_freeze = torch.zeros(num_envs, 1, device=device)
    for i, (mode, ch, jname) in enumerate(conds):
        sel = env_cond == i
        if mode == "clean" or not sel.any():
            continue
        j = DOF.index(jname)
        for key in CHANNELS[ch]:
            hit[key][sel, j] = True
        if mode == "freeze":
            is_freeze[sel] = 1.0

    held, state = {}, {"on": False}

    def apply(key, x):
        m = hit.get(key)
        if m is None:
            return x
        if key not in held:                       # 고장 시점의 값을 붙잡아 둔다
            held[key] = x.clone()
        frozen = held[key]
        repl = frozen * is_freeze                 # freeze -> 붙잡은 값, zero -> 0
        return torch.where(m, repl, x)

    def patched(cls, key_list, buf_dict, obs_scales, noise_scales, cur_scale):
        for obs_key in key_list:
            if obs_key.endswith("_raw"):
                obs_key = obs_key[:-4]
                gate = 0.0
            else:
                gate = noise_scales.get(obs_key, 0.0)
                gate = float(gate) if isinstance(gate, (int, float)) else 1.0
            x = getattr(cls, f"_get_obs_{obs_key}")().clone()
            if state["on"] and gate > 0.0:
                x = apply(obs_key, x)
            buf_dict[obs_key] = x * obs_scales[obs_key]

    H.parse_observation = patched
    import humanoidverse.envs.legged_base_task.legged_robot_base as LRB
    LRB.parse_observation = patched
    return state


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "dropout.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    assert override_config.checkpoint is not None, "+checkpoint=<path/model_X.pt> 가 필요합니다"
    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    with open(config_path) as f:
        train_config = OmegaConf.load(f)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    config = OmegaConf.merge(train_config, override_config)

    conds = build_conditions()
    per_cond = int(config.get("envs_per_cond", 24))
    with open_dict(config):
        config.num_envs = len(conds) * per_cond
        for k in ("dof_pos", "dof_vel"):     # 게이트만 연다 (실제 변환은 patched 가 결정)
            config.env.config.obs.noise_scales[k] = 1.0

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        import argparse

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Joint sensor dropout sweep.")
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        args_cli.video = False
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import numpy as np
    import torch

    from humanoidverse.agents.base_algo.base_algo import BaseAlgo
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)
    device = config.get("device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    env_cond = torch.arange(env.num_envs, device=device) % len(conds)
    state = install_dropout(torch, env.num_envs, device, conds, env_cond)

    cfg = {"vx": float(config.get("vx", 0.5)), "warmup": int(config.get("warmup_steps", 200)),
           "measure": int(config.get("measure_steps", 600))}
    logger.info(f"envs={env.num_envs}, 조건 {len(conds)}종 x {per_cond}, vx={cfg['vx']}, "
                f"warmup={cfg['warmup']}, measure={cfg['measure']}")

    res = run(env, algo, torch, cfg, conds, env_cond, state)

    idx = env_cond.cpu().numpy()
    rows = []
    logger.info(f"{'condition':30s}{'falls/1k':>9s}{'vx':>7s}{'vx_err':>8s}"
                f"{'|vy|':>7s}{'|wz|':>7s}{'act_rate':>9s}")
    for i, (mode, ch, jname) in enumerate(conds):
        m = idx == i
        rate = 1000.0 * res["falls"][m].sum() / max(m.sum() * cfg["measure"], 1)
        label = "clean" if mode == "clean" else f"{mode} {ch} {jname}"
        logger.info(f"{label:30s}{rate:9.2f}{res['vx'][m].mean():7.3f}"
                    f"{res['vx_err'][m].mean():8.3f}{np.abs(res['vy'][m]).mean():7.3f}"
                    f"{np.abs(res['wz'][m]).mean():7.3f}{res['act_rate'][m].mean():9.4f}")
        rows.append((mode, ch, jname, rate, res["vx"][m].mean(), res["vx_err"][m].mean(),
                     np.abs(res["vy"][m]).mean(), np.abs(res["wz"][m]).mean(),
                     res["act_rate"][m].mean()))

    out_path = config.get("out", None)
    out_path = Path(out_path) if out_path else checkpoint.parent / "corruption" / "dropout.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, env_cond=idx, cmd_vx=np.float32(cfg["vx"]),
        measure_steps=np.int32(cfg["measure"]),
        cond_mode=np.array([c[0] for c in conds]), cond_channel=np.array([c[1] for c in conds]),
        cond_joint=np.array([c[2] for c in conds]),
        summary=np.array(rows, dtype=object), **res)
    logger.info(f"저장 완료: {out_path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
