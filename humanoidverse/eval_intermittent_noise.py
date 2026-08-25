"""간헐적 관측 오염(random glitch)에 대한 정책 강건성 평가.

eval_obs_corruption.py 는 매 스텝 오염(지속적 결함)을 봤다. 여기서는 **가끔씩만**
오염이 들어오는 경우를 본다. 두 축을 분리한다:

  duty   : 전체 스텝 중 오염된 스텝의 비율 (0.02 ~ 1.0)
  burst  : 오염이 몇 스텝 연속으로 붙어 오는가 (1 = 고립된 스파이크, 20 = 뭉텅이)

duty 를 고정하고 burst 만 바꾸면 **오염 총량은 같고 시간적 구조만 다른** 비교가
된다. history 5스텝이 고립된 스파이크를 희석한다면 burst=1 에서 격차가 벌어지고,
burst 가 history 길이를 넘어가면 (>=5) 지속적 결함과 같아져 격차가 줄어야 한다.

burst 는 env 마다 카운트다운 타이머로 구현한다. 오염 구간이 끝나면 그 다음 스텝
부터 다시 시작 가능해지므로, 시작 확률 p 와 길이 L 에 대해 실제 오염 비율은
Lp / (Lp + 1 - p) 이다. 원하는 duty d 를 맞추려면 p = d / (L(1-d) + d) 로 둔다
(L=1 이면 p=d, d=1 이면 p=1 로 항상 켜짐 — 두 극단 모두 자연히 성립).

  eval_obs_corruption.py 와 달리 한 스텝 안에서 현재 obs 와 history 저장분이
  **같은 노이즈 실현값**을 본다 (스텝의 첫 parse_observation 호출에서 한 번만
  뽑아 캐시). 실제 센서에 더 가깝고, 그쪽 실험에 있던 이중 추출 문제가 없다.

사용 예:
  python humanoidverse/eval_intermittent_noise.py \
    +checkpoint=ckpt/hist_ablation/rew1/baseline/model_61000.pt +simulator=isaacsim \
    ++headless=True ++vx=0.5 ++envs_per_cond=32
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
from humanoidverse.eval_obs_corruption import TARGETS, measured_sigma, run

DUTIES = [0.02, 0.05, 0.2, 0.5, 1.0]
BURSTS = [1, 5, 20]
MAGNITUDES = [2.0, 4.0]          # 실측 sigma 배수. 4σ 는 지속 오염에서 양쪽 다 무너지는 수준


def build_conditions():
    conds = [(0.0, 0, 0.0)]      # 대조군
    for mag in MAGNITUDES:
        for burst in BURSTS:
            for duty in DUTIES:
                conds.append((duty, burst, mag))
    return conds


def install_intermittent(torch, num_envs, device, sigma, conds, env_cond):
    """parse_observation 을 per-env 간헐 오염 버전으로 교체."""
    import humanoidverse.utils.helpers as H

    start_p = torch.zeros(num_envs, 1, device=device)     # 스텝당 오염 구간 시작 확률
    burst_len = torch.zeros(num_envs, 1, device=device)
    mag = {k: torch.zeros(num_envs, 1, device=device) for k in TARGETS}
    for i, (duty, burst, m) in enumerate(conds):
        sel = env_cond == i
        if duty <= 0.0 or not sel.any():
            continue
        start_p[sel] = duty / (burst * (1.0 - duty) + duty)
        burst_len[sel] = float(burst)
        for k in TARGETS:
            mag[k][sel] = m * sigma[k]

    timer = torch.zeros(num_envs, 1, device=device)
    stats = {"active": torch.zeros(num_envs, device=device), "steps": 0}
    cache, state = {}, {"on": False}

    def advance():
        """스텝당 한 번: 타이머를 줄이고, 꺼진 env 는 확률적으로 새 구간을 연다."""
        timer.clamp_(min=0.0)
        start = (torch.rand_like(timer) < start_p) & (timer <= 0)
        timer[start] = burst_len.expand_as(timer)[start]
        stats["active"] += (timer > 0).float().squeeze(1)
        stats["steps"] += 1

    def patched(cls, key_list, buf_dict, obs_scales, noise_scales, cur_scale):
        # 스텝의 첫 호출(history 경로)에서만 타이머를 굴리고 노이즈를 새로 뽑는다.
        if state["on"] and buf_dict is getattr(cls, "hist_obs_dict", None):
            advance()
            cache.clear()
        active = (timer > 0).float()
        for obs_key in key_list:
            if obs_key.endswith("_raw"):
                obs_key = obs_key[:-4]
                gate = 0.0
            else:
                gate = noise_scales.get(obs_key, 0.0)
                gate = float(gate) if isinstance(gate, (int, float)) else 1.0
            x = getattr(cls, f"_get_obs_{obs_key}")().clone()
            if state["on"] and gate > 0.0 and obs_key in mag:
                if obs_key not in cache:
                    cache[obs_key] = torch.randn_like(x) * mag[obs_key]
                x = x + cache[obs_key] * active
            buf_dict[obs_key] = x * obs_scales[obs_key]

    H.parse_observation = patched
    import humanoidverse.envs.legged_base_task.legged_robot_base as LRB
    LRB.parse_observation = patched
    return state, timer, stats


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "intermittent.log")
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
    per_cond = int(config.get("envs_per_cond", 32))
    with open_dict(config):
        config.num_envs = len(conds) * per_cond
        for k in TARGETS:
            config.env.config.obs.noise_scales[k] = 1.0

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        import argparse

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Intermittent observation noise sweep.")
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

    sigma = measured_sigma(checkpoint.parent, config.env.config.obs.obs_scales,
                           ckpt_num=checkpoint.stem.split("_")[-1],
                           npz_path=config.get("sigma_npz", None),
                           actor_obs_keys=config.env.config.obs.obs_dict.actor_obs,
                           obs_dims=config.env.config.obs.obs_dims,
                           obs_auxiliary=config.env.config.obs.obs_auxiliary)
    logger.info("실측 sigma (물리 단위): " + ", ".join(f"{k}={v:.3f}" for k, v in sigma.items()))

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    env_cond = torch.arange(env.num_envs, device=device) % len(conds)
    state, timer, stats = install_intermittent(torch, env.num_envs, device, sigma, conds, env_cond)

    # 오염 구간이 실제로 소모되도록 매 스텝 타이머를 하나 줄인다.
    orig_step = env.step

    def step_with_timer(*a, **kw):
        out = orig_step(*a, **kw)
        if state["on"]:
            timer.sub_(1.0).clamp_(min=0.0)
        return out

    env.step = step_with_timer

    cfg = {"vx": float(config.get("vx", 0.5)), "warmup": int(config.get("warmup_steps", 200)),
           "measure": int(config.get("measure_steps", 600))}
    logger.info(f"envs={env.num_envs}, 조건 {len(conds)}종 x {per_cond}, vx={cfg['vx']}, "
                f"warmup={cfg['warmup']}, measure={cfg['measure']}")

    res = run(env, algo, torch, cfg, conds, env_cond, state)

    idx = env_cond.cpu().numpy()
    duty_actual = (stats["active"] / max(stats["steps"], 1)).cpu().numpy()
    rows = []
    logger.info(f"{'condition':26s}{'duty(real)':>11s}{'falls/1k':>9s}{'vx':>7s}"
                f"{'vx_err':>8s}{'|vy|':>7s}{'act_rate':>9s}")
    for i, (duty, burst, mag) in enumerate(conds):
        m = idx == i
        rate = 1000.0 * res["falls"][m].sum() / max(m.sum() * cfg["measure"], 1)
        label = "clean" if duty <= 0 else f"duty {duty:g} burst {burst} {mag:g}σ"
        logger.info(f"{label:26s}{duty_actual[m].mean():11.3f}{rate:9.2f}"
                    f"{res['vx'][m].mean():7.3f}{res['vx_err'][m].mean():8.3f}"
                    f"{np.abs(res['vy'][m]).mean():7.3f}{res['act_rate'][m].mean():9.4f}")
        rows.append((duty, burst, mag, rate, res["vx"][m].mean(), res["vx_err"][m].mean(),
                     np.abs(res["vy"][m]).mean(), np.abs(res["wz"][m]).mean(),
                     res["act_rate"][m].mean(), duty_actual[m].mean()))

    out_path = config.get("out", None)
    out_path = Path(out_path) if out_path else checkpoint.parent / "corruption" / "intermittent.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, env_cond=idx, cmd_vx=np.float32(cfg["vx"]),
        measure_steps=np.int32(cfg["measure"]),
        cond_duty=np.array([c[0] for c in conds], dtype=np.float32),
        cond_burst=np.array([c[1] for c in conds], dtype=np.int32),
        cond_mag=np.array([c[2] for c in conds], dtype=np.float32),
        summary=np.array(rows, dtype=object), **res)
    logger.info(f"저장 완료: {out_path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
