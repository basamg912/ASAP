"""정책을 굴리면서 actor 입력 obs 를 그대로 기록한다 (입력 민감도 분석용).

collect_encoder_latents.py 와 같은 방식으로 체크포인트/시뮬레이터를 띄우되,
encoder 출력이 아니라 **정책이 실제로 보는 obs** 를 저장한다. encoder 유무와
무관하므로 baseline(history 를 actor_obs 에 직접 concat) 과 ppo_hist v1/v2/v3
(actor_obs + encoder_obs) 를 모두 지원한다.

용도:
  - obs 차원별 실제 표준편차 -> saliency 를 "gradient x 실제 변동폭" 으로 환산
  - 실제 on-policy 상태에서의 Jacobian (합성 nominal 샘플링 대신)
  - 커맨드별 낙상률 + 커맨드 추종 오차 -> 두 정책이 함께 유효한 커맨드 구간 확인

낙상은 done & ~time_out 으로 센다 (에피소드 시간초과와 구분).

사용 예:
  python humanoidverse/collect_obs_stats.py \
    +checkpoint=logs/.../model_100.pt +simulator=isaacsim \
    ++num_envs=120 ++warmup_steps=300 ++record_steps=600 ++stride=3
"""

import inspect
import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.logging import HydraLoggerBridge

# collect_encoder_latents.py 와 동일한 헬퍼 (커맨드 배정 / eval 범위 확장 / 라벨)
from humanoidverse.collect_encoder_latents import (  # noqa: E402
    assign_commands,
    apply_yaw_rate_command,
    command_label,
    widen_eval_command_ranges,
)

# 정지 / 전후 / 횡보 / 회전을 고루 덮는 격자. baseline 이 못 따라가는 커맨드가
# 어디인지 보려면 좁히지 말고 이 격자 그대로 한 번 돌리는 편이 낫다.
DEFAULT_COMMANDS = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [-0.4, 0.0, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, -0.4, 0.0],
    [0.0, 0.0, 0.6],
    [0.0, 0.0, -0.6],
    [0.5, 0.0, 0.5],
    [0.5, 0.3, 0.0],
]


def make_inference(actor):
    """actor 종류에 맞는 act_inference 호출부를 만든다 (encoder 유무 자동 판별)."""
    n_args = len(inspect.signature(actor.act_inference).parameters)
    if n_args >= 2:                                   # ppo_hist v1/v2/v3
        return lambda obs: actor.act_inference(obs["actor_obs"], obs["encoder_obs"]), True
    return lambda obs: actor.act_inference(obs["actor_obs"]), False


def collect(env, algo, torch, num_warmup, num_record, stride, cmd_grid, device):
    """워밍업 후 num_record 스텝 동안 obs 를 stride 간격으로 기록해 dict(numpy) 반환."""
    import numpy as np

    algo._eval_mode()
    env.set_is_evaluating()
    widen_eval_command_ranges(env)
    obs_dict = env.reset_all()

    per_env_cmd, cmd_idx = assign_commands(env, cmd_grid, device)
    wz = per_env_cmd[:, 2]

    actor = algo.actor
    infer, has_encoder = make_inference(actor)
    logger.info(f"actor={type(actor).__name__}, encoder_obs 사용={has_encoder}")

    buf = {k: [] for k in ("actor_obs", "encoder_obs", "done", "fall", "ep_len",
                           "base_lin_vel", "base_ang_vel", "command")}
    actor_state = {"done_indices": [], "stop": False}
    n_fall = torch.zeros(env.num_envs, device=device)
    total = num_warmup + num_record

    for step in range(total):
        apply_yaw_rate_command(env, wz)
        with torch.no_grad():
            action = infer(obs_dict)

        recording = step >= num_warmup and (step - num_warmup) % stride == 0
        if recording:
            buf["actor_obs"].append(obs_dict["actor_obs"].cpu().numpy().astype(np.float32))
            if has_encoder:
                buf["encoder_obs"].append(obs_dict["encoder_obs"].cpu().numpy().astype(np.float32))
            buf["ep_len"].append(env.episode_length_buf.cpu().numpy().astype(np.int32))
            # 추종 오차용 실측 속도 (base_lin_vel 은 privileged obs 라 actor_obs 에 없다)
            buf["base_lin_vel"].append(env.base_lin_vel.cpu().numpy().astype(np.float32))
            buf["base_ang_vel"].append(env.base_ang_vel.cpu().numpy().astype(np.float32))
            buf["command"].append(env.commands[:, :3].cpu().numpy().astype(np.float32))

        actor_state.update({"obs": obs_dict, "actions": action})
        obs_dict, _, dones, _ = env.step(actor_state)
        obs_dict = {k: v_.to(device) for k, v_ in obs_dict.items()}

        # 시간초과가 아닌 종료 = 낙상. time_out_buf 는 step 안에서 갱신된다.
        time_out = getattr(env, "time_out_buf", None)
        fall = dones.bool() & (~time_out.bool() if time_out is not None else True)
        if step >= num_warmup:
            n_fall += fall.float()
        if recording:
            buf["done"].append(dones.cpu().numpy().astype(bool))
            buf["fall"].append(fall.cpu().numpy().astype(bool))

        if (step + 1) % 200 == 0:
            tag = "warmup" if step < num_warmup else "record"
            logger.info(f"[{tag}] step {step + 1}/{total}")

    out = {k: np.asarray(v_) for k, v_ in buf.items() if v_}
    out["cmd_grid"] = np.asarray(cmd_grid, dtype=np.float32)
    out["env_cmd_idx"] = cmd_idx.cpu().numpy()
    out["labels"] = np.array([command_label(c) for c in cmd_grid])
    out["fall_count"] = n_fall.cpu().numpy()
    out["record_steps"] = np.asarray(num_record)

    # 커맨드별 낙상률 + 추종 오차. 낙상률 = (기록 구간 낙상 횟수) / (env 수 x 기록 스텝),
    # 추종 오차는 리셋 직후 과도구간(ep_len < 20)과 종료 스텝을 뺀 정상 구간에서만 잰다.
    idx = cmd_idx.cpu().numpy()
    falls = n_fall.cpu().numpy()
    ok = (out["ep_len"] >= 20) & (~out["done"])
    v_xy, w_z, cmd = out["base_lin_vel"][..., :2], out["base_ang_vel"][..., 2], out["command"]
    logger.info(f"{'command':16s} {'envs':>5s} {'falls/1k':>9s} "
                f"{'vx err':>8s} {'vy err':>8s} {'wz err':>8s} {'vx':>7s} {'vy':>7s} {'wz':>7s}")
    for i, c in enumerate(cmd_grid):
        m = ok & (idx == i)[None, :]
        e = np.abs(np.stack([v_xy[..., 0] - cmd[..., 0], v_xy[..., 1] - cmd[..., 1],
                             w_z - cmd[..., 2]], axis=-1))[m]
        got = np.stack([v_xy[..., 0], v_xy[..., 1], w_z], axis=-1)[m]
        rate = 1000.0 * falls[idx == i].sum() / max((idx == i).sum() * num_record, 1)
        logger.info(f"{command_label(c):16s} {(idx == i).sum():5d} {rate:9.2f} "
                    f"{e[:, 0].mean():8.3f} {e[:, 1].mean():8.3f} {e[:, 2].mean():8.3f} "
                    f"{got[:, 0].mean():7.3f} {got[:, 1].mean():7.3f} {got[:, 2].mean():7.3f}")
    return out


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "collect_obs.log")
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
    assert config_path.exists(), f"config.yaml 을 찾을 수 없습니다: {config_path}"
    logger.info(f"Loading training config from {config_path}")
    with open(config_path) as f:
        train_config = OmegaConf.load(f)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    config = OmegaConf.merge(train_config, override_config)

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        import argparse

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Collect actor-input obs statistics.")
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

    cmd_grid = config.get("commands", None)
    cmd_grid = [list(c) for c in cmd_grid] if cmd_grid is not None else DEFAULT_COMMANDS
    num_warmup = int(config.get("warmup_steps", 300))
    num_record = int(config.get("record_steps", 600))
    stride = int(config.get("stride", 3))

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    logger.info(
        f"envs={env.num_envs}, 커맨드 {len(cmd_grid)}종 → env당 "
        f"{env.num_envs / len(cmd_grid):.1f}개, warmup={num_warmup}, record={num_record}, "
        f"stride={stride} (샘플 {env.num_envs * (num_record // stride)}개)"
    )

    data = collect(env, algo, torch, num_warmup, num_record, stride, cmd_grid, device)

    ckpt_num = checkpoint.stem.split("_")[-1]
    out_path = config.get("out", None)
    out_path = Path(out_path) if out_path else checkpoint.parent / "obs_stats" / f"obs_ckpt_{ckpt_num}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

    logger.info(f"저장 완료: {out_path}")
    logger.info("  " + ", ".join(f"{k}{v.shape}" for k, v in data.items() if hasattr(v, "shape")))

    # simulation_app.close() 가 간헐적으로 매달리므로 저장이 끝나면 바로 강제 종료한다.
    # (npz 는 위에서 이미 flush 됐고, 이 시점 이후로 할 일이 없다)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
