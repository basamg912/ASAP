"""
History encoder 출력(latent z, vel head v̂) 을 커맨드와 함께 기록한다 (t-SNE 등 군집 분석용).

eval_agent.py 와 같은 방식으로 체크포인트/시뮬레이터를 띄우되, 렌더링·ONNX export 없이
정책을 굴리면서 매 스텝 encoder 출력을 버퍼에 쌓아 npz 로 저장한다.

핵심 설계:
  - env 를 num_envs 개 띄우고 **env 마다 서로 다른 커맨드를 고정 배정**한다.
    (한 env 에서 커맨드를 바꿔가며 기록하면 전환 구간이 섞이고 시간도 오래 걸림)
  - warmup_steps 동안은 버리고(정책이 해당 커맨드의 정상 상태에 도달할 시간),
    이후 record_steps 만큼 기록한다.
  - yaw 커맨드는 env 가 매 스텝 heading 오차로 commands[:,2] 를 덮어쓰므로
    (locomotion.py:_update_tasks_callback) 직접 쓸 수 없다. 목표 heading 을
    `heading + 2*wz` 로 매 스텝 갱신해 원하는 yaw rate 를 만든다.

지원 actor:
  - PPOActorWithStudentEncoder (ppo_hist v2/v3): student(enc_obs) -> (v̂, z)
  - PPOActorWithHistoryEncoder (ppo_hist v1)   : history_encoder.sample(enc_obs) -> (z, mu, logvar)
    (eval 모드에서 z = mu, vel head 없음)

사용 예:
  python humanoidverse/collect_encoder_latents.py \
    +checkpoint=logs/.../model_100.pt +simulator=isaacsim \
    +num_envs=64 +warmup_steps=300 +record_steps=600

  # 커맨드 격자 직접 지정 (vx, vy, wz)
  ... +commands='[[0,0,0],[0.6,0,0],[-0.4,0,0],[0,0.4,0],[0,0,0.5]]'
"""

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

# (vx, vy, wz) — 기본 격자. 정지/전후/좌우/회전/대각을 고루 덮는다.
# DEFAULT_COMMANDS = [
#     [0.0, 0.0, 0.0],    # stand
#     [0.5, 0.0, 0.0],    # forward
#     [0.0, 0.3, 0.0],    # strafe left
#     [0.0, -0.3, 0.0],   # strafe right
# ]


DEFAULT_COMMANDS = [
    [0.3, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.8, 0.0, 0.0],
    [1.0, 0.0, 0.0],
]

def command_label(cmd):
    vx, vy, wz = cmd
    if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1:
        return "stand"
    parts = []
    if abs(vx) >= 0.1:
        parts.append(f"{'fwd' if vx > 0 else 'bwd'}{abs(vx):.1f}")
    if abs(vy) >= 0.1:
        parts.append(f"{'left' if vy > 0 else 'right'}{abs(vy):.1f}")
    if abs(wz) >= 0.1:
        parts.append(f"{'turnL' if wz > 0 else 'turnR'}{abs(wz):.1f}")
    return "+".join(parts)


def encoder_forward(actor, actor_obs, encoder_obs):
    """actor 종류에 맞춰 (action, latent, vel_pred) 반환. act_inference 와 동일한 계산."""
    import torch

    if hasattr(actor, "student"):  # ppo_hist v2 / v3
        v, z = actor.student(encoder_obs)
        action = actor.actor(torch.cat([actor_obs, v, z], dim=-1))
        return action, z, v
    if hasattr(actor, "history_encoder"):  # ppo_hist v1 (VAE, eval 이면 z = mu)
        z, mu, _ = actor.history_encoder.sample(encoder_obs)
        action = actor.actor(torch.cat([actor_obs, z], dim=-1))
        return action, mu, None
    raise TypeError(
        f"{type(actor).__name__} 에 history encoder 가 없습니다 "
        "(ppo_hist v1/v2/v3 체크포인트가 필요합니다)"
    )


def assign_commands(env, cmd_grid, device):
    """env 마다 격자에서 커맨드 하나씩 배정. 반환: env별 (vx,vy,wz) 텐서와 격자 인덱스."""
    import torch

    n = env.num_envs
    idx = torch.arange(n, device=device) % len(cmd_grid)
    grid = torch.tensor(cmd_grid, dtype=torch.float32, device=device)
    per_env = grid[idx]                      # [N, 3]
    env.commands[:, 0] = per_env[:, 0]
    env.commands[:, 1] = per_env[:, 1]
    return per_env, idx


def apply_heading_command(env, wz_per_env):
    """원하는 yaw rate 가 나오도록 목표 heading 을 갱신.

    env 는 매 스텝 commands[:,2] = clip(0.5*wrap_to_pi(commands[:,3] - heading)) 로
    덮어쓰므로, commands[:,3] = heading + 2*wz 로 두면 commands[:,2] ≈ wz 가 된다.
    """
    import torch

    # locomotion.py 와 동일한 2-인자 quat_apply (isaac_utils 쪽은 w_last 인자가 더 있음)
    from humanoidverse.utils.torch_utils import quat_apply

    forward = quat_apply(env.base_quat, env.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    env.commands[:, 3] = heading + 2.0 * wz_per_env


def widen_eval_command_ranges(env):
    """eval 중에는 command curriculum 이 갱신되지 않아 command_ranges 가 progress=0
    (예: yaw ±0.3) 로 남는다. yaw 커맨드가 그 범위로 클립되므로 최종 범위로 넓힌다."""
    final = getattr(env, "final_command_ranges", None)
    if final is None:
        return
    before = dict(env.command_ranges)
    env.command_ranges = {k: list(v) for k, v in final.items()}
    logger.info(f"command_ranges 를 최종 범위로 확장: {before} -> {env.command_ranges}")


def collect(env, algo, torch, num_warmup, num_record, cmd_grid, device):
    """워밍업 후 num_record 스텝 동안 encoder 출력을 기록해 dict(numpy) 반환."""
    import numpy as np

    algo._eval_mode()
    env.set_is_evaluating()
    widen_eval_command_ranges(env)
    obs_dict = env.reset_all()

    per_env_cmd, cmd_idx = assign_commands(env, cmd_grid, device)
    wz = per_env_cmd[:, 2]

    buf = {k: [] for k in
           ("latent", "vel_pred", "base_lin_vel", "command", "phase", "done")}
    actor = algo.actor
    total = num_warmup + num_record
    actor_state = {"done_indices": [], "stop": False}

    for step in range(total):
        apply_heading_command(env, wz)
        with torch.no_grad():
            action, z, v = encoder_forward(
                actor, obs_dict["actor_obs"].to(device), obs_dict["encoder_obs"].to(device)
            )

        recording = step >= num_warmup
        if recording:
            buf["latent"].append(z.cpu().numpy())
            buf["vel_pred"].append(v.cpu().numpy() if v is not None else np.zeros(0))
            buf["base_lin_vel"].append(env.base_lin_vel.detach().cpu().numpy())
            # commands[:,2] 는 env 가 heading 으로 채운 실제 yaw 커맨드
            buf["command"].append(env.commands[:, [0, 1, 2]].detach().cpu().numpy())
            phase = getattr(env, "phase_time", None)
            buf["phase"].append(
                phase.detach().cpu().numpy() if phase is not None
                else np.zeros(env.num_envs, dtype=np.float32))

        actor_state.update({"obs": obs_dict, "actions": action})
        obs_dict, _, dones, _ = env.step(actor_state)
        obs_dict = {k: v_.to(device) for k, v_ in obs_dict.items()}
        if recording:
            buf["done"].append(dones.detach().cpu().numpy().astype(bool))

        if (step + 1) % 100 == 0:
            tag = "warmup" if step < num_warmup else "record"
            logger.info(f"[{tag}] step {step + 1}/{total}")

    out = {k: np.asarray(v_) for k, v_ in buf.items() if k != "vel_pred"}
    if buf["vel_pred"][0].size:
        out["vel_pred"] = np.asarray(buf["vel_pred"])
    out["cmd_grid"] = np.asarray(cmd_grid, dtype=np.float32)
    out["env_cmd_idx"] = cmd_idx.cpu().numpy()
    out["labels"] = np.array([command_label(c) for c in cmd_grid])
    return out


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "collect.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    # ---- 체크포인트 옆 config.yaml 로 학습 설정 복원 (eval_agent.py 와 동일) ----
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

        parser = argparse.ArgumentParser(description="Collect history-encoder latents.")
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

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    logger.info(
        f"envs={env.num_envs}, 커맨드 {len(cmd_grid)}종 → env당 "
        f"{env.num_envs / len(cmd_grid):.1f}개, warmup={num_warmup}, record={num_record} "
        f"(샘플 {env.num_envs * num_record}개)"
    )

    data = collect(env, algo, torch, num_warmup, num_record, cmd_grid, device)

    ckpt_num = checkpoint.stem.split("_")[-1]
    out_path = config.get("out", None)
    out_path = Path(out_path) if out_path else checkpoint.parent / "latents" / f"latents_ckpt_{ckpt_num}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

    logger.info(f"저장 완료: {out_path}")
    logger.info(
        "  " + ", ".join(f"{k}{v.shape}" for k, v in data.items() if hasattr(v, "shape"))
    )
    if "vel_pred" in data:
        err = np.linalg.norm(data["vel_pred"] - data["base_lin_vel"], axis=-1)
        logger.info(f"  vel head MAE={np.abs(data['vel_pred'] - data['base_lin_vel']).mean():.4f}, "
                    f"평균 오차 크기={err.mean():.4f} m/s")

    if simulator_type == "IsaacSim":
        simulation_app.close()


if __name__ == "__main__":
    main()
