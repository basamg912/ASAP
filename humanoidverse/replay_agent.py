'''
deploy_agent.py 로 수집한 rollout 의 action 만을 source simulator(isaaclab/isaacsim)에서
open-loop 으로 재생하여 궤적 재현 가능성을 검증한다. 정책(algo)은 로드하지 않는다.

    python humanoidverse/replay_agent.py +simulator=isaacsim \
        +replay_data=motionData/locomotion_run0.pt headless=True save_video=True

+replay_data 는 파일 하나, 콤마로 구분된 목록, 또는 디렉토리(locomotion_*.pt 전부).
+checkpoint 는 생략 시 replay 데이터에 저장된 checkpoint 경로를 사용한다 (config 탐색용).

초기 상태 정렬 (+init_from_data=True, 기본값):
  reset 시 dof_pos 에 무조건 U(0.5,1.5) 랜덤화가 걸리므로 (legged_robot_base._reset_dofs)
  기본 reset 으로는 수집 당시 초기 상태를 재현할 수 없다. 대신 수집 데이터에서 step-1
  상태를 정확히 복원해 주입한다:
    - 위치: qpos[0] (root pos/quat wxyz + dof pos)
    - 속도: obs[1] 슬라이스에서 복원 (dof_vel, base_ang_vel, critic_obs 의 base_lin_vel;
            이 run 들은 noise_scales 가 전부 0 이라 정확값이다)
  이후 actions[1:] 을 open-loop 재생하고, 수집된 qpos[k] (k>=1) 와 비교한다.
  (qpos[k] 는 actions[k] 적용 직후 상태이므로 주입 상태 = qpos[0] 이 index 0 에 대응)

결과: <stem>_replay_<sim>.pt + <stem>_replay_<sim>_compare.{json,png,mp4}
(비디오는 save_video=True, 좌: 수집 rollout, 우: replay)

기존 파이프라인(train/eval/deploy)과 수집 데이터 포맷은 일절 수정하지 않는다.
'''
import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from utils.config_utils import *  # noqa: E402, F403

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.logging import HydraLoggerBridge


def build_obs_slice_map(obs_config, group):
    """_post_config_observation_callback 과 동일하게 sorted() 순서로 slice 를 계산한다."""
    dims = dict(obs_config.obs_dims)  # pre_process_config 가 dict 로 flatten 해 둠
    for name, comp in obs_config.get("obs_auxiliary", {}).items():
        dims[name] = sum(dims[k] * c for k, c in comp.items())
    slices, offset = {}, 0
    for key in sorted(obs_config.obs_dict[group]):
        base_key = key[:-4] if key.endswith("_raw") else key
        slices[key] = (offset, offset + dims[base_key])
        offset += dims[base_key]
    return slices, offset


def quat_wxyz_to_rotmat(q):
    """(4,) wxyz quaternion -> (3,3) rotation matrix (torch)."""
    import torch

    w, x, y, z = q / q.norm()
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def reconstruct_step1_state(data, config, device):
    """수집 데이터에서 step-1(= qpos[0]) 시점의 full state 를 복원한다.

    Returns (root_states (1,13) [pos, quat wxyz, lin/ang vel world], dof_states (1,D,2)).
    """
    import torch

    obs_config = config.env.config.obs
    group = "critic_obs" if "critic_obs" in data["obs"] else "actor_obs"
    slices, total = build_obs_slice_map(obs_config, group)
    obs1 = data["obs"][group][1][0].to(device)  # obs[k] 는 qpos[k-1] 상태의 관측
    assert obs1.shape[-1] == total, (
        f"{group} dim mismatch: recorded {obs1.shape[-1]} vs config {total}"
    )
    scales = obs_config.obs_scales

    def get(name):
        if name not in slices:
            return None
        a, b = slices[name]
        return obs1[a:b] / scales[name]

    qpos0 = data["qpos"][0].to(device)
    quat_wxyz = qpos0[3:7]
    R = quat_wxyz_to_rotmat(quat_wxyz)  # base -> world

    dof_vel = get("dof_vel")
    ang_vel_base = get("base_ang_vel")
    lin_vel_base = get("base_lin_vel")
    if lin_vel_base is None:
        logger.warning(
            "base_lin_vel not found in recorded obs groups; using zero root lin vel"
        )
        lin_vel_base = torch.zeros(3, device=device)

    # sanity: obs 에서 복원한 dof_pos 가 qpos[0] 과 일치해야 slice/timing 정렬이 맞는 것
    default_dof_pos = torch.tensor(
        [config.robot.init_state.default_joint_angles.get(n, 0.0)
         for n in config.robot.dof_names], device=device)
    dof_pos_from_obs = get("dof_pos") + default_dof_pos
    align_err = (dof_pos_from_obs - qpos0[7:]).abs().max().item()
    if align_err > 1e-4:
        logger.warning(f"obs/qpos alignment check: max dof err {align_err:.2e} (>1e-4)")
    else:
        logger.info(f"obs/qpos alignment check passed (max dof err {align_err:.2e})")

    root_states = torch.cat([
        qpos0[:3], quat_wxyz, R @ lin_vel_base, R @ ang_vel_base
    ]).unsqueeze(0)
    dof_states = torch.stack([qpos0[7:], dof_vel], dim=-1).unsqueeze(0)
    return root_states, dof_states


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "replay.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    if override_config.get("replay_data", None) is None:
        raise ValueError(
            "replay_agent.py requires +replay_data=<rollout.pt | dir | a.pt,b.pt>"
        )
    replay_arg = str(override_config.replay_data)
    if "," in replay_arg:
        replay_files = [Path(p) for p in replay_arg.split(",")]
    elif Path(replay_arg).is_dir():
        replay_files = sorted(Path(replay_arg).glob("locomotion_*.pt"))
        replay_files = [p for p in replay_files if "_replay_" not in p.stem]
    else:
        replay_files = [Path(replay_arg)]
    for p in replay_files:
        if not p.exists():
            raise FileNotFoundError(f"replay data not found: {p}")

    # AppLauncher 는 torch import 이전에 실행되어야 한다 (deploy_agent.py 와 동일 순서).
    # simulator 는 CLI 로 지정되므로 override_config 만으로 분기 가능하다.
    if override_config.get("simulator", None) is None:
        raise ValueError("replay_agent.py requires +simulator=<isaacsim|mujoco|...>")
    simulator_type = override_config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        from isaaclab.app import AppLauncher
        import argparse

        parser = argparse.ArgumentParser(description="Open-loop action replay.")
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = 1
        args_cli.headless = override_config.get("headless", True)
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from utils.replay_compare import compare_and_report  # script-dir import (deploy_agent 와 동일 방식)

    first_data = torch.load(replay_files[0], map_location="cpu", weights_only=False)
    checkpoint = Path(override_config.get("checkpoint", None) or first_data["checkpoint"])
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find training config next to checkpoint: {config_path}")
    logger.info(f"Loading training config file from {config_path}")
    train_config = OmegaConf.load(config_path)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    config = OmegaConf.merge(train_config, override_config)
    config.num_envs = 1
    config.checkpoint = str(checkpoint)

    pre_process_config(config)

    device = config.get("device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    init_from_data = bool(config.get("init_from_data", True))
    save_video = bool(config.get("save_video", False))
    save_dir = Path(config.get("collect_save_dir", replay_files[0].parent))
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving replay logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings" / "replay"
    )
    config.env.config.ckpt_dir = str(checkpoint.parent)
    env = instantiate(config.env, device=device)
    env.set_is_evaluating()

    sim_name = config.simulator.config.name
    mjcf_path = (
        Path(config.robot.motion.asset.assetRoot)
        / config.robot.motion.asset.assetFileName
    )
    dof_names = list(config.robot.dof_names)
    render_fps = round(1.0 / env.dt)
    env_ids = torch.arange(env.num_envs, device=device)

    def capture_qpos():
        """deploy_agent.capture_qpos 와 동일: [pos(3), quat wxyz(4), dof]."""
        root = env.simulator.robot_root_states[0].detach().cpu()
        if sim_name == "isaacsim":
            quat_wxyz = root[3:7]
        else:
            quat_wxyz = root[[6, 3, 4, 5]]
        dof_pos = env.simulator.dof_pos[0].detach().cpu()
        return torch.cat([root[:3], quat_wxyz, dof_pos])

    def inject_state(root_states_wxyz, dof_states):
        """복원한 state 를 env 에 주입한다 (reset_all 의 숨은 step 은 수행하지 않음)."""
        root_states = root_states_wxyz.clone()
        if sim_name != "isaacsim":  # robot_root_states 버퍼는 isaacsim 만 wxyz
            root_states[:, 3:7] = root_states[:, [4, 5, 6, 3]]  # wxyz -> xyzw
        env.reset_envs_idx(
            env_ids,
            target_states={"root_states": root_states, "dof_states": dof_states},
        )
        env.simulator.set_actor_root_state_tensor(env_ids, env.simulator.all_root_states)
        env.simulator.set_dof_state_tensor(env_ids, env.simulator.dof_state)
        env.need_to_refresh_envs[env_ids] = False
        env._refresh_sim_tensors()

    for replay_file in replay_files:
        data = torch.load(replay_file, map_location="cpu", weights_only=False)
        if str(data["checkpoint"]) != str(checkpoint):
            logger.warning(
                f"{replay_file.name}: recorded checkpoint differs from replay config "
                f"({data['checkpoint']} vs {checkpoint})"
            )
        assert data["num_envs"] == 1, "open-loop replay assumes num_envs=1 recordings"
        actions = data["actions"].to(device)  # (T, 1, A)
        # 하체-only policy(PPOLowerBody) 수집 데이터: 상체 action 0 으로 패딩 (배포 시와 동일)
        env_act_dim = int(config.robot.actions_dim)
        if actions.shape[-1] < env_act_dim:
            pad = torch.zeros(*actions.shape[:-1], env_act_dim - actions.shape[-1],
                              dtype=actions.dtype, device=device)
            logger.info(f"{replay_file.name}: padding actions "
                        f"{actions.shape[-1]} -> {env_act_dim} (upper-body zeros)")
            actions = torch.cat([actions, pad], dim=-1)
        T = actions.shape[0]
        command = data.get("command", None)

        env.reset_all()
        if command is not None:
            c = torch.tensor(list(command), dtype=torch.float32, device=device)
            env.commands[:, : len(c)] = c

        start_idx = 0
        qpos_buffer, done_buffer = [], []
        if init_from_data:
            root_states, dof_states = reconstruct_step1_state(data, config, device)
            inject_state(root_states, dof_states)
            start_idx = 1
            injected = capture_qpos()
            inject_err = (injected - data["qpos"][0]).abs().max().item()
            logger.info(f"{replay_file.name}: state injected "
                        f"(round-trip max err {inject_err:.2e})")
            qpos_buffer.append(injected)
            done_buffer.append(torch.zeros(1, dtype=data["dones"].dtype))

        logger.info(
            f"[replay] {replay_file.name}: {T - start_idx} open-loop steps on "
            f"{sim_name} (init_from_data={init_from_data})"
        )
        with torch.no_grad():
            for step in range(start_idx, T):
                _, _, reset_buf, _ = env.step({"actions": actions[step]})
                dones = reset_buf.detach().cpu().clone()
                done_buffer.append(dones.reshape(-1)[:1])
                qpos_buffer.append(capture_qpos())
                if dones.reshape(-1)[0]:
                    logger.info(f"[replay] terminated at step {step} "
                                f"(recorded rollout: {T} steps)")
                    break

        replayed = {
            "qpos": torch.stack(qpos_buffer),
            "dones": torch.stack(done_buffer),
            "actions": actions[:len(qpos_buffer)].detach().cpu().clone(),
            "dof_names": dof_names,
            "mjcf_path": str(mjcf_path),
            "fps": render_fps,
            "source": str(replay_file),
            "checkpoint": str(checkpoint),
            "simulator": sim_name,
            "init_from_data": init_from_data,
            "start_idx": start_idx,
            "num_steps": len(qpos_buffer),
        }
        stem = f"{replay_file.stem}_replay_{sim_name}"
        out_path = save_dir / f"{stem}.pt"
        torch.save(replayed, out_path)
        logger.info(f"saved {out_path} ({len(qpos_buffer)} steps)")

        try:
            metrics = compare_and_report(
                data, replayed, save_dir, stem, no_video=not save_video,
                mjcf_path=str(mjcf_path), log=logger.info,
            )
            logger.info(
                f"[compare] {replay_file.name}: steps rec/rep = "
                f"{metrics['num_steps_recorded']}/{metrics['num_steps_replayed']}, "
                f"dof MAE mean {metrics['dof_mae_mean']:.4f} rad, "
                f"root pos err mean {metrics['root_pos_err_mean']:.4f} m, "
                f"divergence step {metrics['divergence_step_dof_mae']}"
            )
        except Exception as e:
            logger.error(f"comparison failed for {stem}: {e}")

    if simulator_type == "IsaacSim":
        # simulation_app.close() 가 headless/EGL 에서 종종 hang 되므로 timeout 후 강제 종료.
        # 모든 산출물은 이미 저장된 상태라 안전하다.
        import threading

        closer = threading.Thread(target=simulation_app.close, daemon=True)
        closer.start()
        closer.join(timeout=60)
        if closer.is_alive():
            logger.warning("simulation_app.close() timed out; forcing exit")
        os._exit(0)


if __name__ == "__main__":
    main()
