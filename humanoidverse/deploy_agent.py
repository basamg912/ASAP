'''
학습된 정책을 배포(rollout)하며 obs/action/dones/qpos 를 수집한다.
태스크(motion tracking / locomotion)는 env 에 motion library 유무로 자동 분기된다.

[motion tracking] robot.motion.motion_file 의 모든 모션(.pkl)을 순회하며 모션당
    +runs_per_motion(기본 2)회 rollout 을 수집, <모션이름>_run<n>.pt 로 저장.
    MUJOCO_GL=egl python humanoidverse/deploy_agent.py +simulator=isaacsim \
        +checkpoint=logs/MotionTracking/.../model_26100.pt +num_envs=1 +num_collect_steps=2000

[locomotion] velocity command 로 구동하며 +num_rollouts(기본 runs_per_motion)회 수집,
    locomotion_run<n>.pt 로 저장. 커맨드는 아래 중 하나:
      - +algo.config.eval_command=[vx,vy,yaw,heading] → 전 rollout 고정
      - 미지정 시 command range 에서 매 rollout 랜덤 샘플 (+randomize_command=False 로 끄면 0 커맨드)
    MUJOCO_GL=egl python humanoidverse/deploy_agent.py +simulator=mujoco \
        +checkpoint=logs/kapex_locomotion/.../model_36000.pt +num_envs=1 \
        +robot.asset.xml_file=kapex/kapex_play.xml +terrain.mesh_type=plane \
        +domain_rand.push_robots=False +num_rollouts=8

+save_video=True 이면 수집 qpos 를 mujoco 오프스크린 렌더링으로 <stem>.mp4 저장 (헤드리스는 MUJOCO_GL=egl).
'''
import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from utils.config_utils import *  # noqa: E402, F403

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.logging import HydraLoggerBridge


def render_qpos_video(
    mjcf_path, qpos_frames, dof_names, fps, out_path, width=720, height=480
):
    """Offscreen-render collected robot states (root pos+quat wxyz, dof) to mp4.

    Headless 환경에서는 MUJOCO_GL=egl 필요.
    """
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mjdata = mujoco.MjData(model)

    # MJCF offscreen framebuffer 한도 내로 렌더 크기 클램프
    width = min(width, model.vis.global_.offwidth)
    height = min(height, model.vis.global_.offheight)

    has_free_joint = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    dof_qposadr = []
    for name in dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' not found in {mjcf_path}")
        dof_qposadr.append(model.jnt_qposadr[jid])
    dof_qposadr = np.array(dof_qposadr)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.elevation = -15.0
    cam.azimuth = 135.0

    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for qpos in qpos_frames:
            if has_free_joint:
                mjdata.qpos[:7] = qpos[:7]
            mjdata.qpos[dof_qposadr] = qpos[7:]
            mujoco.mj_forward(model, mjdata)
            cam.lookat[:] = qpos[:3]  # follow the root
            renderer.update_scene(mjdata, camera=cam)
            frames.append(renderer.render())

    try:
        import imageio

        imageio.mimsave(str(out_path), frames, fps=fps)
    except ImportError:
        import cv2

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "deploy.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is None:
        raise ValueError("deploy_agent.py requires a trained policy: checkpoint=<path/to/model_XXXX.pt>")

    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find training config next to checkpoint: {config_path}")

    logger.info(f"Loading training config file from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

    config = OmegaConf.merge(train_config, override_config)

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        from isaaclab.app import AppLauncher
        import argparse

        parser = argparse.ArgumentParser(
            description="Collect observations/actions from a trained RL agent."
        )
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        # offscreen rendering is required to capture frames while headless
        if config.get("save_video", False):
            args_cli.enable_cameras = True

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)

    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # collection settings (override on CLI: +num_collect_steps=2000 +collect_save_dir=motionData +runs_per_motion=2)
    num_collect_steps = int(config.get("num_collect_steps", 1000))  # per-rollout step cap
    runs_per_motion = int(config.get("runs_per_motion", 2))
    save_dir = Path(config.get("collect_save_dir", "motionData"))
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving deploy logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
    )
    config.env.config.ckpt_dir = str(checkpoint.parent)
    env = instantiate(config.env, device=device)

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    if env.num_envs != 1:
        logger.warning(
            f"num_envs={env.num_envs}: per-rollout collection assumes num_envs=1."
        )

    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    algo.eval_policy = algo._get_inference_policy()

    # 태스크 판별: motion tracking 은 motion library 로 구동, locomotion 은 velocity command 로 구동
    is_motion_tracking = hasattr(env, "_motion_lib")
    sim_name = config.simulator.config.name

    # 수집 qpos 로 mujoco 오프스크린 비디오를 렌더 (모든 시뮬레이터에서 이식 가능)
    save_video = bool(config.get("save_video", False))
    mjcf_path = (
        Path(config.robot.motion.asset.assetRoot)
        / config.robot.motion.asset.assetFileName
    )
    dof_names = list(config.robot.dof_names)
    render_fps = round(1.0 / env.dt)

    def capture_qpos():
        """robot_root_states → mujoco free-joint qpos [pos(3), quat wxyz(4), dof]."""
        root = env.simulator.robot_root_states[0].detach().cpu()
        if sim_name == "isaacsim":
            quat_wxyz = root[3:7]  # isaacsim: robot_root_states 는 이미 wxyz
        else:
            quat_wxyz = root[[6, 3, 4, 5]]  # isaacgym/mujoco/genesis: xyzw → wxyz
        dof_pos = env.simulator.dof_pos[0].detach().cpu()
        return torch.cat([root[:3], quat_wxyz, dof_pos])

    def rollout():
        """Run one episode until done (or step cap); return buffers."""
        actor_state = algo._create_actor_state()
        obs_dict = env.reset_all()
        init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        obs_buffer = {k: [] for k in obs_dict.keys()}
        act_buffer, done_buffer, qpos_buffer = [], [], []

        with torch.no_grad():
            for step in range(num_collect_steps):
                actor_state["step"] = step
                actor_state = algo._pre_eval_env_step(actor_state)

                # 정책이 본 관측과 그때 낸 액션을 기록
                for k in obs_buffer:
                    obs_buffer[k].append(actor_state["obs"][k].detach().cpu().clone())
                act_buffer.append(actor_state["actions"].detach().cpu().clone())

                actor_state = algo.env_step(actor_state)
                actor_state = algo._post_eval_env_step(actor_state)

                dones = actor_state["dones"].detach().cpu().clone()
                done_buffer.append(dones)
                qpos_buffer.append(capture_qpos())

                if dones[0]:
                    break

        return obs_buffer, act_buffer, done_buffer, qpos_buffer

    def save_rollout(stem, obs_buffer, act_buffer, done_buffer, qpos_buffer, meta):
        data = {
            "obs": {k: torch.stack(v) for k, v in obs_buffer.items()},
            "actions": torch.stack(act_buffer),
            "dones": torch.stack(done_buffer),
            "qpos": torch.stack(qpos_buffer),
            "checkpoint": str(checkpoint),
            "num_envs": env.num_envs,
            "num_steps": len(act_buffer),
            "fps": render_fps,
            **meta,
        }
        data_path = save_dir / f"{stem}.pt"
        torch.save(data, data_path)
        logger.info(
            f"saved {data_path} ({len(act_buffer)} steps, "
            f"actions {tuple(data['actions'].shape)})"
        )
        if save_video:
            video_path = save_dir / f"{stem}.mp4"
            try:
                render_qpos_video(
                    mjcf_path, data["qpos"].numpy(), dof_names, render_fps, video_path
                )
                logger.info(f"saved {video_path}")
            except Exception as e:
                logger.error(f"video rendering failed for {stem}: {e}")

    if is_motion_tracking:
        # --------------------------------------------------------------
        # Motion tracking: 모션 디렉토리를 순회하며 모션당 runs_per_motion 회 수집
        # --------------------------------------------------------------
        num_motions = env.num_motions
        motion_keys = [Path(str(k)).stem for k in env._motion_lib._motion_data_keys]
        logger.info(
            f"[motion_tracking] collecting {runs_per_motion} rollouts x {num_motions} "
            f"motions (per-rollout step cap: {num_collect_steps})"
        )
        for motion_idx in range(num_motions):
            env._motion_lib.load_motions(random_sample=False, start_idx=motion_idx)
            motion_key = motion_keys[motion_idx]
            for run in range(runs_per_motion):
                logger.info(f"[{motion_idx + 1}/{num_motions}] {motion_key} run {run}")
                buffers = rollout()
                save_rollout(
                    f"{motion_key}_run{run}", *buffers,
                    {"motion_key": motion_key, "motion_idx": motion_idx, "run": run},
                )
    else:
        # --------------------------------------------------------------
        # Locomotion: velocity command 로 구동하여 num_rollouts 회 수집
        #   +algo.config.eval_command=[vx,vy,yaw,heading] → 고정 커맨드
        #   미지정 시 매 rollout 마다 command range 에서 랜덤 샘플 (데이터 다양성)
        # --------------------------------------------------------------
        n_rollouts = int(config.get("num_rollouts", runs_per_motion))
        eval_command = None
        try:
            eval_command = config.algo.config.get("eval_command", None)
        except Exception:
            pass
        randomize_command = bool(config.get("randomize_command", eval_command is None))
        logger.info(
            f"[locomotion] collecting {n_rollouts} rollouts "
            f"(fixed_command={None if eval_command is None else list(eval_command)}, "
            f"randomize_command={randomize_command}, step cap: {num_collect_steps})"
        )
        all_env_ids = torch.arange(env.num_envs, device=device)
        for run in range(n_rollouts):
            # 커맨드 설정: eval 모드에서는 reset 이 커맨드를 덮어쓰지 않으므로
            # (locomotion._reset_tasks_callback) rollout 전에 세팅하면 유지된다.
            if eval_command is not None:
                c = torch.tensor(list(eval_command), dtype=torch.float32, device=device)
                env.commands[:, : len(c)] = c
            elif randomize_command:
                env._resample_commands(all_env_ids)
            cmd_used = env.commands[0].detach().cpu().tolist()
            logger.info(f"[locomotion] run {run}/{n_rollouts} command={[round(x, 3) for x in cmd_used]}")
            buffers = rollout()
            save_rollout(
                f"locomotion_run{run}", *buffers,
                {"run": run, "command": cmd_used},
            )

    algo._post_evaluate_policy()

    if simulator_type == "IsaacSim":
        simulation_app.close()


if __name__ == "__main__":
    main()
