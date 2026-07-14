'''
Usage
MUJOCO_GL=egl python humanoidverse/deploy_agent.py +simulator=isaacsim     +checkpoint=logs/MotionTracking/20260713_kapex_walkmotion_tracking/model_26100.pt     +opt=record env.config.save_motion=True env.config.save_note=kapex_walk

robot.motion.motion_file 이 디렉토리면 안의 모든 모션(.pkl)을 순회하며
모션당 +runs_per_motion(기본 2)회 rollout 을 수집하고,
각 rollout 을 <모션이름>_run<n>.pt / <모션이름>_run<n>.mp4 쌍으로 저장한다.
비디오는 수집한 qpos 를 mujoco 오프스크린 렌더링으로 생성 (헤드리스는 MUJOCO_GL=egl).
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

    # ------------------------------------------------------------------
    # Video recording (IsaacSim only): +save_video=True [+video_resolution=[1280,720]]
    # ------------------------------------------------------------------
    save_video = bool(config.get("save_video", False)) and simulator_type == "IsaacSim"
    rgb_annotator = None
    video_frames = []
    if save_video:
        import omni.replicator.core as rep

        resolution = tuple(config.get("video_resolution", (1280, 720)))
        render_product = rep.create.render_product(
            "/OmniverseKit_Persp", resolution=resolution
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach([render_product])
        # camera follows the robot root; offset relative to the robot (override: +camera_offset=[2.0,-2.0,1.0])
        camera_offset = tuple(config.get("camera_offset", (2.0, -2.0, 1.0)))
        logger.info(f"Recording video at {resolution} from the viewport camera")

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    # ------------------------------------------------------------------
    # Rollout: iterate motions, run the policy runs_per_motion times each,
    # and save (obs/act data, mujoco-rendered video) pairs per rollout
    # ------------------------------------------------------------------
    if env.num_envs != 1:
        logger.warning(
            f"num_envs={env.num_envs}: per-motion collection assumes num_envs=1; "
            "each load_motions(start_idx) call assigns consecutive motions to envs."
        )

    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    algo.eval_policy = algo._get_inference_policy()

    # MJCF for offscreen video rendering of collected qpos
    mjcf_path = (
        Path(config.robot.motion.asset.assetRoot)
        / config.robot.motion.asset.assetFileName
    )
    dof_names = list(config.robot.dof_names)
    render_fps = round(1.0 / env.dt)

    num_motions = env.num_motions
    # keys can be full paths when motion_file is a directory — keep only the stem
    motion_keys = [Path(str(k)).stem for k in env._motion_lib._motion_data_keys]
    logger.info(
        f"Collecting {runs_per_motion} rollouts x {num_motions} motions "
        f"(per-rollout step cap: {num_collect_steps})"
    )

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

                # record the observation the policy acted on, and its action
                for k in obs_buffer:
                    obs_buffer[k].append(actor_state["obs"][k].detach().cpu().clone())
                act_buffer.append(actor_state["actions"].detach().cpu().clone())

            if save_video:
                root_pos = env.simulator.robot_root_states[0, :3].detach().cpu().tolist()
                env.simulator.sim.set_camera_view(
                    eye=[root_pos[i] + camera_offset[i] for i in range(3)],
                    target=root_pos,
                )
                # headless render is skipped unless has_rtx_sensors(), which a
                # replicator render product does not set — render explicitly
                env.simulator.sim.render()
                frame = rgb_annotator.get_data()
                if frame is not None and frame.size > 0:
                    video_frames.append(frame[..., :3].copy())

            if (step + 1) % 100 == 0:
                logger.info(f"step {step + 1}/{num_collect_steps}")

    algo._post_evaluate_policy()

    # stack to [num_steps, num_envs, dim] tensors and save
    data = {
        "obs": {k: torch.stack(v) for k, v in obs_buffer.items()},
        "actions": torch.stack(act_buffer),
        "dones": torch.stack(done_buffer),
        "checkpoint": str(checkpoint),
        "num_envs": env.num_envs,
        "num_steps": num_collect_steps,
    }
    save_path = save_dir / f"deploy_data_ckpt_{ckpt_num}.pt"
    torch.save(data, save_path)
    logger.info(f"Saved collected data to {save_path}")
    logger.info(
        f"actions shape: {data['actions'].shape}, "
        + ", ".join(f"obs[{k}]: {tuple(v.shape)}" for k, v in data["obs"].items())
    )

    if save_video and video_frames:
        import imageio

        sim_cfg = config.simulator.config.sim
        fps = int(sim_cfg.fps / sim_cfg.control_decimation)
        video_path = save_dir / f"deploy_video_ckpt_{ckpt_num}.mp4"
        imageio.mimwrite(video_path, video_frames, fps=fps, quality=8)
        logger.info(f"Saved video ({len(video_frames)} frames @ {fps} fps) to {video_path}")

    if simulator_type == "IsaacSim":
        simulation_app.close()


if __name__ == "__main__":
    main()
