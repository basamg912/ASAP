import logging
import os
import sys
import threading
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from utils.config_utils import *  # noqa: E402, F403

# add argparse arguments
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.logging import HydraLoggerBridge

# from pynput import keyboard

try:
    from pynput import keyboard
except ImportError:
    logger.warning("pynput not installed. Keyboard input will not be available.")
    keyboard = None

def on_press(key, env):
    try:
        if key.char == "n":
            # 리스너 스레드에서 next_task()를 직접 부르면 IsaacSim 렌더가
            # 메인 스레드 밖에서 호출되어 RuntimeError가 남 — 플래그만 세우고
            # 실제 전환은 평가 루프(ppo.evaluate_policy)가 수행
            env.next_task_requested = True
            logger.info("Next task requested (will switch on next step).")
        # Force Control
        # Force Control
        if hasattr(key, "char"):
            if key.char == "1":
                env.apply_force_tensor[:, env.left_hand_link_index, 2] += 1.0
                logger.info(
                    f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}"
                )
            elif key.char == "2":
                env.apply_force_tensor[:, env.left_hand_link_index, 2] -= 1.0
                logger.info(
                    f"Left hand force: {env.apply_force_tensor[:, env.left_hand_link_index, :]}"
                )
            elif key.char == "3":
                env.apply_force_tensor[:, env.right_hand_link_index, 2] += 1.0
                logger.info(
                    f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}"
                )
            elif key.char == "4":
                env.apply_force_tensor[:, env.right_hand_link_index, 2] -= 1.0
                logger.info(
                    f"Right hand force: {env.apply_force_tensor[:, env.right_hand_link_index, :]}"
                )
    except AttributeError:
        pass


def listen_for_keypress(env):
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()


# from humanoidverse.envs.base_task.base_task import BaseTask
# from humanoidverse.envs.base_task.omnih2o_cfg import OmniH2OCfg


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        from isaaclab.app import AppLauncher
        import argparse

        parser = argparse.ArgumentParser(
            description="Evaluate an RL agent with RSL-RL."
        )
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        args_cli.video = config.video
        if args_cli.video:
            args_cli.enable_cameras = True
            args_cli.video_length = config.video_length
        # offscreen rendering is required to capture frames while headless
        if config.get("save_video", False):
            args_cli.enable_cameras = True
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == "IsaacGym":
        import isaacgym

    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import (
        export_policy_as_jit,
    )
    # history-encoder deploy path: wrap actor(encoder+decoder+mlp) for ONNX/JIT export
    from humanoidverse.agents.ppo_hist.inference_wrapper import HistEncoderInferenceModule

    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
    )
    config.env.config.ckpt_dir = str(
        checkpoint.parent
    )  # commented out for now, might need it back to save motion
    env = instantiate(config.env, device=device)

    # Start a thread to listen for key press
    if keyboard is not None:
        key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
        key_listener_thread.daemon = True
        key_listener_thread.start()

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    # ------------------------------------------------------------------
    # Video recording (IsaacSim only): +save_video=True [+video_resolution=[1280,720]]
    # evaluate_policy() loops forever, so frames are streamed to the mp4
    # every step and the writer is closed at process exit (Ctrl+C included).
    # ------------------------------------------------------------------
    if config.get("save_video", False) and simulator_type == "IsaacSim":
        import atexit

        import imageio
        import omni.replicator.core as rep

        resolution = tuple(config.get("video_resolution", (1280, 720)))
        render_product = rep.create.render_product(
            "/OmniverseKit_Persp", resolution=resolution
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach([render_product])

        sim_cfg = config.simulator.config.sim
        fps = int(sim_cfg.fps / sim_cfg.control_decimation)
        video_path = eval_log_dir / f"eval_video_ckpt_{ckpt_num}.mp4"
        video_writer = imageio.get_writer(video_path, fps=fps, quality=8)
        atexit.register(video_writer.close)
        logger.info(f"Recording video at {resolution} @ {fps} fps to {video_path}")

        # camera follows the robot root; offset relative to the robot (override: +camera_offset=[2.0,-2.0,1.0])
        camera_offset = tuple(config.get("camera_offset", (2.0, -2.0, 1.0)))

        orig_env_step = algo.env_step
        frame_count = [0]
        step_count = [0]

        def env_step_with_capture(actor_state):
            if step_count[0] == 0:
                logger.info("Entered first env_step (sim stepping OK, waiting on first render...)")
            actor_state = orig_env_step(actor_state)
            step_count[0] += 1
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
                video_writer.append_data(frame[..., :3])
                frame_count[0] += 1
                if frame_count[0] == 1:
                    logger.info("First video frame captured")
            if step_count[0] % 100 == 0:
                logger.info(f"steps: {step_count[0]}, video frames: {frame_count[0]}")
            return actor_state

        algo.env_step = env_step_with_capture

    EXPORT_POLICY = False
    EXPORT_ONNX = True

    checkpoint_path = str(checkpoint)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # from checkpoint path

    HV_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exported_policy_path = os.path.join(HV_ROOT_DIR, checkpoint_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")

    if EXPORT_POLICY:
        export_policy_as_jit(
            algo.alg.actor_critic, exported_policy_path, exported_policy_name
        )
        logger.info(
            "Exported policy as jit script to: ",
            os.path.join(exported_policy_path, exported_policy_name),
        )
    if EXPORT_ONNX:
        # history-encoder deploy: export HistEncoderInferenceModule (encoder(z=mu) ->
        # cat[actor_obs, z] -> actor MLP) instead of the base actor_obs-only wrapper,
        # since export_policy_as_onnx() (inference_helpers.py) assumes a plain PPOActor.
        import copy

        actor_obs_dim = algo.algo_obs_dim_dict['actor_obs']
        encoder_obs_dim = algo.algo_obs_dim_dict[algo.encoder_obs_key]

        actor_cpu = copy.deepcopy(algo.inference_model['actor']).to('cpu')
        wrapper = HistEncoderInferenceModule(actor_cpu, actor_obs_dim, encoder_obs_dim)
        wrapper.eval()

        example_obs_dict = algo.get_example_obs()
        example_input = torch.cat(
            [example_obs_dict['actor_obs'], example_obs_dict[algo.encoder_obs_key]], dim=-1
        )

        onnx_path = os.path.join(exported_policy_path, exported_onnx_name)
        torch.onnx.export(
            wrapper,
            example_input,
            onnx_path,
            verbose=True,
            input_names=["obs"],  # cat([actor_obs, encoder_obs], dim=-1)
            output_names=["action"],
            opset_version=13,
        )
        logger.info(f"Exported policy as onnx to: {onnx_path}")

    logger.info("Starting evaluate_policy (resetting envs, then stepping)...")
    algo.evaluate_policy()


if __name__ == "__main__":
    main()
