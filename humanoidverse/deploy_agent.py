'''
Usage
python ASAP/humanoidverse/deploy_agent.py +simulator=isaacsim num_envs=1 headless=True +num_collect_steps=2000 +collect_save_dir=motionData +checkpoint=

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

    # collection settings (override on CLI: +num_collect_steps=2000 +collect_save_dir=motionData)
    num_collect_steps = int(config.get("num_collect_steps", 1000))
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

    # ------------------------------------------------------------------
    # Rollout: run the trained policy and record observations & actions
    # ------------------------------------------------------------------
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    algo.eval_policy = algo._get_inference_policy()

    actor_state = algo._create_actor_state()
    obs_dict = env.reset_all()
    init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
    actor_state.update({"obs": obs_dict, "actions": init_actions})

    obs_buffer = {k: [] for k in obs_dict.keys()}
    act_buffer = []
    done_buffer = []

    logger.info(f"Collecting {num_collect_steps} steps from {env.num_envs} envs")
    with torch.no_grad():
        for step in range(num_collect_steps):
            actor_state["step"] = step
            actor_state = algo._pre_eval_env_step(actor_state)

            # record the observation the policy acted on, and its action
            for k in obs_buffer:
                obs_buffer[k].append(actor_state["obs"][k].detach().cpu().clone())
            act_buffer.append(actor_state["actions"].detach().cpu().clone())

            actor_state = algo.env_step(actor_state)
            actor_state = algo._post_eval_env_step(actor_state)

            done_buffer.append(actor_state["dones"].detach().cpu().clone())

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

    if simulator_type == "IsaacSim":
        simulation_app.close()


if __name__ == "__main__":
    main()
