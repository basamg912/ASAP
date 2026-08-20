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


from utils.config_utils import *  # noqa: E402, F403


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    simulator_type = config.simulator["_target_"].split(".")[-1]
    # import ipdb; ipdb.set_trace()
    if simulator_type == "IsaacSim":
        from isaaclab.app import AppLauncher
        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        # import ipdb; ipdb.set_trace()
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    # have to import torch after isaacgym
    import torch  # noqa: E402
    from utils.common import seeding
    import wandb
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config, find_resume_checkpoints
    from humanoidverse.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    # auto_load_latest: 재시작으로 timestamp 가 새로 찍혀 dir 이 갈려도 같은
    # experiment 의 이전 run dir 에서 최신 체크포인트를 이어받는다.
    resume_candidates = []
    if config.checkpoint is None and config.auto_load_latest:
        resume_candidates = find_resume_checkpoints(config.experiment_dir)
        if resume_candidates:
            config.checkpoint = str(resume_candidates[0])
            logger.info(f"auto_load_latest: resuming from {config.checkpoint}")
        else:
            logger.info("auto_load_latest: no previous checkpoint found, starting fresh")

    # 스냅샷은 auto-resume 반영 후에 떠야 config.yaml/wandb 에 실제 checkpoint 가 기록된다
    unresolved_conf = OmegaConf.to_container(config, resolve=False)

    if config.use_wandb:
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
        )

    if hasattr(config, "device"):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pre_process_config(config)

    # torch.set_float32_matmul_precision("medium")

    # fabric: Fabric = instantiate(config.fabric)
    # fabric.launch()

    # if config.seed is not None:
    #     rank = fabric.global_rank
    #     if rank is None:
    #         rank = 0
    #     fabric.seed_everything(config.seed + rank)
    #     seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)
    config.env.config.save_rendering_dir = str(
        Path(config.experiment_dir) / "renderings_training"
    )
    env: BaseTask = instantiate(config=config.env, device=device)

    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)

    algo: BaseAlgo = instantiate(
        device=device, env=env, config=config.algo, log_dir=experiment_save_dir
    )
    algo.setup()
    # import ipdb;    ipdb.set_trace()
    if config.checkpoint is not None:
        if resume_candidates:
            # 최신 ckpt 가 저장 도중 전원 차단 등으로 손상됐을 수 있어 순서대로 fallback
            for ckpt in resume_candidates:
                try:
                    algo.load(str(ckpt))
                    config.checkpoint = str(ckpt)
                    break
                except Exception as e:
                    logger.warning(f"auto_load_latest: failed to load {ckpt} ({e}); trying older checkpoint")
            else:
                raise RuntimeError("auto_load_latest: all candidate checkpoints failed to load")
        else:
            algo.load(config.checkpoint)

    # handle saving config
    algo.learn()

    if simulator_type == "IsaacSim":
        simulation_app.close()


if __name__ == "__main__":
    main()
