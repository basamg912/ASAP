import os
import copy
import torch
from torch import nn
import numpy as np
import random

from typing import Any, List, Dict
from termcolor import colored
from loguru import logger

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def pre_process_config(config) -> None:
    
    # compute observation_dim
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1
    
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
    config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {each_dict_obs_dims}")
    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            assert _key in config.env.config.obs.obs_dims.keys()
            auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"): key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
    config.robot.algo_obs_dim_dict = obs_dim_dict
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim
                
    # print the config
    logger.debug(f"PPO CONFIG")
    logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")

def apply_obs_noise(data: torch.Tensor, noise_cfg: Any, curriculum_scale: float = 1.0) -> torch.Tensor:
    """IsaacLab isaaclab.utils.noise 의 NoiseCfg 3종을 대응한다.

    지원 형식 (noise_scales 의 값):
      - scalar x        → UniformNoiseCfg(n_min=-x, n_max=+x)  [기존 config 호환]
      - {n_min, n_max}  → UniformNoiseCfg   (비대칭 가능 — 센서 바이어스 모사)
      - {mean, std}     → GaussianNoiseCfg
      - {bias}          → ConstantNoiseCfg
      - operation: add(기본) | scale | abs — IsaacLab NoiseCfg.operation 과 동일

    curriculum_scale 은 노이즈 크기에만 곱한다(데이터에는 곱하지 않는다).
    """
    if noise_cfg is None:
        return data

    # scalar 형식: 기존 config 와의 하위 호환 (대칭 uniform)
    if isinstance(noise_cfg, (int, float)):
        n = float(noise_cfg) * curriculum_scale
        if n == 0.0:
            return data
        return data + (torch.rand_like(data) * 2.0 - 1.0) * n

    get = noise_cfg.get
    operation = get("operation", "add")

    if "n_min" in noise_cfg or "n_max" in noise_cfg:
        n_min = float(get("n_min", -1.0)) * curriculum_scale
        n_max = float(get("n_max", 1.0)) * curriculum_scale
        noise = torch.rand_like(data) * (n_max - n_min) + n_min
    elif "std" in noise_cfg or "mean" in noise_cfg:
        mean = float(get("mean", 0.0)) * curriculum_scale
        std = float(get("std", 1.0)) * curriculum_scale
        noise = mean + std * torch.randn_like(data)
    elif "bias" in noise_cfg:
        noise = torch.zeros_like(data) + float(get("bias", 0.0)) * curriculum_scale
    else:
        raise ValueError(f"Unknown obs noise cfg: {noise_cfg}")

    if operation == "add":
        return data + noise
    elif operation == "scale":
        return data * noise
    elif operation == "abs":
        return noise
    raise ValueError(f"Unknown operation in noise: {operation}")


def parse_observation(cls: Any,
                      key_list: List,
                      buf_dict: Dict,
                      obs_scales: Dict,
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any) -> None:
    """ Parse observations for the legged_robot_base class

    적용 순서는 IsaacLab ObservationManager 와 동일하다: func → noise → scale
    (observation_manager.py:395-407). 노이즈는 스케일 이전의 물리 단위로 준다.
    """

    for obs_key in key_list:
        if obs_key.endswith("_raw"):
            obs_key = obs_key[:-4]
            noise_cfg = None
        else:
            noise_cfg = noise_scales[obs_key]

        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        noisy_obs = apply_obs_noise(actor_obs, noise_cfg, current_noise_curriculum_value)
        buf_dict[obs_key] = noisy_obs * obs_scales[obs_key]


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
