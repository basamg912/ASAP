import torch
from torch import Tensor
from termcolor import colored
from loguru import logger

class HistoryHandler:
    
    def __init__(
        self,
        num_envs,
        history_config,
        obs_dims,
        device,
        fill_on_first_add=False,
    ):
        self.obs_dims = obs_dims
        self.device = device
        self.num_envs = num_envs
        self.fill_on_first_add = fill_on_first_add
        self.history = {}
        self.num_adds = {}

        self.buffer_config = {}
        for aux_key, aux_config in history_config.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
                else:
                    self.buffer_config[obs_key] = obs_num
        
        for key in self.buffer_config.keys():
            self.history[key] = torch.zeros(num_envs, self.buffer_config[key], obs_dims[key], device=self.device)
            self.num_adds[key] = torch.zeros(
                num_envs, dtype=torch.long, device=self.device
            )

        logger.info(colored("History Handler Initialized", "green"))
        for key, value in self.buffer_config.items():
            logger.info(f"Key: {key}, Value: {value}")

    def reset(self, reset_ids):
        if len(reset_ids)==0:
            return
        assert set(self.buffer_config.keys()) == set(self.history.keys()), f"History keys mismatch\n{self.buffer_config.keys()}\n{self.history.keys()}"
        for key in self.history.keys():
            self.history[key][reset_ids] = 0.
            self.num_adds[key][reset_ids] = 0

    def add(self, key: str, value: Tensor):
        assert key in self.history.keys(), f"Key {key} not found in history"
        history = self.history[key]
        history[:, 1:] = history[:, :-1].clone()
        history[:, 0] = value

        if self.fill_on_first_add:
            first_add = self.num_adds[key] == 0
            if torch.any(first_add):
                history[first_add] = value[first_add].unsqueeze(1).expand(
                    -1, history.shape[1], -1
                )
        self.num_adds[key] += 1
        
    def query(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key].clone()
