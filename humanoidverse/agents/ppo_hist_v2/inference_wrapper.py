import torch
import torch.nn as nn


class HistV2InferenceModule(nn.Module):
    """student(v,z) -> cat[actor_obs, v, z] -> actor MLP -> action mean. For ONNX/JIT export.

    teacher 는 배포에 불필요하므로 포함하지 않는다.
    """
    def __init__(self, actor, actor_obs_dim, encoder_obs_dim):
        super().__init__()
        self.student = actor.student
        self.actor_mlp = actor.actor
        self.actor_obs_dim = actor_obs_dim
        self.encoder_obs_dim = encoder_obs_dim

    def forward(self, obs):
        actor_obs = obs[..., : self.actor_obs_dim]
        encoder_obs = obs[..., self.actor_obs_dim : self.actor_obs_dim + self.encoder_obs_dim]
        v, z = self.student(encoder_obs)
        return self.actor_mlp(torch.cat([actor_obs, v, z], dim=-1))


# v1 이름으로 import 하는 코드와의 호환용 별칭
HistEncoderInferenceModule = HistV2InferenceModule
