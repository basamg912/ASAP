import torch
import torch.nn as nn


class HistEncoderInferenceModule(nn.Module):
    """encoder(z=mu) -> cat[actor_obs, z] -> actor MLP -> action mean. For ONNX/JIT export."""
    def __init__(self, actor, actor_obs_dim, encoder_obs_dim):
        super().__init__()
        self.history_encoder = actor.history_encoder
        self.actor_mlp = actor.actor
        self.actor_obs_dim = actor_obs_dim
        self.encoder_obs_dim = encoder_obs_dim

    def forward(self, obs):
        actor_obs = obs[..., : self.actor_obs_dim]
        encoder_obs = obs[..., self.actor_obs_dim : self.actor_obs_dim + self.encoder_obs_dim]
        mu, _ = self.history_encoder(encoder_obs)   # deterministic mean latent
        return self.actor_mlp(torch.cat([actor_obs, mu], dim=-1))
