"""v4 배포 경로는 v2 와 동일하다 (student(v,z) -> cat[actor_obs, v, z] -> actor MLP).

teacher 는 학습 전용이라 export 대상이 아니므로 v2 wrapper 를 그대로 재사용한다.
"""
from humanoidverse.agents.ppo_hist_v2.inference_wrapper import HistV2InferenceModule

HistV4InferenceModule = HistV2InferenceModule
HistEncoderInferenceModule = HistV2InferenceModule

__all__ = ["HistV4InferenceModule", "HistV2InferenceModule", "HistEncoderInferenceModule"]
