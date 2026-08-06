import torch
from torch import nn
import os
import copy

def export_policy_as_jit(actor_critic, path, exported_policy_name):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def export_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')

        if hasattr(actor, "student") or hasattr(actor, "history_encoder"):
            # ppo_hist v1(PPOActorWithHistoryEncoder) / v2·v3(PPOActorWithStudentEncoder):
            # act_inference(actor_obs, encoder_obs) 2-입력이라 별도 wrapper 로 export.
            # eval() 필수 — v1 encoder 는 train 모드에서 z 를 확률 샘플링하므로
            # 그대로 트레이싱하면 ONNX 에 RandomNormal 노드가 박혀 출력이 비결정적이 된다
            class PPOEncoderWrapper(nn.Module):
                def __init__(self, actor):
                    super().__init__()
                    self.actor = actor

                def forward(self, actor_obs, encoder_obs):
                    return self.actor.act_inference(actor_obs, encoder_obs)

            wrapper = PPOEncoderWrapper(actor)
            wrapper.eval()
            example_input_list = (example_obs_dict["actor_obs"], example_obs_dict["encoder_obs"])
            torch.onnx.export(
                wrapper,
                example_input_list,
                path,
                verbose=True,
                input_names=["actor_obs", "encoder_obs"],
                output_names=["action"],
                opset_version=13
            )
            return

        class PPOWrapper(nn.Module):
            def __init__(self, actor):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOWrapper, self).__init__()
                self.actor = actor

            def forward(self, actor_obs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                return self.actor.act_inference(actor_obs)

        wrapper = PPOWrapper(actor)
        example_input_list = example_obs_dict["actor_obs"]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs"],  # Specify the input names
            output_names=["action"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )

def export_policy_and_estimator_as_onnx(inference_model, path, exported_policy_name, example_obs_dict):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, exported_policy_name)

        actor = copy.deepcopy(inference_model['actor']).to('cpu')
        left_hand_force_estimator = copy.deepcopy(inference_model['left_hand_force_estimator']).to('cpu')
        right_hand_force_estimator = copy.deepcopy(inference_model['right_hand_force_estimator']).to('cpu')

        class PPOForceEstimatorWrapper(nn.Module):
            def __init__(self, actor, left_hand_force_estimator, right_hand_force_estimator):
                """
                model: The original PyTorch model.
                input_keys: List of input names as keys for the input dictionary.
                """
                super(PPOForceEstimatorWrapper, self).__init__()
                self.actor = actor
                self.left_hand_force_estimator = left_hand_force_estimator
                self.right_hand_force_estimator = right_hand_force_estimator

            def forward(self, inputs):
                """
                Dynamically creates a dictionary from the input keys and args.
                """
                actor_obs, history_for_estimator = inputs
                left_hand_force_estimator_output = self.left_hand_force_estimator(history_for_estimator)
                right_hand_force_estimator_output = self.right_hand_force_estimator(history_for_estimator)
                input_for_actor = torch.cat([actor_obs, left_hand_force_estimator_output, right_hand_force_estimator_output], dim=-1)
                return self.actor.act_inference(input_for_actor), left_hand_force_estimator_output, right_hand_force_estimator_output

        wrapper = PPOForceEstimatorWrapper(actor, left_hand_force_estimator, right_hand_force_estimator)
        example_input_list = [example_obs_dict["actor_obs"], example_obs_dict["long_history_for_estimator"]]
        torch.onnx.export(
            wrapper,
            example_input_list,  # Pass x1 and x2 as separate inputs
            path,
            verbose=True,
            input_names=["actor_obs", "long_history_for_estimator"],  # Specify the input names
            output_names=["action", "left_hand_force_estimator_output", "right_hand_force_estimator_output"],       # Name the output
            opset_version=13           # Specify the opset version, if needed
        )