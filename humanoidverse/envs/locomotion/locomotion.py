from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.math import quat_apply_yaw
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator
from scipy.stats import vonmises


class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._init_gait_params()
        self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
        self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
        self.upper_left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
        self.upper_right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]
        self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
        self.init_done = True

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
        # IsaacLab UniformVelocityCommand.is_standing_env 대응 — 정지 env 를 마스크로
        # 영속 보유한다. legged_gym 의 "norm 이 작으면 0" 방식과 달리 정지 비율이
        # command_ranges 넓이와 무관하게 rel_standing_envs 로 고정된다.
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.feet_air_time_positive_biped = torch.zeros_like(self.feet_air_time)
        self.feet_contact_time_positive_biped = torch.zeros_like(self.feet_air_time)

    def _init_gait_params(self):
        # Initialize the normalized period of the swing phase
        self.a_swing = 0.0 # start of the swing phase
        self.b_swing = 0.5 # end of the swing phase
        self.a_stance = 0.5 # start of the stance phase
        self.b_stance = 1.0 # end of the stance phase
        self.kappa = 4.0 # shared variance in Von Mises

        self.T = float(self.config.rewards.get("gait_period", 1.0))
        if self.T <= 0.0:
            raise ValueError("rewards.gait_period must be greater than zero")

        gait_offsets = list(self.config.rewards.get("gait_offsets", [0.0, 0.5]))
        if len(gait_offsets) != len(self.feet_indices):
            raise ValueError(
                "rewards.gait_offsets must contain one offset for each foot "
                f"({len(self.feet_indices)} expected, got {len(gait_offsets)})"
            )
        self.gait_offsets = torch.tensor(
            gait_offsets, dtype=torch.float32, device=self.device
        )
        self.left_offset = float(gait_offsets[0])
        self.right_offset = float(gait_offsets[1])

        self.gait_stance_threshold = float(
            self.config.rewards.get("gait_stance_threshold", 0.55)
        )
        if not 0.0 < self.gait_stance_threshold < 1.0:
            raise ValueError("rewards.gait_stance_threshold must be between zero and one")

        self.left_feet_height = torch.zeros(self.num_envs, device=self.device) # left feet height
        self.right_feet_height = torch.zeros(self.num_envs, device=self.device) # right feet height

        self.phase_time = torch.zeros(self.num_envs, dtype=torch.float32, requires_grad=False, device=self.device)
        self.phase_time_np = np.zeros(self.num_envs, dtype=np.float32)
        self.leg_phase = (
            self.phase_time.unsqueeze(1) + self.gait_offsets.unsqueeze(0)
        ) % 1.0
        self.phase_left = self.leg_phase[:, 0]
        self.phase_right = self.leg_phase[:, 1]
        self.phi_offset = np.zeros(self.num_envs, dtype=np.float32)
        # Initialize the target arm joint positions
        self.swing_arm_joint_pos = torch.tensor([-1.04, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        self.stance_arm_joint_pos = torch.tensor([0.757, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        print("phi_offset: ", self.phi_offset)


    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """Resample direct velocity commands and update the standing mask."""
        super()._update_tasks_callback()

        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        if not self.is_evaluating:
            self.commands[self.is_standing_env] = 0.0

    def _post_physics_step(self):
        super()._post_physics_step()
        self.update_phase_time()

    def update_phase_time(self):
        # Update the phase time
        self.phase_time_np = self._calc_phase_time()
        self.phase_time = torch.tensor(self.phase_time_np, device=self.device, dtype=torch.float, requires_grad=False)
        self.leg_phase = (
            self.phase_time.unsqueeze(1) + self.gait_offsets.unsqueeze(0)
        ) % 1.0
        self.phase_left = self.leg_phase[:, 0]
        self.phase_right = self.leg_phase[:, 1]

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)

        # 정지 env 배정 (IsaacLab UniformVelocityCommand._resample_command 와 동일):
        # 확률로 마스크만 정하고, 실제 0 대입은 매 스텝 _update_tasks_callback 에서 한다.
        #
        # legged_gym 의 "set small commands to zero" (norm < 0.2 면 0) 는 제거했다.
        # 그 방식은 range >> threshold 를 암묵 전제하는데(legged_gym 기본 [-1,1]^2 에서
        # 약 3%), command curriculum 이 range 를 좁히면 전제가 깨진다
        # (initial_ranges [-0.1,0.1]^2 에서는 78.5% 가 정지로 뭉개졌음).
        # IsaacLab 2.3 에는 이 로직이 없고 rel_standing_envs 확률만 쓴다.
        stand_prob = self.config.get("locomotion_stand_still_prob", 0.0)
        self.is_standing_env[env_ids] = (
            torch.rand(len(env_ids), device=self.device) <= stand_prob
        )


    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self.feet_air_time_positive_biped[env_ids] = 0.0
        self.feet_contact_time_positive_biped[env_ids] = 0.0
        if not self.is_evaluating:
            self._resample_commands(env_ids)

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # TODO: haotian: adding command configuration
        if command is not None:
            command = torch.tensor(command, dtype=torch.float32, device=self.device)
            command_size = min(command.numel(), self.commands.shape[1])
            self.commands[:, :command_size] = command[:command_size]

    ########################### TRACKING REWARDS ###########################

    def _get_base_lin_vel_yaw_frame(self):
        """Return world linear velocity in the gravity-aligned yaw frame."""
        root_lin_vel_w = self.simulator.robot_root_states[:, 7:10]
        return quat_apply_yaw(quat_conjugate(self.base_quat), root_lin_vel_w)

    def _get_lin_vel_tracking_error(self):
        return self.commands[:, :2] - self._get_base_lin_vel_yaw_frame()[:, :2]

    def _reward_tracking_lin_vel(self):
        # Unitree RL Lab: compare XY commands in the gravity-aligned yaw frame.
        lin_vel_error = torch.sum(torch.square(self._get_lin_vel_tracking_error()), dim=1)
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity

        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)


    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime = torch.sum((torch.clamp(self.feet_air_time, max=0.45) - 0.3) * first_contact, dim=1)
        rew_airTime *= ~self.is_standing_env  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _gait_motion_gate(self):
        """Gate gait rewards by matching commanded and achieved motion magnitudes.

        커맨드만 보는 게이트는 위상 시계만 맞추면 만점이라 제자리걸음으로
        gait 리워드를 전부 수확할 수 있다 (20260813 런: contact raw 0.85/2.0,
        feet_clearance raw 0.81 인데 tracking_lin_vel 은 0.53 정체).

        실제 속도만 보는 게이트는 작은 커맨드에서도 threshold 속도까지 과속하면
        gait 리워드가 최대가 되는 반대 loophole 이 있다. command와 achieved gate의
        최솟값을 써서 정지 정책은 계속 0점으로 두되, 과속으로 gate를 키울 수 없게
        한다. 선형 이동과 제자리 선회는 서로의 gate를 대신 채우지 않도록 분리한다.
        """
        lin_thr = float(self.config.rewards.get("gait_gate_lin_vel_threshold", 0.1))
        ang_thr = float(self.config.rewards.get("gait_gate_ang_vel_threshold", 0.1))
        if lin_thr <= 0.0 or ang_thr <= 0.0:
            raise ValueError("gait motion gate thresholds must be greater than zero")

        cmd_lin = torch.norm(self.commands[:, :2], dim=1)
        achieved_lin = torch.norm(self.base_lin_vel[:, :2], dim=1)
        cmd_ang = torch.abs(self.commands[:, 2])
        achieved_ang = torch.abs(self.base_ang_vel[:, 2])

        lin_gate = torch.minimum(
            torch.clamp(cmd_lin / lin_thr, max=1.0),
            torch.clamp(achieved_lin / lin_thr, max=1.0),
        )
        ang_gate = torch.minimum(
            torch.clamp(cmd_ang / ang_thr, max=1.0),
            torch.clamp(achieved_ang / ang_thr, max=1.0),
        )
        return (~self.is_standing_env).float() * torch.maximum(lin_gate, ang_gate)

    def _reward_feet_air_time_positive_biped(self):
        """Reward time spent in single stance, following Isaac Lab's biped reward."""
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0

        self.feet_contact_time_positive_biped = torch.where(
            contact,
            self.feet_contact_time_positive_biped + self.dt,
            torch.zeros_like(self.feet_contact_time_positive_biped),
        )
        self.feet_air_time_positive_biped = torch.where(
            contact,
            torch.zeros_like(self.feet_air_time_positive_biped),
            self.feet_air_time_positive_biped + self.dt,
        )

        in_mode_time = torch.where(
            contact,
            self.feet_contact_time_positive_biped,
            self.feet_air_time_positive_biped,
        )
        single_stance = torch.sum(contact.int(), dim=1) == 1
        reward = torch.min(
            torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)),
            dim=1,
        ).values

        threshold = float(self.config.rewards.get("feet_air_time_positive_biped_threshold", 0.4))
        reward = torch.clamp(reward, max=threshold)
        reward *= self._gait_motion_gate()
        return reward

    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward



    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)




    def _reward_base_height(self):
        # Penalize base height away from target

        base_height = self.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)



    def _reward_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5

    def _reward_penalty_feet_slippage(self):
        return self._reward_penalty_slippage()

    def _reward_feet_clearance(self):
        """Reward moving feet for remaining close to the target clearance height."""
        target_height = float(
            self.config.rewards.get(
                "feet_clearance_target_height",
                self.config.rewards.feet_height_target,
            )
        )
        std = float(self.config.rewards.get("feet_clearance_std", 0.05))
        tanh_mult = float(self.config.rewards.get("feet_clearance_tanh_mult", 2.0))

        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        feet_height_error = torch.square(feet_height - target_height)
        feet_xy_speed = torch.norm(
            self.simulator._rigid_body_vel[:, self.feet_indices, :2], dim=2
        )
        velocity_weight = torch.tanh(tanh_mult * feet_xy_speed)
        weighted_error = feet_height_error * velocity_weight
        # 게이트 없이는 발이 멈춰 있을 때 velocity_weight=0 → exp(0)=1.0 만점이라
        # 정지가 최적해가 된다. 실제 이동 중에만 지급한다.
        return torch.exp(-torch.sum(weighted_error, dim=1) / std) * self._gait_motion_gate()


    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        moving = ~self.is_standing_env
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target
        return torch.clip(dif - 0.02, min=0.) * moving

    def _reward_penalty_feet_swing_height(self):
        # 스윙 발의 높이 오차 벌점. 게이트를 실제 접촉(~contact)이 아니라 gait 클록의
        # swing 위상으로 잡는다.
        # 접촉 게이트는 "발을 아예 안 떼면 평가 대상에서 빠지는" 구조라 질질 끄는 보행이
        # 벌점 0 이 되고, 반대로 살짝 뗀 발(제곱오차 최대 지점)만 처벌받아 들기 학습을
        # 막았다. 위상 게이트는 접촉 여부로 도망갈 수 없어 끌기를 직접 벌한다.
        moving = ~self.is_standing_env
        is_swing = self.leg_phase >= self.gait_stance_threshold
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        height_error = torch.square(feet_height - self.config.rewards.feet_height_target)
        return torch.sum(height_error * is_swing, dim=1) * moving

    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0


    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0


    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
        return torch.sum(deviation, dim=1)

    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1)

    def _reward_penalty_hip_pos_l1(self):
        # hip_pos 의 L1 버전 (IsaacLab joint_deviation_l1(legs) 와 동일 커널, roll/yaw 만).
        # L2 는 0 근처 기울기가 소멸해 작은 편차가 공짜 — waist_pos_l1 과 같은 논리.
        # 기존 hip_pos 는 절대각 기준이지만 G1 hip roll/yaw default 가 0 이라 동치이고,
        # default 편차 기준이 다른 로봇에도 안전해 이쪽으로 통일한다.
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        hip_default = self.default_dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.abs(hip_pos - hip_default), dim=1)

    def _reward_penalty_waist_pos(self):
        # 허리(waist_dof_names) 관절이 default 에서 벗어나면 벌점 — 몸통 yaw 비틀림 방지.
        # penalty_torso_ori(중력 기반)는 yaw 불변이라 허리 비틀림을 못 잡음
        waist_pos = self.simulator.dof_pos[:, self.waist_dof_indices]
        waist_default = self.default_dof_pos[:, self.waist_dof_indices]
        return torch.sum(torch.square(waist_pos - waist_default), dim=1)

    def _reward_penalty_waist_pos_l1(self):
        # waist_pos 의 L1 버전 (IsaacLab joint_deviation_l1 과 동일 커널).
        # L2 는 0 근처에서 기울기가 소멸해(∂q²∝q) "약간 젖힌" 자세가 사실상 공짜 —
        # pelvis 를 orientation 으로 강하게 잡으면 waist pitch 가 탈출구가 된다.
        # L1 은 편차 크기와 무관하게 일정한 복원 압력을 유지한다.
        waist_pos = self.simulator.dof_pos[:, self.waist_dof_indices]
        waist_default = self.default_dof_pos[:, self.waist_dof_indices]
        return torch.sum(torch.abs(waist_pos - waist_default), dim=1)

    def _reward_penalty_torso_ori(self):
        # 몸통(torso_name 링크)이 직립에서 기울어지면 벌점.
        # KAPEX WL3는 직립 시 프레임 z축이 월드 +y를 향하므로(z-up 아님)
        # config의 기준 로컬 중력 벡터(torso_upright_local_gravity)와의 편차로 판정
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        torso_gravity = quat_rotate_inverse(torso_quat, self.gravity_vec)
        target = torch.tensor(
            list(self.config.rewards.get("torso_upright_local_gravity", [0.0, 0.0, -1.0])),
            dtype=torch.float32, device=self.device)
        return torch.norm(torso_gravity - target, dim=1)

    def _reward_penalty_stand_still(self):
        # zero command에서 발을 떼거나 움직이면 벌점 — 제자리 gait 스테핑 방지
        zero_cmd = self.is_standing_env
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        feet_off_ground = (~contact).sum(dim=1).float()
        feet_xy_vel = torch.norm(
            self.simulator._rigid_body_vel[:, self.feet_indices, :2], dim=-1).sum(dim=1)
        return (feet_off_ground + feet_xy_vel) * zero_cmd

    ########################### GAIT REWARDS ###########################
    def _calc_phase_time(self):
        # Calculate the phase time
        episode_length_np = self.episode_length_buf.cpu().numpy()
        phase_time = (episode_length_np * self.dt + self.phi_offset) % self.T / self.T
        return phase_time

    def _reward_contact(self):
        moving = self._gait_motion_gate()
        expected_contact = self.leg_phase < self.gait_stance_threshold
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.0
        per_foot_match = torch.sum(contact == expected_contact, dim=1).float()

        # 단일지지 위상에서는 실제로도 한 발만 접촉해야 한다. 기존 per-foot 합산은
        # 양발을 계속 붙여도 stance 쪽 발의 점수(주기 평균 raw 1.1/2.0)를 지급해
        # standing 정책이 contact reward를 수확할 수 있었다. 의도된 double-support
        # 구간(gait_stance_threshold=0.55이면 주기의 10%)은 그대로 허용한다.
        expected_single_support = torch.sum(expected_contact, dim=1) == 1
        actual_single_support = torch.sum(contact, dim=1) == 1
        support_count_valid = ~expected_single_support | actual_single_support
        return per_foot_match * support_count_valid.float() * moving

    def calculate_phase_expectation(self, phi, offset=0, phase="swing"):
        """
        Calculate the expectation value of I_i(φ).

        Parameters:
        phi (float): The given phase time.
        offset (float): The offset of the phase time.

        Returns:
        float: The expectation value of I_i(φ).
        """
        # print("phase_time: ", phi)
        phi = (phi + offset) % 1
        phi *= 2 * np.pi
        # Create Von Mises distribution objects for A_i and B_i
        if phase == "swing":
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_swing)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_swing)
        else:
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_stance)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_stance)
        # Calculate P(A_i < φ) and P(B_i < φ)
        P_A_less_phi = dist_A.cdf(phi)
        P_B_less_phi = dist_B.cdf(phi)
        # Calculate P(A_i < φ < B_i)
        P_A_phi_B = P_A_less_phi * (1 - P_B_less_phi)
        # Calculate the expectation value of I_i
        E_I_i = P_A_phi_B

        return E_I_i

    def _reward_gait_period(self):
        """
        Jonah Siekmann, et al. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
        paper link: https://arxiv.org/abs/2011.01387
        """
        # Calculate the expectation value of I_i of left and right feet
        E_I_l_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="swing")
        E_I_l_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="stance")
        E_I_r_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="swing")
        E_I_r_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="stance")
        # print("E_I_l_swing: ", E_I_l_swing, ", E_I_r_swing: ", E_I_r_swing)
        # print("E_I_l_stance: ", E_I_l_stance, ", E_I_r_stance: ", E_I_r_stance)
        ## Convert to tensor
        E_I_l_swing = torch.tensor(E_I_l_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_swing = torch.tensor(E_I_r_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_l_stance = torch.tensor(E_I_l_stance, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_stance = torch.tensor(E_I_r_stance, device=self.device, dtype=torch.float, requires_grad=False)
        # Get the contact forces and velocities of the feet, and the velocities of the arm ee
        Ff_left = torch.norm(self.simulator.contact_forces[:, self.feet_indices[0], :], dim=-1) # left foot contact force
        Ff_right = torch.norm(self.simulator.contact_forces[:, self.feet_indices[1], :], dim=-1) # right foot contact force
        vf_left = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[0], :], dim=-1) # left foot velocity
        vf_right = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[1], :], dim=-1) # right foot velocity
        # print("Ff_left: ", Ff_left, ", Ff_right: ", Ff_right)
        # print("vf_left: ", vf_left, ", vf_right: ", vf_right)
        reward_gait = E_I_l_swing * torch.exp(-Ff_left**2) + E_I_r_swing * torch.exp(-Ff_right**2) + \
                      E_I_l_stance * torch.exp(-200*vf_left**2) + E_I_r_stance * torch.exp(-200*vf_right**2)
        # Sum up the gait reward
        return reward_gait

    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]

    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]

    def _get_obs_command_stand(self):
        # walk/stand 모드 신호: 1 = standstill(정지 명령), 0 = walk(이동 명령).
        # 커맨드에서 파생 → train/eval/deploy 에서 항상 일관됨
        # (standstill 은 lin vel 정확히 0, walk 는 norm>0.2 이라 0.1 로 분리됨)
        # is_standing_env 마스크를 그대로 쓴다.
        return self.is_standing_env.unsqueeze(1).float()

    def _get_obs_phase_time(self):
        return self.phase_time.unsqueeze(1)

    def _get_obs_sin_phase(self):
        return torch.sin(2 * np.pi * self.phase_time).unsqueeze(1)

    def _get_obs_cos_phase(self):
        return torch.cos(2 * np.pi * self.phase_time).unsqueeze(1)

    def _get_obs_joint_acc(self):
        pass

    def _get_obs_base_acc(self):
        pass
