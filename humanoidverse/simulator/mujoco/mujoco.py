import os
import time

import mujoco as mj
import numpy as np
import torch
from loguru import logger

from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator

# 프레임워크 텐서 컨벤션 (isaacgym/genesis와 동일하게 맞춤):
#   - quaternion: xyzw  (MuJoCo 내부는 wxyz → 경계에서 변환)
#   - root ang vel: world frame (MuJoCo free joint qvel[3:6]은 body frame → 변환)
#   - robot_root_states: [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]


def _wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


class MuJoCo(BaseSimulator):
    """pip-installed MuJoCo backend.

    평가/rollout 수집/sim2sim 검증 용도 (num_envs=1 전용, CPU 물리).
    도메인 랜덤화(link mass/friction/base com)는 미지원 — NO_domain_rand 로 사용할 것.
    """

    def __init__(self, config, device):
        self.cfg = config
        self.sim_cfg = config.simulator.config
        self.robot_cfg = config.robot
        # 텐서는 요청된 device(cuda 가능)에 두고, MuJoCo(CPU)와는 경계에서 복사
        self.device = device
        self.sim_device = device
        self.headless = True
        self.viewer = None
        self._marker_pos = []
        self._last_render_time = None

    # ----- Configuration Setup Methods -----

    def set_headless(self, headless):
        self.headless = headless

    def setup(self):
        self.sim_dt = 1.0 / self.sim_cfg.sim.fps
        if int(self.sim_cfg.sim.substeps) != 1:
            logger.warning("MuJoCo backend ignores substeps != 1")

    # ----- Terrain Setup Methods -----

    def setup_terrain(self, mesh_type):
        if mesh_type != "plane":
            raise NotImplementedError(f"MuJoCo backend supports 'plane' terrain only (got {mesh_type})")

    # ----- Robot Asset Setup Methods -----

    def load_assets(self):
        asset_root = self.robot_cfg.asset.asset_root
        xml_file = self.robot_cfg.asset.xml_file
        if xml_file is None:
            raise ValueError("robot.asset.xml_file must point to an MJCF for the MuJoCo backend")
        asset_path = os.path.join(asset_root, xml_file)
        logger.info(f"[MuJoCo] loading MJCF: {asset_path}")

        self.model = mj.MjModel.from_xml_path(asset_path)
        self.model.opt.timestep = self.sim_dt

        # floor 마찰을 config로 통일
        floor_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        if floor_gid >= 0:
            self.model.geom_friction[floor_gid, 0] = self.sim_cfg.plane.static_friction
        else:
            logger.warning("[MuJoCo] no 'floor' geom found in MJCF — terrain must be part of the model")

        if self.model.njnt == 0 or self.model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("MJCF must have a free joint as the first joint (floating base)")

        # config 순서 → MuJoCo 주소 매핑
        self.dof_names = list(self.robot_cfg.dof_names)
        self.body_names = list(self.robot_cfg.body_names)
        self.num_dof = len(self.dof_names)
        self.num_bodies = len(self.body_names)

        self._qposadr = np.zeros(self.num_dof, dtype=np.int64)
        self._dofadr = np.zeros(self.num_dof, dtype=np.int64)
        for i, name in enumerate(self.dof_names):
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint '{name}' not found in MJCF")
            self._qposadr[i] = self.model.jnt_qposadr[jid]
            self._dofadr[i] = self.model.jnt_dofadr[jid]

        self._body_ids = np.zeros(self.num_bodies, dtype=np.int64)
        for i, name in enumerate(self.body_names):
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"body '{name}' not found in MJCF")
            self._body_ids[i] = bid

        # Isaac은 dof_armature_list를 런타임에 적용 — MJCF에 없으면 여기서 주입
        # (armature 0 + stiff PD + Euler 적분이면 경량 관절이 발산함)
        if self.robot_cfg.get("dof_armature_list", None) is not None:
            self.model.dof_armature[self._dofadr] = np.asarray(self.robot_cfg.dof_armature_list)
        if self.robot_cfg.get("dof_joint_friction_list", None) is not None:
            self.model.dof_frictionloss[self._dofadr] = np.asarray(self.robot_cfg.dof_joint_friction_list)
        self.model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

    # ----- Environment Creation Methods -----

    def create_envs(self, num_envs, env_origins, base_init_state):
        if num_envs != 1:
            raise NotImplementedError(
                f"MuJoCo backend supports num_envs=1 only (got {num_envs}); "
                "use it for evaluation / rollout collection"
            )
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        # env가 extend link를 append하므로 복사본이어야 함
        self._body_list = list(self.body_names)

        self.data = mj.MjData(self.model)
        # base_init_state = [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]
        init = np.asarray(
            base_init_state.cpu() if torch.is_tensor(base_init_state) else base_init_state,
            dtype=np.float64,
        )
        self.data.qpos[:3] = init[:3]
        self.data.qpos[3:7] = init[[6, 3, 4, 5]]  # xyzw -> wxyz
        for i in range(self.num_dof):
            angle = self.robot_cfg.init_state.default_joint_angles.get(self.dof_names[i], 0.0)
            self.data.qpos[self._qposadr[i]] = angle
        mj.mj_forward(self.model, self.data)

        if getattr(self.cfg, "domain_rand", None) is not None:
            for flag in ["randomize_link_mass", "randomize_friction", "randomize_base_com"]:
                if self.cfg.domain_rand.get(flag, False):
                    logger.warning(f"[MuJoCo] domain_rand.{flag} is not supported and will be ignored")
        return None, None

    # ----- Property Retrieval Methods -----

    def find_rigid_body_indice(self, body_name):
        for i, name in enumerate(self._body_list):
            if body_name in name:
                return i
        return None

    def get_dof_limits_properties(self):
        n = self.num_dof
        self.hard_dof_pos_limits = torch.zeros(n, 2, dtype=torch.float, device=self.device)
        self.dof_pos_limits = torch.zeros(n, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(n, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(n, dtype=torch.float, device=self.device)
        for i in range(n):
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
            m_ = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r_ = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m_ - 0.5 * r_ * self.cfg.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m_ + 0.5 * r_ * self.cfg.rewards.reward_limit.soft_dof_pos_limit

        # terminate_when_close_to_dof_pos_limit 용 (isaacgym.py:312-329와 동일 규칙)
        term_scale = None
        try:
            term_scale = self.cfg.env.config.termination_scales.termination_close_to_dof_pos_limit
        except Exception:
            pass
        if term_scale is not None:
            self.dof_pos_limits_termination = torch.zeros(n, 2, dtype=torch.float, device=self.device)
            for i in range(n):
                m_ = (self.hard_dof_pos_limits[i, 0] + self.hard_dof_pos_limits[i, 1]) / 2
                r_ = self.hard_dof_pos_limits[i, 1] - self.hard_dof_pos_limits[i, 0]
                self.dof_pos_limits_termination[i, 0] = m_ - 0.5 * r_ * term_scale
                self.dof_pos_limits_termination[i, 1] = m_ + 0.5 * r_ * term_scale
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        # 영속 버퍼 할당 — env가 이 텐서들의 뷰(base_quat 등)를 init 때 캐시하므로
        # (legged_robot_base._init_buffers) 이후 refresh는 반드시 in-place로만 갱신
        nb = self.num_bodies
        self.robot_root_states = torch.zeros(1, 13, dtype=torch.float, device=self.device)
        self.all_root_states = self.robot_root_states
        self.base_quat = self.robot_root_states[:, 3:7]
        self.dof_pos = torch.zeros(1, self.num_dof, dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros(1, self.num_dof, dtype=torch.float, device=self.device)
        self._rigid_body_pos = torch.zeros(1, nb, 3, dtype=torch.float, device=self.device)
        self._rigid_body_rot = torch.zeros(1, nb, 4, dtype=torch.float, device=self.device)
        self._rigid_body_vel = torch.zeros(1, nb, 3, dtype=torch.float, device=self.device)
        self._rigid_body_ang_vel = torch.zeros(1, nb, 3, dtype=torch.float, device=self.device)
        self.contact_forces = torch.zeros(1, nb, 3, dtype=torch.float, device=self.device)
        self.refresh_sim_tensors()

    def refresh_sim_tensors(self):
        m, d = self.model, self.data
        mj.mj_forward(m, d)

        pos = d.qpos[:3]
        quat_xyzw = np.array([d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[3]])
        lin_vel = d.qvel[:3]  # world frame
        # free joint ang vel은 body frame → world frame으로 회전
        R = d.xmat[self._body_ids[0]].reshape(3, 3)
        ang_vel_world = R @ d.qvel[3:6]

        root = np.concatenate([pos, quat_xyzw, lin_vel, ang_vel_world]).astype(np.float32)
        self.robot_root_states.copy_(torch.from_numpy(root).unsqueeze(0))

        self.dof_pos.copy_(torch.from_numpy(d.qpos[self._qposadr].astype(np.float32)).unsqueeze(0))
        self.dof_vel.copy_(torch.from_numpy(d.qvel[self._dofadr].astype(np.float32)).unsqueeze(0))

        nb = self.num_bodies
        body_pos = d.xpos[self._body_ids].astype(np.float32)
        body_rot = _wxyz_to_xyzw(d.xquat[self._body_ids]).astype(np.float32)
        body_vel = np.zeros((nb, 3), dtype=np.float32)
        body_ang_vel = np.zeros((nb, 3), dtype=np.float32)
        vel6 = np.zeros(6)
        for i, bid in enumerate(self._body_ids):
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_XBODY, int(bid), vel6, 0)  # world frame
            body_ang_vel[i] = vel6[:3]
            body_vel[i] = vel6[3:]
        self._rigid_body_pos.copy_(torch.from_numpy(body_pos).unsqueeze(0))
        self._rigid_body_rot.copy_(torch.from_numpy(body_rot).unsqueeze(0))
        self._rigid_body_vel.copy_(torch.from_numpy(body_vel).unsqueeze(0))
        self._rigid_body_ang_vel.copy_(torch.from_numpy(body_ang_vel).unsqueeze(0))

        # cfrc_ext: [torque(3), force(3)] — force 성분만 사용 (world 방향, com 기준)
        cf = d.cfrc_ext[self._body_ids][:, 3:6].astype(np.float32)
        self.contact_forces.copy_(torch.from_numpy(cf).unsqueeze(0))

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        t = torques.detach().cpu().numpy().reshape(-1)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self._dofadr] = t

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        rs = root_states[0] if root_states.dim() > 1 else root_states
        rs = rs.detach().cpu().numpy().astype(np.float64)
        d = self.data
        d.qpos[:3] = rs[:3]
        d.qpos[3:7] = rs[[6, 3, 4, 5]]  # xyzw -> wxyz
        d.qvel[:3] = rs[7:10]
        mj.mj_forward(self.model, d)
        R = d.xmat[self._body_ids[0]].reshape(3, 3)
        d.qvel[3:6] = R.T @ rs[10:13]  # world -> body frame
        mj.mj_forward(self.model, d)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        ds = dof_states.view(self.num_envs, -1, 2)[0].detach().cpu().numpy().astype(np.float64)
        self.data.qpos[self._qposadr] = ds[:, 0]
        self.data.qvel[self._dofadr] = ds[:, 1]
        mj.mj_forward(self.model, self.data)

    def simulate_at_each_physics_step(self):
        mj.mj_step(self.model, self.data)
        # isaacgym.py:504와 동일 — env의 PD(_compute_torques)가 매 substep fresh한
        # 관절 상태를 읽도록 dof 버퍼를 물리 스텝마다 갱신 (안 하면 50Hz ZOH PD가 발진)
        d = self.data
        self.dof_pos.copy_(torch.from_numpy(d.qpos[self._qposadr].astype(np.float32)).unsqueeze(0))
        self.dof_vel.copy_(torch.from_numpy(d.qvel[self._dofadr].astype(np.float32)).unsqueeze(0))

    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        import mujoco.viewer as mj_viewer

        self.viewer = mj_viewer.launch_passive(self.model, self.data)

    def render(self, sync_frame_time=True):
        if self.viewer is None:
            return
        self.viewer.sync()
        if sync_frame_time:
            # 제어 주기(dt)에 맞춰 실시간 페이싱. realtime_factor < 1.0 이면 슬로모션
            # (예: 0.5 = 절반 속도). 시뮬이 실시간보다 느리면 sleep 없이 그냥 진행.
            factor = float(self.sim_cfg.sim.get("realtime_factor", 1.0))
            dt_wall = self.sim_dt * self.sim_cfg.sim.control_decimation / max(factor, 1e-6)
            now = time.perf_counter()
            if self._last_render_time is not None:
                remain = dt_wall - (now - self._last_render_time)
                if remain > 0:
                    time.sleep(remain)
            self._last_render_time = time.perf_counter()

    # ----- Misc -----

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def add_visualize_entities(self, num_visualize_markers):
        self._marker_pos = [np.zeros(3) for _ in range(num_visualize_markers)]

    def clear_lines(self):
        pass

    def draw_sphere(self, pos, radius, color, env_id, pos_id=0):
        if pos_id < len(self._marker_pos):
            p = pos.detach().cpu().numpy().reshape(-1) if torch.is_tensor(pos) else np.asarray(pos).reshape(-1)
            self._marker_pos[pos_id] = p

    def draw_line(self, start_point, end_point, color, env_id):
        pass
