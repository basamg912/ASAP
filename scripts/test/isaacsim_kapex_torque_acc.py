"""KAPEX torque-injection acceleration inspection (IsaacSim / Isaac Lab standalone).

KAPEX USD 를 스폰하고, PD 를 끈 상태(stiffness=damping=0, ImplicitActuator)에서
무릎 관절(LLJ4/RLJ4)에 사인 토크를 직접 주입한 뒤 매 스텝
robot.data.body_com_acc_w 와 robot.data.joint_acc 를 출력한다.

주의: PD 가 꺼져 있으므로 중력이 켜진 기본 모드에서는 로봇이 주저앉는다(정상).
      순수 토크→가속도 응답만 보려면 --no_gravity 로 공중에 띄워서 실행.

실행 (IsaacSim 환경 소싱 필요):
  conda activate hvlab
  source /home/kist/work/workspace/IsaacSim/_build/linux-x86_64/release/setup_conda_env.sh
  python scripts/test/isaacsim_kapex_torque_acc.py --headless
  python scripts/test/isaacsim_kapex_torque_acc.py --headless --no_gravity --torque 10
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="KAPEX torque injection: print body_com_acc_w / joint_acc")
parser.add_argument("--num_steps", type=int, default=300, help="시뮬 스텝 수")
parser.add_argument("--torque", type=float, default=30.0, help="주입 토크 진폭 [Nm]")
parser.add_argument("--print_every", type=int, default=25, help="출력 주기 [step]")
parser.add_argument("--no_gravity", action="store_true",
                    help="중력 끄고 공중 스폰 — 접촉 없이 토크 응답만 관찰")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- IsaacSim 앱 기동 후에만 import 가능 ----
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KAPEX_USD = os.path.join(
    REPO_ROOT,
    "humanoidverse/data/robots/kapex/KAPEX_wo_hand_head/KAPEX_wo_hand_head.usd")

# kapex_31dof.yaml init_state.default_joint_angles 와 동일한 pose (전 관절 명시)
DEFAULT_JOINT_POS = {f"{s}J{i}": 0.0 for s in ("LL", "RL", "LA", "RA") for i in range(1, 8)}
DEFAULT_JOINT_POS.update({f"WLJ{i}": 0.0 for i in range(1, 4)})
DEFAULT_JOINT_POS.update({
    "LLJ3": -0.035, "LLJ4": 0.38, "LLJ5": -0.33,
    "RLJ3": 0.035, "RLJ4": -0.38, "RLJ5": 0.33,
    "LAJ1": 0.2, "LAJ2": 0.2, "LAJ3": 0.18, "LAJ4": -0.35,
    "RAJ1": -0.2, "RAJ2": -0.2, "RAJ3": -0.18, "RAJ4": 0.35,
})

KAPEX_CFG = ArticulationCfg(
    prim_path="/World/KAPEX",
    spawn=sim_utils.UsdFileCfg(
        usd_path=KAPEX_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=args_cli.no_gravity,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5 if args_cli.no_gravity else 0.91),  # 0.91 = kapex base 높이
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    # 순수 토크 주입: PD 게인 0 -> set_joint_effort_target 가 그대로 관절 토크가 됨
    actuators={
        "all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
    },
)


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args_cli.device))  # fps 200 (학습 config 동일)

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)

    robot = Articulation(KAPEX_CFG)
    sim.reset()

    print(f"[INFO] num_joints={robot.num_joints}, num_bodies={robot.num_bodies}")
    print(f"[INFO] joint_names={robot.joint_names}")
    print(f"[INFO] body_names={robot.body_names}")

    knee_ids, knee_names = robot.find_joints(["LLJ4", "RLJ4"])
    print(f"[INFO] torque 주입 대상: {knee_names} (ids={knee_ids})")

    dt = sim.get_physics_dt()
    efforts = torch.zeros(robot.num_instances, robot.num_joints, device=robot.device)

    for step in range(args_cli.num_steps):
        # 0.5 Hz 사인 토크를 무릎에 주입
        efforts.zero_()
        efforts[:, knee_ids] = args_cli.torque * math.sin(2.0 * math.pi * 0.5 * step * dt)
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if step % args_cli.print_every == 0:
            joint_acc = robot.data.joint_acc              # [N, num_joints]  (joint_vel 차분)
            # body_com_acc_w: [N, num_bodies, 6] (lin 3 + ang 3), world frame, PhysX link acc
            body_acc = getattr(robot.data, "body_com_acc_w", None)
            if body_acc is None:  # 구버전 IsaacLab 호환
                body_acc = robot.data.body_acc_w

            lin_norm = body_acc[0, :, :3].norm(dim=-1)
            max_body = int(lin_norm.argmax())
            print(f"\n[step {step:4d}] knee torque = "
                  f"{[f'{v:.2f}' for v in efforts[0, knee_ids].tolist()]} Nm")
            print(f"  joint_acc {knee_names}: "
                  f"{[f'{v:.3f}' for v in joint_acc[0, knee_ids].tolist()]} rad/s^2"
                  f" | all-joint |max|: {joint_acc[0].abs().max().item():.3f}")
            print(f"  body_com_acc_w shape: {tuple(body_acc.shape)}")
            print(f"  pelvis com acc lin: "
                  f"{[f'{v:.3f}' for v in body_acc[0, 0, :3].tolist()]} m/s^2")
            print(f"  |lin acc| max body: {robot.body_names[max_body]}"
                  f" = {lin_norm[max_body].item():.3f} m/s^2")

    simulation_app.close()


if __name__ == "__main__":
    main()
