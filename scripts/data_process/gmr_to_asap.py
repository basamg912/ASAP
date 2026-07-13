"""GMR 출력 pkl → ASAP 모션 pkl 변환. hvgym 환경에서 실행할 것."""

import glob
import os
import sys

import numpy
import pickle

import joblib
import torch
from hydra import compose, initialize
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

# numpy 2.x로 저장된 pickle을 numpy 1.x에서 열기 위한 shim.
# 주의: torch import 전에 등록하면 torch 초기화가 segfault 나므로 반드시 이후에 둘 것.
sys.modules["numpy._core"] = numpy.core
sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
sys.modules["numpy._core.numeric"] = numpy.core.numeric
sys.modules["numpy._core.umath"] = numpy.core.umath


def build_pose_aa(humanoid_fk, num_extend, root_rot_xyzw, dof):
    N = dof.shape[0]
    root_aa = torch.from_numpy(
        sRot.from_quat(root_rot_xyzw).as_rotvec()
    ).float()  # (N,3)
    dof_t = torch.from_numpy(dof).float()[None, :, :, None]  # (1,N,31,1)
    pose_aa = torch.cat(
        [
            root_aa[None, :, None],  # (1,N,1,3)  루트 회전
            humanoid_fk.dof_axis * dof_t,  # (1,N,31,3) 관절축 × 관절각
            torch.zeros((1, N, num_extend, 3)),  # extend: left_hand, right_hand, head
        ],
        axis=2,
    )
    return pose_aa.squeeze(0).numpy()  # (N, 35, 3)


def convert(gmr_pkl_path, out_dir, humanoid_fk, num_extend):
    with open(gmr_pkl_path, "rb") as f:
        d = pickle.load(f)
    root_rot = numpy.ascontiguousarray(d["root_rot"])
    dof = numpy.ascontiguousarray(d["dof_pos"])
    key = "0-" + os.path.splitext(os.path.basename(gmr_pkl_path))[0]
    asap_data = {
        key: {
            "root_trans_offset": numpy.ascontiguousarray(d["root_pos"]),
            "root_rot": root_rot,  # xyzw 그대로
            "dof": dof,  # 순서 1:1 일치 확인됨
            "pose_aa": build_pose_aa(humanoid_fk, num_extend, root_rot, dof),
            "fps": int(d["fps"]),
        }
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, key + ".pkl")
    joblib.dump(asap_data, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    # fit_smpl_motion.py와 동일한 config 로드 방식 (경로는 이 파일 기준 상대경로)
    with initialize(config_path="../../humanoidverse/config", version_base=None):
        cfg = compose(config_name="base", overrides=["+robot=kapex/kapex_31dof"])
    humanoid_fk = Humanoid_Batch(cfg.robot.motion)
    num_extend = len(cfg.robot.motion.extend_config)

    # dof 순서 1회 검증용: yaml dof_names 순서(LLJ1-7, RLJ1-7, WLJ1-3, LAJ1-7, RAJ1-7)와 비교
    print("Humanoid_Batch body order:", humanoid_fk.body_names)
    gmr_motion_folder = cfg.robot.motion.gmr_motion_file
    gmr_motion_folder = gmr_motion_folder + "/*.pkl"
    print(gmr_motion_folder)
    out_dir = "../motionData/"
    for p in sys.argv[1:] or glob.glob(
        gmr_motion_folder,
        recursive=True,
    ):
        convert(p, out_dir, humanoid_fk, num_extend)
