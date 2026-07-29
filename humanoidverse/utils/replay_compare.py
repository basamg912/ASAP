'''
deploy_agent.py 로 수집한 rollout(.pt)과 replay_agent.py 의 open-loop replay 결과(.pt)를
비교한다: qpos 궤적 오차 metric(json) + 오차 곡선(png) + side-by-side 비디오(mp4).

qpos layout 은 deploy_agent.capture_qpos 와 동일: [root pos(3), root quat wxyz(4), dof(D)].

standalone 실행 (isaacsim/isaaclab 불필요, torch/numpy/matplotlib/mujoco 만 사용):
    python humanoidverse/utils/replay_compare.py \
        motionData/locomotion_run0.pt motionData/locomotion_run0_replay_isaacsim.pt \
        --out-dir motionData/replay_report [--no-video]
XML/dof_names/fps 는 replay .pt 의 메타에서 읽고, CLI 인자로 덮어쓸 수 있다.
'''
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# 고정 색 할당 (Okabe-Ito, CVD-safe): 좌=수집(mujoco), 우=재현(isaaclab), 오차 계열
COLOR_RECORDED = "#0072B2"   # blue
COLOR_REPLAYED = "#E69F00"   # orange
COLOR_ERROR = "#D55E00"      # vermillion
COLOR_ERROR_AUX = "#999999"  # gray


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def quat_angle_deg(q1, q2):
    """Geodesic angle (deg) between wxyz quaternion arrays of shape (T, 4)."""
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.clip(np.abs((q1 * q2).sum(-1)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def first_step_above(err, threshold):
    idx = np.nonzero(err > threshold)[0]
    return int(idx[0]) if len(idx) else None


def compute_metrics(qpos_rec, qpos_rep, dof_names=None, dof_err_threshold=0.1):
    """Per-step trajectory errors + summary scalars. qpos: (T, 7 + D)."""
    qpos_rec, qpos_rep = to_numpy(qpos_rec), to_numpy(qpos_rep)
    T = min(len(qpos_rec), len(qpos_rep))
    rec, rep = qpos_rec[:T], qpos_rep[:T]

    dof_abs_err = np.abs(rec[:, 7:] - rep[:, 7:])          # (T, D)
    dof_mae = dof_abs_err.mean(-1)                          # (T,)
    dof_max = dof_abs_err.max(-1)                           # (T,)
    root_pos_err = np.linalg.norm(rec[:, :3] - rep[:, :3], axis=-1)
    height_err = np.abs(rec[:, 2] - rep[:, 2])
    quat_err_deg = quat_angle_deg(rec[:, 3:7], rep[:, 3:7])

    per_joint_mae = dof_abs_err.mean(0)                     # (D,)
    if dof_names is None:
        dof_names = [f"dof_{i}" for i in range(dof_abs_err.shape[1])]

    return {
        "num_steps_recorded": int(len(qpos_rec)),
        "num_steps_replayed": int(len(qpos_rep)),
        "num_steps_compared": int(T),
        "dof_err_threshold_rad": dof_err_threshold,
        "divergence_step_dof_mae": first_step_above(dof_mae, dof_err_threshold),
        "dof_mae_mean": float(dof_mae.mean()),
        "dof_mae_final": float(dof_mae[-1]),
        "root_pos_err_mean": float(root_pos_err.mean()),
        "root_pos_err_final": float(root_pos_err[-1]),
        "quat_err_deg_mean": float(quat_err_deg.mean()),
        "quat_err_deg_final": float(quat_err_deg[-1]),
        "per_joint_mae": {n: float(v) for n, v in zip(dof_names, per_joint_mae)},
        "curves": {
            "dof_mae": dof_mae.tolist(),
            "dof_max": dof_max.tolist(),
            "root_pos_err": root_pos_err.tolist(),
            "height_err": height_err.tolist(),
            "quat_err_deg": quat_err_deg.tolist(),
        },
    }


def plot_comparison(metrics, fps, out_png, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = metrics["curves"]
    t = np.arange(metrics["num_steps_compared"]) / fps
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title or "open-loop replay vs recorded rollout")

    ax = axes[0, 0]
    ax.plot(t, c["dof_mae"], color=COLOR_ERROR, label="mean |Δdof|")
    ax.plot(t, c["dof_max"], color=COLOR_ERROR_AUX, label="max |Δdof|")
    thr = metrics["dof_err_threshold_rad"]
    ax.axhline(thr, color=COLOR_ERROR_AUX, ls="--", lw=1)
    div = metrics["divergence_step_dof_mae"]
    if div is not None:
        ax.axvline(div / fps, color=COLOR_ERROR_AUX, ls=":", lw=1)
        ax.annotate(f"diverge @{div / fps:.2f}s", (div / fps, thr),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title("DOF position error [rad]")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, c["root_pos_err"], color=COLOR_ERROR, label="‖Δroot pos‖")
    ax.plot(t, c["height_err"], color=COLOR_ERROR_AUX, label="|Δheight|")
    ax.set_title("root position error [m]")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, c["quat_err_deg"], color=COLOR_ERROR)
    ax.set_title("root orientation error [deg]")

    ax = axes[1, 1]
    per_joint = sorted(metrics["per_joint_mae"].items(), key=lambda kv: -kv[1])[:10]
    names = [n for n, _ in per_joint][::-1]
    vals = [v for _, v in per_joint][::-1]
    ax.barh(names, vals, color=COLOR_ERROR)
    ax.set_title("worst 10 joints, mean |Δdof| [rad]")
    ax.tick_params(axis="y", labelsize=7)

    for ax in axes[:, :2].flat[:3]:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_side_by_side(mjcf_path, qpos_rec, qpos_rep, dof_names, fps, out_path,
                        width=640, height=480):
    """좌: 수집 rollout, 우: replay. 두 qpos 시퀀스를 같은 mujoco 모델로 렌더 후 hstack.

    Headless 환경에서는 MUJOCO_GL=egl 필요.
    """
    import mujoco

    qpos_rec, qpos_rep = to_numpy(qpos_rec), to_numpy(qpos_rep)
    T = min(len(qpos_rec), len(qpos_rep))

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mjdata = mujoco.MjData(model)
    width = min(width, model.vis.global_.offwidth)
    height = min(height, model.vis.global_.offheight)

    has_free_joint = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    dof_qposadr = []
    for name in dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' not found in {mjcf_path}")
        dof_qposadr.append(model.jnt_qposadr[jid])
    dof_qposadr = np.array(dof_qposadr)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0

    def render_frame(renderer, qpos):
        if has_free_joint:
            mjdata.qpos[:7] = qpos[:7]
        mjdata.qpos[dof_qposadr] = qpos[7:]
        mujoco.mj_forward(model, mjdata)
        cam.lookat[:] = qpos[:3]
        renderer.update_scene(mjdata, camera=cam)
        return renderer.render()

    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(T):
            left = render_frame(renderer, qpos_rec[i])
            right = render_frame(renderer, qpos_rep[i])
            frames.append(np.hstack([left, right]))

    write_video(frames, fps, out_path)


def write_video(frames, fps, out_path):
    """imageio(ffmpeg) 우선, 없으면 cv2 로 저장 (deploy_agent.render_qpos_video 와 동일한 전략)."""
    try:
        import imageio

        imageio.mimsave(str(out_path), frames, fps=fps)
    except (ImportError, ValueError):
        import cv2

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()


def compare_and_report(recorded, replayed, out_dir, stem, no_video=False,
                       mjcf_path=None, log=print):
    """두 rollout dict 를 비교해 out_dir 에 <stem>_compare.{json,png,mp4} 를 남긴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dof_names = replayed.get("dof_names")
    fps = replayed.get("fps", recorded.get("fps", 50))

    metrics = compute_metrics(recorded["qpos"], replayed["qpos"], dof_names)
    metrics["recorded_done_step"] = int(to_numpy(recorded["dones"]).argmax()) \
        if to_numpy(recorded["dones"]).any() else None
    metrics["replayed_done_step"] = int(to_numpy(replayed["dones"]).argmax()) \
        if to_numpy(replayed["dones"]).any() else None

    json_path = out_dir / f"{stem}_compare.json"
    curves = metrics.pop("curves")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    metrics["curves"] = curves
    log(f"saved {json_path}")

    png_path = out_dir / f"{stem}_compare.png"
    plot_comparison(metrics, fps, png_path, title=stem)
    log(f"saved {png_path}")

    mjcf_path = mjcf_path or replayed.get("mjcf_path")
    if not no_video and mjcf_path and dof_names:
        mp4_path = out_dir / f"{stem}_compare.mp4"
        render_side_by_side(
            mjcf_path, recorded["qpos"], replayed["qpos"], dof_names, fps, mp4_path
        )
        log(f"saved {mp4_path} (left: recorded, right: replayed)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recorded", help="deploy_agent.py rollout .pt")
    parser.add_argument("replayed", help="replay_agent.py replay .pt")
    parser.add_argument("--out-dir", default=None, help="default: alongside replayed")
    parser.add_argument("--xml", default=None, help="MJCF path override for video")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    recorded = torch.load(args.recorded, map_location="cpu", weights_only=False)
    replayed = torch.load(args.replayed, map_location="cpu", weights_only=False)
    out_dir = args.out_dir or Path(args.replayed).parent
    stem = Path(args.replayed).stem

    metrics = compare_and_report(
        recorded, replayed, out_dir, stem, no_video=args.no_video, mjcf_path=args.xml
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "curves"}, indent=2))


if __name__ == "__main__":
    main()
