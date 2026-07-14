"""Build a delta-action-ready motion file (motion pkl with an extra "action" key).

The delta action model training consumes the same motion pkl format as naive
motion tracking (see check_motion.py), plus a per-frame "action" array. MotionLib
picks it up automatically when the key exists (motion_lib_base.py: has_action).

Two ways to get there:

[A] convert  - from an eval/deploy run with +env.config.save_motion=True.
    That dump already contains "action" recorded in the same step as the state
    (states and actions are appended together in _post_physics_step), so this
    mode just segments episodes at `terminate`, keeps the delta-a keys, and
    re-saves. THIS IS THE RELIABLE PATH.

[B] merge    - graft actions from a deploy_agent.py .pt onto an action-less
    motion pkl. ONLY valid when both artifacts were recorded in the SAME
    rollout; states from run A + actions from run B are wrong training pairs
    for the delta action model. Episode lengths are cross-checked and the
    script refuses ambiguous matches.

Usage:
    # A: env save_motion dump -> delta-a motion file
    python scripts/data_process/make_delta_a_motion.py convert \
        --motion <ckpt_dir>/motions/<dump>.pkl --out motionData/delta_a/<name>.pkl

    # B: merge deploy_agent actions into an action-less motion pkl
python scripts/data_process/make_delta_a_motion.py merge \
    --motion motionData/0-B4_-_Stand_to_Walk_backwards_stageii.pkl \
    --actions_pt motionData/deploy_data_ckpt_148300_deployData.pt \
    --out motionData/delta_a/B4_with_action.pkl
"""

import argparse
import pathlib

import joblib
import numpy as np

DELTA_A_KEYS = ["root_trans_offset", "root_rot", "dof", "pose_aa", "action"]


def describe(name, motion):
    print(f"  {name}:")
    for k, v in motion.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"    {k}: {v}")


def save(out_path, data):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, out_path)
    print(f"\nsaved {out_path}")
    print("== result (check_motion.py view) ==")
    for name, motion in data.items():
        describe(name, motion)


def convert(args):
    src = joblib.load(args.motion)
    out = {}
    for name, motion in src.items():
        missing = [k for k in DELTA_A_KEYS if k not in motion]
        if missing:
            raise SystemExit(
                f"[{name}] missing {missing} — this dump was not recorded with "
                "save_motion=True on current code (action is recorded there). "
                "Re-run eval/deploy with +env.config.save_motion=True, or use `merge`."
            )
        T = len(motion["dof"])
        # segment at terminations; keep segments long enough to be useful
        term = np.asarray(motion.get("terminate", np.zeros(T))).astype(bool)
        ends = list(np.where(term)[0]) or [T - 1]
        start, seg_i = 0, 0
        for end in ends:
            length = end - start + 1
            if length >= args.min_len:
                seg = {k: np.asarray(motion[k])[start : end + 1] for k in DELTA_A_KEYS}
                seg["fps"] = motion["fps"]
                out[f"{name}_seg{seg_i}"] = seg
                seg_i += 1
            start = end + 1
        print(f"[{name}] {T} frames -> {seg_i} segment(s) (min_len={args.min_len})")
    save(args.out, out)


def merge(args):
    import torch

    src = joblib.load(args.motion)
    dep = torch.load(args.actions_pt, map_location="cpu")
    dep = dep["episodes"][1]
    actions = dep["actions"][:, args.env].numpy()  # (T, dof)
    dones = dep["dones"][:, args.env].numpy().astype(bool)

    # episodes in the deploy rollout
    ends = np.where(dones)[0]
    starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.array([0])
    episodes = [(s, e) for s, e in zip(starts, ends)]
    print(f"deploy rollout: {len(episodes)} episodes, lengths={[e - s + 1 for s, e in episodes]}")

    out = {}
    for name, motion in src.items():
        if "action" in motion:
            print(f"[{name}] already has action — copied as-is")
            out[name] = motion
            continue
        L = len(motion["dof"])
        matches = [(s, e) for s, e in episodes if abs((e - s + 1) - L) <= args.tol]
        if not matches:
            raise SystemExit(
                f"[{name}] no deploy episode matches motion length {L} "
                f"(±{args.tol}). States and actions are probably from different "
                "rollouts — that pairing is invalid for delta-a training. "
                "Prefer re-collecting with +env.config.save_motion=True (mode A)."
            )
        s, e = matches[0]
        if len(matches) > 1:
            print(f"[{name}] {len(matches)} episodes match length {L}; using the first (steps {s}-{e})")
        act = actions[s : s + L]
        if len(act) < L:  # episode slightly shorter than motion: pad with last action
            act = np.concatenate([act, np.repeat(act[-1:], L - len(act), axis=0)])
        new_motion = dict(motion)
        new_motion["action"] = act.astype(np.float32)
        out[name] = new_motion
        print(f"[{name}] merged action {act.shape} from deploy steps {s}-{s + L - 1}")
    save(args.out, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("convert", help="save_motion dump -> delta-a motion file")
    a.add_argument("--motion", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--min_len", type=int, default=60, help="drop segments shorter than this (frames)")
    a.set_defaults(func=convert)

    b = sub.add_parser("merge", help="graft deploy_agent actions onto an action-less motion pkl")
    b.add_argument("--motion", required=True)
    b.add_argument("--actions_pt", required=True)
    b.add_argument("--env", type=int, default=0)
    b.add_argument("--tol", type=int, default=2, help="episode/motion length match tolerance")
    b.add_argument("--out", required=True)
    b.set_defaults(func=merge)

    args = ap.parse_args()
    args.func(args)
