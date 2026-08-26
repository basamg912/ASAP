"""Visualize history-encoder latent NPZ files with UMAP.

Examples:
  python scripts/plot_latent_umap.py logs/.../latents/latents_ckpt_30000.npz
  python scripts/plot_latent_umap.py <npz> --color phase --max-samples 12000
  python scripts/plot_latent_umap.py <npz> --n-neighbors 50 --min-dist 0.05

Colors:
  command : assigned command-grid label (default)
  vx/vy/wz: commanded velocity component
  phase   : gait phase

Dependency:
  pip install umap-learn
"""
import argparse
from pathlib import Path

import numpy as np


COLORS = ("command", "vx", "vy", "wz", "phase")


def reset_keep_mask(done, drop_after_reset):
    """Keep samples except the N frames immediately following each reset."""
    keep = np.ones(done.shape, dtype=bool)
    if drop_after_reset <= 0:
        return keep
    time_steps = done.shape[0]
    for reset_time, env_index in zip(*np.nonzero(done)):
        start = reset_time + 1
        end = min(start + drop_after_reset, time_steps)
        keep[start:end, env_index] = False
    return keep


def load_samples(npz_path, drop_after_reset=10, max_samples=12000, seed=0):
    """Load, mask, flatten and deterministically subsample collector output."""
    raw = np.load(npz_path, allow_pickle=True)
    required = {"latent", "command", "env_cmd_idx", "labels"}
    missing = sorted(required.difference(raw.files))
    if missing:
        raise ValueError(f"missing NPZ arrays: {missing}")

    latent = raw["latent"]
    command = raw["command"]
    if latent.ndim != 3:
        raise ValueError(f"latent must have shape [T,N,L], got {latent.shape}")
    time_steps, num_envs, _ = latent.shape
    if command.shape != (time_steps, num_envs, 3):
        raise ValueError(
            f"command must have shape {(time_steps, num_envs, 3)}, got {command.shape}")

    if "done" in raw:
        done = np.asarray(raw["done"], dtype=bool)
        if done.shape != (time_steps, num_envs):
            raise ValueError(
                f"done must have shape {(time_steps, num_envs)}, got {done.shape}")
        keep = reset_keep_mask(done, drop_after_reset)
    else:
        keep = np.ones((time_steps, num_envs), dtype=bool)

    flat_keep = keep.reshape(-1)
    samples = {
        "latent": latent.reshape(time_steps * num_envs, -1)[flat_keep],
        "command": command.reshape(time_steps * num_envs, 3)[flat_keep],
        "command_index": np.tile(
            np.asarray(raw["env_cmd_idx"]), (time_steps, 1)).reshape(-1)[flat_keep],
        "phase": (
            np.asarray(raw["phase"]).reshape(-1)[flat_keep]
            if "phase" in raw else np.zeros(int(flat_keep.sum()), dtype=np.float32)
        ),
        "labels": np.asarray(raw["labels"]),
        "total_count": time_steps * num_envs,
        "kept_count": int(flat_keep.sum()),
    }

    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    sample_count = len(samples["latent"])
    if sample_count > max_samples:
        selected = np.random.default_rng(seed).choice(
            sample_count, max_samples, replace=False)
        for key in ("latent", "command", "command_index", "phase"):
            samples[key] = samples[key][selected]
    return samples


def import_umap():
    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "UMAP dependency is missing. Install it in the active environment with "
            "`pip install umap-learn`.") from error
    return umap


def fit_umap(samples, args):
    features = samples["latent"].astype(np.float32, copy=False)
    if len(features) < 3:
        raise ValueError(f"UMAP needs at least 3 samples, got {len(features)}")
    if args.standardize:
        from sklearn.preprocessing import StandardScaler
        features = StandardScaler().fit_transform(features).astype(np.float32)

    n_neighbors = min(args.n_neighbors, len(features) - 1)
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be at least 2, got {args.n_neighbors}")
    if not 0.0 <= args.min_dist <= args.spread:
        raise ValueError(
            f"min_dist must satisfy 0 <= min_dist <= spread, got "
            f"{args.min_dist} and {args.spread}")

    umap = import_umap()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        spread=args.spread,
        metric=args.metric,
        init="spectral",
        random_state=args.seed,
        n_jobs=1,
        low_memory=True,
        verbose=args.verbose,
    )
    return reducer.fit_transform(features), n_neighbors


def plot_embedding(embedding, samples, args, n_neighbors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#fcfcfb"
    ink, muted = "#242321", "#77746f"
    fig, axis = plt.subplots(figsize=(9, 8), facecolor=surface)
    axis.set_facecolor(surface)

    if args.color == "command":
        labels = samples["labels"]
        command_index = samples["command_index"]
        cmap = plt.get_cmap("tab20")
        for index, label in enumerate(labels):
            selected = command_index == index
            if selected.any():
                axis.scatter(
                    embedding[selected, 0], embedding[selected, 1],
                    s=5, alpha=0.58, color=cmap(index % 20),
                    edgecolors="none", label=str(label), rasterized=True)
        axis.legend(
            markerscale=3, fontsize=8, loc="best", ncol=2,
            frameon=False, labelcolor=ink)
    else:
        command = samples["command"]
        values = {
            "vx": command[:, 0],
            "vy": command[:, 1],
            "wz": command[:, 2],
            "phase": samples["phase"],
        }[args.color]
        scatter = axis.scatter(
            embedding[:, 0], embedding[:, 1], s=5, alpha=0.58, c=values,
            cmap="twilight" if args.color == "phase" else "coolwarm",
            edgecolors="none", rasterized=True)
        colorbar = fig.colorbar(scatter, ax=axis, label=args.color, fraction=0.046, pad=0.04)
        colorbar.outline.set_visible(False)

    standardize_note = " · standardized" if args.standardize else ""
    axis.set_title(
        f"history encoder latent UMAP · colored by {args.color}\n"
        f"n_neighbors={n_neighbors}, min_dist={args.min_dist:g}, metric={args.metric}"
        f"{standardize_note}\n{args.npz.name}",
        fontsize=12, color=ink, pad=14)
    axis.set_xlabel("UMAP-1", fontsize=9, color=muted)
    axis.set_ylabel("UMAP-2", fontsize=9, color=muted)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    output = args.out or args.npz.with_name(
        f"{args.npz.stem}_umap_{args.color}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    return output


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--color", default="command", choices=COLORS)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--drop-after-reset", type=int, default=10,
                        help="exclude N frames immediately following a reset")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--standardize", action="store_true",
                        help="z-score each latent dimension before UMAP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        samples = load_samples(
            args.npz, drop_after_reset=args.drop_after_reset,
            max_samples=args.max_samples, seed=args.seed)
        print(
            f"UMAP input: {samples['latent'].shape} "
            f"(kept {samples['kept_count']} / total {samples['total_count']})")
        embedding, n_neighbors = fit_umap(samples, args)
        output = plot_embedding(embedding, samples, args, n_neighbors)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"saved: {output}")
    raw = np.load(args.npz, allow_pickle=True)
    if "vel_pred" in raw and "base_lin_vel" in raw:
        error = np.abs(raw["vel_pred"] - raw["base_lin_vel"])
        print(
            f"vel head error: MAE={error.mean():.4f} m/s, "
            f"per-axis={error.mean(axis=(0, 1)).round(4)}")


if __name__ == "__main__":
    main()
