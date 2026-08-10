"""
collect_encoder_latents.py 가 저장한 npz 를 t-SNE 로 시각화한다.

  python scripts/plot_latent_tsne.py logs/.../latents/latents_ckpt_100.npz
  python scripts/plot_latent_tsne.py <npz> --color phase --max-samples 8000

--color:
  command : 배정된 커맨드 격자 라벨로 색칠 (기본) — latent 가 커맨드를 구분하는지
  vx / vy / wz : 커맨드 성분 연속값
  phase : gait 위상 — latent 가 커맨드 대신 위상만 인코딩하는지 확인용
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--color", default="command",
                    choices=["command", "vx", "vy", "wz", "phase"])
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--drop-after-reset", type=int, default=10,
                    help="리셋 직후 N 스텝은 과도 상태라 제외")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    d = np.load(args.npz, allow_pickle=True)
    lat, cmd = d["latent"], d["command"]          # [T,N,L], [T,N,3]
    T, N, _ = lat.shape

    # 리셋 직후 구간 마스킹
    keep = np.ones((T, N), dtype=bool)
    if "done" in d and args.drop_after_reset > 0:
        done = d["done"]
        for t, n in zip(*np.nonzero(done)):
            keep[t + 1:t + 1 + args.drop_after_reset, n] = False

    X = lat.reshape(T * N, -1)[keep.reshape(-1)]
    C = cmd.reshape(T * N, 3)[keep.reshape(-1)]
    idx = np.tile(d["env_cmd_idx"], (T, 1)).reshape(-1)[keep.reshape(-1)]
    ph = (d["phase"].reshape(-1)[keep.reshape(-1)] if "phase" in d
          else np.zeros(len(X), dtype=np.float32))

    if len(X) > args.max_samples:
        sel = np.random.default_rng(0).choice(len(X), args.max_samples, replace=False)
        X, C, idx, ph = X[sel], C[sel], idx[sel], ph[sel]
    print(f"t-SNE 입력: {X.shape} (전체 {T*N} 중)")

    emb = TSNE(n_components=2, perplexity=args.perplexity,
               init="pca", random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 8))
    if args.color == "command":
        labels = d["labels"]
        cmap = plt.get_cmap("tab20")
        for k, lab in enumerate(labels):
            m = idx == k
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.6,
                           color=cmap(k % 20), label=str(lab))
        ax.legend(markerscale=3, fontsize=8, loc="best", ncol=2)
    else:
        vals = {"vx": C[:, 0], "vy": C[:, 1], "wz": C[:, 2], "phase": ph}[args.color]
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.6, c=vals,
                        cmap="twilight" if args.color == "phase" else "coolwarm")
        fig.colorbar(sc, ax=ax, label=args.color)

    ax.set_title(f"history encoder latent t-SNE — colored by {args.color}\n{args.npz.name}")
    ax.set_xticks([]); ax.set_yticks([])
    out = args.out or args.npz.with_name(f"{args.npz.stem}_tsne_{args.color}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"저장: {out}")

    if "vel_pred" in d:
        vp, gt = d["vel_pred"], d["base_lin_vel"]
        print(f"vel head 오차: MAE={np.abs(vp - gt).mean():.4f} m/s, "
              f"축별={np.abs(vp - gt).mean(axis=(0, 1)).round(4)}")


if __name__ == "__main__":
    main()
