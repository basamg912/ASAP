"""Observation-corruption robustness: Baseline vs History Encoder (vx = 0.5 m/s).

Reads the sweep npz written by humanoidverse/eval_obs_corruption.py for both
checkpoints and renders a 2x5 small-multiples grid: fall rate vs corruption
level, one row per noise type (i.i.d. gaussian / constant bias), one column per
corrupted observation group.

Regenerate the sweeps (Isaac Lab, ~2 min per model on one GPU) from ASAP/:
  for M in baseline v3; do
    python humanoidverse/eval_obs_corruption.py \
      +checkpoint=ckpt/hist_ablation/rew1/$M/model_61000.pt +simulator=isaacsim \
      ++headless=True ++vx=0.5 ++envs_per_cond=24
  done
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "rew1")
OUT = os.path.join(HERE, "corruption_robustness.png")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NAME = {"baseline": "Baseline", "v3": "History Encoder"}
CLR = {"baseline": BLUE, "v3": ORANGE}

TARGETS = ["dof_pos", "dof_vel", "projected_gravity", "base_ang_vel", "all"]
DISP = {"dof_pos": "dof_pos", "dof_vel": "dof_vel", "projected_gravity": "proj_gravity",
        "base_ang_vel": "base_ang_vel", "all": "all four together"}
TYPES = [("gauss", "i.i.d. gaussian noise — resampled every step"),
         ("bias", "constant bias — a fixed offset, the same every step")]
LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0]

# summary rows: (type, target, level, falls_per_1k, vx, vx_err, |vy|, |wz|, act_rate)
tab = {}
for m in ("baseline", "v3"):
    d = np.load(os.path.join(ROOT, m, "corruption", "sweep.npz"), allow_pickle=True)
    tab[m] = {(r[0], r[1], float(r[2])): r for r in d["summary"]}
    tab[m]["_vx"] = float(d["cmd_vx"])

def series(m, typ, target, col):
    clean = tab[m][("clean", "none", 0.0)]
    return [float(clean[col])] + [float(tab[m][(typ, target, lv)][col]) for lv in LEVELS[1:]]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
fig = plt.figure(figsize=(15.0, 7.6), dpi=180)
gs = fig.add_gridspec(2, 5, left=0.062, right=0.985, top=0.755, bottom=0.125,
                      hspace=0.42, wspace=0.16)
ymax = max(max(series(m, t, g, 3)) for m in tab for t, _ in TYPES for g in TARGETS) * 1.12
x = np.arange(len(LEVELS))

for r, (typ, _) in enumerate(TYPES):
    for c, g in enumerate(TARGETS):
        ax = fig.add_subplot(gs[r, c])
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(AXIS)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(color=AXIS)
        ax.set_xticks(x, [f"{v:g}" if v else "0" for v in LEVELS])
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.25, len(LEVELS) - 0.75)
        if c:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("falls per 1k env-steps", fontsize=8, color=MUTED)
        if r:
            ax.set_xlabel("corruption level (× real σ)", fontsize=8, color=MUTED)
        for m in ("baseline", "v3"):
            y = series(m, typ, g, 3)
            ax.plot(x, y, color=CLR[m], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.8,
                    solid_capstyle="round", label=NAME[m], zorder=3)
        ax.set_title(DISP[g], loc="left", fontsize=9.5, color=INK, pad=6)
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left",
                      handlelength=1.4)

# 행 머리말은 그 행 축의 실제 상단에서 띄운다 (패널 제목과 겹치지 않게).
for r, (_, blurb) in enumerate(TYPES):
    top = gs[r, 0].get_position(fig).y1
    fig.text(0.062, top + 0.052, blurb, fontsize=9, color=INK, ha="left",
             weight="semibold")

fig.text(0.062, 0.955, "Observation corruption — Baseline vs History Encoder", fontsize=15,
         color=INK, ha="left", weight="semibold")
fig.text(0.062, 0.917,
         f"hist_ablation/rew1 · model_61000 · vx = {tab['baseline']['_vx']:g} m/s held for every env · "
         "200 clean warm-up steps, then 600 corrupted steps · 24 envs per condition · "
         "level is a multiple of that observation's measured on-policy σ",
         fontsize=8.5, color=MUTED, ha="left")
fig.text(0.062, 0.040,
         "Corruption uses the env's own observation-noise path, so it contaminates both the current "
         "observation and what the history buffer stores · neither policy saw observation noise in training.",
         fontsize=7.5, color=MUTED, ha="left")
fig.text(0.062, 0.018,
         "Caveat: the env draws the noise for the current observation and for the history entry separately, "
         "so within one step those two draws differ under gaussian noise — this slightly favours the model "
         "that averages over history.",
         fontsize=7.5, color=MUTED, ha="left")

fig.savefig(OUT)
print("saved:", OUT)
