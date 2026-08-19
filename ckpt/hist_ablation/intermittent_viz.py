"""Intermittent observation glitches: Baseline vs History Encoder (vx = 0.5 m/s).

Reads the intermittent npz written by humanoidverse/eval_intermittent_noise.py for
both checkpoints and renders a 2x3 small-multiples grid: fall rate vs duty cycle,
one row per spike magnitude, one column per burst length.

Reading the columns left to right holds the total amount of corruption fixed and
only changes how it is clustered in time — isolated single-step spikes on the
left, 20-step bursts on the right.

Regenerate the sweeps (Isaac Lab, ~2 min per model on one GPU) from ASAP/:
  for M in baseline v3; do
    python humanoidverse/eval_intermittent_noise.py \
      +checkpoint=ckpt/hist_ablation/rew1/$M/model_61000.pt +simulator=isaacsim \
      ++headless=True ++vx=0.5 ++envs_per_cond=32
  done
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "rew1")
OUT = os.path.join(HERE, "intermittent_robustness.png")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NAME = {"baseline": "Baseline", "v3": "History Encoder"}
CLR = {"baseline": "#2a78d6", "v3": "#eb6834"}

DUTIES = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
BURSTS = [1, 5, 20]
MAGS = [4.0, 8.0]

tab, CMD = {}, None
for m in ("baseline", "v3"):
    d = np.load(os.path.join(ROOT, m, "corruption", "intermittent.npz"), allow_pickle=True)
    tab[m] = {(float(r[0]), int(r[1]), float(r[2])): r for r in d["summary"]}
    CMD = float(d["cmd_vx"])

def series(m, burst, mag, col=3):
    return [float(tab[m][(d, burst, mag)][col]) for d in DUTIES]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
fig = plt.figure(figsize=(13.4, 8.4), dpi=180)
gs = fig.add_gridspec(2, 3, left=0.075, right=0.982, top=0.725, bottom=0.115,
                      hspace=0.54, wspace=0.13)
ymax = max(max(series(m, b, g)) for m in tab for b in BURSTS for g in MAGS) * 1.12
x = np.arange(len(DUTIES))

for r, mag in enumerate(MAGS):
    for c, burst in enumerate(BURSTS):
        ax = fig.add_subplot(gs[r, c])
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(AXIS)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(color=AXIS)
        ax.set_xticks(x, [f"{d:g}" for d in DUTIES])
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.25, len(DUTIES) - 0.75)
        if c:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("falls per 1k env-steps", fontsize=8, color=MUTED)
        ax.set_xlabel("duty cycle (fraction of steps corrupted)", fontsize=8, color=MUTED)
        for m in ("baseline", "v3"):
            ax.plot(x, series(m, burst, mag), color=CLR[m], lw=2, marker="o", ms=6,
                    mec=SURFACE, mew=1.8, solid_capstyle="round", label=NAME[m], zorder=3)
        head = "isolated spikes" if burst == 1 else f"{burst}-step bursts"
        ax.set_title(f"{head}  (burst = {burst})", loc="left", fontsize=9.5, color=INK, pad=6)
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left",
                      handlelength=1.4)

for r, mag in enumerate(MAGS):
    top = gs[r, 0].get_position(fig).y1
    fig.text(0.075, top + 0.055, f"spike magnitude {mag:g}σ", fontsize=9, color=INK,
             ha="left", weight="semibold")

fig.text(0.075, 0.955, "Intermittent observation glitches — Baseline vs History Encoder",
         fontsize=15, color=INK, ha="left", weight="semibold")
fig.text(0.075, 0.918,
         f"hist_ablation/rew1 · model_61000 · vx = {CMD:g} m/s · gaussian spikes on dof_pos, dof_vel, "
         "proj_gravity and base_ang_vel at once · 200 clean warm-up steps, then 600 steps · 32 envs per point",
         fontsize=8.5, color=MUTED, ha="left")
fig.text(0.075, 0.888,
         "Across a row the total corruption is held fixed and only its clustering in time changes · at duty 1 "
         "all three columns are the same experiment, and they agree — a calibration check",
         fontsize=8.5, color=MUTED, ha="left")
fig.text(0.075, 0.038,
         "Within a step the current observation and the history entry get the same noise draw here, so a spike "
         "is one physical glitch rather than two independent ones · magnitude is a multiple of each "
         "observation's measured on-policy σ.",
         fontsize=7.5, color=MUTED, ha="left")
fig.text(0.075, 0.018,
         "Neither policy was trained with observation noise, so every point is out of distribution.",
         fontsize=7.5, color=MUTED, ha="left")

fig.savefig(OUT)
print("saved:", OUT)
