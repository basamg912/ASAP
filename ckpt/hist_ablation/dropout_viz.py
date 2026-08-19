"""Single-joint sensor dropout: Baseline vs History Encoder (vx = 0.5 m/s).

Reads the dropout npz written by humanoidverse/eval_joint_dropout.py for both
checkpoints and renders a 2x2 heatmap grid: joints (rows) x failure mode
(columns), one row of panels per metric (fall rate, speed error), one column of
panels per model.

Falls alone understate the damage — most single-joint dropouts do not topple the
robot but do wreck locomotion (walking backwards, stalling, running away), so the
speed panel carries as much of the story as the fall panel.

Regenerate the sweeps (Isaac Lab, ~2 min per model on one GPU) from ASAP/:
  for M in baseline v3; do
    python humanoidverse/eval_joint_dropout.py \
      +checkpoint=ckpt/hist_ablation/rew1/$M/model_61000.pt +simulator=isaacsim \
      ++headless=True ++vx=0.5 ++envs_per_cond=16
  done
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "rew1")
OUT = os.path.join(HERE, "dropout_robustness.png")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS = "#c3c2b7"
NAME = {"baseline": "Baseline", "v3": "History Encoder"}
# 단일 색상 sequential (밝음 -> 진함). 색은 심각도만 뜻하고 모델 정체성이 아니다.
RAMP = LinearSegmentedColormap.from_list("sev", ["#f6efe8", "#f3b48f", "#eb6834", "#8f2f0d"])

JOINTS = ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
          "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch", "L_sh_pitch"]
COLS = [("zero", "pos+vel"), ("zero", "pos"), ("zero", "vel"),
        ("freeze", "pos+vel"), ("freeze", "pos"), ("freeze", "vel")]
COL_LAB = ["pos+vel", "pos", "vel", "pos+vel", "pos", "vel"]

tab, CMD = {}, None
for m in ("baseline", "v3"):
    d = np.load(os.path.join(ROOT, m, "corruption", "dropout.npz"), allow_pickle=True)
    tab[m] = {(r[0], r[1], r[2]): r for r in d["summary"]}
    CMD = float(d["cmd_vx"])

def grid(m, col):
    """(joint x condition) 행렬. col=3 낙상률, col=4 실측 vx."""
    return np.array([[float(tab[m][(mode, ch, j)][col]) for mode, ch in COLS] for j in JOINTS])

FALLS = {m: grid(m, 3) for m in tab}
SPEED = {m: np.abs(grid(m, 4) - CMD) for m in tab}          # 명령 대비 속도 오차
CLEAN = {m: (float(tab[m][("clean", "none", "none")][3]),
             abs(float(tab[m][("clean", "none", "none")][4]) - CMD)) for m in tab}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
})
fig = plt.figure(figsize=(13.6, 12.2), dpi=180)
gs = fig.add_gridspec(2, 2, left=0.135, right=0.90, top=0.845, bottom=0.075,
                      hspace=0.20, wspace=0.10)

METRICS = [("falls per 1k env-steps", FALLS, "{:.0f}", 0.6),
           ("|measured vx − command| (m/s)", SPEED, "{:.2f}", 0.25)]

for r, (mlabel, data, fmt, hide_below) in enumerate(METRICS):
    vmax = max(d.max() for d in data.values())
    for c, model in enumerate(("baseline", "v3")):
        ax = fig.add_subplot(gs[r, c])
        M = data[model]
        ax.imshow(M, cmap=RAMP, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(COLS)), COL_LAB, fontsize=8)
        ax.set_yticks(range(len(JOINTS)))
        ax.set_yticklabels(JOINTS if c == 0 else [], fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        # 셀 구분은 배경색 격자로 (2px surface gap)
        ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(JOINTS), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.6)
        ax.tick_params(which="minor", length=0)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] < hide_below:
                    continue
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.8,
                        color=SURFACE if M[i, j] > vmax * 0.55 else INK2)
        # zero / freeze 블록 경계
        ax.axvline(2.5, color=SURFACE, lw=3)
        ax.text(1.0, -0.95, "zero", ha="center", fontsize=8.5, color=INK2, weight="semibold")
        ax.text(4.0, -0.95, "freeze", ha="center", fontsize=8.5, color=INK2, weight="semibold")
        title = f"{NAME[model]} · {mlabel}"
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=22)
        base = CLEAN[model][r]
        ax.text(1.0, 1.006, f"no failure: {fmt.format(base)}", transform=ax.transAxes,
                fontsize=7.5, color=MUTED, ha="right", va="bottom")
    cax = fig.add_axes([0.915, gs[r, 0].get_position(fig).y0, 0.013,
                        gs[r, 0].get_position(fig).height])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=RAMP,
                                            norm=plt.Normalize(0, vmax)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5, color=MUTED)

fig.text(0.135, 0.955, "Single-joint sensor dropout — Baseline vs History Encoder",
         fontsize=15, color=INK, ha="left", weight="semibold")
fig.text(0.135, 0.925,
         f"hist_ablation/rew1 · model_61000 · vx = {CMD:g} m/s · one joint's observation is broken for 600 "
         "steps after a 200-step clean warm-up · 16 envs per cell",
         fontsize=8.5, color=MUTED, ha="left")
fig.text(0.135, 0.895,
         "zero = the reading dies at 0 (dof_pos is measured from the default pose, so 0 reads as \"neutral\") · "
         "freeze = the reading sticks at its value when the sensor failed",
         fontsize=8.5, color=MUTED, ha="left")
fig.text(0.135, 0.038,
         "Falls alone understate the damage: most dropouts keep the robot upright but break locomotion, which "
         "is what the lower row shows · L_sh_pitch (shoulder) is a control — a joint locomotion should not need.",
         fontsize=7.5, color=MUTED, ha="left")
fig.text(0.135, 0.018,
         "The break is applied to both the current observation and the history buffer, and is deterministic, so "
         "both paths always see the same broken value.",
         fontsize=7.5, color=MUTED, ha="left")

fig.savefig(OUT)
print("saved:", OUT)
