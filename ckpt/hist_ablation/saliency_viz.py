"""Visualize actor input sensitivity: baseline vs v3 (hist_ablation/rew1, model_61000).

Recomputes end-to-end mean |Jacobian| for both checkpoints (same sampling as
input_saliency.py / input_saliency_v3.py) and renders one 4-panel PNG:
  A. sensitivity share by input group (grouped hbar)
  B. per-dim sensitivity vs history depth (line)
  C/D. top-12 individual input dims per model (hbar, current vs history shade)
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch

ROOT = "/home/kist/work/workspace/ACM/ASAP/ckpt/hist_ablation/rew1"
OUT = "/home/kist/work/workspace/ACM/ASAP/ckpt/hist_ablation/saliency_baseline_v3.png"

# ---------------- palette (validated: see dataviz skill) ----------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LT = "#2a78d6", "#86b6ef"      # baseline: current / history
ORANGE, ORANGE_LT = "#eb6834", "#f29e74"  # v3:       current / history

# ---------------- obs metadata ----------------
DOF = ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
       "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
       "waist_yaw", "waist_roll", "waist_pitch",
       "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow",
       "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow"]
COMP = {"actions": (23, DOF), "base_ang_vel": (3, ["x", "y", "z"]),
        "command_ang_vel": (1, ["yaw"]), "command_lin_vel": (2, ["vx", "vy"]),
        "command_stand": (1, ["stand"]), "dof_pos": (23, DOF), "dof_vel": (23, DOF),
        "projected_gravity": (3, ["x", "y", "z"])}
CUR_KEYS = sorted(COMP.keys())
STD = {"actions": 1.0, "base_ang_vel": 0.25, "command_ang_vel": 0.5, "command_lin_vel": 0.5,
       "command_stand": 0.5, "dof_pos": 0.5, "dof_vel": 0.25, "projected_gravity": 0.2}
T = 5

def make_layout(hist_keys):
    names, groups = [], []
    for k in CUR_KEYS:
        d, lab = COMP[k]
        names += [f"{k}:{l}" for l in lab]; groups += [k] * d
    for hk in hist_keys:
        d, lab = COMP[hk]
        for t in range(T):
            names += [f"hist[t-{t+1}] {hk}:{l}" for l in lab]
            groups += [f"hist_{hk}"] * d
    return names, groups

def nominal(names):
    x0 = torch.zeros(len(names))
    for i, nm in enumerate(names):
        if "projected_gravity:z" in nm:
            x0[i] = -1.0
    return x0

def jac_saliency(fn, xs):
    xs = xs.clone().requires_grad_(True)
    mu = fn(xs)
    acc = torch.zeros(xs.shape[1])
    for j in range(mu.shape[1]):
        g, = torch.autograd.grad(mu[:, j].sum(), xs, retain_graph=(j < mu.shape[1] - 1))
        acc += g.abs().mean(dim=0)
    return acc.detach().numpy()

def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)

# ---------------- baseline: actor(474) ----------------
sd_b = torch.load(f"{ROOT}/baseline/model_61000.pt", map_location="cpu",
                  weights_only=False)["actor_model_state_dict"]
actor_b = mlp([474, 512, 256, 128, 23])
actor_b.load_state_dict({k.replace("actor_module.module.", ""): v for k, v in sd_b.items()
                         if k.startswith("actor_module")}, strict=True)
actor_b.eval()
names_b, groups_b = make_layout(CUR_KEYS)          # short_history has all 8 comps
assert len(names_b) == 474

# ---------------- v3: actor(cat[obs79, student(hist375)]) ----------------
HIST_V3 = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
KEY_DIMS = [COMP[k][0] for k in HIST_V3]

class MLPMixer(nn.Module):
    def __init__(self, F=75, C=5, out=19):
        super().__init__()
        self.ln, self.ln2 = nn.LayerNorm(F), nn.LayerNorm(F)
        self.feature_mixing = nn.Sequential(nn.Linear(F, 256), nn.ELU(),
                                            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, F))
        self.channel_mixing = nn.Sequential(nn.Linear(C, 64), nn.ELU(), nn.Linear(64, C))
        self.output_layer = nn.Linear(C * F, out)

    def forward(self, x):
        y = self.feature_mixing(self.ln(x))
        x = x + y
        y = self.channel_mixing(self.ln2(x).transpose(1, 2))
        x = x + y.transpose(1, 2)
        return self.output_layer(x.flatten(1))

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLPMixer()
    def forward(self, hist):
        Nb = hist.shape[0]
        chunks = torch.split(hist, [d * T for d in KEY_DIMS], dim=1)
        x = torch.cat([c.view(Nb, T, d) for c, d in zip(chunks, KEY_DIMS)], dim=-1)
        out = self.net(x)
        return out[..., :3], out[..., 3:]

sd_v = torch.load(f"{ROOT}/v3/model_61000.pt", map_location="cpu",
                  weights_only=False)["actor_model_state_dict"]
student = Student()
student.load_state_dict({k.replace("student.", ""): v for k, v in sd_v.items()
                         if k.startswith("student.")}, strict=True)
actor_v = mlp([98, 512, 256, 128, 23])
actor_v.load_state_dict({k.replace("actor_module.module.", ""): v for k, v in sd_v.items()
                         if k.startswith("actor_module")}, strict=True)
student.eval(); actor_v.eval()
names_v, groups_v = make_layout(HIST_V3)
assert len(names_v) == 454

def fwd_v3(x):
    v, z = student(x[:, 79:])
    return actor_v(torch.cat([x[:, :79], v, z], dim=-1))

# ---------------- compute saliency (nominal-state sampling, B=1024) ----------------
B = 1024
results = {}
for tag, fn, names, groups in [("baseline", actor_b, names_b, groups_b),
                               ("v3", fwd_v3, names_v, groups_v)]:
    torch.manual_seed(0)
    std_vec = torch.tensor([STD[g.replace("hist_", "")] for g in groups])
    xs = nominal(names) + torch.randn(B, len(names)) * std_vec
    sal = jac_saliency(fn, xs)
    results[tag] = dict(sal=sal, names=names, groups=groups)
    print(f"{tag}: total {sal.sum():.1f}")

def group_share(res):
    tot, out = res["sal"].sum(), {}
    for g in set(res["groups"]):
        out[g] = 100.0 * res["sal"][[i for i, x in enumerate(res["groups"]) if x == g]].sum() / tot
    return out

def depth_mean(res):
    buckets = {k: [] for k in ["current"] + [f"t-{t}" for t in range(1, 6)]}
    for i, nm in enumerate(res["names"]):
        key = nm.split("]")[0].split("[")[1] if nm.startswith("hist") else "current"
        buckets[key].append(res["sal"][i])
    return {k: float(np.mean(v)) for k, v in buckets.items()}

share_b, share_v = group_share(results["baseline"]), group_share(results["v3"])
depth_b, depth_v = depth_mean(results["baseline"]), depth_mean(results["v3"])

def topk(res, k=12):
    idx = np.argsort(-res["sal"])[:k]
    return [(res["names"][i], float(res["sal"][i]), res["names"][i].startswith("hist")) for i in idx]

top_b, top_v = topk(results["baseline"]), topk(results["v3"])

# ---------------- figure ----------------
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
fig = plt.figure(figsize=(15.2, 11.2), dpi=180)
gs = fig.add_gridspec(2, 2, left=0.105, right=0.975, top=0.895, bottom=0.06,
                      hspace=0.34, wspace=0.42, height_ratios=[1.12, 1.0])
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

def style_barh(ax, xmax, xlabel):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=AXIS)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)

def rounded_barh(ax, y, w, h, color, r_px=4.0):
    """barh with 4px-rounded data end, square at the baseline."""
    p0, p1 = ax.transData.transform((0, 0)), ax.transData.transform((1, 1))
    ppx, ppy = p1[0] - p0[0], p1[1] - p0[1]
    r = r_px / ppx
    if w <= 2.2 * r:
        ax.add_patch(Rectangle((0, y - h / 2), w, h, fc=color, ec="none"))
        return
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ppx / ppy, fc=color, ec="none"))
    ax.add_patch(Rectangle((0, y - h / 2), w - r, h, fc=color, ec="none"))

# ---- panel A: group share ----
DISP = {"dof_vel": "dof_vel", "dof_pos": "dof_pos", "projected_gravity": "proj_gravity",
        "base_ang_vel": "base_ang_vel", "actions": "prev_actions",
        "command_lin_vel": "cmd_lin_vel", "command_ang_vel": "cmd_ang_vel",
        "command_stand": "cmd_stand"}
rows = sorted(set(share_b) | set(share_v), key=lambda g: -share_b.get(g, 0.0))
labels = [("hist " if g.startswith("hist_") else "") + DISP[g.replace("hist_", "")] for g in rows]
amax = max(max(share_b.values()), max(share_v.values())) * 1.22
axA.set_ylim(-0.55, len(rows) - 0.45)
axA.set_yticks(range(len(rows)), labels[::-1])
axA.invert_yaxis(); axA.set_yticks(range(len(rows))); axA.set_yticklabels(labels)
axA.set_ylim(len(rows) - 0.45, -0.55)
style_barh(axA, amax, "share of total sensitivity (%)")
fig.canvas.draw()
H = 0.34
for i, g in enumerate(rows):
    if g in share_b:
        rounded_barh(axA, i - 0.19, share_b[g], H, BLUE)
    if g in share_v:
        rounded_barh(axA, i + 0.19, share_v[g], H, ORANGE)
    else:
        axA.text(0.35, i + 0.19, "not in v3 (no cmd in history)", fontsize=6.5,
                 color=MUTED, style="italic", va="center")
    if max(share_b.get(g, 0), share_v.get(g, 0)) >= 3.0:  # label the big ones only
        if g in share_b:
            axA.text(share_b[g] + amax * 0.012, i - 0.19, f"{share_b[g]:.1f}",
                     fontsize=7, color=MUTED, va="center")
        if g in share_v:
            axA.text(share_v[g] + amax * 0.012, i + 0.19, f"{share_v[g]:.1f}",
                     fontsize=7, color=MUTED, va="center")
axA.set_title("A · Sensitivity share by input group", loc="left", fontsize=11,
              color=INK, pad=10)
axA.legend(handles=[Patch(fc=BLUE, label="baseline"), Patch(fc=ORANGE, label="v3")],
           frameon=False, fontsize=8, labelcolor=INK2, loc="lower right",
           handlelength=1.1, handleheight=1.1)

# ---- panel B: sensitivity vs history depth ----
xs = ["current", "t-1", "t-2", "t-3", "t-4", "t-5"]
yb, yv = [depth_b[k] for k in xs], [depth_v[k] for k in xs]
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    axB.spines[s].set_color(AXIS)
axB.yaxis.grid(True, color=GRID, lw=0.8)
axB.set_axisbelow(True)
axB.tick_params(color=AXIS)
axB.plot(xs, yb, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
         solid_capstyle="round", label="baseline")
axB.plot(xs, yv, color=ORANGE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
         solid_capstyle="round", label="v3")
axB.set_ylim(0, max(max(yb), max(yv)) * 1.15)
axB.set_ylabel("mean |∂a/∂x| per input dim", fontsize=8, color=MUTED)
axB.set_xlabel("input recency", fontsize=8, color=MUTED)
axB.text(5.08, yb[-1], f"baseline  {yb[-1]:.1f}", fontsize=8, color=INK2, va="bottom")
axB.text(5.08, yv[-1], f"v3  {yv[-1]:.1f}", fontsize=8, color=INK2, va="top")
axB.set_xlim(-0.3, 6.4)
axB.set_title("B · Per-dim sensitivity vs input recency", loc="left", fontsize=11,
              color=INK, pad=10)
axB.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right",
           handlelength=1.4)
axB.text(0.0, 1.005, "v3 leans harder on the current step; its mixer treats t-2…t-5 evenly",
         transform=axB.transAxes, fontsize=8, color=MUTED, va="bottom")

# ---- panels C/D: top-12 dims ----
def top_panel(ax, top, dark, light, model):
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    hist = [t[2] for t in top][::-1]
    xmax = max(vals) * 1.18
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_ylim(-0.55, len(names) - 0.45)
    style_barh(ax, xmax, "mean |∂action/∂input|")
    fig.canvas.draw()
    for i, (v, h) in enumerate(zip(vals, hist)):
        rounded_barh(ax, i, v, 0.55, light if h else dark)
        ax.text(v + xmax * 0.012, i, f"{v:.1f}", fontsize=7, color=MUTED, va="center")
    ax.set_title(f"{'C' if model == 'baseline' else 'D'} · Top-12 input dims — {model}",
                 loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(handles=[Patch(fc=dark, label="current step"),
                       Patch(fc=light, label="history")],
              frameon=False, fontsize=8, labelcolor=INK2, loc="lower right",
              handlelength=1.1, handleheight=1.1)

top_panel(axC, top_b, BLUE, BLUE_LT, "baseline")
top_panel(axD, top_v, ORANGE, ORANGE_LT, "v3")

fig.text(0.105, 0.955, "Actor input sensitivity — baseline vs v3", fontsize=15,
         color=INK, ha="left", weight="semibold")
fig.text(0.105, 0.925,
         "hist_ablation/rew1 · model_61000 · mean |∂action/∂input| over 1024 samples around the nominal "
         "standing state, in scaled-obs space · v3 gradients flow end-to-end through its history encoder",
         fontsize=8.5, color=MUTED, ha="left")
fig.text(0.105, 0.012,
         "Notes: dof_vel observations are pre-scaled ×0.05 (per-physical-unit sensitivity is 1/20 of shown) · "
         "v3's history contains no command inputs by design · sensitivity ≠ closed-loop causal contribution.",
         fontsize=7.5, color=MUTED, ha="left")

fig.savefig(OUT)
print("saved:", OUT)
