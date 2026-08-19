"""Visualize actor input sensitivity: baseline vs v3 (hist_ablation/rew1, model_61000).

Recomputes mean |Jacobian| for both checkpoints and renders three PNGs:
  saliency_baseline_vs_hist_encoder.png  6 panels, the two side by side
  saliency_baseline.png                  4 panels, Baseline alone
  saliency_hist_encoder.png              5 panels, History Encoder alone (adds the latent view)
"v3" is the checkpoint directory name; in every figure it is labelled
"History Encoder". Panels:
  A. sensitivity share by input group (grouped hbar)
  B. per-dim sensitivity vs history depth (line)
  C/D. top-20 individual input dims per model (hbar, current vs history shade)
  E. actor-input view: what each first layer receives, weighted by real sigma
  F. per-latent-dim sensitivity, scaled by the latent's real spread (encoder only)

Two views, deliberately different:
  A-D are end-to-end. v3's gradients flow through the student encoder, so its
  375 history dims get credit and the latent is an interior node with no bar of
  its own -- that is why the latent is missing from A/C/D by construction.
  E-F stop at each actor's own first-layer inputs (baseline 474 raw dims, v3
  obs79 + latent19), which is the only view where the latent has a bar, and
  scale every gradient by that input's measured sigma so inputs with different
  units are comparable.

States come from collected rollouts when available, so each policy is evaluated
on the states it actually visits; without them the script falls back to
synthetic samples around the nominal standing state and says so in the caption.
Collect them (Isaac Lab, ~40 s per model on one GPU) from ASAP/:
  for M in baseline v3; do
    python humanoidverse/collect_obs_stats.py \
      +checkpoint=ckpt/hist_ablation/rew1/$M/model_61000.pt +simulator=isaacsim \
      ++num_envs=120 ++warmup_steps=300 ++record_steps=600 ++stride=3 ++headless=True
  done
The default 10-command grid (stand / fwd / bwd / strafe / turn) covers the
operating range; a forward-only grid understates the spread of the velocity head.
That script also reports per-command fall rate and tracking error -- worth a look
before trusting the sigma, since a command the policy cannot follow contributes
fall dynamics rather than normal operation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "rew1")
OUT_CMP = os.path.join(HERE, "saliency_baseline_vs_hist_encoder.png")
OUT_BASE = os.path.join(HERE, "saliency_baseline.png")
OUT_HIST = os.path.join(HERE, "saliency_hist_encoder.png")
OBS_NPZ = {m: os.path.join(ROOT, m, "obs_stats", "obs_ckpt_61000.npz")
           for m in ("baseline", "v3")}

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

# ---------------- state ensemble ----------------
def real_states(model, names):
    """수집된 롤아웃에서 policy 가 실제로 본 입력 행렬. 없으면 None."""
    path = OBS_NPZ[model]
    if not os.path.exists(path):
        return None
    d = np.load(path)
    ok = (d["ep_len"] >= 20) & (~d["done"])      # 리셋 직후 과도구간·종료 스텝 제외
    x = (d["actor_obs"] if "encoder_obs" not in d
         else np.concatenate([d["actor_obs"], d["encoder_obs"]], axis=-1))[ok]
    assert x.shape[1] == len(names), f"{model}: obs {x.shape[1]} != layout {len(names)}"
    return torch.from_numpy(x)

def state_ensemble(model, names, groups):
    """B개 상태를 뽑는다. 실제 롤아웃이 있으면 그쪽, 없으면 nominal 주변 합성 샘플."""
    xr = real_states(model, names)
    if xr is None:
        std_vec = torch.tensor([STD[g.replace("hist_", "")] for g in groups])
        return nominal(names) + torch.randn(B, len(names)) * std_vec, "synthetic nominal-state samples"
    sel = torch.randperm(xr.shape[0])[:B]
    return xr[sel], f"real on-policy states ({xr.shape[0]} collected)"

# ---------------- compute saliency (B=1024 states) ----------------
B = 1024
results = {}
for tag, fn, names, groups in [("baseline", actor_b, names_b, groups_b),
                               ("v3", fwd_v3, names_v, groups_v)]:
    torch.manual_seed(0)
    xs, src = state_ensemble(tag, names, groups)
    sal = jac_saliency(fn, xs)
    # 실제 변동폭. gradient x sigma 로 환산해야 스케일이 다른 입력끼리 비교가 된다.
    results[tag] = dict(sal=sal, names=names, groups=groups,
                        sigma=xs.std(dim=0).numpy(), xs=xs, src=src)
    print(f"{tag}: total {sal.sum():.1f}  [{src}]")
STATE_SRC = results["v3"]["src"]

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

def topk(res, k=20):
    idx = np.argsort(-res["sal"])[:k]
    return [(res["names"][i], float(res["sal"][i]), res["names"][i].startswith("hist")) for i in idx]

top_b, top_v = topk(results["baseline"]), topk(results["v3"])

# ---------------- actor-input view (실제 변동폭으로 가중) ----------------
# A-D 는 end-to-end 라 latent 이 내부 노드로 묻힌다. 여기서는 각 actor 가 1층에서
# 실제로 받는 입력(baseline 474, v3 = obs79 + latent19)만 놓고 비교한다.
xs_v = results["v3"]["xs"]
with torch.no_grad():
    v_s, z_s = student(xs_v[:, 79:])
lat0 = torch.cat([v_s, z_s], dim=-1)
sal_lat = jac_saliency(lambda l: actor_v(torch.cat([xs_v[:, :79], l], dim=-1)), lat0)
lat_std = lat0.std(dim=0).numpy()
LAT_NAMES = [f"v:{c}" for c in "xyz"] + [f"z{i}" for i in range(16)]

# obs 쪽 actor-input gradient 는 end-to-end 값과 같다 (history 경로가 obs 를 안 지난다).
eff = {m: results[m]["sal"] * results[m]["sigma"] for m in ("baseline", "v3")}
eff_lat = sal_lat * lat_std
tot_eff = {"baseline": eff["baseline"].sum(),
           "v3": eff["v3"][:79].sum() + eff_lat.sum()}
print(f"actor-input effective sensitivity | baseline {tot_eff['baseline']:.1f} "
      f"(hist {100 * eff['baseline'][79:].sum() / tot_eff['baseline']:.1f}%) | "
      f"v3 obs {eff['v3'][:79].sum():.1f} + latent {eff_lat.sum():.1f} "
      f"(latent {100 * eff_lat.sum() / tot_eff['v3']:.1f}%)")

# ---------------- figure ----------------
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
NAME = {"baseline": "Baseline", "v3": "History Encoder"}
CLR = {"baseline": (BLUE, BLUE_LT), "v3": (ORANGE, ORANGE_LT)}
DISP = {"dof_vel": "dof_vel", "dof_pos": "dof_pos", "projected_gravity": "proj_gravity",
        "base_ang_vel": "base_ang_vel", "actions": "prev_actions",
        "command_lin_vel": "cmd_lin_vel", "command_ang_vel": "cmd_ang_vel",
        "command_stand": "cmd_stand"}
DISP2 = dict(DISP, latent_v="latent v̂", latent_z="latent z")

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

def model_legend(ax, models):
    """2개 모델일 때만 범례. 단일 계열은 제목이 이미 이름을 말해준다."""
    if len(models) < 2:
        return
    ax.legend(handles=[Patch(fc=CLR[m][0], label=NAME[m]) for m in models],
              frameon=False, fontsize=8, labelcolor=INK2, loc="lower right",
              handlelength=1.1, handleheight=1.1)

# 그룹별 차원 수. A/E 의 막대는 합이라 차원 수를 모르면 크기를 잘못 읽는다.
NDIM = {}
for m in ("baseline", "v3"):
    for g in results[m]["groups"]:
        NDIM[g] = NDIM.get(g, 0) + 0
for m in ("baseline", "v3"):
    for g in set(results[m]["groups"]):
        NDIM[g] = sum(1 for x in results[m]["groups"] if x == g)
NDIM["latent_v"], NDIM["latent_z"] = 3, 16

def glabel(g):
    base = ("hist " if g.startswith("hist_") else "") + DISP2[g.replace("hist_", "")]
    return f"{base} ({NDIM[g]})"

# ---- panel: sensitivity share by input group ----
def panel_group_share(fig, ax, models, letter):
    share = {m: group_share(results[m]) for m in models}
    lead = models[0]
    rows = sorted(set().union(*[set(s) for s in share.values()]),
                  key=lambda g: -share[lead].get(g, 0.0))
    amax = max(max(s.values()) for s in share.values()) * 1.22
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([glabel(g) for g in rows], fontsize=7.5)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    style_barh(ax, amax, "share of total sensitivity (%)")
    fig.canvas.draw()
    offs = [0.0] if len(models) == 1 else [-0.19, 0.19]
    h = 0.55 if len(models) == 1 else 0.34
    for i, g in enumerate(rows):
        for m, off in zip(models, offs):
            if g in share[m]:
                rounded_barh(ax, i + off, share[m][g], h, CLR[m][0])
                if max(s.get(g, 0) for s in share.values()) >= 3.0:
                    ax.text(share[m][g] + amax * 0.012, i + off, f"{share[m][g]:.1f}",
                            fontsize=7, color=MUTED, va="center")
            elif len(models) > 1:
                ax.text(0.35, i + off, f"not in {NAME[m]} (no cmd in history)", fontsize=6.5,
                        color=MUTED, style="italic", va="center")
    ax.set_title(f"{letter} · Sensitivity share by input group", loc="left", fontsize=11,
                 color=INK, pad=10)
    ax.text(0.0, 1.005, "bars are sums — (n) is the group's dim count, so wide groups add up",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    model_legend(ax, models)

# ---- panel: per-dim sensitivity vs input recency ----
def panel_recency(fig, ax, models, letter):
    xs_lab = ["current", "t-1", "t-2", "t-3", "t-4", "t-5"]
    ys = {m: [depth_mean(results[m])[k] for k in xs_lab] for m in models}
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(color=AXIS)
    for m in models:
        ax.plot(xs_lab, ys[m], color=CLR[m][0], lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
                solid_capstyle="round", label=NAME[m])
        ax.text(5.08, ys[m][-1], f"{NAME[m]}  {ys[m][-1]:.1f}", fontsize=8, color=INK2,
                va="bottom" if m == "baseline" else "top")
    ax.set_ylim(0, max(max(v) for v in ys.values()) * 1.15)
    ax.set_ylabel("mean |∂a/∂x| per input dim", fontsize=8, color=MUTED)
    ax.set_xlabel("input recency", fontsize=8, color=MUTED)
    ax.set_xlim(-0.3, 6.4)
    ax.set_title(f"{letter} · Per-dim sensitivity vs input recency", loc="left", fontsize=11,
                 color=INK, pad=10)
    note = ("the encoder leans harder on the current step; it treats t-2…t-5 evenly"
            if models == ["v3"] else
            "per dim, not summed — every past step is worth far less than the current one")
    ax.text(0.0, 1.005, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    if len(models) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right", handlelength=1.4)

# ---- panel: top-k individual dims ----
def panel_top(fig, ax, model, letter, k=20):
    top = topk(results[model], k)
    dark, light = CLR[model]
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
    ax.set_title(f"{letter} · Top-{k} input dims — {NAME[model]}", loc="left", fontsize=11,
                 color=INK, pad=10)
    ax.legend(handles=[Patch(fc=dark, label="current step"), Patch(fc=light, label="history")],
              frameon=False, fontsize=8, labelcolor=INK2, loc="lower right",
              handlelength=1.1, handleheight=1.1)

# ---- panel: actor-input view (gradient x real sigma) ----
def eff_rows_for(model):
    out = {}
    for g in sorted(set(results[model]["groups"])):
        sel = [i for i, gg in enumerate(results[model]["groups"]) if gg == g]
        out[g] = 100.0 * eff[model][sel].sum() / tot_eff[model]
    if model == "v3":
        out = {g: v for g, v in out.items() if not g.startswith("hist_")}
        out["latent_v"] = 100.0 * eff_lat[:3].sum() / tot_eff["v3"]
        out["latent_z"] = 100.0 * eff_lat[3:].sum() / tot_eff["v3"]
    return out

def panel_actor_input(fig, ax, models, letter):
    row = {m: eff_rows_for(m) for m in models}
    # 값 순 정렬 대신 current / raw history / latent 3블록. 두 모델의 입력 공간이
    # 달라서 (baseline 은 raw history, encoder 는 latent) 값 순으로 섞으면 구조가 안 보인다.
    cur = sorted(CUR_KEYS, key=lambda g: -max(r.get(g, 0) for r in row.values()))
    hist = sorted([g for g in row.get("baseline", {}) if g.startswith("hist_")],
                  key=lambda g: -row["baseline"][g])
    lat = ["latent_v", "latent_z"] if "v3" in models else []
    rows = cur + hist + lat
    emax = max(max(r.values()) for r in row.values()) * 1.20
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([glabel(g) for g in rows], fontsize=7.5)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    style_barh(ax, emax, "share of that actor's own input sensitivity (%)")
    fig.canvas.draw()
    for y, tag in ((len(cur) - 0.5, "raw history"), (len(cur) + len(hist) - 0.5, "encoder latent")):
        if (tag == "raw history" and hist) or (tag == "encoder latent" and lat):
            ax.axhline(y, color=AXIS, lw=0.7, ls=(0, (3, 3)), zorder=0)
            owner = "Baseline-only: " if len(models) > 1 and tag == "raw history" else ""
            owner = "History-Encoder-only: " if len(models) > 1 and tag == "encoder latent" else owner
            ax.text(emax * 0.985, y - 0.12, owner + tag, fontsize=6.5, color=MUTED,
                    style="italic", ha="right", va="bottom")
    offs = [0.0] if len(models) == 1 else [-0.19, 0.19]
    h = 0.55 if len(models) == 1 else 0.34
    for i, g in enumerate(rows):
        for m, off in zip(models, offs):
            if g in row[m]:
                rounded_barh(ax, i + off, row[m][g], h, CLR[m][0])
        if max(r.get(g, 0) for r in row.values()) >= 2.0:
            for m, off in zip(models, offs):
                if g in row[m]:
                    ax.text(row[m][g] + emax * 0.012, i + off, f"{row[m][g]:.1f}",
                            fontsize=6.5, color=MUTED, va="center")
    ax.set_title(f"{letter} · Actor-input view — what the first layer actually receives",
                 loc="left", fontsize=11, color=INK, pad=10)
    note = {("baseline",): "474 raw dims straight into the trunk · gradient × real σ",
            ("v3",): "375 history dims arrive only as 19 latent dims · gradient × real σ"}.get(
        tuple(models), "Baseline reads 474 raw dims; the encoder's 375 history dims "
                       "arrive as 19 latent dims · gradient × real σ")
    ax.text(0.0, 1.005, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    model_legend(ax, models)

# ---- panel: per-latent-dim sensitivity ----
def panel_latent(fig, ax, letter):
    order = np.argsort(-eff_lat)
    f_lab = [LAT_NAMES[i] for i in order]
    f_val = [float(eff_lat[i]) for i in order]
    f_isv = [i < 3 for i in order]
    ax.set_yticks(range(len(f_lab)))
    ax.set_yticklabels(f_lab, fontsize=7.5)
    ax.set_ylim(-0.55, len(f_lab) - 0.45)
    ax.invert_yaxis()
    style_barh(ax, max(f_val) * 1.18, "mean |∂action/∂latent| × σ(latent)")
    fig.canvas.draw()
    for i, (v, isv) in enumerate(zip(f_val, f_isv)):
        rounded_barh(ax, i, v, 0.55, ORANGE if isv else ORANGE_LT)
        ax.text(v + max(f_val) * 0.014, i, f"{v:.2f}", fontsize=7, color=MUTED, va="center")
    ax.set_title(f"{letter} · Latent dims — sensitivity × real spread", loc="left",
                 fontsize=11, color=INK, pad=10)
    ax.text(0.0, 1.005, "σ(latent) measured on the same on-policy states",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    ax.legend(handles=[Patch(fc=ORANGE, label="v̂ — supervised velocity head"),
                       Patch(fc=ORANGE_LT, label="z — free latent")],
              frameon=False, fontsize=8, labelcolor=INK2, loc="lower right",
              handlelength=1.1, handleheight=1.1)

# ---- shared header / footer ----
def header(fig, title, subtitle, notes, y_title, y_sub):
    fig.text(0.105, y_title, title, fontsize=15, color=INK, ha="left", weight="semibold")
    fig.text(0.105, y_sub, subtitle, fontsize=8.5, color=MUTED, ha="left")
    for i, n in enumerate(notes):
        fig.text(0.105, 0.046 - 0.016 * i, n, fontsize=7.5, color=MUTED, ha="left")

SUB = (f"hist_ablation/rew1 · model_61000 · mean |∂action/∂input| over 1024 {STATE_SRC}, "
       "in scaled-obs space · evaluated on the states the policy actually visits, "
       "over a 10-command grid")
NOTE_SCALE = ("Notes: dof_vel observations are pre-scaled ×0.05 (per-physical-unit sensitivity is "
              "1/20 of shown) · sensitivity ≠ closed-loop causal contribution · both policies "
              "under-track yaw here (0.6 → 0.21 Baseline, 0.32 History Encoder).")
NOTE_E2E = ("Group/recency/top-N panels are end-to-end: for the History Encoder, history is "
            "credited through the encoder, so the latent is an interior node with no bar of its own.")
NOTE_EFF = ("The actor-input panel stops at the first layer and scales every gradient by that "
            "input's measured σ on the same states, so inputs with different units are comparable.")

# ================= 1. comparison =================
fig = plt.figure(figsize=(15.2, 18.8), dpi=180)
gs = fig.add_gridspec(3, 2, left=0.105, right=0.975, top=0.925, bottom=0.082,
                      hspace=0.26, wspace=0.42, height_ratios=[1.12, 1.55, 1.28])
BOTH = ["baseline", "v3"]
panel_group_share(fig, fig.add_subplot(gs[0, 0]), BOTH, "A")
panel_recency(fig, fig.add_subplot(gs[0, 1]), BOTH, "B")
panel_top(fig, fig.add_subplot(gs[1, 0]), "baseline", "C")
panel_top(fig, fig.add_subplot(gs[1, 1]), "v3", "D")
panel_actor_input(fig, fig.add_subplot(gs[2, 0]), BOTH, "E")
panel_latent(fig, fig.add_subplot(gs[2, 1]), "F")
header(fig, "Actor input sensitivity — Baseline vs History Encoder", SUB,
       [NOTE_SCALE, NOTE_E2E, NOTE_EFF], 0.972, 0.952)
fig.savefig(OUT_CMP)
plt.close(fig)
print("saved:", OUT_CMP)

# ================= 2. baseline only =================
fig = plt.figure(figsize=(15.2, 13.4), dpi=180)
gs = fig.add_gridspec(2, 2, left=0.105, right=0.975, top=0.900, bottom=0.105,
                      hspace=0.26, wspace=0.42, height_ratios=[1.0, 1.42])
panel_group_share(fig, fig.add_subplot(gs[0, 0]), ["baseline"], "A")
panel_recency(fig, fig.add_subplot(gs[0, 1]), ["baseline"], "B")
panel_top(fig, fig.add_subplot(gs[1, 0]), "baseline", "C")
panel_actor_input(fig, fig.add_subplot(gs[1, 1]), ["baseline"], "D")
header(fig, "Actor input sensitivity — Baseline", SUB, [NOTE_SCALE, NOTE_EFF], 0.962, 0.935)
fig.savefig(OUT_BASE)
plt.close(fig)
print("saved:", OUT_BASE)

# ================= 3. history encoder only =================
fig = plt.figure(figsize=(15.2, 17.2), dpi=180)
gs = fig.add_gridspec(3, 2, left=0.105, right=0.975, top=0.918, bottom=0.085,
                      hspace=0.26, wspace=0.42, height_ratios=[1.0, 1.42, 0.78])
panel_group_share(fig, fig.add_subplot(gs[0, 0]), ["v3"], "A")
panel_recency(fig, fig.add_subplot(gs[0, 1]), ["v3"], "B")
panel_top(fig, fig.add_subplot(gs[1, 0]), "v3", "C")
panel_latent(fig, fig.add_subplot(gs[1, 1]), "D")
panel_actor_input(fig, fig.add_subplot(gs[2, :]), ["v3"], "E")
header(fig, "Actor input sensitivity — History Encoder", SUB,
       [NOTE_SCALE, NOTE_E2E, NOTE_EFF], 0.970, 0.948)
fig.savefig(OUT_HIST)
plt.close(fig)
print("saved:", OUT_HIST)
