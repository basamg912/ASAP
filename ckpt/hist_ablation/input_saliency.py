"""Input-dimension saliency analysis for ASAP baseline actor (model_61000.pt).

Methods:
  A) First-layer weight column L2 norm  (layer-1 only, no data needed)
  B) Mean |Jacobian| of action mean w.r.t. input, sampled around nominal standing
     state (and an N(0,1) robustness check) -- full-network sensitivity.
"""
import torch
import torch.nn as nn
import numpy as np

CKPT = "/home/kist/work/workspace/ACM/ASAP/ckpt/hist_ablation/rew1/baseline/model_61000.pt"

DOF = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow",
]

# component -> (dim, per-dim labels)
COMP = {
    "actions":           (23, DOF),
    "base_ang_vel":      (3,  ["x(roll)", "y(pitch)", "z(yaw)"]),
    "command_ang_vel":   (1,  ["yaw"]),
    "command_lin_vel":   (2,  ["vx", "vy"]),
    "command_stand":     (1,  ["stand"]),
    "dof_pos":           (23, DOF),
    "dof_vel":           (23, DOF),
    "projected_gravity": (3,  ["x", "y", "z"]),
}
SORTED_KEYS = sorted(COMP.keys())  # concat order in legged_robot_base.py (sorted)
HIST_LEN = 5  # short_history: 5 steps each, index 0 = t-1 (most recent)

# ---- build dim -> name/group map for the 474-dim actor input ----
names, groups = [], []
for k in sorted(list(COMP.keys()) + ["short_history"]):
    if k == "short_history":
        for hk in SORTED_KEYS:
            d, labels = COMP[hk]
            for t in range(HIST_LEN):
                for i in range(d):
                    names.append(f"hist[t-{t+1}] {hk}:{labels[i]}")
                    groups.append(f"hist_{hk}")
    else:
        d, labels = COMP[k]
        for i in range(d):
            names.append(f"{k}:{labels[i]}")
            groups.append(k)
N = len(names)
assert N == 474, N

# nominal standing state: all zeros except projected_gravity z = -1 (current + history)
x0 = torch.zeros(N)
for i, nm in enumerate(names):
    if "projected_gravity:z" in nm:
        x0[i] = -1.0

# plausible per-dim noise std in the model's (scaled) input space
STD = {
    "actions": 1.0, "base_ang_vel": 0.25,
    "command_ang_vel": 0.5, "command_lin_vel": 0.5, "command_stand": 0.5,
    "dof_pos": 0.5, "dof_vel": 0.25, "projected_gravity": 0.2,
}
std_vec = torch.tensor([STD[g.replace("hist_", "")] for g in groups])

# ---- load actor ----
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt["actor_model_state_dict"]
net = nn.Sequential(
    nn.Linear(474, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 23),
)
net.load_state_dict({k.replace("actor_module.module.", ""): v
                     for k, v in sd.items() if k.startswith("actor_module")})
net.eval()
torch.manual_seed(0)

# ---- Method A: first-layer column norms ----
W0 = sd["actor_module.module.0.weight"]          # (512, 474)
colnorm = W0.norm(dim=0).numpy()                  # (474,)

# ---- Method B: mean |Jacobian| over samples ----
def jacobian_saliency(xs):
    """xs: (B, N). Returns (474,) = mean_b sum_j |d mu_j / d x_i|."""
    xs = xs.clone().requires_grad_(True)
    mu = net(xs)                                  # (B, 23)
    acc = torch.zeros(xs.shape[1])
    for j in range(mu.shape[1]):
        g, = torch.autograd.grad(mu[:, j].sum(), xs, retain_graph=(j < mu.shape[1] - 1))
        acc += g.abs().mean(dim=0)
    return acc.detach().numpy()

B = 1024
xs_nom = x0 + torch.randn(B, N) * std_vec         # around nominal standing state
sal_nom = jacobian_saliency(xs_nom)
sal_iso = jacobian_saliency(torch.randn(B, N))    # N(0,1) robustness check
sal_x0  = jacobian_saliency(x0.unsqueeze(0))      # exactly at nominal point

# ---- report ----
def agg(vals, keyfn):
    out = {}
    for i, v in enumerate(vals):
        out.setdefault(keyfn(i), []).append(v)
    return {k: (float(np.sum(v)), float(np.mean(v)), len(v)) for k, v in out.items()}

def show(title, vals):
    print(f"\n=== {title} ===")
    a = agg(vals, lambda i: groups[i])
    tot = sum(s for s, _, _ in a.values())
    print(f"{'group':22s} {'dims':>4s} {'sum':>8s} {'share%':>7s} {'mean/dim':>9s}")
    for k, (s, m, n) in sorted(a.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:22s} {n:4d} {s:8.3f} {100*s/tot:6.1f}% {m:9.4f}")

show("Method A: first-layer |W| column L2 norm", colnorm)
show("Method B: mean |Jacobian| (nominal-state sampling)", sal_nom)
show("Method B': mean |Jacobian| (N(0,1) check)", sal_iso)

# current-vs-history and per-timestep breakdown (Method B)
cur = sum(sal_nom[i] for i in range(N) if not groups[i].startswith("hist_"))
hist = sum(sal_nom[i] for i in range(N) if groups[i].startswith("hist_"))
print(f"\ncurrent-step total: {cur:.3f} ({100*cur/(cur+hist):.1f}%)   "
      f"history total: {hist:.3f} ({100*hist/(cur+hist):.1f}%)")
tstep = agg(sal_nom, lambda i: names[i].split("]")[0] + "]" if names[i].startswith("hist") else "current")
print(f"{'time':12s} {'sum':>8s} {'mean/dim':>9s}")
for k, (s, m, n) in sorted(tstep.items()):
    print(f"{k:12s} {s:8.3f} {m:9.4f}")

print("\n=== Top 30 individual input dims (Method B, nominal sampling) ===")
order = np.argsort(-sal_nom)
print(f"{'rank':4s} {'dim':>4s}  {'name':40s} {'B:jac_nom':>9s} {'B:jac@x0':>9s} {'A:|W|col':>9s}")
for r, i in enumerate(order[:30]):
    print(f"{r+1:4d} {i:4d}  {names[i]:40s} {sal_nom[i]:9.4f} {sal_x0[i]:9.4f} {colnorm[i]:9.4f}")

# rank correlation between methods
from scipy.stats import spearmanr
print(f"\nSpearman rank corr  A vs B(nom): {spearmanr(colnorm, sal_nom).statistic:.3f}   "
      f"B(nom) vs B(N01): {spearmanr(sal_nom, sal_iso).statistic:.3f}")
