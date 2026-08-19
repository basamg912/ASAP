"""Input-dimension saliency for ASAP hist_ablation rew1/v3 (model_61000.pt).

Architecture (PPOHistV3, deploy path = act_inference):
    v(3), z(16) = StudentEncoder(history 375)          # MLP-mixer
    action(23)  = actor MLP( cat[actor_obs 79, v, z] ) # 98 -> 512-256-128 -> 23
Attribution is computed through the FULL composed function (grad flows
through the student encoder), so history dims get end-to-end credit.
"""
import numpy as np
import torch
import torch.nn as nn

DIR = "/home/kist/work/workspace/ACM/ASAP/ckpt/hist_ablation/rew1/v3"
CKPT = f"{DIR}/model_61000.pt"
ONNX = f"{DIR}/exported/model_61000.onnx"
NPZ = f"{DIR}/latents/latents_ckpt_61000.npz"

DOF = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow",
]
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
CUR_KEYS = sorted(COMP.keys())                      # actor_obs concat order (79)
HIST_KEYS = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
KEY_DIMS = [COMP[k][0] for k in HIST_KEYS]          # [23,3,23,23,3] per-step 75
T = 5                                               # index 0 = t-1 (most recent)

names, groups = [], []
for k in CUR_KEYS:
    d, labels = COMP[k]
    names += [f"{k}:{l}" for l in labels]; groups += [k] * d
N_CUR = len(names)                                   # 79
for hk in HIST_KEYS:                                 # env flat layout: comp-major, t-major inside
    d, labels = COMP[hk]
    for t in range(T):
        names += [f"hist[t-{t+1}] {hk}:{l}" for l in labels]
        groups += [f"hist_{hk}"] * d
N = len(names)
assert N_CUR == 79 and N == 454, (N_CUR, N)

x0 = torch.zeros(N)
for i, nm in enumerate(names):
    if "projected_gravity:z" in nm:
        x0[i] = -1.0
STD = {"actions": 1.0, "base_ang_vel": 0.25, "command_ang_vel": 0.5,
       "command_lin_vel": 0.5, "command_stand": 0.5, "dof_pos": 0.5,
       "dof_vel": 0.25, "projected_gravity": 0.2}
std_vec = torch.tensor([STD[g.replace("hist_", "")] for g in groups])

# ---- exact reimplementation of MLP_mixer / StudentEncoder (ppo_hist_modules.py) ----
class MLPMixer(nn.Module):
    def __init__(self, F=75, C=5, out=19, hidden=(256, 128), ch_hidden=(64,)):
        super().__init__()
        self.ln, self.ln2 = nn.LayerNorm(F), nn.LayerNorm(F)
        fl = [nn.Linear(F, hidden[0]), nn.ELU()]
        for l in range(len(hidden)):
            fl += [nn.Linear(hidden[l], F)] if l == len(hidden) - 1 else [nn.Linear(hidden[l], hidden[l + 1]), nn.ELU()]
        self.feature_mixing = nn.Sequential(*fl)
        cl = [nn.Linear(C, ch_hidden[0]), nn.ELU()]
        for l in range(len(ch_hidden)):
            cl += [nn.Linear(ch_hidden[l], C)] if l == len(ch_hidden) - 1 else [nn.Linear(ch_hidden[l], ch_hidden[l + 1]), nn.ELU()]
        self.channel_mixing = nn.Sequential(*cl)
        self.output_layer = nn.Linear(C * F, out)

    def forward(self, x):                            # x: (B, T, F)
        y = self.feature_mixing(self.ln(x))
        x = x + y
        y = self.channel_mixing(self.ln2(x).transpose(1, 2))
        x = x + y.transpose(1, 2)
        return self.output_layer(x.flatten(1))

def unflatten_history(flat, key_dims, T):
    Nb = flat.shape[0]
    chunks = torch.split(flat, [d * T for d in key_dims], dim=1)
    return torch.cat([c.view(Nb, T, d) for c, d in zip(chunks, key_dims)], dim=-1)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLPMixer()
    def forward(self, hist):
        out = self.net(unflatten_history(hist, KEY_DIMS, T))
        return out[..., :3], out[..., 3:]            # v(3), z(16)

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["actor_model_state_dict"]
student = Student()
student.load_state_dict({k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")}, strict=True)
actor = nn.Sequential(nn.Linear(98, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                      nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 23))
actor.load_state_dict({k.replace("actor_module.module.", ""): v for k, v in sd.items() if k.startswith("actor_module")}, strict=True)
student.eval(); actor.eval()

def full_forward(x):                                  # x: (B, 454) -> action mean (B, 23)
    v, z = student(x[:, N_CUR:])
    return actor(torch.cat([x[:, :N_CUR], v, z], dim=-1))

# ---- validate reimplementation against ONNX export ----
import onnxruntime as ort
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
ins = {i.name: i.shape for i in sess.get_inputs()}
torch.manual_seed(0)
xt = torch.randn(8, N)
def onnx_run(xrow):                                   # onnx export is batch-1
    feeds = {}
    for i in sess.get_inputs():
        dim = i.shape[-1]
        if dim == N_CUR: feeds[i.name] = xrow[None, :N_CUR].numpy()
        elif dim == N - N_CUR: feeds[i.name] = xrow[None, N_CUR:].numpy()
        elif dim == N: feeds[i.name] = xrow[None, :].numpy()
        else: raise RuntimeError(f"unexpected onnx input {i.name} {i.shape}")
    return sess.run(None, feeds)[0]
onnx_out = np.concatenate([onnx_run(xt[b]) for b in range(xt.shape[0])], axis=0)
with torch.no_grad():
    mine = full_forward(xt).numpy()
err = np.abs(onnx_out - mine).max()
print(f"ONNX inputs: {ins} | reimpl vs onnx max|diff| = {err:.2e}")
assert err < 1e-4, "reimplementation mismatch!"

# ---- Method B: mean |Jacobian| end-to-end ----
def jacobian_saliency(xs):
    xs = xs.clone().requires_grad_(True)
    mu = full_forward(xs)
    acc = torch.zeros(xs.shape[1])
    for j in range(mu.shape[1]):
        g, = torch.autograd.grad(mu[:, j].sum(), xs, retain_graph=(j < mu.shape[1] - 1))
        acc += g.abs().mean(dim=0)
    return acc.detach().numpy()

B = 1024
torch.manual_seed(0)
xs_nom = x0 + torch.randn(B, N) * std_vec
sal_nom = jacobian_saliency(xs_nom)
sal_iso = jacobian_saliency(torch.randn(B, N))
sal_x0 = jacobian_saliency(x0.unsqueeze(0))

# ---- Method A: actor first-layer col norms (98 actor inputs: 79 obs + v3 + z16) ----
W0 = sd["actor_module.module.0.weight"]              # (512, 98)
colnorm98 = W0.norm(dim=0).numpy()

# ---- latent bottleneck: d action / d (v,z) at sampled states, x real z stats ----
with torch.no_grad():
    v_s, z_s = student(xs_nom[:, N_CUR:])
lat = torch.cat([v_s, z_s], dim=-1).clone().requires_grad_(True)
mu = actor(torch.cat([xs_nom[:, :N_CUR], lat], dim=-1))
acc_lat = torch.zeros(19)
for j in range(23):
    g, = torch.autograd.grad(mu[:, j].sum(), lat, retain_graph=(j < 22))
    acc_lat += g.abs().mean(dim=0)
acc_lat = acc_lat.detach().numpy()

d = np.load(NPZ)
done = d["done"].astype(bool)
zr = d["latent"].reshape(-1, 16)[~done.reshape(-1)]
vr = d["vel_pred"].reshape(-1, 3)[~done.reshape(-1)]
real_std = np.concatenate([vr.std(0), zr.std(0)])
syn_std = torch.cat([v_s, z_s], dim=-1).std(0).numpy()

# ---- report ----
def agg(vals, keyfn):
    out = {}
    for i, v in enumerate(vals): out.setdefault(keyfn(i), []).append(v)
    return {k: (float(np.sum(v)), float(np.mean(v)), len(v)) for k, v in out.items()}

def show(title, vals):
    print(f"\n=== {title} ===")
    a = agg(vals, lambda i: groups[i])
    tot = sum(s for s, _, _ in a.values())
    print(f"{'group':22s} {'dims':>4s} {'sum':>8s} {'share%':>7s} {'mean/dim':>9s}")
    for k, (s, m, n) in sorted(a.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:22s} {n:4d} {s:8.3f} {100*s/tot:6.1f}% {m:9.4f}")

show("Method B: end-to-end mean |Jacobian| (nominal sampling)", sal_nom)
show("Method B': end-to-end mean |Jacobian| (N(0,1) check)", sal_iso)

cur = sal_nom[:N_CUR].sum(); hist = sal_nom[N_CUR:].sum()
print(f"\ncurrent-step total: {cur:.3f} ({100*cur/(cur+hist):.1f}%)   "
      f"history(through encoder): {hist:.3f} ({100*hist/(cur+hist):.1f}%)")
tstep = agg(sal_nom, lambda i: names[i].split("]")[0] + "]" if names[i].startswith("hist") else "current")
print(f"{'time':12s} {'sum':>8s} {'mean/dim':>9s}")
for k, (s, m, n) in sorted(tstep.items()):
    print(f"{k:12s} {s:8.3f} {m:9.4f}")

print("\n=== Top 30 individual input dims (end-to-end, nominal sampling) ===")
order = np.argsort(-sal_nom)
print(f"{'rank':4s} {'dim':>4s}  {'name':40s} {'jac_nom':>8s} {'jac@x0':>8s}")
for r, i in enumerate(order[:30]):
    print(f"{r+1:4d} {i:4d}  {names[i]:40s} {sal_nom[i]:8.4f} {sal_x0[i]:8.4f}")

print("\n=== Actor-input view (98 dims): first-layer |W| col norm & d(action)/d(v,z) ===")
lat_names = [f"v:{c}" for c in "xyz"] + [f"z{i}" for i in range(16)]
obs_sum = colnorm98[:N_CUR].sum(); lat_sum = colnorm98[N_CUR:].sum()
print(f"|W|col sum  obs79: {obs_sum:.2f} ({100*obs_sum/(obs_sum+lat_sum):.1f}%)  "
      f"latent19: {lat_sum:.2f} ({100*lat_sum/(obs_sum+lat_sum):.1f}%)")
print(f"{'latent dim':10s} {'|W|col':>7s} {'|dA/dl|':>8s} {'realstd':>8s} {'grad*std':>9s} {'syn_std':>8s}")
for i in range(19):
    print(f"{lat_names[i]:10s} {colnorm98[N_CUR+i]:7.3f} {acc_lat[i]:8.3f} "
          f"{real_std[i]:8.3f} {acc_lat[i]*real_std[i]:9.3f} {syn_std[i]:8.3f}")

from scipy.stats import spearmanr
print(f"\nSpearman B(nom) vs B(N01): {spearmanr(sal_nom, sal_iso).statistic:.3f}")
