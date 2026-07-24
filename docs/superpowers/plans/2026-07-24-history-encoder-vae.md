# History-Encoder (Concurrent VAE Context Encoder) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an RMA-style history encoder (VAE context encoder) that is trained end-to-end with PPO for KAPEX locomotion, as a fully additive parallel pipeline.

**Architecture:** A `HistoryEncoder` MLP maps a 5-step proprio+action history to a latent distribution `(μ, logvar)`; the sampled latent `z` is concatenated to the base-policy input. A `StatePredictor` decodes `z` to the next-step proprioceptive observation. The next-obs reconstruction MSE and a KL(q‖N(0,I)) term are added to the PPO actor loss (`total = ppo_loss + recon + β·KL`), with `z` NOT detached (joint gradient). Everything ships as new files (new agent subclass, new modules file, new algo/obs/exp configs); no existing training code/config/env is edited.

**Tech Stack:** PyTorch, Hydra/OmegaConf, IsaacGym (conda env `hvgym`), the humanoidverse PPO framework.

## Global Constraints

- **Additive only.** Do NOT edit any existing file. New files only. (spec §2.1)
- Existing untouched: `humanoidverse/agents/ppo/ppo.py`, `humanoidverse/agents/modules/ppo_modules.py`, `humanoidverse/agents/modules/modules.py`, `config/algo/ppo.yaml`, existing obs configs, `humanoidverse/envs/locomotion/locomotion.py`, `humanoidverse/envs/legged_base_task/legged_robot_base.py`.
- Branch: `master` (no feature branch).
- Reconstruction target = next-step proprio `[base_ang_vel, projected_gravity, dof_pos, dof_vel]` only.
- Gradient flow = joint (latent not detached).
- Encoder = Flatten → MLP.
- Defaults: `latent_dim=16`, encoder `[256,128]`, decoder `[128,256]`, `vae_beta=1.0`, `recon_coef=1.0`. Concurrent stability knobs (already tuned elsewhere): `entropy_coef=0.001`, `init_noise_std=0.5`.
- Python interpreter for tests/training: `/home/kist/anaconda3/envs/hvgym/bin/python`.
- **pytest invocation:** always prefix `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (a ROS2 `launch_testing` pytest plugin in this env needs `lark` and otherwise breaks collection). E.g. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -v`.
- Create ONLY the files named in each task — no extra directories or files.
- TDD, frequent commits, DRY, YAGNI.

---

## File Structure

- Create: `humanoidverse/agents/modules/ppo_hist_modules.py` — `MLP`, `HistoryEncoder`, `StatePredictor`, `PPOActorWithHistoryEncoder`, and the pure-function VAE losses `vae_kl_loss`, `recon_loss_masked`.
- Create: `humanoidverse/agents/ppo_hist/__init__.py`
- Create: `humanoidverse/agents/ppo_hist/ppo_hist.py` — `PPOHistoryEncoder(PPO)`.
- Create: `humanoidverse/config/algo/ppo_history_encoder.yaml`
- Create: `humanoidverse/config/obs/loco/leggedloco_obs_history_encoder.yaml`
- Create: `humanoidverse/config/exp/locomotion_history_encoder.yaml`
- Create: `tests/agents/test_ppo_hist_modules.py` — pure-torch unit tests.

Tasks 1–4 are pure-torch and unit-tested. Task 5 is config. Task 6 wires the agent (verified by a short sim smoke-run, since it needs GPU/IsaacGym). Task 7 adds the deploy/export path.

---

### Task 1: `MLP` + `HistoryEncoder`

**Files:**
- Create: `humanoidverse/agents/modules/ppo_hist_modules.py`
- Test: `tests/agents/test_ppo_hist_modules.py`

**Interfaces:**
- Produces:
  - `MLP(input_dim:int, output_dim:int, hidden_dims:list[int], activation:str="ELU").forward(x)->Tensor`
  - `HistoryEncoder(history_dim:int, latent_dim:int, hidden_dims=(256,128), activation="ELU")`
    - `.forward(history:Tensor)->(mu:Tensor[...,L], logvar:Tensor[...,L])`
    - `.sample(history:Tensor)->(z:Tensor[...,L], mu, logvar)` — `z=mu+std*ε` in train mode, `z=mu` in eval mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ppo_hist_modules.py
import torch
from humanoidverse.agents.modules.ppo_hist_modules import MLP, HistoryEncoder


def test_mlp_shape():
    net = MLP(10, 4, [16, 8])
    out = net(torch.zeros(5, 10))
    assert out.shape == (5, 4)


def test_history_encoder_shapes():
    enc = HistoryEncoder(history_dim=40, latent_dim=16)
    mu, logvar = enc(torch.zeros(7, 40))
    assert mu.shape == (7, 16)
    assert logvar.shape == (7, 16)


def test_history_encoder_sample_train_is_stochastic():
    torch.manual_seed(0)
    enc = HistoryEncoder(history_dim=40, latent_dim=16)
    enc.train()
    h = torch.randn(3, 40)
    z1, _, _ = enc.sample(h)
    z2, _, _ = enc.sample(h)
    assert not torch.allclose(z1, z2)


def test_history_encoder_sample_eval_is_mean():
    enc = HistoryEncoder(history_dim=40, latent_dim=16)
    enc.eval()
    h = torch.randn(3, 40)
    z, mu, _ = enc.sample(h)
    assert torch.allclose(z, mu)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (module not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# humanoidverse/agents/modules/ppo_hist_modules.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .ppo_modules import PPOActor


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="ELU"):
        super().__init__()
        act = getattr(nn, activation)()
        layers = [nn.Linear(input_dim, hidden_dims[0]), act]
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers += [nn.Linear(hidden_dims[l], hidden_dims[l + 1]), act]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HistoryEncoder(nn.Module):
    def __init__(self, history_dim, latent_dim, hidden_dims=(256, 128), activation="ELU"):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MLP(history_dim, 2 * latent_dim, list(hidden_dims), activation)

    def forward(self, history):
        out = self.net(history)
        mu = out[..., : self.latent_dim]
        logvar = out[..., self.latent_dim :]
        return mu, logvar

    def sample(self, history):
        mu, logvar = self.forward(history)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add humanoidverse/agents/modules/ppo_hist_modules.py tests/agents/test_ppo_hist_modules.py
git commit -m "feat(ppo_hist): add MLP and HistoryEncoder modules"
```

---

### Task 2: `StatePredictor` (decoder)

**Files:**
- Modify: `humanoidverse/agents/modules/ppo_hist_modules.py` (append class)
- Test: `tests/agents/test_ppo_hist_modules.py` (append test)

**Interfaces:**
- Produces: `StatePredictor(latent_dim:int, recon_dim:int, hidden_dims=(128,256), activation="ELU").forward(z:Tensor[...,L])->Tensor[...,recon_dim]`

- [ ] **Step 1: Write the failing test**

```python
def test_state_predictor_shape():
    from humanoidverse.agents.modules.ppo_hist_modules import StatePredictor
    dec = StatePredictor(latent_dim=16, recon_dim=68)
    out = dec(torch.zeros(9, 16))
    assert out.shape == (9, 68)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py::test_state_predictor_shape -v`
Expected: FAIL — `ImportError: cannot import name 'StatePredictor'`.

- [ ] **Step 3: Write minimal implementation** (append to `ppo_hist_modules.py`)

```python
class StatePredictor(nn.Module):
    def __init__(self, latent_dim, recon_dim, hidden_dims=(128, 256), activation="ELU"):
        super().__init__()
        self.net = MLP(latent_dim, recon_dim, list(hidden_dims), activation)

    def forward(self, z):
        return self.net(z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py::test_state_predictor_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add humanoidverse/agents/modules/ppo_hist_modules.py tests/agents/test_ppo_hist_modules.py
git commit -m "feat(ppo_hist): add StatePredictor decoder"
```

---

### Task 3: VAE loss functions (`vae_kl_loss`, `recon_loss_masked`)

**Files:**
- Modify: `humanoidverse/agents/modules/ppo_hist_modules.py` (append functions)
- Test: `tests/agents/test_ppo_hist_modules.py` (append tests)

**Interfaces:**
- Produces:
  - `vae_kl_loss(mu:Tensor[B,L], logvar:Tensor[B,L])->scalar Tensor` — mean over batch of `-0.5·Σ(1+logvar−mu²−e^logvar)`.
  - `recon_loss_masked(pred:Tensor[B,D], target:Tensor[B,D], valid_mask:Tensor[B] or [B,1])->scalar Tensor` — mean-squared error per sample, averaged over valid (mask=1) samples only.

- [ ] **Step 1: Write the failing test**

```python
def test_vae_kl_zero_at_standard_normal():
    from humanoidverse.agents.modules.ppo_hist_modules import vae_kl_loss
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)  # var = 1
    assert torch.allclose(vae_kl_loss(mu, logvar), torch.tensor(0.0), atol=1e-6)


def test_vae_kl_positive_when_offset():
    from humanoidverse.agents.modules.ppo_hist_modules import vae_kl_loss
    mu = torch.ones(4, 16)
    logvar = torch.zeros(4, 16)
    assert vae_kl_loss(mu, logvar).item() > 0.0


def test_recon_mask_all_zero_gives_zero_over_valid():
    from humanoidverse.agents.modules.ppo_hist_modules import recon_loss_masked
    pred = torch.ones(3, 5)
    target = torch.zeros(3, 5)
    mask = torch.zeros(3)          # no valid samples
    assert torch.allclose(recon_loss_masked(pred, target, mask), torch.tensor(0.0))


def test_recon_mask_selects_valid_samples():
    from humanoidverse.agents.modules.ppo_hist_modules import recon_loss_masked
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.zeros(2, 2)
    mask = torch.tensor([1.0, 0.0])   # only first sample counts, its MSE = 1.0
    assert torch.allclose(recon_loss_masked(pred, target, mask), torch.tensor(1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -k "kl or recon" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `ppo_hist_modules.py`)

```python
def vae_kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def recon_loss_masked(pred, target, valid_mask):
    per_sample = ((pred - target) ** 2).mean(dim=-1)   # (B,)
    w = valid_mask.reshape(-1).to(per_sample.dtype)
    denom = torch.clamp(w.sum(), min=1.0)
    return (per_sample * w).sum() / denom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -k "kl or recon" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add humanoidverse/agents/modules/ppo_hist_modules.py tests/agents/test_ppo_hist_modules.py
git commit -m "feat(ppo_hist): add VAE kl and masked reconstruction losses"
```

---

### Task 4: `PPOActorWithHistoryEncoder`

**Files:**
- Modify: `humanoidverse/agents/modules/ppo_hist_modules.py` (append class)
- Test: `tests/agents/test_ppo_hist_modules.py` (append tests)

**Interfaces:**
- Consumes: `PPOActor` (base), `HistoryEncoder`, `StatePredictor`, `BaseModule` (via `PPOActor.__init__`).
- Produces: `PPOActorWithHistoryEncoder(obs_dim_dict, module_config_dict, num_actions, init_noise_std, encoder_config, decoder_config, latent_dim)`
  - `.act(actor_obs, encoder_obs)->actions Tensor[B,num_actions]` (sets `.distribution`, stashes `_last_latent`)
  - `.act_inference(actor_obs, encoder_obs)->action_mean Tensor[B,num_actions]`
  - `.update_distribution(actor_obs, encoder_obs)->None`
  - `.predict_next_state()->Tensor[B,recon_dim]` (uses stashed `z`)
  - `.get_latent_stats()->dict(mu, logvar, z)`
  - `encoder_config = {'input_key':str, 'hidden_dims':list, 'activation':str}`
  - `decoder_config = {'target_key':str, 'hidden_dims':list, 'activation':str}`
  - `module_config_dict.actor.input_dim` MUST be `[<actor_obs_key>, <latent_dim int>]` so `BaseModule` sizes the trunk as `actor_obs_dim + latent_dim`.

- [ ] **Step 1: Write the failing test**

```python
def _make_actor():
    from omegaconf import OmegaConf
    from humanoidverse.agents.modules.ppo_hist_modules import PPOActorWithHistoryEncoder
    obs_dim_dict = {"actor_obs": 30, "encoder_obs": 40, "recon_target": 12}
    latent_dim = 16
    module_config_dict = OmegaConf.create({
        "input_dim": ["actor_obs", latent_dim],
        "output_dim": ["robot_action_dim"],
        "layer_config": {"type": "MLP", "hidden_dims": [64, 32], "activation": "ELU"},
    })
    enc_cfg = {"input_key": "encoder_obs", "hidden_dims": [64, 32], "activation": "ELU"}
    dec_cfg = {"target_key": "recon_target", "hidden_dims": [32, 64], "activation": "ELU"}
    return PPOActorWithHistoryEncoder(obs_dim_dict, module_config_dict, num_actions=8,
                                      init_noise_std=0.5, encoder_config=enc_cfg,
                                      decoder_config=dec_cfg, latent_dim=latent_dim), latent_dim


def test_actor_act_shapes_and_latent_stash():
    torch.manual_seed(0)
    actor, L = _make_actor()
    actor.train()
    a = actor.act(torch.randn(4, 30), torch.randn(4, 40))
    assert a.shape == (4, 8)
    stats = actor.get_latent_stats()
    assert stats["mu"].shape == (4, L)
    assert stats["logvar"].shape == (4, L)
    assert actor.predict_next_state().shape == (4, 12)


def test_actor_inference_is_deterministic():
    actor, _ = _make_actor()
    actor.eval()
    ao, eo = torch.randn(4, 30), torch.randn(4, 40)
    m1 = actor.act_inference(ao, eo)
    m2 = actor.act_inference(ao, eo)
    assert m1.shape == (4, 8)
    assert torch.allclose(m1, m2)


def test_actor_gradient_flows_into_encoder():
    actor, _ = _make_actor()
    actor.train()
    actor.act(torch.randn(4, 30), torch.randn(4, 40))
    loss = actor.action_mean.pow(2).mean()      # policy-side signal only
    loss.backward()
    enc_grad = next(actor.history_encoder.parameters()).grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0   # joint gradient path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -k actor -v`
Expected: FAIL — `ImportError: cannot import name 'PPOActorWithHistoryEncoder'`.

- [ ] **Step 3: Write minimal implementation** (append to `ppo_hist_modules.py`)

```python
class PPOActorWithHistoryEncoder(PPOActor):
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std,
                 encoder_config, decoder_config, latent_dim):
        # base builds the actor trunk from module_config_dict; input_dim must already
        # include the numeric latent_dim, e.g. ["actor_obs", 16]
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std)
        self.latent_dim = latent_dim
        history_dim = obs_dim_dict[encoder_config["input_key"]]
        recon_dim = obs_dim_dict[decoder_config["target_key"]]
        self.history_encoder = HistoryEncoder(
            history_dim, latent_dim,
            tuple(encoder_config["hidden_dims"]), encoder_config["activation"])
        self.state_predictor = StatePredictor(
            latent_dim, recon_dim,
            tuple(decoder_config["hidden_dims"]), decoder_config["activation"])
        self._last_latent = {}

    def _encode(self, encoder_obs):
        z, mu, logvar = self.history_encoder.sample(encoder_obs)
        self._last_latent = {"z": z, "mu": mu, "logvar": logvar}
        return z

    def update_distribution(self, actor_obs, encoder_obs):
        z = self._encode(encoder_obs)
        mean = self.actor(torch.cat([actor_obs, z], dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs, encoder_obs, **kwargs):
        self.update_distribution(actor_obs, encoder_obs)
        return self.distribution.sample()

    def act_inference(self, actor_obs, encoder_obs):
        z, _, _ = self.history_encoder.sample(encoder_obs)  # eval mode -> z = mu
        return self.actor(torch.cat([actor_obs, z], dim=-1))

    def predict_next_state(self):
        return self.state_predictor(self._last_latent["z"])

    def get_latent_stats(self):
        return self._last_latent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py -k actor -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add humanoidverse/agents/modules/ppo_hist_modules.py tests/agents/test_ppo_hist_modules.py
git commit -m "feat(ppo_hist): add PPOActorWithHistoryEncoder (joint-grad latent)"
```

---

### Task 5: New configs (obs, algo, exp)

**Files:**
- Create: `humanoidverse/config/obs/loco/leggedloco_obs_history_encoder.yaml`
- Create: `humanoidverse/config/algo/ppo_history_encoder.yaml`
- Create: `humanoidverse/config/exp/locomotion_history_encoder.yaml`

**Interfaces:**
- Produces obs groups consumed by Task 6: `actor_obs` (current step, no history), `critic_obs` (privileged, unchanged), `encoder_obs` (= `history` aux), `recon_target` (next-obs target fields). `obs_dims` must define every atomic key used.
- `${robot.dof_obs_size}` = per-robot dof count `N` (KAPEX: 31), resolved by Hydra.

- [ ] **Step 1: Create the obs config**

```yaml
# humanoidverse/config/obs/loco/leggedloco_obs_history_encoder.yaml
# @package _global_
# Additive copy of leggedloco_obs_history_wolinvel.yaml, restructured for the
# history-encoder pipeline: short_history removed from actor_obs; a dedicated
# encoder_obs (history) and recon_target group added.
obs:
  obs_dict:
    actor_obs: [
      base_ang_vel,
      projected_gravity,
      command_lin_vel,
      command_ang_vel,
      command_stand,
      dof_pos,
      dof_vel,
      actions
    ]
    critic_obs: [
      base_lin_vel,
      base_ang_vel,
      projected_gravity,
      command_lin_vel,
      command_ang_vel,
      command_stand,
      dof_pos,
      dof_vel,
      actions,
      short_history
    ]
    encoder_obs: [ history ]
    recon_target: [
      base_ang_vel,
      projected_gravity,
      dof_pos,
      dof_vel
    ]

  obs_auxiliary:
    history: {
      base_ang_vel: 5,
      projected_gravity: 5,
      dof_pos: 5,
      dof_vel: 5,
      actions: 5,
    }
    short_history: {
      base_ang_vel: 5,
      projected_gravity: 5,
      dof_pos: 5,
      dof_vel: 5,
      actions: 5,
      command_lin_vel: 5,
      command_ang_vel: 5,
      command_stand: 5,
    }

  obs_scales: {
    base_lin_vel: 2.0,
    base_ang_vel: 0.25,
    projected_gravity: 1.0,
    command_lin_vel: 1.0,
    command_ang_vel: 1.0,
    command_stand: 1.0,
    dof_pos: 1.0,
    dof_vel: 0.05,
    actions: 1.0,
    history: 1.0,
    short_history: 1.0,
  }
  add_noise_currculum: False
  noise_initial_value: 0.05
  noise_value_max: 1.00
  noise_value_min: 0.00001
  soft_dof_pos_curriculum_degree: 0.00001
  soft_dof_pos_curriculum_level_down_threshold: 100
  soft_dof_pos_curriculum_level_up_threshold: 900
  noise_scales: {
    base_lin_vel: 0.0,
    base_ang_vel: 0.0,
    projected_gravity: 0.0,
    command_lin_vel: 0.0,
    command_ang_vel: 0.0,
    command_stand: 0.0,
    dof_pos: 0.0,
    dof_vel: 0.0,
    actions: 0.0,
    history: 0.0,
    short_history: 0.0,
  }

  obs_dims:
    - base_lin_vel: 3
    - base_ang_vel: 3
    - projected_gravity: 3
    - command_lin_vel: 2
    - command_ang_vel: 1
    - command_stand: 1
    - dof_pos: ${robot.dof_obs_size}
    - dof_vel: ${robot.dof_obs_size}
    - actions: ${robot.dof_obs_size}
```

Notes: `recon_target` dim = `3 + 3 + N + N` (KAPEX N=31 → 68). `encoder_obs` (`history`) dim = `(3+3+N+N+N)*5` (KAPEX → `(6+3·31)·5 = 495`). These are auto-computed by `helpers.pre_process_config` into `algo_obs_dim_dict`.

- [ ] **Step 2: Create the algo config**

```yaml
# humanoidverse/config/algo/ppo_history_encoder.yaml
# @package _global_
# Additive copy of ppo.yaml pointing at the new PPOHistoryEncoder agent, with
# encoder/decoder module configs and VAE hyperparameters.
algo:
  _target_: humanoidverse.agents.ppo_hist.ppo_hist.PPOHistoryEncoder
  _recursive_: False
  config:
    num_learning_epochs: 5
    num_mini_batches: 4
    clip_param: 0.2
    gamma: 0.99
    lam: 0.95
    value_loss_coef: 1.0
    entropy_coef: 0.001
    actor_learning_rate: 1.e-3
    critic_learning_rate: 1.e-3
    max_grad_norm: 1.0
    use_clipped_value_loss: True
    schedule: "adaptive"
    desired_kl: 0.01
    num_steps_per_env: 24
    save_interval: 100
    load_optimizer: True
    init_noise_std: 0.5
    num_learning_iterations: 1000000
    init_at_random_ep_len: True
    eval_callbacks: null

    # ---- history-encoder additions ----
    latent_dim: 16
    vae_beta: 1.0
    recon_coef: 1.0
    encoder_obs_key: encoder_obs
    recon_target_key: recon_target
    encoder_config:
      input_key: encoder_obs
      hidden_dims: [256, 128]
      activation: ELU
    decoder_config:
      target_key: recon_target
      hidden_dims: [128, 256]
      activation: ELU

    module_dict:
      actor:
        input_dim: [actor_obs, 16]   # actor_obs_dim + latent_dim; keep in sync with latent_dim
        output_dim: [robot_action_dim]
        layer_config:
          type: MLP
          hidden_dims: [512, 256, 128]
          activation: ELU
      critic:
        type: MLP
        input_dim: [critic_obs]
        output_dim: [1]
        layer_config:
          type: MLP
          hidden_dims: [512, 256, 128]
          activation: ELU
```

- [ ] **Step 3: Create the exp config**

```yaml
# humanoidverse/config/exp/locomotion_history_encoder.yaml
# @package _global_
# Additive copy of exp/locomotion.yaml, only swapping the algo group.
# NOTE: obs is selected on the CLI (+obs=loco/leggedloco_obs_history_encoder),
# exactly as the existing run does — do NOT add a `- /obs:` default here.
defaults:
  - /algo: ppo_history_encoder
  - /env: locomotion

experiment_name: hist_encoder
log_task_name: locomotion_history_encoder
```

Note: `exp/locomotion.yaml` sets only `/algo` and `/env`; obs is passed at launch via `+obs=...`. Mirror that exactly — the new exp only swaps `/algo` to `ppo_history_encoder`.

- [ ] **Step 4: Verify config composition (dim wiring)**

Run:
```bash
/home/kist/anaconda3/envs/hvgym/bin/python -c "
import hydra
from hydra import compose, initialize_config_dir
from humanoidverse.utils.helpers import pre_process_config
import os
cfg_dir = os.path.abspath('humanoidverse/config')
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(config_name='base', overrides=[
        '+simulator=isaacsim',
        '+exp=locomotion_history_encoder',
        '+domain_rand=domain_rand_base',
        '+rewards=loco/reward_kapex_locomotion',
        '+robot=kapex/kapex_31dof',
        '+terrain=terrain_locomotion_plane',
        '+obs=loco/leggedloco_obs_history_encoder',
    ])
    pre_process_config(cfg)
    d = cfg.robot.algo_obs_dim_dict
    print('actor_obs', d['actor_obs'], 'encoder_obs', d['encoder_obs'], 'recon_target', d['recon_target'])
    assert d['recon_target'] == 3+3+cfg.robot.dof_obs_size*2
    assert d['encoder_obs'] == (3+3+cfg.robot.dof_obs_size*3)*5
    print('OK')
"
```
Expected: prints dims and `OK` (KAPEX dof=31 → recon_target 68, encoder_obs 510). These are the exact override groups the existing working run used (from its `.hydra/overrides.yaml`), with `+obs` swapped to the new file. If importing `humanoidverse.utils.helpers` fails in `hvgym` due to a sim import, report it — compose/`pre_process_config` should be sim-free (the simulator is only instantiated later via its `_target_`).

- [ ] **Step 5: Commit**

```bash
git add humanoidverse/config/obs/loco/leggedloco_obs_history_encoder.yaml \
        humanoidverse/config/algo/ppo_history_encoder.yaml \
        humanoidverse/config/exp/locomotion_history_encoder.yaml
git commit -m "feat(ppo_hist): add obs/algo/exp configs for history-encoder pipeline"
```

---

### Task 6: `PPOHistoryEncoder` agent (overrides + joint loss)

**Files:**
- Create: `humanoidverse/agents/ppo_hist/__init__.py` (empty)
- Create: `humanoidverse/agents/ppo_hist/ppo_hist.py`

**Interfaces:**
- Consumes: `PPO` (base), `PPOActorWithHistoryEncoder`, `PPOCritic`, `vae_kl_loss`, `recon_loss_masked`. Config keys from Task 5 (`latent_dim`, `vae_beta`, `recon_coef`, `encoder_config`, `decoder_config`, `encoder_obs_key`, `recon_target_key`).
- Produces: agent class `PPOHistoryEncoder(PPO)` resolvable by `_target_`.
- Storage key added: `next_obs_target` shape `(recon_dim,)`.

- [ ] **Step 1: Create `__init__.py`**

```python
# humanoidverse/agents/ppo_hist/__init__.py
```

- [ ] **Step 2: Write the agent (copied overrides from `PPO`, additive)**

```python
# humanoidverse/agents/ppo_hist/ppo_hist.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.ppo_modules import PPOCritic
from humanoidverse.agents.modules.ppo_hist_modules import (
    PPOActorWithHistoryEncoder, vae_kl_loss, recon_loss_masked,
)


class PPOHistoryEncoder(PPO):
    """PPO + concurrent VAE history encoder (RMA-style, end-to-end, joint grad)."""

    # ---- config passthrough ----
    def _init_config(self):
        super()._init_config()
        self.latent_dim = self.config.latent_dim
        self.vae_beta = self.config.vae_beta
        self.recon_coef = self.config.recon_coef
        self.encoder_obs_key = self.config.encoder_obs_key
        self.recon_target_key = self.config.recon_target_key

    # ---- add VAE keys so _training_step averages + _logging_to_writer emits them ----
    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['recon'] = 0
        loss_dict['vae_kl'] = 0
        loss_dict['latent_std'] = 0
        return loss_dict

    # ---- models: actor carries encoder+decoder ----
    def _setup_models_and_optimizer(self):
        self.actor = PPOActorWithHistoryEncoder(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            encoder_config=self.config.encoder_config,
            decoder_config=self.config.decoder_config,
            latent_dim=self.latent_dim,
        ).to(self.device)
        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    # ---- storage: add shifted next-obs reconstruction target ----
    def _setup_storage(self):
        super()._setup_storage()
        recon_dim = self.algo_obs_dim_dict[self.recon_target_key]
        self.storage.register_key('next_obs_target', shape=(recon_dim,), dtype=torch.float)

    # ---- act uses both actor_obs and encoder_obs ----
    def _actor_act_step(self, obs_dict):
        return self.actor.act(obs_dict["actor_obs"], obs_dict[self.encoder_obs_key])

    # ---- rollout: copy of base PPO._rollout_step + next_obs_target capture ----
    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])
                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # ---- ADDED: next-step reconstruction target for the step just stored ----
                self.storage.update_key('next_obs_target',
                                        obs_dict[self.recon_target_key])

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'),
                                       dones=self.storage.query_key('dones'),
                                       rewards=self.storage.query_key('rewards')))
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)
        return obs_dict

    # ---- update: copy of base PPO._update_ppo + recon + KL terms ----
    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)  # recomputes distribution + stashes latent
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)
                for pg in self.actor_optimizer.param_groups:
                    pg['lr'] = self.actor_learning_rate
                for pg in self.critic_optimizer.param_groups:
                    pg['lr'] = self.critic_learning_rate

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()

        # ---- ADDED: VAE reconstruction + KL (joint grad; latent not detached) ----
        latent = self.actor.get_latent_stats()
        pred_next = self.actor.predict_next_state()
        target_next = policy_state_dict['next_obs_target']
        valid_mask = (~policy_state_dict['dones'].bool()).float()
        recon_loss = recon_loss_masked(pred_next, target_next, valid_mask)
        kl_vae = vae_kl_loss(latent['mu'], latent['logvar'])

        actor_loss = (surrogate_loss - self.entropy_coef * entropy_loss
                      + self.recon_coef * recon_loss + self.vae_beta * kl_vae)
        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['recon'] = loss_dict.get('recon', 0.0) + recon_loss.item()
        loss_dict['vae_kl'] = loss_dict.get('vae_kl', 0.0) + kl_vae.item()
        loss_dict['latent_std'] = loss_dict.get('latent_std', 0.0) + \
            torch.exp(0.5 * latent['logvar']).mean().item()
        return loss_dict
```

Resolved facts (verified against base `ppo.py` — no guessing needed):
- `_init_config` IS a method (`ppo.py:63`) called from `__init__`; the override above correctly calls `super()._init_config()` then reads the extra keys.
- Logging is AUTOMATIC and needs NO logging override: `_training_step` (`ppo.py:354`) averages every key in `loss_dict` by `num_updates` (`ppo.py:366-367`), and `_logging_to_writer` (`ppo.py:545`) emits every `loss_dict` key as `Loss/<key>` (`ppo.py:547-548`). The subclass's `_init_loss_dict_at_training_step` override adds `recon`/`vae_kl`/`latent_std` (=0) and `_update_ppo` accumulates them, so they surface as `Loss/recon`, `Loss/vae_kl`, `Loss/latent_std` (latent_std under `Loss/`, not `Policy/` — acceptable).
- `_update_algo_step` (`ppo.py:378`) calls `self._update_ppo(...)`, which resolves to the subclass override — no need to touch it.

- [ ] **Step 3: Import smoke test (no sim)**

Run:
```bash
/home/kist/anaconda3/envs/hvgym/bin/python -c "
from humanoidverse.agents.ppo_hist.ppo_hist import PPOHistoryEncoder
print('import OK', PPOHistoryEncoder.__name__)
"
```
Expected: `import OK PPOHistoryEncoder`.

- [ ] **Step 4: Short training smoke-run (needs GPU/IsaacGym)**

Run with the SAME override groups the existing run used (`.hydra/overrides.yaml`), only swapping `+exp`, `+obs`, and shrinking envs/iters. Use the sim/GPU python env you normally train with (the existing run used `+simulator=isaacsim`, i.e. the isaaclab env — NOT necessarily `hvgym`):
```bash
python humanoidverse/train_agent.py \
  +simulator=isaacsim \
  +exp=locomotion_history_encoder \
  +domain_rand=domain_rand_base \
  +rewards=loco/reward_kapex_locomotion \
  +robot=kapex/kapex_31dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_history_encoder \
  num_envs=64 project_name=smoke experiment_name=hist_enc_smoke \
  algo.config.num_learning_iterations=5 headless=True
```
Expected: training starts, per-iteration logs include finite `recon` and `vae_kl`; no shape errors. This step needs the GPU + isaacsim environment; if the executor lacks it, mark this step "deferred to user" and rely on the Step 3 import smoke test + static review for acceptance.

- [ ] **Step 5: Regression guard — confirm nothing existing was modified**

Run: `git status --porcelain`
Expected: only new files under `humanoidverse/agents/ppo_hist/`, `humanoidverse/agents/modules/ppo_hist_modules.py`, `humanoidverse/config/**/*history_encoder*.yaml`, `tests/`, and `docs/`. No `M` (modified) lines for existing source files.

- [ ] **Step 6: Commit**

```bash
git add humanoidverse/agents/ppo_hist/ tests/
git commit -m "feat(ppo_hist): PPOHistoryEncoder agent with joint VAE recon+KL loss"
```

---

### Task 7: Deploy / ONNX export path (additive)

**Files:**
- Create: `humanoidverse/agents/ppo_hist/inference_wrapper.py` — `HistEncoderInferenceModule(nn.Module)` that takes concatenated `[actor_obs, encoder_obs]` and returns the action mean (encoder→cat→actor MLP), for ONNX export.
- Create: `humanoidverse/eval_agent_hist.py` — copy of `eval_agent.py` that uses the wrapper for `_pre_eval_env_step` (pass both obs) and ONNX export. (New file; `eval_agent.py` untouched.)

**Interfaces:**
- Consumes: trained `PPOHistoryEncoder.actor` (`PPOActorWithHistoryEncoder`).
- Produces: `HistEncoderInferenceModule(actor).forward(x)` where `x = cat([actor_obs, encoder_obs], dim=-1)`; and an eval entrypoint that reproduces sim behavior with `z=μ`.

- [ ] **Step 1: Write the inference wrapper**

```python
# humanoidverse/agents/ppo_hist/inference_wrapper.py
import torch
import torch.nn as nn


class HistEncoderInferenceModule(nn.Module):
    """encoder(z=mu) -> cat[actor_obs, z] -> actor MLP -> action mean. For ONNX/JIT export."""
    def __init__(self, actor, actor_obs_dim, encoder_obs_dim):
        super().__init__()
        self.history_encoder = actor.history_encoder
        self.actor_mlp = actor.actor
        self.actor_obs_dim = actor_obs_dim
        self.encoder_obs_dim = encoder_obs_dim

    def forward(self, obs):
        actor_obs = obs[..., : self.actor_obs_dim]
        encoder_obs = obs[..., self.actor_obs_dim : self.actor_obs_dim + self.encoder_obs_dim]
        mu, _ = self.history_encoder(encoder_obs)   # deterministic mean latent
        return self.actor_mlp(torch.cat([actor_obs, mu], dim=-1))
```

- [ ] **Step 2: Unit test the wrapper (no sim)**

Add to `tests/agents/test_ppo_hist_modules.py`:

```python
def test_inference_wrapper_matches_act_inference():
    from humanoidverse.agents.ppo_hist.inference_wrapper import HistEncoderInferenceModule
    actor, _ = _make_actor()
    actor.eval()
    wrap = HistEncoderInferenceModule(actor, actor_obs_dim=30, encoder_obs_dim=40)
    ao, eo = torch.randn(4, 30), torch.randn(4, 40)
    ref = actor.act_inference(ao, eo)
    got = wrap(torch.cat([ao, eo], dim=-1))
    assert torch.allclose(ref, got, atol=1e-6)
```

Run: `/home/kist/anaconda3/envs/hvgym/bin/python -m pytest tests/agents/test_ppo_hist_modules.py::test_inference_wrapper_matches_act_inference -v`
Expected: PASS.

- [ ] **Step 3: Create `eval_agent_hist.py`**

Copy `humanoidverse/eval_agent.py` verbatim, then change only: (a) `_pre_eval_env_step` must call the policy with both `obs['actor_obs']` and `obs['encoder_obs']` — simplest is to build the concatenated input and call the wrapper; (b) the ONNX export block to export `HistEncoderInferenceModule` with an example input of width `actor_obs_dim + encoder_obs_dim`. Because `eval_agent.py`'s eval loop calls `self.eval_policy(actor_state["obs"]['actor_obs'])` in `ppo.py:641` (base), also override `_pre_eval_env_step` and `_get_inference_policy` in `PPOHistoryEncoder` (Task 6 file) to pass both obs:

```python
    # add to PPOHistoryEncoder (Task 6 file)
    def _get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference

    def _pre_eval_env_step(self, actor_state):
        obs = actor_state["obs"]
        actions = self.eval_policy(obs['actor_obs'], obs[self.encoder_obs_key])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
```

(These two overrides belong in Task 6's `ppo_hist.py`; add them there. `eval_agent_hist.py` is only needed for the ONNX/JIT export wrapper.)

- [ ] **Step 4: Eval smoke-run (needs GPU/IsaacGym)**

Run:
```bash
/home/kist/anaconda3/envs/hvgym/bin/python humanoidverse/eval_agent_hist.py \
  +checkpoint=<path to a model_*.pt from Task 6 smoke-run> num_envs=1
```
Expected: the policy runs in sim without shape errors; robot behavior matches the trained checkpoint.

- [ ] **Step 5: Commit**

```bash
git add humanoidverse/agents/ppo_hist/inference_wrapper.py humanoidverse/eval_agent_hist.py \
        humanoidverse/agents/ppo_hist/ppo_hist.py tests/agents/test_ppo_hist_modules.py
git commit -m "feat(ppo_hist): add inference wrapper + eval/export entrypoint"
```

---

## Self-Review

- **Spec coverage:** encoder (T1), decoder (T2), VAE losses (T3), actor-with-encoder joint grad (T4), obs/algo/exp configs incl. recon target + latent wiring (T5), agent loss integration + rollout target capture + storage + adaptive-KL note (T6), deploy/ONNX + `z=μ` inference (T7), new logging `recon`/`vae_kl`/`latent_std` (T6), additive/no-edit regression guard (T6 Step 5). All spec §3–§7 items map to a task.
- **Placeholder scan:** none — all code steps contain full code; sim-dependent steps give exact commands + expected observations.
- **Type consistency:** `HistoryEncoder.sample`→`(z,mu,logvar)`, actor stashes `_last_latent={z,mu,logvar}`, `get_latent_stats()` returns it, `predict_next_state()` uses `_last_latent['z']`, loss fns consume `(mu,logvar)` / `(pred,target,mask)` — consistent across T1/T3/T4/T6. `input_dim:[actor_obs,16]` matches `latent_dim:16`.

## Known implementer follow-ups (flagged, not placeholders)

- Confirm base `PPO` config-init hook name for `_init_config` (else read extra keys in `_setup_models_and_optimizer`).
- Confirm `loss_dict` init + TensorBoard logging hook to surface `Loss/recon`, `Loss/vae_kl`, `Policy/latent_std` without editing base (override the logging method in the subclass).
- `latent_dim` appears twice (algo `config.latent_dim` and `module_dict.actor.input_dim[1]`); keep in sync (documented in the algo config comment).
