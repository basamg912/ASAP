# History-Encoder (Concurrent VAE Context Encoder) — Design

- **Date:** 2026-07-24
- **Author:** Sangyun Bae (design assisted)
- **Scope:** KAPEX locomotion, PPO (`humanoidverse/agents/ppo/ppo.py`), IsaacGym backend (`hvgym`)
- **Status:** Approved design — pending implementation plan

## 1. Motivation

The KAPEX locomotion policy is `wolinvel` (the actor does not observe base linear
velocity) and consumes a flat 5-step `short_history` directly. We want the policy to
infer the unobserved dynamics/state from a history of proprioception + actions through a
learned latent, **RMA-style**, but **without the two-phase (teacher→adaptation) training**
used by the delta-action pipeline (`humanoidverse.agents.delta_a.train_delta_a.PPODeltaA`).

The encoder is trained **concurrently and end-to-end** with the base policy. This is the
DreamWaQ-style *concurrent VAE context encoder* pattern: a history encoder outputs a latent
distribution, the latent feeds the base policy, and a self-supervised next-observation
reconstruction (plus a KL term) is added to the PPO loss.

This is a separate feature from — but complementary to — the current training-stability
fixes (lower `entropy_coef`, lower `init_noise_std`, penalty-curriculum floor).

## 2. Key Decisions (locked)

| Decision | Choice |
|---|---|
| Reconstruction target | **Next proprioceptive obs `o_{t+1}` only** (pure self-supervised; no privileged targets) |
| Gradient flow | **Joint** — latent is *not* detached; PPO surrogate grad + VAE grad both train the encoder |
| Encoder architecture | **Flatten → MLP** over the 5-step history |
| Loss integration | Single combined loss: `total = ppo_loss + vae_loss`, one backward through `actor.parameters()` |

## 3. Architecture & Data Flow

```
 history H_t (5 steps: proprio + action, flattened)
      │
      ▼
 history_encoder (MLP)  ──▶  μ_z(L), logσ_z(L)
      │
      ▼   z = μ_z + σ_z⊙ε  (train) / z = μ_z (inference)
      ├──────────────────────────────┐
      ▼ (joint grad)                  ▼ (recon + KL)
 base_policy (actor)             decoder (MLP)
   [o_t ⊕ z] → MLP(512,256,128)    z → ô_{t+1}
   → action mean, std (existing)   loss = MSE(o_{t+1}, ô_{t+1}) + β·KL(q(z|H)‖N(0,I))

 critic: unchanged (privileged critic_obs: base_lin_vel + short_history)
```

- **encoder input** `H_t`: 5-step stack of `[base_ang_vel, projected_gravity, dof_pos, dof_vel, actions]` (proprio + action; commands excluded — external goals, not state).
- **encoder output**: latent distribution `(μ_z, logσ_z)`, dim `L`.
- **actor input**: current single-step obs `o_t` (base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, actions) concatenated with `z`. Replaces the old `short_history` entry in `actor_obs`.
- **decoder input**: `z` only (forces `z` to carry predictive information). Output = `ô_{t+1}`.
- **reconstruction target** `o_{t+1}`: next-step proprio `[base_ang_vel, projected_gravity, dof_pos, dof_vel]` (`recon_dim = 3 + 3 + N + N`, N = dof count). Commands and actions excluded.
- **latent at inference**: `z = μ_z` (no sampling). Decoder dropped at deploy.

## 4. Loss Integration & Gradient Flow

Encoder + decoder live **inside** the actor module (new class
`PPOActorWithHistoryEncoder(PPOActor)`), so `self.actor.parameters()` includes them and the
existing `actor_optimizer` (`ppo.py:113`) trains them. No new optimizer; critic untouched.
`max_grad_norm=1.0` clip already applies to `actor.parameters()` (`ppo.py:453`).

**Actor forward** (`update_distribution`, `ppo_modules.py:62`):
```
(μ_z, logσ_z) = encoder(H_t)
z = reparameterize(μ_z, logσ_z)         # train; μ_z at inference
mean = actor_mlp(cat[o_t, z])
self.distribution = Normal(mean, self.std)
# stash z, μ_z, logσ_z for KL and decode
```

**Aux loss** (in `_update_ppo`, near `ppo.py:441`). `_actor_act_step` (`ppo.py:397`)
recomputes the distribution from stored obs during the update — the encoder (and thus `z`)
is recomputed there, so the importance ratio `π_new/π_old` remains correct with the encoder
as part of the policy `θ`:
```
recon      = decoder(z)
recon_loss = MSE(recon, next_obs_target_batch) * (~dones mask)
kl_vae     = -0.5 * mean( 1 + logσ_z² − μ_z² − σ_z² )
actor_loss = surrogate_loss − entropy_coef·entropy
             + recon_coef · recon_loss + vae_beta · kl_vae
```
Because encoder + decoder ∈ `actor.parameters()`, the single `actor_loss.backward()`
(`ppo.py:449`) trains policy + encoder + decoder jointly. `z` is not detached, so
`surrogate_loss` also flows into the encoder (the chosen joint behavior).

**Adaptive-KL note**: the LR adaptation (`ppo.py:405-421`) uses the *action* distribution KL,
independent of the VAE KL. The encoder now affects the action mean, so encoder updates are
reflected in the action-KL — acceptable, but large VAE-driven updates can spike action-KL
and shrink LR. Mitigated by modest `recon_coef`/`vae_beta` and grad clipping.

## 5. Config / Obs / Storage Changes

**New obs config** (copy of `config/obs/loco/leggedloco_obs_history_wolinvel.yaml`):
- `actor_obs`: remove `short_history` → `[base_ang_vel, projected_gravity, command_lin_vel, command_ang_vel, command_stand, dof_pos, dof_vel, actions]`.
- New aux `history`: `{base_ang_vel:5, projected_gravity:5, dof_pos:5, dof_vel:5, actions:5}`.
- `critic_obs`: unchanged.
- `obs_dims`: add `recon_dim` entry for the reconstruction target.

**module_dict** (algo config):
```
history_encoder: input=[history] → out=2·L (μ,logσ)   MLP [256,128]
decoder:         input=[latent]  → out=recon_dim        MLP [128,256]
actor:           input=[actor_obs]+latent(L) → action   MLP [512,256,128]
latent_dim: 16,  vae_beta: 1.0,  recon_coef: 1.0
```

**Storage** (`_setup_storage`, `ppo.py:116`): `history` auto-registers via the
`algo_obs_dim_dict` loop (`ppo.py:119`). Add `register_key('next_obs_target', shape=(recon_dim,))`;
during rollout fill `next_obs_target[t]` with the post-step proprio, masked by `dones`.

**Input plumbing**: `actor.act` / `_actor_act_step` receive both `actor_obs` and `history`.

## 6. Hyperparameters (all config-exposed)

- `latent_dim = 16`; encoder `[256,128]`, decoder `[128,256]`
- `vae_beta = 1.0`, `recon_coef = 1.0`
- Concurrent stability fixes: `entropy_coef = 0.001`, `init_noise_std = 0.5`

## 7. Risks & Mitigations

1. **Joint-grad instability** — VAE grad shifts `z`→action mean, spiking action-KL and
   collapsing adaptive LR. → modest `recon_coef`/`vae_beta`, grad clip, optional warmup
   (stop-grad `z`→actor for first K iters, then open gradually). Monitor action-KL, `mean_noise_std`.
2. **Surrogate is tiny (~0.001 observed)** — `recon_coef·recon_loss` may dominate the
   encoder gradient, making it effectively "encoder = predictor, policy adapts" (near-detach).
   Not fatal (decoder/recon do not touch actor-MLP weights directly), but balance via
   `recon_coef`; monitor `Loss/Surrogate` vs `recon_loss` ratio.
3. **Posterior collapse** (KL→0, `z` ignored) — β annealing or free-bits; log KL magnitude.
4. **next_obs_target across resets** — mask terminal transitions with `dones`.
5. **Deployment** — encoder is a small MLP (negligible latency); update ONNX export path in
   `eval_agent.py` to include encoder→actor.

**New logging**: `Loss/recon`, `Loss/vae_kl`, `Policy/latent_std` (collapse watch).

## 8. Out of Scope

- Two-phase teacher/adaptation training (explicitly avoided).
- Privileged env-parameter regression (friction/mass/terrain).
- Recurrent (GRU/LSTM) or 1D-CNN encoders (may revisit if history length grows).
- Non-locomotion tasks (motion_tracking, delta-a).

## 9. Success Criteria

- Training runs end-to-end with the combined loss; `Loss/recon` decreases, `Loss/vae_kl`
  stays bounded (no collapse), `Policy/latent_std` > 0.
- `mean_noise_std` converges lower than the current stuck ~2.0 baseline; episode return and
  velocity tracking improve vs the no-encoder baseline under identical reward/curriculum.
- Deployed (ONNX) policy uses encoder(`z=μ`)→actor and reproduces sim behavior.
