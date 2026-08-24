"""관측 오염(observation corruption)에 대한 정책 강건성 평가.

가설: history 를 raw 로 받는 baseline 은 오염된 관측이 history 버퍼에 5스텝 남아
계속 영향을 주는 반면, history encoder 는 오염이 19차원 latent 로 압축되면서
희석돼 더 잘 버틸 것이다. (반대 가능성도 있다 — encoder 쪽이 현재 스텝 obs 에
차원당 2배 민감하고, baseline 의 history 는 같은 신호의 시간 복사본이라 i.i.d.
노이즈에 대해서는 저역통과 필터로 작동할 수 있다.)

주입 지점: env 가 이미 가진 관측 노이즈 경로(helpers.parse_observation)를 그대로
쓴다. 이 경로는 actor 가 읽는 현재 obs 와 history 버퍼에 들어가는 값 **양쪽**에
노이즈를 넣고, critic/teacher/recon_target 은 깨끗하게 유지한다. 즉 실제 센서
결함과 같은 방식으로 전파된다.

  주의: env 는 현재 스텝 obs 와 history 저장분에 대해 parse_observation 을 각각
  호출하므로 같은 스텝에 대해 노이즈를 독립적으로 두 번 뽑는다. bias 조건에서는
  둘이 같은 값이라 문제가 없지만, gauss 조건에서는 history 에 남는 값이 현재 obs
  와 다른 실현값이다 (같은 참값의 독립 샘플 2개).

조건: env 마다 (노이즈 종류, 대상 obs, 세기) 하나를 고정 배정해 한 번의 시뮬레이션
으로 전 조건을 동시에 돌린다. 세기는 각 obs 의 **실측 표준편차 배수**로 준다
(obs_stats npz 에서 계산, 물리 단위 — 노이즈는 obs_scales 적용 전에 더해진다).

사용 예:
  python humanoidverse/eval_obs_corruption.py \
    +checkpoint=ckpt/hist_ablation/rew1/baseline/model_61000.pt +simulator=isaacsim \
    ++headless=True ++vx=0.5 ++envs_per_cond=24
"""

import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, open_dict

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from humanoidverse.utils.logging import HydraLoggerBridge

TARGETS = ["dof_pos", "dof_vel", "projected_gravity", "base_ang_vel"]
TARGET_SETS = {t: [t] for t in TARGETS}
TARGET_SETS["all"] = list(TARGETS)
LEVELS = [0.5, 1.0, 2.0, 4.0]        # 실측 sigma 배수
TYPES = ["gauss", "bias"]            # i.i.d. 백색 / 지속적 상수 오프셋


def build_conditions():
    """[(type, target_name, level)] — 첫 항목은 오염 없음(대조군)."""
    conds = [("clean", "none", 0.0)]
    for t in TYPES:
        for name in TARGET_SETS:
            for lv in LEVELS:
                conds.append((t, name, lv))
    return conds


LEGACY_79_SPANS = {
    "actions": (0, 23),
    "base_ang_vel": (23, 26),
    "command_ang_vel": (26, 27),
    "command_lin_vel": (27, 29),
    "command_stand": (29, 30),
    "dof_pos": (30, 53),
    "dof_vel": (53, 76),
    "projected_gravity": (76, 79),
}


def actor_obs_spans(actor_obs_keys, obs_dims, obs_auxiliary=None):
    """Reproduce actor_obs layout, expanding auxiliary history blocks."""
    if hasattr(obs_dims, "items"):
        dims = {str(k): int(v) for k, v in obs_dims.items()}
    else:
        dims = {str(k): int(v) for item in obs_dims for k, v in item.items()}
    auxiliary = dict(obs_auxiliary or {})

    spans = {}
    offset = 0
    for configured_key in sorted(str(k) for k in actor_obs_keys):
        key = configured_key[:-4] if configured_key.endswith("_raw") else configured_key
        if key in dims:
            next_offset = offset + dims[key]
            # Prefer a direct current-state component if the same signal also
            # appears inside an auxiliary history block.
            spans[key] = (offset, next_offset)
        elif key in auxiliary:
            next_offset = offset
            for history_key, history_length in sorted(auxiliary[key].items()):
                history_key = str(history_key)
                if history_key not in dims:
                    raise ValueError(
                        f"auxiliary '{key}'의 항목 '{history_key}' 차원을 찾을 수 없습니다")
                history_end = next_offset + dims[history_key] * int(history_length)
                spans.setdefault(history_key, (next_offset, history_end))
                next_offset = history_end
        else:
            raise ValueError(
                f"actor_obs 항목 '{key}'의 dimension/auxiliary 구성을 찾을 수 없습니다")
        offset = next_offset
    return spans, offset


def measured_sigma(ckpt_dir, obs_scales, ckpt_num=None, npz_path=None,
                   actor_obs_keys=None, obs_dims=None, obs_auxiliary=None):
    """obs_stats npz 에서 obs 키별 실측 sigma (물리 단위) 계산.

    npz 의 actor_obs 는 obs_scales 가 곱해진 뒤의 값이라, 노이즈를 넣는 지점
    (스케일 이전)과 단위를 맞추려면 obs_scales 로 나눠야 한다.

    actor_obs span은 학습 config의 obs_dict/obs_dims/obs_auxiliary로부터 env와 같은
    사전순 concat 규칙으로 계산한다. history를 직접 받는 baseline은 각 대상 신호의
    전체 history block에서 sigma를 계산한다.
    """
    import numpy as np

    if npz_path is not None:
        path = Path(npz_path)
    else:
        path = (Path(ckpt_dir) / "obs_stats" / f"obs_ckpt_{ckpt_num}.npz"
                if ckpt_num is not None else None)
        if path is None or not path.exists():
            cand = sorted((Path(ckpt_dir) / "obs_stats").glob("obs_ckpt_*.npz"))
            if cand:
                path = cand[-1]
    assert path is not None and path.exists(), (
        f"obs_stats npz 가 없습니다 ({path}). collect_obs_stats.py 로 수집하거나 "
        "++sigma_npz=<path> 로 지정하세요.")
    d = np.load(path)
    ok = (d["ep_len"] >= 20) & (~d["done"])
    o = d["actor_obs"][ok]
    obs_width = o.shape[-1]
    span = None
    expected_width = None
    if actor_obs_keys is not None and obs_dims is not None:
        span, expected_width = actor_obs_spans(
            actor_obs_keys, obs_dims, obs_auxiliary=obs_auxiliary)
        if expected_width != obs_width:
            span = None

    # Backward compatibility for a 79-D sigma_npz shared by another policy.
    if span is None and obs_width == 79:
        span = LEGACY_79_SPANS
        if expected_width is not None:
            logger.warning(
                f"sigma NPZ는 79차원, 평가 정책 actor_obs는 {expected_width}차원입니다. "
                "79차원 encoder 모델의 canonical span을 사용합니다.")

    assert span is not None, (
        f"actor_obs가 {obs_width}차원이지만 config에서 계산한 layout은 "
        f"{expected_width if expected_width is not None else '미지정'}차원입니다. "
        "현재 checkpoint와 같은 obs config로 obs_stats를 다시 수집하거나, "
        "++sigma_npz=<79차원 encoder 모델 NPZ>를 지정하세요.")
    missing = [key for key in TARGETS if key not in span]
    assert not missing, f"actor_obs에 corruption 대상 항목이 없습니다: {missing}"
    out = {}
    for k in TARGETS:
        a, b = span[k]
        out[k] = float(o[:, a:b].std(0).mean()) / float(obs_scales[k])
    return out


def install_corruption(torch, num_envs, device, sigma, conds, env_cond):
    """parse_observation 을 per-env 오염 버전으로 교체하고 제어 핸들을 돌려준다."""
    import humanoidverse.utils.helpers as H

    # 조건별 (세기 x sigma) 와 종류 마스크를 obs 키마다 per-env 벡터로 펼친다.
    mag = {k: torch.zeros(num_envs, 1, device=device) for k in TARGETS}
    is_gauss = torch.zeros(num_envs, 1, device=device)
    for i, (typ, name, lv) in enumerate(conds):
        sel = env_cond == i
        if typ == "clean" or not sel.any():
            continue
        for k in TARGET_SETS[name]:
            mag[k][sel] = lv * sigma[k]
        if typ == "gauss":
            is_gauss[sel] = 1.0

    # bias 조건용 고정 방향 (env x dim). 매 스텝 같은 값이 더해진다 = 센서 오프셋.
    bias_dir, state = {}, {"on": False}

    def draw(key, x):
        if key not in mag:
            return None
        if key not in bias_dir:
            g = torch.Generator(device="cpu").manual_seed(1234 + len(bias_dir))
            bias_dir[key] = torch.randn(x.shape, generator=g).to(device)
        return (torch.randn_like(x) * is_gauss
                + bias_dir[key] * (1.0 - is_gauss)) * mag[key]

    def patched(cls, key_list, buf_dict, obs_scales, noise_scales, cur_scale):
        for obs_key in key_list:
            if obs_key.endswith("_raw"):
                obs_key = obs_key[:-4]
                gate = 0.0
            else:
                gate = noise_scales.get(obs_key, 0.0)
                gate = float(gate) if isinstance(gate, (int, float)) else 1.0
            x = getattr(cls, f"_get_obs_{obs_key}")().clone()
            if state["on"] and gate > 0.0:
                n = draw(obs_key, x)
                if n is not None:
                    x = x + n
            buf_dict[obs_key] = x * obs_scales[obs_key]

    H.parse_observation = patched
    # legged_robot_base 는 `from ...helpers import *` 로 이름을 가져가므로 그쪽도 교체
    import humanoidverse.envs.legged_base_task.legged_robot_base as LRB
    LRB.parse_observation = patched
    return state


def run(env, algo, torch, cfg, conds, env_cond, state):
    import inspect

    import numpy as np

    algo._eval_mode()
    env.set_is_evaluating()
    obs_dict = env.reset_all()

    # 커맨드 고정: 전 env 동일. is_standing_env 는 env 생성 시 확률적으로 켜져
    # eval 에서 해제되지 않으므로(=그 env 는 걷지 않음) 반드시 끈다.
    env.is_standing_env[:] = False
    env.commands[:, 0] = cfg["vx"]
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0

    actor = algo.actor
    n_args = len(inspect.signature(actor.act_inference).parameters)
    infer = ((lambda o: actor.act_inference(o["actor_obs"], o["encoder_obs"]))
             if n_args >= 2 else (lambda o: actor.act_inference(o["actor_obs"])))

    N = env.num_envs
    acc = {k: torch.zeros(N, device=env.device) for k in
           ("falls", "steps", "vx", "vy", "wz", "vx_err", "act_rate", "n_ok")}
    first_fall = torch.full((N,), float("nan"), device=env.device)
    prev_action = None
    warm, meas = cfg["warmup"], cfg["measure"]

    for step in range(warm + meas):
        env.is_standing_env[:] = False
        env.commands[:, 0] = cfg["vx"]
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        state["on"] = step >= warm            # 워밍업은 깨끗한 관측으로 보행 확립

        with torch.no_grad():
            action = infer(obs_dict)

        recording = step >= warm
        if recording:
            ok = (env.episode_length_buf >= 20).float()   # 리셋 직후 과도구간 제외
            acc["n_ok"] += ok
            acc["vx"] += env.base_lin_vel[:, 0] * ok
            acc["vy"] += env.base_lin_vel[:, 1] * ok
            acc["wz"] += env.base_ang_vel[:, 2] * ok
            acc["vx_err"] += (env.base_lin_vel[:, 0] - cfg["vx"]).abs() * ok
            if prev_action is not None:
                acc["act_rate"] += (action - prev_action).abs().mean(-1) * ok
            acc["steps"] += 1
        prev_action = action.clone()

        obs_dict, _, dones, _ = env.step({"obs": obs_dict, "actions": action,
                                          "done_indices": [], "stop": False})
        obs_dict = {k: v.to(env.device) for k, v in obs_dict.items()}

        time_out = getattr(env, "time_out_buf", None)
        fall = dones.bool() & (~time_out.bool() if time_out is not None else True)
        if recording:
            acc["falls"] += fall.float()
            newly = fall & torch.isnan(first_fall)
            first_fall[newly] = float(step - warm)

        if (step + 1) % 200 == 0:
            logger.info(f"[{'warmup' if step < warm else 'corrupt'}] {step + 1}/{warm + meas}")

    n = acc["n_ok"].clamp(min=1)
    per_env = {"falls": acc["falls"], "vx": acc["vx"] / n, "vy": acc["vy"] / n,
               "wz": acc["wz"] / n, "vx_err": acc["vx_err"] / n,
               "act_rate": acc["act_rate"] / n, "n_ok": acc["n_ok"],
               "first_fall": first_fall}
    return {k: v.cpu().numpy() for k, v in per_env.items()}


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "corruption.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level=os.environ.get("LOGURU_LEVEL", "INFO").upper(), colorize=True)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    assert override_config.checkpoint is not None, "+checkpoint=<path/model_X.pt> 가 필요합니다"
    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    with open(config_path) as f:
        train_config = OmegaConf.load(f)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    config = OmegaConf.merge(train_config, override_config)

    conds = build_conditions()
    per_cond = int(config.get("envs_per_cond", 24))
    with open_dict(config):
        config.num_envs = len(conds) * per_cond
        # 오염 대상 키의 노이즈 게이트를 연다. 실제 크기는 patched parse_observation
        # 이 per-env 로 정하고, 여기 값은 on/off 플래그로만 쓰인다.
        for k in TARGETS:
            config.env.config.obs.noise_scales[k] = 1.0

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        import argparse

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="Observation corruption sweep.")
        AppLauncher.add_app_launcher_args(parser)
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        args_cli.video = False
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import numpy as np
    import torch

    from humanoidverse.agents.base_algo.base_algo import BaseAlgo
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)
    device = config.get("device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")

    sigma = measured_sigma(checkpoint.parent, config.env.config.obs.obs_scales,
                           ckpt_num=checkpoint.stem.split("_")[-1],
                           npz_path=config.get("sigma_npz", None),
                           actor_obs_keys=config.env.config.obs.obs_dict.actor_obs,
                           obs_dims=config.env.config.obs.obs_dims,
                           obs_auxiliary=config.env.config.obs.obs_auxiliary)
    logger.info("실측 sigma (물리 단위): " + ", ".join(f"{k}={v:.3f}" for k, v in sigma.items()))

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    env_cond = torch.arange(env.num_envs, device=device) % len(conds)
    state = install_corruption(torch, env.num_envs, device, sigma, conds, env_cond)

    cfg = {"vx": float(config.get("vx", 0.5)), "warmup": int(config.get("warmup_steps", 200)),
           "measure": int(config.get("measure_steps", 600))}
    logger.info(f"envs={env.num_envs}, 조건 {len(conds)}종 x {per_cond}, "
                f"vx={cfg['vx']}, warmup={cfg['warmup']}, measure={cfg['measure']}")

    res = run(env, algo, torch, cfg, conds, env_cond, state)

    idx = env_cond.cpu().numpy()
    rows = []
    logger.info(f"{'condition':26s}{'falls/1k':>9s}{'vx':>7s}{'vx_err':>8s}"
                f"{'|vy|':>7s}{'|wz|':>7s}{'act_rate':>9s}")
    for i, (typ, name, lv) in enumerate(conds):
        m = idx == i
        rate = 1000.0 * res["falls"][m].sum() / max(m.sum() * cfg["measure"], 1)
        label = "clean" if typ == "clean" else f"{typ} {name} {lv:g}σ"
        logger.info(f"{label:26s}{rate:9.2f}{res['vx'][m].mean():7.3f}"
                    f"{res['vx_err'][m].mean():8.3f}{np.abs(res['vy'][m]).mean():7.3f}"
                    f"{np.abs(res['wz'][m]).mean():7.3f}{res['act_rate'][m].mean():9.4f}")
        rows.append((typ, name, lv, rate, res["vx"][m].mean(), res["vx_err"][m].mean(),
                     np.abs(res["vy"][m]).mean(), np.abs(res["wz"][m]).mean(),
                     res["act_rate"][m].mean()))

    out_path = config.get("out", None)
    out_path = Path(out_path) if out_path else checkpoint.parent / "corruption" / "sweep.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, env_cond=idx, sigma=np.array([sigma[k] for k in TARGETS]),
        sigma_keys=np.array(TARGETS), cmd_vx=np.float32(cfg["vx"]),
        measure_steps=np.int32(cfg["measure"]),
        cond_type=np.array([c[0] for c in conds]), cond_target=np.array([c[1] for c in conds]),
        cond_level=np.array([c[2] for c in conds], dtype=np.float32),
        summary=np.array(rows, dtype=object), **res)
    logger.info(f"저장 완료: {out_path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
