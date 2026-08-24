"""외란(external push)에 대한 정책 강건성 평가.

관측을 망가뜨리는 corruption 계열(eval_obs_corruption / eval_joint_dropout /
eval_intermittent_noise)과 달리, 여기서는 관측은 깨끗하게 두고 **몸에 직접
외란을 준다**: 훈련 때의 push 와 같은 방식으로 base 수평 속도에 Δv 를 더한다
(legged_robot_base._push_robots 와 동일한 주입 지점 · IsaacLab
push_by_setting_velocity 방식의 '더하기').

훈련과 다른 점:
  - 훈련 push 는 랜덤 방향 / |Δv|<=max_push_vel_xy(0.5) / 5~10s 랜덤 간격.
    여기서는 방향과 크기를 조건으로 고정하고 일정 간격(150스텝)으로 준다.
  - 방향은 world 가 아니라 **로봇 heading(yaw) 프레임** 기준이다. 리셋 yaw 가
    ±π 랜덤이라 world 방향으로 주면 로봇마다 다른 상대 방향을 맞게 된다.
  - env 자체의 랜덤 push 는 끈다 (domain_rand.push_robots=False).

집계:
  - 각 push 를 하나의 사건으로 보고, 다음 push 전(150스텝) 낙상하면 그 push 에
    귀속시킨다 → P(fall | push). 리셋 직후(ep_len<20)의 env 는 밀지도 세지도
    않는다.
  - push 후 50스텝의 |vx-cmd| 평균 = 회복 품질. 오프셋별 vx 평균 곡선도 저장
    (회복 궤적 시각화용).

사용 예:
python humanoidverse/eval_push_robustness.py \
+checkpoint=ckpt/hist_ablation/cur/baseline/model_10000.pt +simulator=isaacsim \
++headless=True ++vx=0.5 ++envs_per_cond=20
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

# heading(yaw) 프레임 Δv. left = 로봇 기준 왼쪽(+y).
DIRS = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
MAGS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]   # m/s. 훈련 push 는 |Δv|<=0.5 (그 이상은 OOD)
PUSH_EVERY = 150                          # 스텝. 사건 창 = 다음 push 까지
RECOVERY_WINDOW = 50                      # push 후 |vx-cmd| 평균 구간


def build_conditions():
    conds = [("clean", 0.0)]
    for mag in MAGS:
        for d in DIRS:
            conds.append((d, mag))
    return conds


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "push.log")
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
    per_cond = int(config.get("envs_per_cond", 20))
    with open_dict(config):
        config.num_envs = len(conds) * per_cond
        config.env.config.domain_rand.push_robots = False   # 랜덤 push 차단 (외란은 우리가 준다)

    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacSim":
        import argparse

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="External push robustness sweep.")
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

    import inspect

    import numpy as np
    import torch
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo
    from humanoidverse.utils.math import quat_apply_yaw
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)
    device = config.get("device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    N = env.num_envs
    env_cond = torch.arange(N, device=device) % len(conds)

    # 조건별 heading-frame Δv 를 per-env (N,3) 로 펼친다. clean 은 0.
    dv_body = torch.zeros(N, 3, device=device)
    for i, (d, mag) in enumerate(conds):
        if d == "clean":
            continue
        sel = env_cond == i
        dv_body[sel, 0] = DIRS[d][0] * mag
        dv_body[sel, 1] = DIRS[d][1] * mag

    # push 주입: 훈련 _push_robots 와 같은 지점(_update_tasks_callback, refresh 직전).
    # base_quat 은 직전의 _pre_compute_observations_callback 에서 갱신된 상태다.
    push_ctl = {"fire": False, "counted": None}
    orig_cb = env._update_tasks_callback

    def cb_with_push():
        orig_cb()
        if not push_ctl["fire"]:
            return
        push_ctl["fire"] = False
        alive = env.episode_length_buf >= 20
        sel = alive & (dv_body.abs().sum(-1) > 0)
        dv_world = quat_apply_yaw(env.base_quat, dv_body)
        env.simulator.robot_root_states[:, 7:9] += torch.where(
            sel.unsqueeze(1), dv_world[:, :2], torch.zeros_like(dv_world[:, :2]))
        env.need_to_refresh_envs |= sel
        push_ctl["counted"] = sel.clone()

    env._update_tasks_callback = cb_with_push

    algo._eval_mode()
    env.set_is_evaluating()
    obs_dict = env.reset_all()

    cfg_vx = float(config.get("vx", 0.5))
    warm = int(config.get("warmup_steps", 200))
    meas = int(config.get("measure_steps", 600))

    actor = algo.actor
    n_args = len(inspect.signature(actor.act_inference).parameters)
    infer = ((lambda o: actor.act_inference(o["actor_obs"], o["encoder_obs"]))
             if n_args >= 2 else (lambda o: actor.act_inference(o["actor_obs"])))

    n_conds = len(conds)
    events = torch.zeros(n_conds, device=device)          # 성립한 push 사건 수
    falls_attr = torch.zeros(n_conds, device=device)      # 창 안에서 낙상한 사건 수
    falls_all = torch.zeros(N, device=device)             # 측정 구간 전체 낙상 (falls/1k 용)
    vx_sum = torch.zeros(N, device=device)                # 정상 구간 속도 통계
    vy_sum = torch.zeros(N, device=device)
    wz_sum = torch.zeros(N, device=device)
    n_ok = torch.zeros(N, device=device)
    err50_sum = torch.zeros(n_conds, device=device)       # push 후 50스텝 |vx-cmd|
    err50_cnt = torch.zeros(n_conds, device=device)
    rec_vx = torch.zeros(n_conds * PUSH_EVERY, device=device)   # 오프셋별 vx 회복 곡선
    rec_cnt = torch.zeros(n_conds * PUSH_EVERY, device=device)

    event_open = torch.zeros(N, dtype=torch.bool, device=device)
    offset = torch.zeros(N, dtype=torch.long, device=device)

    logger.info(f"envs={N}, 조건 {n_conds}종 x {per_cond}, vx={cfg_vx}, warmup={warm}, "
                f"measure={meas}, push 간격 {PUSH_EVERY}스텝 (사건 {meas // PUSH_EVERY}회/env)")

    for step in range(warm + meas):
        env.is_standing_env[:] = False
        env.commands[:, 0] = cfg_vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        recording = step >= warm
        if recording and (step - warm) % PUSH_EVERY == 0:
            push_ctl["fire"] = True                      # 이번 스텝 post-physics 에서 주입

        with torch.no_grad():
            action = infer(obs_dict)

        if recording:
            ok = (env.episode_length_buf >= 20).float()
            n_ok += ok
            vx_sum += env.base_lin_vel[:, 0] * ok
            vy_sum += env.base_lin_vel[:, 1] * ok
            wz_sum += env.base_ang_vel[:, 2] * ok

        obs_dict, _, dones, _ = env.step({"obs": obs_dict, "actions": action,
                                          "done_indices": [], "stop": False})
        obs_dict = {k: v.to(env.device) for k, v in obs_dict.items()}

        time_out = getattr(env, "time_out_buf", None)
        fall = dones.bool() & (~time_out.bool() if time_out is not None else True)

        if recording:
            falls_all += fall.float()
            # 열린 사건에 낙상 귀속 (창 = 다음 push 전 PUSH_EVERY 스텝)
            hit = fall & event_open
            if hit.any():
                falls_attr.index_add_(0, env_cond[hit], torch.ones(int(hit.sum()), device=device))
            event_open &= ~dones.bool()                  # 낙상이든 타임아웃이든 사건 종료

            # 회복 곡선 / 추종 오차 누적 (사건이 열려 있는 동안)
            if event_open.any():
                m = event_open
                idx = env_cond[m] * PUSH_EVERY + offset[m]
                rec_vx.index_add_(0, idx, env.base_lin_vel[m, 0])
                rec_cnt.index_add_(0, idx, torch.ones(int(m.sum()), device=device))
                e50 = m & (offset < RECOVERY_WINDOW)
                if e50.any():
                    err50_sum.index_add_(0, env_cond[e50],
                                         (env.base_lin_vel[e50, 0] - cfg_vx).abs())
                    err50_cnt.index_add_(0, env_cond[e50], torch.ones(int(e50.sum()), device=device))
            offset[event_open] += 1
            expired = event_open & (offset >= PUSH_EVERY)
            event_open &= ~expired

            # 방금 주입된 push 로 새 사건 개시
            if push_ctl["counted"] is not None:
                sel = push_ctl["counted"]
                push_ctl["counted"] = None
                events.index_add_(0, env_cond[sel], torch.ones(int(sel.sum()), device=device))
                event_open |= sel
                offset[sel] = 0

        if (step + 1) % 200 == 0:
            logger.info(f"[{'warmup' if step < warm else 'push'}] {step + 1}/{warm + meas}")

    idx = env_cond.cpu().numpy()
    n = n_ok.clamp(min=1)
    vx_m, vy_m, wz_m = (vx_sum / n).cpu().numpy(), (vy_sum / n).cpu().numpy(), (wz_sum / n).cpu().numpy()
    falls_np = falls_all.cpu().numpy()
    events_np, falls_attr_np = events.cpu().numpy(), falls_attr.cpu().numpy()
    err50 = (err50_sum / err50_cnt.clamp(min=1)).cpu().numpy()

    rows = []
    logger.info(f"{'condition':16s}{'events':>7s}{'P(fall|push)':>13s}{'falls/1k':>9s}"
                f"{'vx_err50':>9s}{'vx':>7s}{'|vy|':>7s}{'|wz|':>7s}")
    for i, (d, mag) in enumerate(conds):
        m = idx == i
        rate = 1000.0 * falls_np[m].sum() / max(m.sum() * meas, 1)
        p_fall = falls_attr_np[i] / max(events_np[i], 1)
        label = "clean" if d == "clean" else f"{d} {mag:g} m/s"
        logger.info(f"{label:16s}{events_np[i]:7.0f}{p_fall:13.3f}{rate:9.2f}"
                    f"{err50[i]:9.3f}{vx_m[m].mean():7.3f}"
                    f"{np.abs(vy_m[m]).mean():7.3f}{np.abs(wz_m[m]).mean():7.3f}")
        rows.append((d, mag, p_fall, events_np[i], rate, err50[i], vx_m[m].mean(),
                     np.abs(vy_m[m]).mean(), np.abs(wz_m[m]).mean()))

    out_path = config.get("out", None)
    out_path = Path(out_path) if out_path else checkpoint.parent / "perturb" / "push.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, env_cond=idx, cmd_vx=np.float32(cfg_vx),
        measure_steps=np.int32(meas), push_every=np.int32(PUSH_EVERY),
        cond_dir=np.array([c[0] for c in conds]),
        cond_mag=np.array([c[1] for c in conds], dtype=np.float32),
        events=events_np, falls_attr=falls_attr_np,
        recovery_vx=(rec_vx / rec_cnt.clamp(min=1)).cpu().numpy().reshape(n_conds, PUSH_EVERY),
        recovery_cnt=rec_cnt.cpu().numpy().reshape(n_conds, PUSH_EVERY),
        summary=np.array(rows, dtype=object))
    logger.info(f"저장 완료: {out_path}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
