"""auto_load_latest 용 체크포인트 탐색 (utils/helpers.py:find_resume_checkpoints).

experiment_dir 이름 규약: {timestamp}-{experiment_name}-{log_task_name}-{robot_type}
(timestamp 는 %Y%m%d_%H%M%S 라 '-' 를 포함하지 않음 → 첫 '-' 이후가 실험 식별자)
"""
from pathlib import Path

from humanoidverse.utils.helpers import find_resume_checkpoints

SUFFIX = "hist_v1-locomotion-g1_29dof_anneal_23dof"


def _make_run_dir(project_dir: Path, timestamp: str, iters, suffix: str = SUFFIX) -> Path:
    d = project_dir / f"{timestamp}-{suffix}"
    d.mkdir(parents=True)
    for it in iters:
        (d / f"model_{it}.pt").touch()
    return d


def test_no_previous_runs_returns_empty(tmp_path):
    exp_dir = tmp_path / "logs" / "PROJ" / f"20260820_120000-{SUFFIX}"
    assert find_resume_checkpoints(exp_dir) == []


def test_picks_highest_iteration_across_sibling_dirs(tmp_path):
    project = tmp_path / "logs" / "PROJ"
    _make_run_dir(project, "20260818_100000", [100, 2800])
    old_best = _make_run_dir(project, "20260819_180000", [100, 300])
    exp_dir = project / f"20260820_120000-{SUFFIX}"

    candidates = find_resume_checkpoints(exp_dir)

    assert candidates[0] == project / f"20260818_100000-{SUFFIX}" / "model_2800.pt"
    # 후순위 후보들은 iteration 내림차순 (손상 ckpt fallback 용)
    assert [c.name for c in candidates] == [
        "model_2800.pt", "model_300.pt", "model_100.pt", "model_100.pt",
    ]
    assert candidates[1] == old_best / "model_300.pt"


def test_ignores_other_experiments_and_non_numeric_files(tmp_path):
    project = tmp_path / "logs" / "PROJ"
    _make_run_dir(project, "20260819_000000", [7000], suffix="hist_v2-locomotion-g1_29dof_anneal_23dof")
    mine = _make_run_dir(project, "20260819_010000", [500])
    (mine / "model_final.pt").touch()
    (mine / "config.yaml").touch()
    exp_dir = project / f"20260820_120000-{SUFFIX}"

    candidates = find_resume_checkpoints(exp_dir)

    assert candidates == [mine / "model_500.pt"]


def test_iteration_tie_prefers_newest_run_dir(tmp_path):
    project = tmp_path / "logs" / "PROJ"
    _make_run_dir(project, "20260818_100000", [500])
    newer = _make_run_dir(project, "20260819_180000", [500])
    exp_dir = project / f"20260820_120000-{SUFFIX}"

    candidates = find_resume_checkpoints(exp_dir)

    assert candidates[0] == newer / "model_500.pt"
