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


class FakeAlgo:
    """load() 가 경로별로 성공/실패를 흉내내는 대역."""
    def __init__(self, mismatch_dirs=(), corrupt_files=()):
        self.mismatch_dirs = set(mismatch_dirs)
        self.corrupt_files = set(corrupt_files)
        self.attempted = []

    def load(self, ckpt_path):
        p = Path(ckpt_path)
        self.attempted.append(p)
        if p.parent in self.mismatch_dirs:
            raise RuntimeError(
                "Error(s) in loading state_dict for PPOActorWithStudentEncoder:\n"
                "\tsize mismatch for student.net.channel_mixing.0.weight")
        if p in self.corrupt_files:
            raise RuntimeError("PytorchStreamReader failed reading zip archive")


def test_load_skips_whole_dir_on_state_dict_mismatch(tmp_path):
    from humanoidverse.utils.helpers import load_resume_checkpoint
    old = _make_run_dir(tmp_path, "20260814_081157", [25700, 25800])  # 구조 변경 전
    new = _make_run_dir(tmp_path, "20260819_180100", [300, 500])      # 현재 구조
    candidates = find_resume_checkpoints(tmp_path / f"20260820_120000-{SUFFIX}")
    algo = FakeAlgo(mismatch_dirs=[old])

    loaded = load_resume_checkpoint(algo, candidates)

    assert loaded == new / "model_500.pt"
    # 불일치 dir 은 첫 실패 후 통째로 스킵 — 25700 재시도 없음
    assert algo.attempted == [old / "model_25800.pt", new / "model_500.pt"]


def test_load_falls_back_within_dir_on_corrupt_file(tmp_path):
    from humanoidverse.utils.helpers import load_resume_checkpoint
    d = _make_run_dir(tmp_path, "20260819_180100", [400, 500])
    candidates = find_resume_checkpoints(tmp_path / f"20260820_120000-{SUFFIX}")
    algo = FakeAlgo(corrupt_files=[d / "model_500.pt"])

    loaded = load_resume_checkpoint(algo, candidates)

    assert loaded == d / "model_400.pt"


def test_load_returns_none_when_nothing_loadable(tmp_path):
    from humanoidverse.utils.helpers import load_resume_checkpoint
    old = _make_run_dir(tmp_path, "20260814_081157", [100])
    candidates = find_resume_checkpoints(tmp_path / f"20260820_120000-{SUFFIX}")
    algo = FakeAlgo(mismatch_dirs=[old])

    assert load_resume_checkpoint(algo, candidates) is None
