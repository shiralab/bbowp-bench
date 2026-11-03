from __future__ import annotations

import warnings
from pathlib import Path

import pytest

"""
search pairs of (task.json, eval_config.json) in {BENCHMARK_ROOT} and inject them as pytest fixtures.
"""

BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark" / "config"
TASK_JSON = "task.json"
EVAL_CONFIG_JSON = "eval_config.json"


def collect_task_dirs(root: Path) -> list[Path]:
    task_dirs = []
    if not root.exists():
        warnings.warn(
            f"Benchmark root does not exist: {root}",
            UserWarning,
            stacklevel=2,
        )
        return task_dirs

    list_of_domain_dirs = sorted(root.iterdir())
    print(f"Found {len(list_of_domain_dirs)} domain directories.")
    for domain_dir in list_of_domain_dirs:
        if not domain_dir.is_dir():
            continue

        list_of_task_dirs = sorted(domain_dir.iterdir())
        print(
            f"Domain '{domain_dir.name}': Found {len(list_of_task_dirs)} task directories."
        )
        for task_dir in list_of_task_dirs:
            if not task_dir.is_dir():
                continue
            task_dirs.append(task_dir)

    print(f"Total collected task directories: {len(task_dirs)} to go through.")
    return task_dirs


TASK_DIRS = collect_task_dirs(BENCHMARK_ROOT)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "task_dir" in metafunc.fixturenames:
        metafunc.parametrize(
            "task_dir",
            TASK_DIRS,
            ids=[f"{d.parent.name}/{d.name}" for d in TASK_DIRS],
        )


# ---------------------------------------------------------------------------
# raw JSON fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def task_json_raw(task_dir: Path) -> str:
    path = task_dir / TASK_JSON
    if not path.exists():
        pytest.fail(f"task.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def eval_config_json_raw(task_dir: Path) -> str:
    path = task_dir / EVAL_CONFIG_JSON
    if not path.exists():
        pytest.fail(f"eval_config.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Pydantic model fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def task_model(task_json_raw: str):
    """parse task.json into Pydantic model and return it."""
    try:
        from bbowp.schema.bbowp import BBOWP

        return BBOWP.model_validate_json(task_json_raw)
    except ImportError:
        pytest.fail("BBOWP schema not importable; cannot create task_model fixture")


@pytest.fixture
def eval_config_model(eval_config_json_raw: str):
    """parse eval_config.json into Pydantic model and return it."""
    try:
        from bbowp.schema.bbowp import BBOWPEvalConfig

        return BBOWPEvalConfig.model_validate_json(eval_config_json_raw)
    except ImportError:
        pytest.fail(
            "BBOWPEvalConfig schema not importable; cannot create eval_config_model fixture"
        )
