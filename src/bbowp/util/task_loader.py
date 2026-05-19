import pathlib

from bbowp.schema.bbowp import BBOWP, BBOWPEvalConfig


def get_task(task_path: pathlib.Path) -> tuple[BBOWPEvalConfig, BBOWP]:
    execution_json_path = task_path / "eval_config.json"
    if not execution_json_path.exists():
        raise FileNotFoundError(f"Task JSON file not found: {execution_json_path}")
    execution = BBOWPEvalConfig.model_validate_json(
        execution_json_path.read_text(encoding="utf-8")
    )

    task_path = task_path / "task.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Task JSON file not found: {task_path}")
    task = BBOWP.model_validate_json(task_path.read_text(encoding="utf-8"))

    return execution, task
