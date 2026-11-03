from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

"""
ERROR-level checks (must pass):
- TestFileExistence: task.json / eval_config.json file pair existence

WARN-level checks (should pass, but not mandatory):
- TestAlgorithmCompleteness: TODO/TBA tokens in algorithm names/reasons and search_space reasons
- TestDescriptionCompleteness: TBA tokens in description texts
- TestComprehensiveCompleteness: Any TODO/TBA tokens anywhere in task.json or eval_config.json (comprehensive scan)
"""

TODO_TOKENS = ("TODO", "TBA", "todo", "tba")
TASK_JSON = "task.json"
EVAL_CONFIG_JSON = "eval_config.json"


def _warn(msg: str) -> None:
    warnings.warn(msg, UserWarning, stacklevel=3)


class TestFileExistence:
    def test_task_json_exists(self, task_dir: Path):
        if not (task_dir / TASK_JSON).exists():
            raise FileNotFoundError(
                f"[{task_dir.name}] task.json not found in {task_dir}"
            )

    def test_eval_config_json_exists(self, task_dir: Path):
        if not (task_dir / EVAL_CONFIG_JSON).exists():
            raise FileNotFoundError(
                f"[{task_dir.name}] eval_config.json not found in {task_dir}"
            )


class TestDescriptionCompleteness:
    def test_no_todo_in_description(self, task_dir: Path, task_json_raw: str):
        task_json_raw_dict = json.loads(task_json_raw)
        try:
            description_entries = task_json_raw_dict["description"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

        for entry in description_entries:
            try:
                difficulty = entry["difficulty"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            try:
                text = entry["text"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            if text in TODO_TOKENS:
                _warn(
                    f"[{task_dir.name}] description[{difficulty}].text is '{text}' — implementation pending"
                )


class TestAlgorithmCompleteness:
    def test_no_todo_in_variables_search_space(
        self, task_dir: Path, task_json_raw: str
    ):
        task_json_raw_dict = json.loads(task_json_raw)

        try:
            answer_entries = task_json_raw_dict["answer"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

        for entry in answer_entries:  # answer_item
            try:
                amount = entry["amount"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            try:
                search_space = entry["answer_item"]["search_space"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            assert search_space not in TODO_TOKENS, (
                f"[{task_dir.name}] answer[{amount}].search_space is '{search_space}'"
            )

            try:
                variables = search_space["variables"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            is_variables_todo = (
                (isinstance(variables, list) and variables[0] in TODO_TOKENS)
                or variables in TODO_TOKENS
                or variables == []
            )
            assert not is_variables_todo, (
                f"[{task_dir.name}] answer[{amount}].search_space.variables is '{variables}'"
            )

            try:
                reason = search_space["reason"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            assert reason not in TODO_TOKENS and reason != "", (
                f"[{task_dir.name}] answer[{amount}].search_space.reason is '{reason}'"
            )

    def test_no_todo_in_search_space_reason(self, task_dir: Path, task_json_raw: str):
        task_json_raw_dict = json.loads(task_json_raw)

        try:
            answer_entries = task_json_raw_dict["answer"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

        for entry in answer_entries:
            try:
                amount = entry["amount"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                reason = entry["answer_item"]["search_space"]["reason"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            assert reason not in TODO_TOKENS and reason != "", (
                f"[{task_dir.name}] answer[{amount}].search_space.reason is '{reason}'"
            )

    def test_no_todo_in_algorithm_name(self, task_dir: Path, task_json_raw: str):
        task_json_raw_dict = json.loads(task_json_raw)
        try:
            answer_entries = task_json_raw_dict["answer"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
        for entry in answer_entries:
            try:
                amount = entry["amount"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                algo = entry["answer_item"]["algorithm"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                name = algo["name"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            assert name not in TODO_TOKENS and name != "", (
                f"[{task_dir.name}] answer[{amount}].algorithm name is '{name}'"
            )

    def test_no_todo_in_algorithm_reason(self, task_dir: Path, task_json_raw: str):
        task_json_raw_dict = json.loads(task_json_raw)

        try:
            answer_entries = task_json_raw_dict["answer"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

        for entry in answer_entries:
            try:
                amount = entry["amount"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                algo = entry["answer_item"]["algorithm"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                reason = algo["reason"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

            assert reason not in TODO_TOKENS and reason != "", (
                f"[{task_dir.name}] answer[{amount}].algorithm reason is '{reason}'"
            )

    def test_warn_score_is_in_todo_tokens(self, task_dir: Path, task_json_raw: str):
        task_json_raw_dict = json.loads(task_json_raw)

        try:
            answer_entries = task_json_raw_dict["answer"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

        for entry in answer_entries:
            try:
                amount = entry["amount"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                algo = entry["answer_item"]["algorithm"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            try:
                score = algo["score"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            if score in TODO_TOKENS:
                _warn(
                    f"[{task_dir.name}] answer[{amount}].algorithm score "
                    f"is '{algo['score']}' — implementation pending"
                )


def _find_todos_in_obj(obj, path: str = "") -> list[str]:
    """
    - recursively search for TODO/TBA tokens in any string value within the given object
    - returning a list of paths to the hits.
    """

    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(_find_todos_in_obj(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(_find_todos_in_obj(item, f"{path}[{i}]"))
    elif isinstance(obj, str) and obj in TODO_TOKENS:
        hits.append(f"{path} = '{obj}'")
    return hits


class TestOverallCompleteness:
    def test_no_todo_anywhere_in_task_json(self, task_dir: Path, task_json_raw: str):
        task_json_raw_dict = json.loads(task_json_raw)
        hits = _find_todos_in_obj(task_json_raw_dict)
        for hit in hits:
            _warn(f"[{task_dir.name}] TODO/TBA in task.json: {hit}")

    def test_no_todo_anywhere_in_eval_config(
        self, task_dir: Path, eval_config_json_raw: str
    ):
        eval_config_json_raw_dict = json.loads(eval_config_json_raw)
        hits = _find_todos_in_obj(eval_config_json_raw_dict)
        for hit in hits:
            _warn(f"[{task_dir.name}] TODO/TBA in eval_config.json: {hit}")
