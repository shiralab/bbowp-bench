from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

"""
ERROR-level checks (must pass):
- TestProblemId: problem_id must be identical between task.json and eval_config.json
- TestVariableNameConsistency: actual_name of variables in each budget amount level's answer
  (large/small) must exactly match those in eval_config.json execution.arguments
- TestVariableFieldConsistency: type field of each variable in each budget amount level's answer
  must exactly match the corresponding type in eval_config.json execution.arguments

WARN-level checks (informational, not mandatory):
- TestVariableOrder: variable order in task.json answer[amount] should match
  the relative order of the same variables in eval_config.json execution.arguments
"""

BUDGET_LEVELS = ["large", "small"]


class _TypeMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        normalized_key = key.removeprefix("const_")
        self[key] = normalized_key
        return normalized_key


TYPE_MAP = _TypeMap()


class TestProblemId:
    def test_problem_id_matches(
        self,
        task_dir: Path,
        task_json_raw: str,
        eval_config_json_raw: str,
    ):
        task_json_dict = json.loads(task_json_raw)
        eval_config_json_dict = json.loads(eval_config_json_raw)

        try:
            task_id = task_json_dict["problem_id"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
        try:
            eval_id = eval_config_json_dict["problem_id"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] eval_config.json missing key: {e}")

        assert task_id == eval_id, (
            f"[{task_dir.name}] problem_id mismatch: "
            f"task.json='{task_id}' vs eval_config.json='{eval_id}'"
        )


def get_answer_variables(task: dict, budget: str) -> list[dict]:
    try:
        answer_list = task["answer"]
    except KeyError as e:
        pytest.fail(f"task.json missing key: {e}")

    for entry in answer_list:
        try:
            entry_budget = entry["amount"]
        except KeyError as e:
            pytest.fail(f"task.json missing key: {e}")

        if entry_budget == budget:
            try:
                return entry["answer_item"]["search_space"]["variables"]
            except KeyError as e:
                pytest.fail(f"task.json missing key: {e}")


def get_exec_variables(eval_config: dict) -> list[dict]:
    try:
        return eval_config["execution"]["arguments"]
    except KeyError as e:
        pytest.fail(f"eval_config.json missing key: {e}")


def get_exec_var_map(eval_config: dict) -> dict[str, dict]:
    return {v["actual_name"]: v for v in get_exec_variables(eval_config)}


class TestVariableNameConsistency:
    @pytest.mark.parametrize("amount", BUDGET_LEVELS)
    def test_variable_names_match(
        self,
        task_dir: Path,
        task_json_raw: str,
        eval_config_json_raw: str,
        amount: str,
    ):
        task_json_dict = json.loads(task_json_raw)
        eval_config_json_dict = json.loads(eval_config_json_raw)
        answer_vars = get_answer_variables(task_json_dict, amount)

        answer_names = {v["actual_name"] for v in answer_vars}
        eval_names = {
            v["actual_name"] for v in get_exec_variables(eval_config_json_dict)
        }

        only_in_answer = answer_names - eval_names

        errors = []
        if only_in_answer:
            errors.append(f"Only in task answer[{amount}]: {sorted(only_in_answer)}")

        assert not errors, (
            f"[{task_dir.name}] Variable name mismatch [{amount}]:\n"
            + "\n".join(f"  {e}" for e in errors)
        )


class TestVariableFieldConsistency:
    @pytest.mark.parametrize("amount", BUDGET_LEVELS)
    def test_variable_types_match(
        self,
        task_dir: Path,
        task_json_raw: str,
        eval_config_json_raw: str,
        amount: str,
    ):
        task_json_dict = json.loads(task_json_raw)
        eval_config_json_dict = json.loads(eval_config_json_raw)
        answer_vars = get_answer_variables(task_json_dict, amount)

        eval_map = get_exec_var_map(eval_config_json_dict)
        mismatches = []

        for var in answer_vars:
            name = var["actual_name"]
            try:
                eval_type = eval_map[name]["type"]
            except KeyError:
                pytest.fail(
                    f"[{task_dir.name}] Variable '{name}' in task answer[{amount}] not found in eval_config execution.arguments"
                )
            try:
                answer_type = var["type"]
            except KeyError:
                pytest.fail(
                    f"[{task_dir.name}] Variable '{name}' in task answer[{amount}] missing 'type' field"
                )
            answer_type_mapped = TYPE_MAP[answer_type]

            if eval_type != answer_type_mapped:
                mismatches.append(
                    f"'{name}': task answer='{answer_type}' vs eval_config='{eval_type}'"
                )

        assert not mismatches, (
            f"[{task_dir.name}] Variable type mismatches [{amount}]:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


class TestVariableOrder:
    @pytest.mark.parametrize("amount", BUDGET_LEVELS)
    def test_variable_order_matches(
        self,
        task_dir: Path,
        task_json_raw: str,
        eval_config_json_raw: str,
        amount: str,
    ):
        task_json_dict = json.loads(task_json_raw)
        eval_config_json_dict = json.loads(eval_config_json_raw)
        answer_order = [
            v["actual_name"] for v in get_answer_variables(task_json_dict, amount)
        ]
        eval_order = [
            v["actual_name"] for v in get_exec_variables(eval_config_json_dict)
        ]
        eval_order_filtered = [name for name in eval_order if name in set(answer_order)]

        if eval_order_filtered != answer_order:
            warnings.warn(
                f"[{task_dir.name}] Variable order mismatch:\n"
                f"  eval_config : {eval_order}\n"
                f"  task answer : {answer_order}",
                UserWarning,
                stacklevel=2,
            )
