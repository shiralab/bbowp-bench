from __future__ import annotations

import json
from pathlib import Path

import pytest

from bbowp.util.preprocess import convert_snake_case

"""
ERROR-level checks (must pass):
- TestBounds: for numeric variables in eval_config.json execution.arguments,
  low and high must satisfy low < high
- TestBudget: task.json budget must contain all three amounts (small/large),
  and their max_eval_value must satisfy small < large
- TestAliases: for each variable in eval_config.json execution.arguments:
  - actual_name must be included in its own aliases list
  - no duplicate aliases within the same variable
  - no alias shared across multiple variables
- TestPlaceholder: for each difficulty level's description.text in task.json:
  - {MAX_EVAL_VALUE} placeholder must be present
  - {MAX_EVAL_VALUE} placeholder must be substitutable without KeyError

WARN-level checks (should pass, but not mandatory):
- TestDifficultyCoverage (currently commented out): description and answer in task.json
  should each contain all three difficulty levels (hard/intermediate/easy)
"""


DIFFICULTIES = ("hard", "intermediate", "easy")
BUDGET_AMOUNTS = ("small", "large")
PLACEHOLDER = "{MAX_EVAL_VALUE}"
TODO_TOKENS = frozenset({"TODO", "TBA", "todo", "tba"})


def get_exec_variables(eval_config: dict) -> list[dict]:
    try:
        return eval_config["execution"]["arguments"]
    except KeyError as e:
        pytest.fail(f"eval_config.json missing key: {e}")


def get_budget_map(task: dict) -> dict[str, int]:
    try:
        budget_list = task["budget"]
    except KeyError as e:
        pytest.fail(f"task.json missing key: {e}")

    return {
        b["amount"]: b["max_eval_value"]
        for b in budget_list
        if "amount" in b and "max_eval_value" in b
    }


class TestBounds:
    def test_low_less_than_high(self, task_dir: Path, eval_config_json_raw: str):
        eval_config_dict = json.loads(eval_config_json_raw)
        for var in get_exec_variables(eval_config_dict):
            if var["type"] in ("float", "int"):
                try:
                    low = var["low"]
                    high = var["high"]
                except KeyError as e:
                    pytest.fail(f"eval_config.json missing key: {e}")
                if low is None or high is None:
                    continue  # covered by test_schema.py

                assert low < high, (
                    f"[{task_dir.name}] Variable '{var['actual_name']}': "
                    f"low ({low}) must be less than high ({high})"
                )


class TestBudget:
    def test_budget_has_all_amounts(self, task_dir: Path, task_json_raw: str):
        task_dict = json.loads(task_json_raw)
        try:
            amounts = [b["amount"] for b in task_dict["budget"]]
        except KeyError as e:
            pytest.fail(f"task.json missing key: {e}")
        for required in BUDGET_AMOUNTS:
            assert required in amounts, (
                f"[{task_dir.name}] budget is missing amount: '{required}'"
            )

    def test_budget_ordering(self, task_dir: Path, task_json_raw: str):
        task_dict = json.loads(task_json_raw)
        bmap = get_budget_map(task_dict)
        if not all(k in bmap for k in BUDGET_AMOUNTS):
            pytest.skip(
                "budget amounts incomplete; "
            )  # covered by test_budget_has_all_amounts

        assert bmap["small"] < bmap["large"], (
            f"[{task_dir.name}] budget amounts are not in ascending order: "
            f"small ({bmap['small']}) >= large ({bmap['large']})"
        )


def _check_duplicate_aliases_within_list(
    aliases: list[str], preprocess_fn=convert_snake_case
) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for alias in [preprocess_fn(alias) for alias in aliases]:
        if alias in seen:
            duplicates.append(alias)
        seen.add(alias)

    return duplicates


class TestAliases:
    def test_actual_name_in_own_aliases(
        self, task_dir: Path, eval_config_json_raw: str
    ):
        """checking if actual_name is included in its own aliases list"""
        eval_config_dict = json.loads(eval_config_json_raw)
        for var in get_exec_variables(eval_config_dict):
            name = var["actual_name"]
            try:
                aliases = var["aliases"]
            except KeyError as e:
                pytest.fail(f"eval_config.json missing key: {e}")

            assert name in aliases, (
                f"[{task_dir.name}] Variable '{name}': actual_name must be included in its own aliases list"
            )

    def test_no_duplicate_aliases_within_variable(
        self, task_dir: Path, eval_config_json_raw: str
    ):
        """checking duplicates within the same variable"""
        vars_duplicates: dict[str, list[str]] = {}
        eval_config_dict = json.loads(eval_config_json_raw)
        for var in get_exec_variables(eval_config_dict):
            try:
                aliases = var["aliases"]
            except KeyError as e:
                pytest.fail(f"eval_config.json missing key: {e}")

            duplicates = _check_duplicate_aliases_within_list(
                aliases, preprocess_fn=convert_snake_case
            )

            if len(duplicates) > 0:
                vars_duplicates[var["actual_name"]] = duplicates

        assert len(vars_duplicates) == 0, (
            f"[{task_dir.name}] Variables with duplicate aliases: \n"
            + "\n".join(
                f"  - {var}: {aliases}" for var, aliases in vars_duplicates.items()
            )
        )

    def test_no_cross_variable_alias_collision(
        self, task_dir: Path, eval_config_json_raw: str
    ):
        """checking if any alias is shared by multiple variables"""

        alias_to_vars: dict[str, list[str]] = {}

        eval_config_dict = json.loads(eval_config_json_raw)
        for var in get_exec_variables(eval_config_dict):
            var_name = var["actual_name"]
            try:
                preprocess_fn = convert_snake_case
                aliases_set = set([preprocess_fn(alias) for alias in var["aliases"]])
            except KeyError as e:
                pytest.fail(f"eval_config.json missing key: {e}")

            for alias in aliases_set:
                alias_to_vars.setdefault(alias, []).append(var_name)

        collisions = [
            f"'{alias}' is shared by {owners}"
            for alias, owners in alias_to_vars.items()
            if len(owners) > 1
        ]

        assert len(collisions) == 0, (
            f"[{task_dir.name}] Cross-variable alias collisions found:\n"
            + "\n".join(f"  - {c}" for c in collisions)
        )

    def test_no_duplicate_aliases_within_choice(
        self, task_dir: Path, eval_config_json_raw: str
    ):
        """checking duplicates within the same choice for categorical variables"""
        eval_config_dict = json.loads(eval_config_json_raw)
        choice_duplicates: dict[str, list[str]] = {}
        for var in get_exec_variables(eval_config_dict):
            if var["type"] != "cat":
                continue

            name = var["actual_name"]
            for choice in var["choices"]:
                choice_name = choice["actual_name"]
                duplicates = _check_duplicate_aliases_within_list(
                    choice["aliases"], preprocess_fn=convert_snake_case
                )
                if len(duplicates) > 0:
                    choice_duplicates[f"{name} -> {choice_name}"] = duplicates

        assert len(choice_duplicates) == 0, (
            f"[{task_dir.name}] Choices with duplicate aliases: \n"
            + "\n".join(
                f"  - {choice}: {aliases}"
                for choice, aliases in choice_duplicates.items()
            )
        )

    def test_no_cross_variable_alias_collision_in_choices(
        self, task_dir: Path, eval_config_json_raw: str
    ):
        """checking if any alias is shared by multiple choices within one categorical variables"""
        eval_config_dict = json.loads(eval_config_json_raw)
        collisions: list[str] = []
        for var in get_exec_variables(eval_config_dict):
            if var["type"] != "cat":
                continue

            var_name = var["actual_name"]
            alias_to_choices: dict[str, list[str]] = {}
            for choice in var["choices"]:
                preprocess_fn = convert_snake_case
                aliases_set = set([preprocess_fn(alias) for alias in choice["aliases"]])
                for alias in aliases_set:
                    alias_to_choices.setdefault(alias, []).append(
                        f"{var_name} -> {choice['actual_name']}"
                    )

            collisions.extend(
                [
                    f"'{alias}' is shared by {owners}"
                    for alias, owners in alias_to_choices.items()
                    if len(owners) > 1
                ]
            )

        assert len(collisions) == 0, (
            f"[{task_dir.name}] Cross-variable alias collisions found:\n"
            + "\n".join(f"  - {c}" for c in collisions)
        )


class TestPlaceholder:
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_placeholder_formattable(
        self, task_dir: Path, task_json_raw: str, difficulty: str
    ):
        """
        - checking existense {MAX_EVAL_VALUE} in description > text for every difficulty
        - checking if the placeholder can be substituted with an actual value without KeyError or leaving the placeholder intact.
        """
        task_json_dict = json.loads(task_json_raw)
        try:
            description_entries = task_json_dict["description"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")

        for desc in description_entries:
            try:
                desc_difficulty = desc["difficulty"]
            except KeyError as e:
                pytest.fail(f"[{task_dir.name}] task.json missing key: {e}")
            if desc_difficulty == difficulty:
                try:
                    text = desc["text"]
                except KeyError:
                    pytest.fail(
                        f"[{task_dir.name}] {difficulty} description is missing 'text' field"
                    )

                if sum(token in text for token in TODO_TOKENS) > 0:
                    pytest.xfail(
                        f"[{task_dir.name}] {difficulty} description contains {text}, caught by test_completeness.py"
                    )

                assert PLACEHOLDER in text, (
                    f"[{task_dir.name}] {difficulty} description must contain '{PLACEHOLDER}', got:\n{text}"
                )
                try:
                    result = text.format(MAX_EVAL_VALUE=1000)

                    assert PLACEHOLDER not in result, (
                        f"[{task_dir.name}] {difficulty} description: Placeholder '{PLACEHOLDER}' was not substituted after format(), got:\n{text}"
                    )

                except KeyError as e:
                    pytest.fail(
                        f"[{task_dir.name}] {difficulty} description: Unexpected placeholder key: {e}\ntext: {text}"
                    )
