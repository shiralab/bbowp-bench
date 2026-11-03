from __future__ import annotations

import json
from pathlib import Path

import pytest

"""
ERROR-level checks (must pass):
- TestPydanticParse: task.json and eval_config.json must be parseable by their respective
  Pydantic models (BBOWP and BBOWPEvalConfig)
- TestVariableSchema: each variable in eval_config.json execution.arguments must satisfy:
  - type is one of ("float", "int", "cat", "bool")
  - cat variables must have non-empty choices, and low/high must be null
  - float/int variables must have choices=null, and non-null low/high
- TestTargetSchema: target.direction must be one of ("minimize", "maximize")
"""


def get_exec_variables(eval_config: dict) -> list[dict]:
    try:
        return eval_config["execution"]["arguments"]
    except KeyError as e:
        pytest.fail(f"eval_config.json missing key: {e}")


class TestPydanticParse:
    def test_task_json_parses(self, task_dir: Path, task_json_raw: str):
        """checking if task.json can be parsed by Pydantic model."""

        try:
            from bbowp.schema.bbowp import BBOWP

            BBOWP.model_validate_json(task_json_raw)
        except ImportError:
            pytest.fail("BBOWP not importable")
        except Exception as e:
            pytest.fail(f"[{task_dir.name}] task.json Pydantic parse failed: {e}")

    def test_eval_config_json_parses(self, task_dir: Path, eval_config_json_raw: str):
        """checking if eval_config.json can be parsed by Pydantic model."""
        try:
            from bbowp.schema.bbowp import BBOWPEvalConfig

            BBOWPEvalConfig.model_validate_json(eval_config_json_raw)
        except ImportError:
            pytest.fail("BBOWPEvalConfig not importable")
        except Exception as e:
            pytest.fail(
                f"[{task_dir.name}] eval_config.json Pydantic parse failed: {e}"
            )


class TestVariableSchema:
    def test_variable_types_valid(self, task_dir: Path, eval_config_json_raw: str):
        try:
            from bbowp.schema.bbowp import BBOWPEvalConfig

            _ = BBOWPEvalConfig.model_validate_json(eval_config_json_raw)
        except ImportError:
            pytest.fail("BBOWPEvalConfig not importable")
        except Exception as e:
            pytest.fail(f"[{task_dir.name}] Variable schema validation failed: {e}")


class TestTargetSchema:
    def test_direction_valid(self, task_dir: Path, eval_config_json_raw: str):
        """checking if target.direction is valid."""
        eval_config_json_dict = json.loads(eval_config_json_raw)

        try:
            direction = eval_config_json_dict["target"]["direction"]
        except KeyError as e:
            pytest.fail(f"[{task_dir.name}] eval_config.json missing key: {e}")

        assert direction in ("minimize", "maximize"), (
            f"[{task_dir.name}] target.direction must be one of ('minimize', 'maximize'), got '{direction}'"
        )
