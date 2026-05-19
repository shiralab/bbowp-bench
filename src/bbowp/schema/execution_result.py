from typing import Any

from pydantic import BaseModel


class ExecutionResult(BaseModel, frozen=True):
    best_params: dict[str, Any]
    best_fval: float
