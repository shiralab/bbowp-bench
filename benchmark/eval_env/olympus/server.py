import importlib
import logging
import pathlib
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

app = FastAPI()
objectives = {}


class EvalRequest(BaseModel):
    params: dict[str, Any]


def load_objective(task_name: str):
    module_path = pathlib.Path(f"/app/{task_name.split('_', 1)[1]}/actual_objective.py")
    logger.info(f"Loading objective from: {module_path}")
    if not module_path.exists():
        raise FileNotFoundError(f"Objective file not found: {module_path}")

    try:
        spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.ActualObjective()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load task '{task_name}': {e}")


@app.post("/evaluate/{task_name}")
def evaluate(task_name: str, req: EvalRequest):
    try:
        if task_name not in objectives:
            logger.info(f"Loading objective for task: {task_name}")
            objectives[task_name] = load_objective(task_name)
        objective = objectives[task_name]
        result = objective(req.params)
        return {"score": float(result)}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
