import json
import logging
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append("/app/MECHBench")
from MECHBench.src import sob

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

app = FastAPI(
    title="MECHBench API Server",
    description="API for running OpenRadioss simulations in containerized environment",
    version="1.0.0",
)

ORSS_MAIN_PATH = "/app/OpenRadioss/"
sim_id = 175

task_info_dict = {
    "Star_shaped_crash_box": {"id": 1, "metric": "penalized_sea"},
    "Three_point_bending_model": {"id": 2, "metric": "penalized_mass"},
    "Long_crash_tube": {"id": 3, "metric": "load_uniformity"},
}


class EvalRequest(BaseModel):
    params: dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str
    service: str
    orss_path: str


def get_valid_params(params: dict[str, float], task_name: str) -> list[float]:
    """
    Scale parameters to [-5, 5] range with boundary clipping
    Args:
        params: Dictionary of parameter names and values
        task_name: Task name used to identify the JSON file
    Returns:
        List of parameter values scaled to [-5, 5]
    Raises:
        ValueError: If required parameters are missing
    """
    task_json_path = f"/app/simulator_variable_bounds/{task_name}.json"
    with open(task_json_path, "r") as f:
        task_bounds = json.load(f)

    scaled_args = {}

    for variable in task_bounds["search_space"]["variables"]:
        key = variable["actual_name"]
        lower_bound = variable["low"]
        upper_bound = variable["high"]

        # validate all required parameters are provided
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

        value = params.get(key)
        value_clipped = max(min(value, upper_bound), lower_bound)
        scaled_args[key] = (
            10 * (value_clipped - lower_bound) / (upper_bound - lower_bound) - 5
        )

    return list(scaled_args.values())


def merge_runner_options(num_threads: int) -> dict[str, Any]:
    """
        Args:
        num_threads: Number of threads to use for the simulation
    Returns:
        Merged runner options dictionary
    """
    merged_options = {
        "open_radioss_main_path": ORSS_MAIN_PATH,
        "write_vtk": True,
        "np": 8,  # TODO make configurable
        "nt": 1,
        "h_level": 1,
        "gmsh_verbosity": 0,
    }
    merged_options["nt"] = num_threads

    return merged_options


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
def health_check():
    # Health check endpoint for container startup verification.
    # Used by BenchmarkContainer._wait_for_server() to confirm API server is ready.

    return HealthResponse(
        status="healthy", service="MECHBench", orss_path=ORSS_MAIN_PATH
    )


@app.post("/evaluate/{task_name}")
def evaluate(task_name: str, req: EvalRequest):
    task_id, objective_key = (
        task_info_dict.get(task_name, {}).get("id"),
        task_info_dict.get(task_name, {}).get("metric"),
    )

    try:
        runner_options = merge_runner_options(num_threads=1)

        valid_params = get_valid_params(req.params, task_name)

        f = sob.get_problem(
            task_id,
            len(valid_params),  # dimension
            runner_options,
            output_data=objective_key,  # see documentation for get_problem
            sequential_id_numbering=False,
        )

        return {"score": float(f(valid_params, sim_id))}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found.")
    except ValueError as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Simulation execution failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e!s}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
