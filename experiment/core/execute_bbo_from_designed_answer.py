import json
import logging
import pathlib

from pydantic import ValidationError

from bbowp.executor import Executor
from bbowp.schema.designed_answer import DesignedAnswer
from bbowp.util.container import BenchmarkContainer
from bbowp.util.objective import Objective
from bbowp.util.preprocess import convert_snake_case
from bbowp.util.task_loader import get_task

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def setup_logger(log_path: str | pathlib.Path) -> logging.Logger:
    logger_name = f"{log_path.parent!s}"
    run_logger = logging.getLogger(logger_name)
    run_logger.setLevel(logging.DEBUG)
    run_logger.propagate = False

    if not run_logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        file_fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)
        run_logger.addHandler(file_handler)

    return run_logger


def get_image_name(execution) -> str:
    image_name = execution.execution.docker_image
    return image_name


def execute_bbo_from_designed_answer(
    designed_answer_path: str | pathlib.Path,
    task_name: str,
    budget: str,
    seed: int,
    data_dir: str | pathlib.Path = "benchmark/config",
):
    output_dir = (
        pathlib.Path(designed_answer_path).parent / f"optimization_seed_{seed:02}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "optimization.log"
    logger = setup_logger(log_path)

    # load path of designed answer
    designed_answer_path = pathlib.Path(designed_answer_path)
    try:
        designed_answer_json_str = designed_answer_path.read_text(encoding="utf-8")
    except IsADirectoryError:
        logger.error(
            f"Designed answer path is a directory, expected a file: {designed_answer_path}"
        )
        raise
    except FileNotFoundError:
        logger.error(f"Designed answer file not found: {designed_answer_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading designed answer: {e}")
        raise

    # load designed answer
    try:
        designed_answer = DesignedAnswer.model_validate_json(designed_answer_json_str)
    except ValidationError as e:
        logger.error(f"Validation error in designed answer: {e}")
        raise

    # load task
    dataset_name = task_name.split("_")[0]
    task_path = pathlib.Path(data_dir) / dataset_name / task_name

    execution, task = get_task(task_path)
    image_name = get_image_name(execution)
    direction = execution.target.direction

    budget_amount = next((b for b in task.budget if b.amount == budget), None)
    if budget_amount is None:
        available_budgets = ", ".join(b.amount for b in task.budget)
        logger.error(
            f"Unknown budget '{budget}' for task '{task_name}'. Available budgets: {available_budgets}"
        )
    max_eval_value = budget_amount.max_eval_value

    # start optimization
    try:
        with BenchmarkContainer(image_name, logger) as (host, port):
            objective = Objective(execution, host, port, logger)
            executor = Executor(
                objective,
                direction,
                max_eval_value,
                designed_answer,
                seed,
                logger,
                preprocess_name_fn=convert_snake_case,
            )
            result = executor.execute()

        if max_eval_value != objective.n_calls:
            raise RuntimeError(
                f"Number of evaluations ({objective.n_calls}) does not match given budget ({max_eval_value})."
            )

        # save result
        output_path = output_dir / "optimization_result.json"
        result_content = {
            "best_params": result.best_params,
            "best_fval": result.best_fval,
        }
        with open(output_path, "w") as f:
            json.dump(result_content, f, indent=4)
    except Exception:
        logger.exception(f"Optimization failed for {designed_answer_path}.")
        raise
