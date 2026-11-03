import csv
import logging
import pathlib

import numpy as np
from scipy.stats import qmc

from bbowp.schema.designed_answer import DesignedAnswer
from bbowp.schema.variable_type import VariableType
from bbowp.util.container import BenchmarkContainer
from bbowp.util.objective import Objective
from bbowp.util.preprocess import convert_snake_case
from bbowp.util.task_loader import get_task

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def sample_point_and_score(
    task_name: str,
    designed_answer: DesignedAnswer,
    n_sample: int,
    out_dir: str | pathlib.Path,
    data_dir: str | pathlib.Path = "benchmark/config",
):
    points = sample_uniform_random_points(designed_answer, n_sample)

    # load task
    dataset_name = task_name.split("_")[0]
    task_path = pathlib.Path(data_dir) / dataset_name / task_name

    execution, _ = get_task(task_path)
    image_name = get_image_name(execution)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "uniform_random_sample.csv"

    data = []
    header_var_name = list(points[0].keys())
    # start optimization
    try:
        with BenchmarkContainer(image_name, logger) as (host, port):
            objective = Objective(execution, host, port, logger)
            for i, var in enumerate(points):
                score = objective(var, convert_snake_case)
                data.append({**var, "score": score})
                if (i + 1) % 10000 == 0:
                    logger.info(f"Sampled and scored {i + 1} points so far...")
                    with open(
                        out_path,
                        mode="w",
                        encoding="utf-8",
                        newline="",
                    ) as f:
                        fieldnames = header_var_name + ["score"]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
    except Exception:
        logger.exception("Failed.")
        raise

    with open(
        out_path,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = header_var_name + ["score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"Saved sampled points and scores to {out_path}.")


def get_image_name(execution) -> str:
    image_name = execution.execution.docker_image
    return image_name


def sample_sobol_points(
    designed_answer: DesignedAnswer,
    n_sample: int,
) -> list[dict]:
    """
    Sample n points from the search space using a Sobol sequence.
    """

    def map_continuous(
        low: float, high: float, x: float, log_scale: bool = False
    ) -> float:
        """
        Map a [0, 1) sample x to a continuous range.
        """
        if log_scale:
            low_log = np.log(low)
            high_log = np.log(high)
            return float(np.exp(low_log + x * (high_log - low_log)))
        else:
            return float(low + x * (high - low))

    def map_integer(low: int, high: int, x: float, log_scale: bool = False) -> int:
        """
        Map a [0, 1) sample x to an integer range.
        """
        if log_scale:
            # map in log space with widened bounds to reduce discretization bias
            safe_low = max(1e-10, low - 0.5)
            low_log = np.log(safe_low)
            high_log = np.log(high + 0.5)
            val = np.exp(low_log + x * (high_log - low_log))
            return int(np.clip(np.round(val), low, high))
        else:
            # scale then floor so each integer in [low, high] can be selected
            val = int(np.floor(low + x * (high - low + 1)))
            return min(val, high)

    def map_categorical(categories: list, x: float):
        """
        Map a [0, 1) sample x to a categorical choice.
        """
        n_categories = len(categories)
        idx = int(np.floor(x * n_categories))
        idx = min(idx, n_categories - 1)
        return categories[idx]

    variables = designed_answer.search_space.variables

    # split searchable variables from constants to determine the Sobol dimension
    active_vars = []
    const_vars = []
    for var in variables:
        if var.type in [
            VariableType.FLOAT,
            VariableType.INT,
            VariableType.CATEGORICAL,
            VariableType.BOOL,
        ]:
            active_vars.append(var)
        else:
            const_vars.append(var)

    D = len(active_vars)

    # generate a D-dimensional Sobol sequence
    if D > 0:
        sampler = qmc.Sobol(d=D, scramble=True)
        # Shape: (n_sample, D), range: [0, 1)
        sobol_sequence = sampler.random(n=n_sample)
    else:
        sobol_sequence = np.zeros((n_sample, 0))

    # map [0, 1) samples into the actual variable domains
    sampled_points = []
    for i in range(n_sample):
        point_variables = {}
        vector = sobol_sequence[i]

        # map active variables
        for dim_idx, var in enumerate(active_vars):
            x = vector[dim_idx]  # [0, 1) sample

            if var.type == VariableType.FLOAT:
                value = map_continuous(var.low, var.high, x, log_scale=var.log)
            elif var.type == VariableType.INT:
                value = map_integer(var.low, var.high, x, log_scale=var.log)
            elif var.type == VariableType.CATEGORICAL:
                choices = [choice.actual_name for choice in var.choices]
                value = map_categorical(choices, x)
            elif var.type == VariableType.BOOL:
                value = map_categorical([True, False], x)

            point_variables[var.actual_name] = value

        # assign constant variables
        for var in const_vars:
            point_variables[var.actual_name] = var.value

        sampled_points.append(point_variables)

    return sampled_points


def sample_uniform_random_points(
    designed_answer: DesignedAnswer,
    n_sample: int,
) -> list[dict]:
    """
    Sample n points from the search space uniformly at random.
    """
    variables = designed_answer.search_space.variables
    sampled_points = []
    for _ in range(n_sample):
        point_variables = {}
        for var in variables:
            if var.type == VariableType.FLOAT:
                if var.log:
                    low_log = np.log(var.low)
                    high_log = np.log(var.high)
                    value = float(np.exp(np.random.uniform(low_log, high_log)))
                else:
                    value = float(np.random.uniform(var.low, var.high))
            elif var.type == VariableType.INT:
                if var.log:
                    safe_low = max(1e-10, var.low - 0.5)
                    low_log = np.log(safe_low)
                    high_log = np.log(var.high + 0.5)
                    value = int(
                        np.clip(
                            np.round(np.exp(np.random.uniform(low_log, high_log))),
                            var.low,
                            var.high,
                        )
                    )
                else:
                    value = int(np.random.randint(var.low, var.high + 1))
            elif var.type == VariableType.CATEGORICAL:
                choices = [choice.actual_name for choice in var.choices]
                value = np.random.choice(choices)
            elif var.type == VariableType.BOOL:
                value = np.random.choice([True, False])
            else:
                value = var.value

            point_variables[var.actual_name] = value

        sampled_points.append(point_variables)

    return sampled_points
