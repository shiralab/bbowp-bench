import argparse
import json
import logging
import pathlib
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from core.execute_bbo_from_designed_answer import execute_bbo_from_designed_answer

from bbowp.schema.designed_answer import DesignedAnswer


def setup_logger(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("main_optimization_runner")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    # output to console
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    stream_fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(stream_fmt)
    logger.addHandler(handler)

    # output to file
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


def get_setting_list(config: dict) -> list[dict]:
    setting_list = []

    data_dir = config["data_dir"]
    out_dir = config["out_dir"]
    task_list = config["task"]
    budget_list = config["budget"]

    seed_start = config["seed"]["start"]
    seed_end = config["seed"]["end"]
    seed_list = list(range(seed_start, seed_end + 1))

    out_dir = pathlib.Path(out_dir)

    for budget in budget_list:
        for task in task_list:
            # get designed answer of expert from task config
            task_config_path = get_task_config_path(task, data_dir)
            task_config_path = pathlib.Path(task_config_path)

            with open(task_config_path, "r", encoding="utf-8") as f:
                task_config = json.load(f)

            answer = next(
                (a for a in task_config["answer"] if a["amount"] == budget), None
            )
            designed_answer = DesignedAnswer.model_validate(answer["answer_item"])

            for seed in seed_list:
                # save designed answer to output directory for record
                designed_answer_path = (
                    out_dir / f"budget_{budget}" / task / "designed_answer.json"
                )
                designed_answer_path.parent.mkdir(parents=True, exist_ok=True)
                with open(designed_answer_path, "w", encoding="utf-8") as f:
                    json.dump(designed_answer.model_dump(), f, indent=4)

                setting_list.append(
                    {
                        "designed_answer_path": designed_answer_path,
                        "task_name": task,
                        "budget": budget,
                        "seed": seed,
                    }
                )

    return setting_list


def get_task_config_path(task_name: str, data_dir: str) -> pathlib.Path:
    data_dir = pathlib.Path(data_dir)

    benchmark_name = task_name.split("_")[0]
    task_config_path = data_dir / benchmark_name / task_name / "task.json"

    return task_config_path


def run_single_optimization(setting: dict):
    designed_answer_path = setting["designed_answer_path"]
    task_name = setting["task_name"]
    budget = setting["budget"]
    seed = setting["seed"]

    # add logging here for each optimization run?

    execute_bbo_from_designed_answer(
        designed_answer_path=designed_answer_path,
        task_name=task_name,
        budget=budget,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    # load experiment config
    config_path = pathlib.Path(args.exp_config)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

            if config is None:
                raise ValueError("Experiment config file is empty.")
    except FileNotFoundError:
        parser.error(f"Experiment config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        parser.error(f"Error parsing experiment config YAML: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        parser.error(f"Error loading experiment config: {e}")
        sys.exit(1)

    # get result directory and check existence
    try:
        out_dir = pathlib.Path(config["out_dir"])
    except KeyError:
        parser.error("Missing 'out_dir' in experiment config.")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # setup logger
    log_path = out_dir / "run_optimization.log"
    logger = setup_logger(log_path)

    # get setting list
    setting_list = get_setting_list(config)

    # start optimization for each designed answer file
    max_workers = args.num_workers

    logger.info(
        f"Start optimization: total_settings={len(setting_list)}, max_workers={max_workers}"
    )

    failed_settings: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_setting = {
            executor.submit(run_single_optimization, setting): setting
            for setting in setting_list
        }

        for future in as_completed(future_to_setting):
            setting = future_to_setting[future]

            try:
                future.result()
            except Exception:  # noqa: BLE001
                logger.error(
                    f"Optimization failed for setting {setting}: {traceback.format_exc()}"
                )
                failed_settings.append(setting)

    if failed_settings:
        logger.error(
            f"Finished optimization with failures: {len(failed_settings)}/{len(setting_list)} settings failed."
        )
        sys.exit(1)

    logger.info(
        f"Finished optimization successfully: all {len(setting_list)} settings completed."
    )
