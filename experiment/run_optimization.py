import argparse
import logging
import pathlib
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from core.execute_bbo_from_designed_answer import execute_bbo_from_designed_answer


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

    result_dir = config["result_dir"]
    task_list = config["task"]
    diff_list = config["difficulty"]
    budget_list = config["budget"]

    seed_start = config["seed"]["start"]
    seed_end = config["seed"]["end"]
    seed_list = list(range(seed_start, seed_end + 1))

    result_dir = pathlib.Path(result_dir)
    all_designed_answer_files = list(
        pathlib.Path(result_dir).glob(
            "diff_*/budget_*/*/**/designed_answer.json"
        )  # expected path format: {result_dir}/diff_{diff}/budget_{budget}/{task_name}/**/designed_answer.json
    )

    for designed_answer_path in all_designed_answer_files:
        parts = designed_answer_path.relative_to(result_dir).parts
        diff = parts[0].split("_")[1]  # extract diff from "diff_{diff}"
        budget = parts[1].split("_")[1]  # extract budget from "budget_{budget}"
        task = parts[2]

        if task in task_list and diff in diff_list and budget in budget_list:
            for seed in seed_list:
                setting_list.append(
                    {
                        "designed_answer_path": designed_answer_path,
                        "task_name": task,
                        "budget": budget,
                        "seed": seed,
                    }
                )

    return setting_list


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
    except Exception as e:
        parser.error(f"Error loading experiment config: {e}")
        sys.exit(1)

    # get result directory and check existence
    try:
        result_dir = pathlib.Path(config["result_dir"])
    except KeyError:
        parser.error("Missing 'result_dir' in experiment config.")
        sys.exit(1)

    if not result_dir.exists():
        parser.error(f"Result directory does not exist: {result_dir}")

    # setup logger
    log_path = result_dir / "run_optimization.log"
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
            except Exception:
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
