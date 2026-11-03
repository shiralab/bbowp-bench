import argparse
import json
import logging
import pathlib
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from core.sample_point_and_score import sample_point_and_score

from bbowp.schema.designed_answer import DesignedAnswer


def setup_logger(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger()
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
    out_dir = pathlib.Path(out_dir)
    task_list = config["task"]
    n_sample = config["n_sample"]

    for task in task_list:
        task_config_path = get_task_config_path(task, data_dir)
        task_config_path = pathlib.Path(task_config_path)

        with open(task_config_path, "r", encoding="utf-8") as f:
            task_config = json.load(f)

        answer = next(
            (a for a in task_config["answer"] if a["amount"] == "large"), None
        )
        designed_answer = DesignedAnswer.model_validate(answer["answer_item"])

        setting_list.append(
            {
                "task_name": task,
                "designed_answer": designed_answer,
                "n_sample": n_sample,
                "out_dir": out_dir / task,
                "data_dir": data_dir,
            }
        )

    return setting_list


def get_task_config_path(task_name: str, data_dir: str) -> pathlib.Path:
    data_dir = pathlib.Path(data_dir)

    benchmark_name = task_name.split("_")[0]
    task_config_path = data_dir / benchmark_name / task_name / "task.json"

    return task_config_path


def run_single_task(setting: dict):
    task_name = setting["task_name"]
    designed_answer = setting["designed_answer"]
    n_sample = setting["n_sample"]
    out_dir = setting["out_dir"]
    data_dir = setting["data_dir"]

    sample_point_and_score(task_name, designed_answer, n_sample, out_dir, data_dir)


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
    except (OSError, ValueError) as e:
        parser.error(f"Error loading experiment config: {e}")
        sys.exit(1)

    # get result directory and check existence
    try:
        out_dir = pathlib.Path(config["out_dir"])
    except KeyError:
        parser.error("Missing 'out_dir' in experiment config.")
        sys.exit(1)

    # setup logger
    log_path = out_dir / "run_sample_point.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)

    # get setting list
    setting_list = get_setting_list(config)

    # start optimization for each designed answer file
    max_workers = args.num_workers

    logger.info(
        f"Start sampling: total_settings={len(setting_list)}, max_workers={max_workers}"
    )

    failed_settings: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_setting = {
            executor.submit(run_single_task, setting): setting
            for setting in setting_list
        }

        for future in as_completed(future_to_setting):
            setting = future_to_setting[future]

            try:
                future.result()
            except Exception:  # noqa: BLE001
                logger.error(
                    f"Sampling failed for setting {setting}: {traceback.format_exc()}"
                )
                failed_settings.append(setting)

    if failed_settings:
        logger.error(
            f"Finished sampling with failures: {len(failed_settings)}/{len(setting_list)} settings failed."
        )
        sys.exit(1)

    logger.info(
        f"Finished sampling successfully: all {len(setting_list)} settings completed."
    )
