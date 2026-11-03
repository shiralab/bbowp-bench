import argparse
import logging
import pathlib
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import yaml
from core.execute_designed_answer_generation import execute_designed_answer_generation


def setup_logger(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
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

    task_list = config["task"]
    diff_list = config["difficulty"]
    budget_list = config["budget"]
    model_list = config["model"]

    prompt = config["prompt"]
    example_task_list = config.get("example_task", None)

    seed_start = config["seed"]["start"]
    seed_end = config["seed"]["end"]
    seed_list = list(range(seed_start, seed_end + 1))

    for task, diff, budget, model, seed in product(
        task_list, diff_list, budget_list, model_list, seed_list
    ):
        task_config_path = get_task_config_path(task, data_dir)

        base_out_dir = config["out_dir"]
        out_dir = (
            pathlib.Path(base_out_dir)
            / f"diff_{diff}"
            / f"budget_{budget}"
            / task
            / model
            / f"seed_{seed:02d}"
        )

        setting = {
            "task_config_path": task_config_path,
            "diff": diff,
            "budget": budget,
            "prompt": prompt,
            "model": model,
            "out_dir": out_dir,
            "seed": seed,
        }

        if example_task_list is not None:
            if len(example_task_list) == 1:
                example_task = example_task_list[0]
                example_task_config_path = get_task_config_path(example_task, data_dir)
                setting["example_task_config_path"] = example_task_config_path
            else:
                logger.warning(
                    "Multiple example tasks provided, but only one is supported. Ignoring example tasks."
                )

        setting_list.append(setting)

    return setting_list


def get_task_config_path(task_name: str, data_dir: str) -> pathlib.Path:
    data_dir = pathlib.Path(data_dir)

    benchmark_name = task_name.split("_")[0]
    task_config_path = data_dir / benchmark_name / task_name / "task.json"

    return task_config_path


def run_single_setting(setting: dict) -> tuple[dict, bool, str | None]:
    try:
        execute_designed_answer_generation(**setting)
        return setting, True, None
    except Exception as exc:  # noqa: BLE001
        error_trace = "".join(traceback.format_exception(exc))
        return setting, False, error_trace


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

    # get setting list
    setting_list = get_setting_list(config)

    if not setting_list:
        parser.error("No execution setting was generated from experiment config.")
        sys.exit(1)

    # setup logger
    out_dir = pathlib.Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generate_designed_answer.log"
    logger = setup_logger(log_path)

    # start execution of designed answer generation for each setting
    max_workers = args.num_workers

    logger.info(
        f"Start designed answer generation: total_settings={len(setting_list)}, max_workers={max_workers}"
    )

    failed_settings: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_setting = {
            executor.submit(run_single_setting, setting): setting
            for setting in setting_list
        }

        for future in as_completed(future_to_setting):
            setting = future_to_setting[future]
            setting_label = (
                f"task_config_path={setting['task_config_path']}, diff={setting['diff']}, "
                f"budget={setting['budget']}, model={setting['model']}, seed={setting['seed']}"
            )

            try:
                _, success, error_trace = future.result()
            except Exception:  # noqa: BLE001
                logger.error(
                    f"Failed to execute designed answer generation for setting: {setting_label}\n"
                    f"{traceback.format_exc()}"
                )
                failed_settings.append(setting)
                continue

            if success:
                logger.info(f"Completed setting: {setting_label}")
            else:
                logger.error(
                    f"Failed to execute designed answer generation for setting: {setting_label}\n{error_trace}"
                )
                failed_settings.append(setting)

    if failed_settings:
        logger.error(
            f"Finished with failures: {len(failed_settings)}/{len(setting_list)} settings failed."
        )
        sys.exit(1)

    logger.info(f"Finished successfully: all {len(setting_list)} settings completed.")
