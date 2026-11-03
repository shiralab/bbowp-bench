import argparse
import csv
import itertools
import math
import pathlib
import sys

import pandas as pd
import yaml

TASK_NAME_MAPPING = {
    "yahpo_lcbench_3945": "Task 1 (↑)",
    "yahpo_rbv2_aknn_41138": "Task 2 (↑)",
    "yahpo_rbv2_ranger_4135": "Task 3 (↑)",
    "yahpo_rbv2_rpart_41138": "Task 4 (↑)",
    "yahpo_rbv2_svm_40981": "Task 5 (↑)",
    "yahpo_rbv2_xgboost_16": "Task 6 (↑)",
    "yahpo_rbv2_super_41138": "Task 7 (↑)",
    "olympus_benzylation_bayesnn": "Task 8 (↓)",
    "olympus_colors_bob_bayesnn": "Task 9 (↓)",
    "olympus_colors_n9_bayesnn": "Task 10 (↓)",
    "olympus_fullerenes_bayesnn": "Task 11 (↑)",
    "olympus_photo_pce10_bayesnn": "Task 12 (↓)",
    "olympus_photo_wf3_bayesnn": "Task 13 (↓)",
    "olympus_snar_bayesnn": "Task 14 (↓)",
    "olympus_hplc_bayesnn": "Task 15 (↑)",
    "olympus_suzuki_bayesnn": "Task 16 (↑)",
    "MECHBench_Star_shaped_crash_box": "Task 17 (↓)",
    "MECHBench_Three_point_bending_model": "Task 18 (↓)",
    "MuJoCo_Inverted_Pendulum": "Task 19 (↑)",
    "MuJoCo_Inverted_Double_Pendulum": "Task 20 (↑)",
    "MuJoCo_Hopper": "Task 21 (↑)",
    "MuJoCo_Reacher": "Task 22 (↑)",
    "MuJoCo_Swimmer": "Task 23 (↑)",
}


def is_maximization_task(task_name: str) -> bool | None:
    """Return True for maximization (↑), False for minimization (↓)."""
    display_name = TASK_NAME_MAPPING.get(task_name)
    if display_name is None:
        return None
    if "↑" in display_name:
        return True
    if "↓" in display_name:
        return False
    return None


def frs_for_score(
    task_name: str, best_score: float, sampled_best_scores: list[float]
) -> float | None:
    maximize = is_maximization_task(task_name)
    if maximize is None:
        return None
    if maximize:
        n_beat_sampled_best_scores = sum(
            [best_score >= score for score in sampled_best_scores]
        )
    else:
        n_beat_sampled_best_scores = sum(
            [best_score <= score for score in sampled_best_scores]
        )
    return n_beat_sampled_best_scores / len(sampled_best_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # load config
    config_path = pathlib.Path(args.config)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

            if config is None:
                raise ValueError("Config file is empty.")
    except FileNotFoundError:
        parser.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        parser.error(f"Error parsing config YAML: {e}")
        sys.exit(1)
    except Exception as e:
        parser.error(f"Error loading config: {e}")
        sys.exit(1)

    task_list = config["task"]
    sample_data_dir = config["sample_data_dir"]
    result_dir = config["result_dir"]
    model_list = config["model"]
    diff_list = config["difficulty"]
    budget_list = config.get("budget", ["small"])
    out_dir = pathlib.Path(config.get("out_dir", pathlib.Path(result_dir) / "frs"))

    # Per-seed best scores come from the CSV produced by fval/aggregate_score.py.
    score_csv_path = pathlib.Path(
        config.get(
            "score_csv", pathlib.Path(result_dir) / "fval" / "aggregate_score.csv"
        )
    )
    if not score_csv_path.is_file():
        parser.error(
            f"Aggregated score CSV not found: {score_csv_path}. "
            "Run experiment/eval/fval/aggregate_score.py first, "
            "or set 'score_csv' in the config."
        )
    score_df = pd.read_csv(score_csv_path)

    # load data of sampled designed answers
    sampled_scores_by_task: dict[str, list[float]] = {}
    for task_name in task_list:
        if is_maximization_task(task_name) is None:
            print(
                f"Warning: Unknown task name: {task_name}. "
                "Its optimization direction is undefined, so it is skipped."
            )
            continue

        csv_path = pathlib.Path(sample_data_dir) / f"{task_name}_scores.csv"
        if not csv_path.is_file():
            print(f"Warning: No sampled score data for {task_name}: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            sampled_scores_by_task[task_name] = [float(row[0]) for row in reader]

    records: list[dict[str, object]] = []
    combis = itertools.product(task_list, diff_list, budget_list, model_list)
    for task_name, diff, budget, model in combis:
        if task_name not in sampled_scores_by_task:
            continue  # already reported while loading the sampled score data

        rows = score_df[
            (score_df["task"] == task_name)
            & (score_df["difficulty"] == diff)
            & (score_df["budget"] == budget)
            & (score_df["model"] == model)
        ]

        if rows.empty:
            print(
                f"Warning: No row found in {score_csv_path.name} for "
                f"{task_name}, {diff}, {budget}, {model}"
            )
            continue

        for row in rows.to_dict("records"):
            best_score = row["best_score"]
            frs = (
                None
                if math.isnan(best_score)
                else frs_for_score(
                    task_name, best_score, sampled_scores_by_task[task_name]
                )
            )
            records.append(
                {
                    "task": task_name,
                    "difficulty": diff,
                    "budget": budget,
                    "model": model,
                    "designed_seed": row.get("designed_seed"),
                    "seed": row["seed"],
                    "best_score": best_score,
                    "frs": frs,
                }
            )

    df_frs = pd.DataFrame(
        records,
        columns=[
            "task",
            "difficulty",
            "budget",
            "model",
            "designed_seed",
            "seed",
            "best_score",
            "frs",
        ],
    )

    out_path = out_dir / "frs_score.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_frs.to_csv(out_path, index=False)
    print(f"Per-seed FRS saved to {out_path}")
