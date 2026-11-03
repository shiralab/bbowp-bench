import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml


def designed_seed_from_path(path: Path) -> int | None:
    """Extract the designed-answer seed from the result path."""
    match = re.fullmatch(r"seed_(\d+)", path.parent.parent.name)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # load config
    config_path = Path(args.config)
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
    except (OSError, ValueError) as e:
        parser.error(f"Error loading config: {e}")
        sys.exit(1)

    result_dir = config["result_dir"]
    task_list = config["task"]
    diff_list = config["difficulty"]
    budget_list = config["budget"]
    model_list = config["model"]
    seed_start = config["seed"]["start"]
    seed_end = config["seed"]["end"]
    seed_list = list(range(seed_start, seed_end + 1))

    result_dir = Path(result_dir)
    out_dir = Path(result_dir) / "fval"

    records: list[dict] = []
    combis = itertools.product(task_list, diff_list, budget_list, model_list, seed_list)
    for task, diff, budget, model, seed in combis:
        if model == "human_baseline":
            optimization_result_path_list = sorted(
                Path(result_dir).glob(
                    f"**/budget_{budget}/{task}/optimization_seed_{seed:02}/optimization_result.json"
                )
            )
        else:
            optimization_result_path_list = sorted(
                Path(result_dir).glob(
                    f"**/diff_{diff}/budget_{budget}/{task}/{model}/seed_*/optimization_seed_{seed:02}/optimization_result.json"
                )
            )
        if len(optimization_result_path_list) == 0:
            print(
                f"Warning: No optimization result file found for {task}, {diff}, {budget}, {model}, seed {seed}"
            )
            records.append(
                {
                    "task": task,
                    "difficulty": diff,
                    "budget": budget,
                    "model": model,
                    "designed_seed": None,
                    "seed": seed,
                    "best_score": None,
                }
            )
            continue

        seen_designed_seeds: set[int | None] = set()
        for optimization_result_path in optimization_result_path_list:
            designed_seed = designed_seed_from_path(optimization_result_path)
            if designed_seed in seen_designed_seeds:
                print(
                    f"Warning: Multiple optimization result files found for {task}, {diff}, {budget}, {model}, "
                    f"designed seed {designed_seed}, seed {seed}. Ignoring {optimization_result_path}"
                )
                continue
            seen_designed_seeds.add(designed_seed)

            score = None
            try:
                with open(optimization_result_path, "r", encoding="utf-8") as f:
                    optimization_result = json.load(f)

                if optimization_result is None:
                    print(
                        f"Warning: Optimization result file is empty: {optimization_result_path}"
                    )
                else:
                    score = optimization_result.get("best_fval")
                    if score is None:
                        print(
                            f"Warning: 'best_fval' not found in optimization result file: {optimization_result_path}"
                        )
            except (OSError, ValueError, AttributeError) as e:
                print(
                    f"Error reading optimization result file {optimization_result_path}: {e}"
                )

            records.append(
                {
                    "task": task,
                    "difficulty": diff,
                    "budget": budget,
                    "model": model,
                    "designed_seed": designed_seed,
                    "seed": seed,
                    "best_score": score,
                }
            )

    df_result = pd.DataFrame(
        records,
        columns=[
            "task",
            "difficulty",
            "budget",
            "model",
            "designed_seed",
            "seed",
            "best_score",
        ],
    )

    out_path = Path(out_dir) / "aggregate_score.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(out_path, index=False)
    print(f"Aggregate score saved to {out_path}")
