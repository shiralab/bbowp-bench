import argparse
import itertools
import pathlib
import re

import pandas as pd

DOMAIN_ORDER = ["YAHPO", "Olympus", "MECHBench", "MuJoCo", "Other"]

MODEL_NAME_MAPPING = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openai/gpt-5-mini": "GPT-5 mini",
    "openai/gpt-5.2": "GPT-5.2",
    "human_baseline": "Human Baseline",
}

TASK_NAME_MAPPING = {
    "yahpo_lcbench_3945": "task 1",
    "yahpo_rbv2_aknn_41138": "task 2",
    "yahpo_rbv2_ranger_4135": "task 3",
    "yahpo_rbv2_rpart_41138": "task 4",
    "yahpo_rbv2_svm_40981": "task 5",
    "yahpo_rbv2_xgboost_16": "task 6",
    "yahpo_rbv2_super_41138": "task 7",
    "olympus_benzylation_bayesnn": "task 8",
    "olympus_colors_bob_bayesnn": "task 9",
    "olympus_colors_n9_bayesnn": "task 10",
    "olympus_fullerenes_bayesnn": "task 11",
    "olympus_photo_pce10_bayesnn": "task 12",
    "olympus_photo_wf3_bayesnn": "task 13",
    "olympus_snar_bayesnn": "task 14",
    "olympus_hplc_bayesnn": "task 15",
    "olympus_suzuki_bayesnn": "task 16",
    "MECHBench_Star_shaped_crash_box": "task 17",
    "MECHBench_Three_point_bending_model": "task 18",
    "MuJoCo_Inverted_Pendulum": "task 19",
    "MuJoCo_Inverted_Double_Pendulum": "task 20",
    "MuJoCo_Hopper": "task 21",
    "MuJoCo_Reacher": "task 22",
    "MuJoCo_Swimmer": "task 23",
}


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)
    return escaped


def format_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def extract_task_number(display_name: str) -> float:
    """Extract the task number used to order rows, e.g. "task 20" -> 20."""
    match = re.search(r"task\s+(\d+)", display_name, re.IGNORECASE)
    return int(match.group(1)) if match else float("inf")


def get_domain(task_name: str) -> str:
    """Extract domain from task name."""
    if task_name.startswith("yahpo"):
        return "YAHPO"
    elif task_name.startswith("olympus"):
        return "Olympus"
    elif task_name.startswith("MECHBench"):
        return "MECHBench"
    elif task_name.startswith("MuJoCo"):
        return "MuJoCo"
    return "Other"


def render_table(
    caption: str,
    label: str,
    model_list: list[str],
    task_rows: list[tuple[str, str, list[str]]],
    average_row: list[str],
    merge_cells: bool = True,
) -> str:
    column_spec = "ll" + "c" * len(model_list)
    model_headers = " & ".join(
        latex_escape(MODEL_NAME_MAPPING.get(m, m)) for m in model_list
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"\toprule")
    lines.append(f"Domain & Task & {model_headers} \\\\")
    lines.append(r"\midrule")

    grouped = itertools.groupby(task_rows, key=lambda row: row[0])
    for i_domain, (domain, group) in enumerate(grouped):
        rows = list(group)
        if i_domain > 0:  # rule separating one domain from the next
            lines.append(r"\midrule")
        for j, (_, task_display_name, values) in enumerate(rows):
            if not merge_cells:
                domain_cell = latex_escape(domain)
            elif j == 0:
                domain_cell = (
                    f"\\multirow{{{len(rows)}}}{{*}}{{{latex_escape(domain)}}}"
                )
            else:
                domain_cell = ""
            lines.append(
                f"{domain_cell} & {latex_escape(task_display_name)} & "
                + " & ".join(values)
                + r" \\"
            )

    lines.append(r"\midrule")
    average_label = (
        r"\multicolumn{2}{l}{Average}" if merge_cells else "Average & Average"
    )
    lines.append(average_label + " & " + " & ".join(average_row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_tables(csv_path: str, merge_cells: bool = True) -> list[pathlib.Path]:
    df = pd.read_csv(csv_path)

    def task_sort_key(task_name: str) -> tuple[int, float, str]:
        domain = get_domain(task_name)
        domain_rank = (
            DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else len(DOMAIN_ORDER)
        )
        display_name = TASK_NAME_MAPPING.get(task_name, task_name)
        return (domain_rank, extract_task_number(display_name), task_name)

    task_list = sorted(dict.fromkeys(df["task"]), key=task_sort_key)
    diff_list = list(dict.fromkeys(df["difficulty"]))
    budget_list = list(dict.fromkeys(df["budget"]))
    model_list = list(dict.fromkeys(df["model"]))

    cell_mean = df.groupby(["task", "difficulty", "budget", "model"], sort=False)[
        "frs"
    ].mean()
    seed_counts = set(
        df.dropna(subset=["frs"])
        .groupby(["task", "difficulty", "budget", "model"], sort=False)
        .size()
    )

    def cell(task_name: str, diff: str, budget: str, model: str) -> float | None:
        value = cell_mean.get((task_name, diff, budget, model))
        return None if value is None or pd.isna(value) else float(value)

    seed_desc = (
        f"{next(iter(seed_counts))}" if len(seed_counts) == 1 else "varying numbers of"
    )
    metric_desc = (
        "Each value is the fraction of sampled answers matched or beaten, "
        f"computed per seed and averaged over {seed_desc} seeds."
    )

    tables_by_group: dict[str, list[str]] = {}
    for budget in budget_list:
        label_budget = budget.replace("/", "_")

        for diff in diff_list:
            task_rows = []
            per_model_scores: dict[str, list[float]] = {m: [] for m in model_list}
            for task_name in task_list:
                values = []
                for model in model_list:
                    score = cell(task_name, diff, budget, model)
                    values.append(format_metric(score))
                    if score is not None:
                        per_model_scores[model].append(score)
                task_rows.append(
                    (
                        get_domain(task_name),
                        TASK_NAME_MAPPING.get(task_name, task_name),
                        values,
                    )
                )

            tables_by_group.setdefault(diff, []).append(
                render_table(
                    caption=(
                        f"Performance of designed answers "
                        f"({diff.capitalize()}, budget: {budget}). {metric_desc}"
                    ),
                    label=f"tab:metric_{diff}_{label_budget}",
                    model_list=model_list,
                    task_rows=task_rows,
                    average_row=[
                        format_metric(mean_or_none(per_model_scores[m]))
                        for m in model_list
                    ],
                    merge_cells=merge_cells,
                )
            )

        # Overall average across difficulties
        task_rows = []
        per_model_scores = {m: [] for m in model_list}
        for task_name in task_list:
            values = []
            for model in model_list:
                scores = [cell(task_name, d, budget, model) for d in diff_list]
                valid_scores = [s for s in scores if s is not None]
                values.append(format_metric(mean_or_none(valid_scores)))
                per_model_scores[model].extend(valid_scores)
            task_rows.append(
                (
                    get_domain(task_name),
                    TASK_NAME_MAPPING.get(task_name, task_name),
                    values,
                )
            )

        tables_by_group.setdefault("overall", []).append(
            render_table(
                caption=(
                    f"Performance of designed answers "
                    f"(averaged over all difficulties, budget: {budget}). {metric_desc}"
                ),
                label=f"tab:metric_overall_{label_budget}",
                model_list=model_list,
                task_rows=task_rows,
                average_row=[
                    format_metric(mean_or_none(per_model_scores[m])) for m in model_list
                ],
                merge_cells=merge_cells,
            )
        )

    output_dir = pathlib.Path(csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[pathlib.Path] = []
    for group, group_tables in tables_by_group.items():
        output_path = output_dir / f"frs_tables_{group}.tex"
        output_path.write_text("\n\n".join(group_tables) + "\n", encoding="utf-8")
        print(f"Generated: {output_path}")
        output_paths.append(output_path)
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--no-merged-cells",
        action="store_true",
        help="Emit no \\multirow/\\multicolumn; repeat the value in every spanned cell",
    )
    args = parser.parse_args()

    generate_tables(args.csv_path, merge_cells=not args.no_merged_cells)
