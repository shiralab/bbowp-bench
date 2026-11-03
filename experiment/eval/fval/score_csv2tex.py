import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Task name mapping for display in LaTeX tables
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

# Model name mapping for display in LaTeX tables
MODEL_NAME_MAPPING = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openai/gpt-5-mini": "GPT-5 mini",
    "openai/gpt-5.2": "GPT-5.2",
    "human_baseline": "Human Baseline",
}

# Select tasks to include in the output. Comment out the tasks you want to exclude.
SELECTED_TASKS = [
    "yahpo_lcbench_3945",
    "yahpo_rbv2_aknn_41138",
    "yahpo_rbv2_ranger_4135",
    "yahpo_rbv2_rpart_41138",
    "yahpo_rbv2_svm_40981",
    "yahpo_rbv2_xgboost_16",
    "yahpo_rbv2_super_41138",
    "olympus_benzylation_bayesnn",
    "olympus_colors_bob_bayesnn",
    "olympus_colors_n9_bayesnn",
    "olympus_fullerenes_bayesnn",
    "olympus_photo_pce10_bayesnn",
    "olympus_photo_wf3_bayesnn",
    "olympus_snar_bayesnn",
    "olympus_hplc_bayesnn",
    "olympus_suzuki_bayesnn",
    "MECHBench_Star_shaped_crash_box",
    "MECHBench_Three_point_bending_model",
    "MuJoCo_Inverted_Pendulum",
    "MuJoCo_Inverted_Double_Pendulum",
    "MuJoCo_Hopper",
    "MuJoCo_Reacher",
    "MuJoCo_Swimmer",
]


def format_value(val):
    """Format a value for display in LaTeX table.

    Handles inf/-inf and rounds floats to appropriate precision.
    """
    if pd.isna(val):
        return "--"
    elif np.isinf(val):
        return r"$\infty$" if val > 0 else r"$-\infty$"
    else:
        # Round to 2 decimal places
        return f"{val:.2f}"


def format_mean_std(mean_str, std_val, n_seeds=None):
    """Append a standard deviation to an already-formatted mean string."""
    if mean_str == "--":
        return "--"

    if pd.isna(std_val) or np.isinf(std_val):
        return f"{mean_str}$^{{\\dagger}}$" if n_seeds == 1 else mean_str

    return f"{mean_str} {{\\scriptsize $\\pm$ {std_val:.2f}}}"


def extract_task_number(task_name):
    """Extract task number from task name for sorting.

    E.g., "task 1 (↑)" -> 1, "task 23 (↑)" -> 23
    """
    match = re.search(r"Task\s+(\d+)", task_name)
    if match:
        return int(match.group(1))
    return float("inf")  # Unknown tasks go to the end


def is_maximization_task(task_name):
    """Determine if a task is maximization (↑) or minimization (↓).

    Returns True for maximization (↑), False for minimization (↓).
    """
    return "↑" in task_name


def highlight_best_indices(row, task_name, skip_universal_ties=False):
    """Return indices of best values based on task optimization direction.

    Args:
        row: Series of values for a task row
        task_name: Task name containing direction (↑ or ↓)
        skip_universal_ties: If True, don't highlight when all values are equal
    """
    numeric_values = pd.to_numeric(row, errors="coerce")
    valid_mask = ~(numeric_values.isna() | np.isinf(numeric_values))

    if not valid_mask.any():
        return []

    display_values = numeric_values[valid_mask].round(2)

    # If skip_universal_ties is enabled and all values are equal, return empty
    if skip_universal_ties and display_values.nunique() == 1:
        return []

    if is_maximization_task(task_name):
        best_value = display_values.max()
    else:
        best_value = display_values.min()

    return display_values[display_values == best_value].index.tolist()


def sort_index_by_task_number(index):
    """Sort index by extracted task numbers."""
    return index.sort_values(key=lambda x: x.map(extract_task_number))


def make_score_pivot(df: pd.DataFrame, aggfunc: str):
    """Pivot best_score into (domain, task) x model, aggregated with `aggfunc`."""
    return df.pivot_table(
        index=["domain", "task"],
        columns="model",
        values="best_score",
        aggfunc=aggfunc,
        dropna=True,
        observed=True,
    )


def get_domain(task_name):
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


def generate_latex_tables(
    csv_path: str,
    output_dir: str = "experiment/metric",
    skip_universal_ties: bool = False,
):
    """Generate LaTeX tables from aggregate_score.csv grouped by difficulty.

    Args:
        csv_path: Path to aggregate_score.csv
        output_dir: Directory to save output LaTeX files
        skip_universal_ties: If True, don't highlight when all model values are equal
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter tasks based on SELECTED_TASKS
    df = df[df["task"].isin(SELECTED_TASKS)].copy()

    # Add domain column
    df["domain"] = df["task"].apply(get_domain)

    # Get unique values, filtering out nan
    difficulties = sorted(df["difficulty"].dropna().unique())
    budgets = sorted(df["budget"].dropna().unique())

    output_dir = Path(csv_path).parent

    domain_order = ["YAHPO", "Olympus", "MECHBench", "MuJoCo", "Other"]

    # For each difficulty and budget, create a separate table
    for difficulty in difficulties:
        for budget in budgets:
            df_filtered = df[
                (df["difficulty"] == difficulty) & (df["budget"] == budget)
            ].copy()

            if df_filtered.empty:
                continue

            # Enforce domain ordering
            df_filtered["domain"] = pd.Categorical(
                df_filtered["domain"],
                categories=[
                    d for d in domain_order if d in df_filtered["domain"].values
                ],
                ordered=True,
            )
            # Create pivot table: (domain, task) x model.
            # Only `seed` varies within a cell here, so mean/std/count are over seeds.
            pivot_df = make_score_pivot(df_filtered, "mean")
            # std/count pivots can drop all-NaN columns, so align them to the mean.
            pivot_std = make_score_pivot(df_filtered, "std").reindex(
                index=pivot_df.index, columns=pivot_df.columns
            )
            pivot_count = make_score_pivot(df_filtered, "count").reindex(
                index=pivot_df.index, columns=pivot_df.columns
            )

            # Sort by task number within each domain
            sorted_index = sorted(
                pivot_df.index,
                key=lambda x: (
                    domain_order.index(x[0]) if x[0] in domain_order else 999,
                    extract_task_number(TASK_NAME_MAPPING.get(x[1], x[1])),
                ),
            )
            pivot_df = pivot_df.reindex(sorted_index)
            pivot_std = pivot_std.reindex(sorted_index)
            pivot_count = pivot_count.reindex(sorted_index)

            # Convert task names using mapping
            pivot_df.index = pivot_df.index.set_levels(
                pivot_df.index.levels[1].map(lambda x: TASK_NAME_MAPPING.get(x, x)),
                level=1,
            )

            # Convert model names using mapping
            pivot_df.columns = pivot_df.columns.map(
                lambda x: MODEL_NAME_MAPPING.get(x, x)
            )

            # std/count are row/column aligned with the mean, so mirror the labels
            pivot_std.index = pivot_df.index
            pivot_std.columns = pivot_df.columns
            pivot_count.index = pivot_df.index
            pivot_count.columns = pivot_df.columns

            # Format values (mean only for now; std is appended after highlighting)
            formatted_df = pivot_df.map(format_value)

            # Highlight best values per task (across columns)
            for idx in pivot_df.index:
                best_model_indices = highlight_best_indices(
                    pivot_df.loc[idx],
                    task_name=idx[1],  # The display task name has the direction
                    skip_universal_ties=skip_universal_ties,
                )
                for col in best_model_indices:
                    formatted_df.loc[idx, col] = (
                        f"\\textbf{{{formatted_df.loc[idx, col]}}}"
                    )

            # Append the seed-wise standard deviation to each (already bold-marked) mean
            single_seed_cells = 0
            for idx in formatted_df.index:
                for col in formatted_df.columns:
                    n_seeds = pivot_count.loc[idx, col]
                    n_seeds = None if pd.isna(n_seeds) else int(n_seeds)
                    if n_seeds == 1:
                        single_seed_cells += 1
                    formatted_df.loc[idx, col] = format_mean_std(
                        formatted_df.loc[idx, col],
                        pivot_std.loc[idx, col],
                        n_seeds=n_seeds,
                    )

            n_seed_values = sorted(
                {int(v) for v in pivot_count.to_numpy().ravel() if not pd.isna(v)}
            )
            seed_desc = (
                f"{n_seed_values[0]}"
                if len(n_seed_values) == 1
                else "varying numbers of"
            )
            caption = (
                f"Model performance comparison ({difficulty.capitalize()}, budget={budget}). "
                f"Values are mean $\\pm$ sample standard deviation over {seed_desc} seeds."
            )
            if single_seed_cells:
                caption += (
                    " $\\dagger$ marks cells backed by a single seed, "
                    "for which the standard deviation is undefined."
                )

            # Generate LaTeX table
            latex_str = formatted_df.to_latex(
                multirow=True,
                escape=False,
                caption=caption,
                label=f"tab:results_{difficulty}_{budget}",
                column_format="l" + "l" + "r" * len(formatted_df.columns),
            )

            # Save to file
            output_file = output_dir / f"latex_table_{difficulty}_{budget}.tex"
            with open(output_file, "w") as f:
                f.write(latex_str)

            print(f"Generated: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--skip-universal-ties",
        action="store_true",
        help="Skip highlighting when all model values are equal for a task",
    )
    args = parser.parse_args()
    csv_path = args.csv_path

    # Generate all tables
    generate_latex_tables(csv_path, skip_universal_ties=args.skip_universal_ties)

    print("\nAll LaTeX tables have been generated successfully!")
    if args.skip_universal_ties:
        print("(Universal ties highlighting is disabled)")
