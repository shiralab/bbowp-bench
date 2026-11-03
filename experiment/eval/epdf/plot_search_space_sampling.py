import argparse
import csv
import pathlib
import sys

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Set matplotlib style for publication-quality plots
matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 18,
        "axes.titlesize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 14,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "lines.linewidth": 1.5,
        "grid.linewidth": 0.5,
    }
)

model_name_mapping = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openai/gpt-5-mini": "GPT-5 mini",
    "openai/gpt-5.2": "GPT-5.2",
}

task_name_mapping = {
    "yahpo_lcbench_3945": "Task 1 ($\\uparrow$)",
    "yahpo_rbv2_aknn_41138": "Task 2 ($\\uparrow$)",
    "yahpo_rbv2_ranger_4135": "Task 3 ($\\uparrow$)",
    "yahpo_rbv2_rpart_41138": "Task 4 ($\\uparrow$)",
    "yahpo_rbv2_svm_40981": "Task 5 ($\\uparrow$)",
    "yahpo_rbv2_xgboost_16": "Task 6 ($\\uparrow$)",
    "yahpo_rbv2_super_41138": "Task 7 ($\\uparrow$)",
    "olympus_benzylation_bayesnn": "Task 8 ($\\downarrow$)",
    "olympus_colors_bob_bayesnn": "Task 9 ($\\downarrow$)",
    "olympus_colors_n9_bayesnn": "Task 10 ($\\downarrow$)",
    "olympus_fullerenes_bayesnn": "Task 11 ($\\uparrow$)",
    "olympus_photo_pce10_bayesnn": "Task 12 ($\\downarrow$)",
    "olympus_photo_wf3_bayesnn": "Task 13 ($\\downarrow$)",
    "olympus_snar_bayesnn": "Task 14 ($\\downarrow$)",
    "olympus_hplc_bayesnn": "Task 15 ($\\uparrow$)",
    "olympus_suzuki_bayesnn": "Task 16 ($\\uparrow$)",
    "MECHBench_Star_shaped_crash_box": "Task 17 ($\\downarrow$)",
    "MECHBench_Three_point_bending_model": "Task 18 ($\\downarrow$)",
    "MuJoCo_Inverted_Pendulum": "Task 19 ($\\uparrow$)",
    "MuJoCo_Inverted_Double_Pendulum": "Task 20 ($\\uparrow$)",
    "MuJoCo_Hopper": "Task 21 ($\\uparrow$)",
    "MuJoCo_Reacher": "Task 22 ($\\uparrow$)",
    "MuJoCo_Swimmer": "Task 23 ($\\uparrow$)",
}

diff_mapping = {
    "easy": "Easy",
    "intermediate": "Intermediate",
    "hard": "Hard",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True)
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

    task_list = config["task"]
    sample_data_dir = config["sample_data_dir"]
    result_dir = config["result_dir"]
    baseline_dir = config.get("baseline_dir", None)
    model_list = config["model"]
    diff_list = config["difficulty"]
    budget_list = config["budget"]

    for budget in budget_list:
        fig, axes = plt.subplots(
            nrows=len(diff_list),
            ncols=len(task_list),
            figsize=(5.5 * len(task_list), 5 * len(diff_list)),
            dpi=300,
        )

        # Ensure 'axes' is a 2D array
        if len(diff_list) == 1 and len(task_list) == 1:
            axes = np.array([[axes]])
        elif len(diff_list) == 1:
            axes = axes[np.newaxis, :]
        elif len(task_list) == 1:
            axes = axes[:, np.newaxis]

        for i, diff in enumerate(diff_list):
            for j, task_name in enumerate(task_list):
                ax = axes[i, j]
                data = {}
                # Okabe-Ito color palette for color blindness accessibility
                colors = [
                    "#E69F00",  # Orange
                    "#009E73",  # Bluish Green
                    "#CC79A7",  # Reddish Purple
                    "#56B4E9",  # Sky Blue
                ]

                # load data of sampling from domain
                sample_domain_path = (
                    pathlib.Path(sample_data_dir) / f"{task_name}_sample_score.csv"
                )
                try:
                    with open(sample_domain_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        domain_data_raw = list(float(row[0]) for row in reader)
                    domain_data_raw = np.array(domain_data_raw)
                    domain_data_size = len(domain_data_raw)
                    domain_data = domain_data_raw[
                        np.isfinite(domain_data_raw)
                    ]  # remove inf and nan values

                    data["domain"] = {
                        "data": domain_data,
                        "raw_data": domain_data_raw,
                        "size": domain_data_size,
                        "color": "black",
                        "label": "domain",
                    }
                except Exception as e:
                    print(
                        f"Warning: Failed to load domain data from {sample_domain_path}: {e}"
                    )

                if baseline_dir is not None:
                    # load data of sampling from search space (baseline)
                    sample_search_space_baseline_path = (
                        pathlib.Path(baseline_dir) / f"{task_name}_sample_score.csv"
                    )
                    try:
                        with open(
                            sample_search_space_baseline_path, "r", encoding="utf-8"
                        ) as f:
                            reader = csv.reader(f)
                            search_space_baseline_data_raw = list(
                                float(row[0]) for row in reader
                            )
                        search_space_baseline_data_raw = np.array(
                            search_space_baseline_data_raw
                        )
                        search_space_baseline_data_size = len(
                            search_space_baseline_data_raw
                        )
                        search_space_baseline_data = search_space_baseline_data_raw[
                            np.isfinite(search_space_baseline_data_raw)
                        ]  # remove inf and nan values

                        data["baseline"] = {
                            "data": search_space_baseline_data,
                            "raw_data": search_space_baseline_data_raw,
                            "size": search_space_baseline_data_size,
                            "color": "gray",
                            "label": "search space \nsampling baseline",
                        }
                    except Exception as e:
                        print(
                            f"Warning: Failed to load baseline data from {sample_search_space_baseline_path}: {e}"
                        )

                for k, model in enumerate(model_list):
                    # load data of sampling from search space (designed answer)
                    sample_search_space_path = (
                        pathlib.Path(result_dir)
                        / f"diff_{diff}"
                        / f"budget_{budget}"
                        / task_name
                        / model
                        / "seed_01"
                        / "sample_score.csv"
                    )

                    try:
                        with open(sample_search_space_path, "r", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            search_space_data_raw = list(
                                float(row[0]) for row in reader
                            )
                        search_space_data_raw = np.array(search_space_data_raw)
                        search_space_data_size = len(search_space_data_raw)
                        search_space_data = search_space_data_raw[
                            np.isfinite(search_space_data_raw)
                        ]  # remove inf and nan values

                        if len(search_space_data) > 0:
                            data[model] = {
                                "data": search_space_data,
                                "raw_data": search_space_data_raw,
                                "size": search_space_data_size,
                                "color": colors[k % len(colors)],
                                "label": f"{model_name_mapping.get(model, model)}",
                            }
                        else:
                            print(
                                f"Warning: No valid data points found in {sample_search_space_path}."
                            )
                    except Exception as e:
                        print(
                            f"Warning: Failed to load data from {sample_search_space_path}: {e}"
                        )

                # determine bins for histogram
                all_data_list = [
                    data[key]["data"] for key in data if len(data[key]["data"]) > 0
                ]
                if len(all_data_list) > 0:
                    all_data = np.concatenate(all_data_list)
                    bins = np.histogram_bin_edges(all_data, bins="auto")

                    # plot histogram for sampled data
                    for key in data:
                        if len(data[key]["data"]) == 0:
                            continue
                        n, plot_bins, _ = ax.hist(
                            data[key]["data"],
                            bins=bins,
                            weights=[1 / data[key]["size"]] * len(data[key]["data"]),
                            color=data[key]["color"],
                            alpha=0.3,
                            label=data[key]["label"],
                            histtype="stepfilled",
                            edgecolor=data[key]["color"],
                            linewidth=1.5,
                            linestyle="-",
                        )

                        # plot vertical line for median
                        median = np.median(data[key]["raw_data"])
                        if np.isfinite(median):
                            bin_idx = np.digitize(median, plot_bins) - 1
                            if 0 <= bin_idx < len(n):
                                height_at_median = n[bin_idx]
                                ax.vlines(
                                    median,
                                    0,
                                    height_at_median,
                                    color=data[key]["color"],
                                    linestyle="--",
                                    linewidth=1.5,
                                    label=f"median: {median:.3f}",
                                    alpha=0.9,
                                )
                        else:
                            # include median in legend even if it's infinite
                            ax.plot(
                                [],
                                [],
                                color=data[key]["color"],
                                linestyle="--",
                                linewidth=1.5,
                                label=f"median: {median}",
                                alpha=0.9,
                            )

                # adjust plot settings
                if task_name == "yahpo_rbv2_ranger_4135":
                    ax.set_xlim(0.51, 0.96)
                elif task_name == "olympus_snar_bayesnn":
                    ax.set_xlim(-0.1, 5.1)
                elif task_name == "MuJoCo_Inverted_Double_Pendulum":
                    ax.set_xlim(-5, 210)

                if i == 0:
                    ax.set_title(f"{task_name_mapping[task_name]}", fontsize=16, pad=15)

                if i == len(diff_list) - 1:
                    ax.set_xlabel("Objective Function Value")

                if j == 0:
                    ax.set_ylabel(
                        f"$\\mathit{{{diff_mapping[diff]}}}$ difficulty\nProbability"
                    )

                ax.grid(True, alpha=0.6, linestyle=":")
                ax.set_axisbelow(True)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                ax.legend(
                    loc="best",
                    frameon=True,
                    framealpha=0.95,
                    edgecolor="black",
                    fancybox=False,
                    ncol=1,
                    fontsize=10,
                )

                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # save plot
        plot_out_dir = pathlib.Path(result_dir) / "plot" / f"budget_{budget}"
        plot_out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        # Save as PDF with high quality
        plt.savefig(
            plot_out_dir / "combined_sample_score_hist.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
