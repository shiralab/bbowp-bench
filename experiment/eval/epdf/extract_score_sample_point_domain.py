import argparse
import csv
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()

    sample_point_path = (
        pathlib.Path(args.result_dir) / args.task_name / "uniform_random_sample.csv"
    )
    out_path = pathlib.Path(args.result_dir) / f"{args.task_name}_sample_score.csv"

    with (
        open(sample_point_path, "r", encoding="utf-8") as f_in,
        open(out_path, "w", encoding="utf-8", newline="") as f_out,
    ):
        reader = csv.DictReader(f_in)
        f_out.writelines(row["score"] + "\n" for row in reader)
