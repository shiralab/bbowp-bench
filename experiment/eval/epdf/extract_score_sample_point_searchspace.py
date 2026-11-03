import argparse
import csv
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    for sample_point_path in pathlib.Path(args.result_dir).glob(
        f"diff_*/budget_small/{args.task_name}/**/uniform_random_sample.csv"
    ):
        rel_path = sample_point_path.relative_to(args.result_dir)
        out_path = pathlib.Path(args.out_dir) / rel_path.parent / "sample_score.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(sample_point_path, "r", encoding="utf-8") as f_in,
            open(out_path, "w", encoding="utf-8", newline="") as f_out,
        ):
            reader = csv.DictReader(f_in)
            f_out.writelines(row["score"] + "\n" for row in reader)
