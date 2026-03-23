import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "repl" / "src"))

from value_families import VALUE_TO_FAMILY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prompt sensitivity for Task 1 and Task 2.")
    parser.add_argument("--task1", type=str, required=True, help="Parsed Task 1 CSV path.")
    parser.add_argument("--task2", type=str, required=True, help="Raw Task 2 CSV path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path.")
    return parser.parse_args()


def build_output_path(task2_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return task2_path.parents[1] / "metrics" / f"{task2_path.stem}_prompt_sensitivity.csv"


def main() -> None:
    args = parse_args()
    task1 = pd.read_csv(args.task1)
    task2 = pd.read_csv(args.task2)

    task1_ok = task1[task1["parse_success"] == True].copy()
    t1 = (
        task1_ok.groupby(["model", "country", "topic", "value"], dropna=False)
        .agg(
            task1_mean=("likert_score", "mean"),
            task1_std=("likert_score", "std"),
            task1_min=("likert_score", "min"),
            task1_max=("likert_score", "max"),
            task1_n=("likert_score", "count"),
        )
        .reset_index()
    )
    t1["task1_range"] = t1["task1_max"] - t1["task1_min"]

    task2["positive_numeric"] = task2["selected_polarity"].map({"positive": 1, "negative": 0})
    t2 = (
        task2.groupby(["model", "country", "topic", "value"], dropna=False)
        .agg(
            task2_mean=("positive_numeric", "mean"),
            task2_std=("positive_numeric", "std"),
            task2_n=("positive_numeric", "count"),
            positive_choices=("positive_numeric", "sum"),
        )
        .reset_index()
    )
    t2["task2_flip_count"] = t2.apply(lambda r: min(r["positive_choices"], r["task2_n"] - r["positive_choices"]), axis=1)

    merged = t1.merge(t2, on=["model", "country", "topic", "value"], how="inner")
    merged["family"] = merged["value"].map(VALUE_TO_FAMILY)
    merged = merged.sort_values(["task2_flip_count", "task1_range", "value"], ascending=[False, False, True])

    out_path = build_output_path(Path(args.task2), args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote prompt sensitivity analysis to {out_path}")


if __name__ == "__main__":
    main()
