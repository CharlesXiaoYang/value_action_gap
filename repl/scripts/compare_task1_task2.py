import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Task 1 value ratings with Task 2 action choices.")
    parser.add_argument("--task1", type=str, required=True, help="Parsed Task 1 CSV path.")
    parser.add_argument("--task2", type=str, required=True, help="Raw Task 2 CSV path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path.")
    return parser.parse_args()


def build_output_path(task2_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return task2_path.parents[1] / "metrics" / f"{task2_path.stem}_vs_task1.csv"


def main() -> None:
    args = parse_args()
    task1_path = Path(args.task1)
    task2_path = Path(args.task2)

    task1 = pd.read_csv(task1_path)
    task2 = pd.read_csv(task2_path)

    task1_ok = task1[task1["parse_success"] == True].copy()
    task1_summary = (
        task1_ok.groupby(["model", "country", "topic", "value"], dropna=False)
        .agg(
            mean_likert=("likert_score", "mean"),
            std_likert=("likert_score", "std"),
            task1_n=("likert_score", "count"),
        )
        .reset_index()
    )
    task1_summary["endorsement_strength"] = 5 - task1_summary["mean_likert"]
    task1_summary["endorsement_rate"] = (task1_summary["endorsement_strength"] - 1) / 3

    task2_summary = (
        task2.groupby(["model", "country", "topic", "value"], dropna=False)
        .agg(
            prompts=("prompt_index", "count"),
            positive_choices=("selected_polarity", lambda s: (s == "positive").sum()),
            negative_choices=("selected_polarity", lambda s: (s == "negative").sum()),
            parse_failures=("selected_polarity", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    task2_summary["positive_choice_rate"] = task2_summary["positive_choices"] / task2_summary["prompts"]
    task2_summary["negative_choice_rate"] = task2_summary["negative_choices"] / task2_summary["prompts"]
    task2_summary["parse_failure_rate"] = task2_summary["parse_failures"] / task2_summary["prompts"]

    merged = task1_summary.merge(task2_summary, on=["model", "country", "topic", "value"], how="inner")
    merged["gap_signed"] = merged["positive_choice_rate"] - merged["endorsement_rate"]
    merged["gap_abs"] = (merged["gap_signed"]).abs()
    merged = merged.sort_values(["country", "topic", "value"]).reset_index(drop=True)

    out_path = build_output_path(task2_path, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote Task 1 vs Task 2 comparison to {out_path}")


if __name__ == "__main__":
    main()
