import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task 2 raw evaluations.")
    parser.add_argument("--input", type=str, required=True, help="Task 2 raw CSV path.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional summary CSV path. Defaults to repl/outputs/metrics/<stem>_summary.csv",
    )
    return parser.parse_args()


def build_output_path(input_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return input_path.parents[1] / "metrics" / f"{input_path.stem}_summary.csv"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    df["is_positive_choice"] = df["selected_polarity"].eq("positive")
    df["is_negative_choice"] = df["selected_polarity"].eq("negative")
    df["is_parse_failure"] = df["selected_polarity"].isna()

    summary = (
        df.groupby(["model", "country", "topic", "value"], dropna=False)
        .agg(
            prompts=("prompt_index", "count"),
            positive_choices=("is_positive_choice", "sum"),
            negative_choices=("is_negative_choice", "sum"),
            parse_failures=("is_parse_failure", "sum"),
        )
        .reset_index()
    )
    summary["positive_choice_rate"] = summary["positive_choices"] / summary["prompts"]
    summary["negative_choice_rate"] = summary["negative_choices"] / summary["prompts"]
    summary["parse_failure_rate"] = summary["parse_failures"] / summary["prompts"]

    out_path = build_output_path(input_path, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
