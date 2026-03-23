import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "repl" / "src"))

from value_families import VALUE_TO_FAMILY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Task 1 vs Task 2 comparison by Schwartz family.")
    parser.add_argument("--comparison", type=str, required=True, help="Comparison CSV path.")
    parser.add_argument("--output", type=str, default=None, help="Optional family-level output path.")
    return parser.parse_args()


def build_output_path(comparison_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return comparison_path.parents[1] / "metrics" / f"{comparison_path.stem}_families.csv"


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison)
    df = pd.read_csv(comparison_path)
    df["family"] = df["value"].map(VALUE_TO_FAMILY)

    family = (
        df.groupby(["model", "country", "topic", "family"], dropna=False)
        .agg(
            n_values=("value", "count"),
            mean_likert=("mean_likert", "mean"),
            endorsement_rate=("endorsement_rate", "mean"),
            positive_choice_rate=("positive_choice_rate", "mean"),
            gap_signed=("gap_signed", "mean"),
            gap_abs=("gap_abs", "mean"),
        )
        .reset_index()
        .sort_values(["country", "topic", "gap_abs"], ascending=[True, True, False])
    )

    out_path = build_output_path(comparison_path, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    family.to_csv(out_path, index=False)
    print(f"Wrote family aggregation to {out_path}")


if __name__ == "__main__":
    main()
