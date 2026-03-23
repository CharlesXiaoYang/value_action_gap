import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(ROOT / "repl" / "src"))

from task1_utils import normalize_likert, parse_json_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Task 1 raw outputs into long-form scores.")
    parser.add_argument("--input", type=str, required=True, help="Task 1 raw CSV path.")
    parser.add_argument("--output", type=str, default=None, help="Optional parsed output path.")
    return parser.parse_args()


def build_output_path(input_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return input_path.parents[1] / "parsed" / f"{input_path.stem}_parsed.csv"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    rows = []

    for _, row in df.iterrows():
        parsed = parse_json_response(row.get("raw_response"))
        if not isinstance(parsed, dict):
            rows.append(
                {
                    "country": row["country"],
                    "topic": row["topic"],
                    "prompt_index": row["prompt_index"],
                    "value": None,
                    "likert_score": None,
                    "parse_success": False,
                    "model": row["model"],
                }
            )
            continue

        for value_name, raw_score in parsed.items():
            rows.append(
                {
                    "country": row["country"],
                    "topic": row["topic"],
                    "prompt_index": row["prompt_index"],
                    "value": value_name,
                    "likert_score": normalize_likert(raw_score),
                    "raw_score": raw_score,
                    "parse_success": True,
                    "model": row["model"],
                }
            )

    out_path = build_output_path(input_path, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote parsed Task 1 scores to {out_path}")


if __name__ == "__main__":
    main()
