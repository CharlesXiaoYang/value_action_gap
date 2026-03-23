import argparse
import sys
import time
from pathlib import Path

import aisuite as ai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "repl" / "src"))

from task1_prompting import Task1Prompting
from task1_utils import parse_json_response

DEFAULT_COUNTRIES = [
    "United States", "India", "Pakistan", "Nigeria", "Philippines", "United Kingdom",
    "Germany", "Uganda", "Canada", "Egypt", "France", "Australia"
]

DEFAULT_TOPICS = [
    "Politics",
    "Social Networks",
    "Social Inequality",
    "Family & Changing Gender Roles",
    "Work Orientation",
    "Religion",
    "Environment",
    "National Identity",
    "Citizenship",
    "Leisure Time and Sports",
    "Health and Health Care",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 value statement evaluation in repl.")
    parser.add_argument("--model", type=str, required=True, help="Aisuite model id.")
    parser.add_argument("--output", type=str, default=None, help="Optional raw output CSV path.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated country list.")
    parser.add_argument("--topics", type=str, default=None, help="Comma-separated topic list.")
    parser.add_argument("--limit-countries", type=int, default=None, help="Keep only the first N countries.")
    parser.add_argument("--limit-topics", type=int, default=None, help="Keep only the first N topics.")
    parser.add_argument("--prompt-indices", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated prompt indices.")
    return parser.parse_args()


def pick_list(raw: str | None, default: list[str], limit: int | None) -> list[str]:
    values = [item.strip() for item in raw.split(",")] if raw else list(default)
    values = [item for item in values if item]
    return values[:limit] if limit is not None else values


def build_output_path(model: str, output: str | None) -> Path:
    if output:
        return Path(output)
    safe_name = model.replace(":", "__").replace("/", "__")
    return ROOT / "repl" / "outputs" / "raw" / f"{safe_name}_task1_raw.csv"


def main() -> None:
    load_dotenv()
    args = parse_args()
    countries = pick_list(args.countries, DEFAULT_COUNTRIES, args.limit_countries)
    topics = pick_list(args.topics, DEFAULT_TOPICS, args.limit_topics)
    prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(",") if x.strip()]

    prompting = Task1Prompting()
    client = ai.Client()
    rows = []

    for country in countries:
        for topic in tqdm(topics, desc=f"topics:{country}"):
            for prompt_index in prompt_indices:
                prompt = prompting.generate_prompt(country=country, scenario=topic, index=prompt_index)
                start = time.time()
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                )
                duration = time.time() - start
                content = response.choices[0].message.content
                parsed = parse_json_response(content)
                rows.append(
                    {
                        "country": country,
                        "topic": topic,
                        "prompt_index": prompt_index,
                        "raw_response": content,
                        "parsed_json": None if parsed is None else str(parsed),
                        "duration_sec": duration,
                        "model": args.model,
                    }
                )

    out_path = build_output_path(args.model, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} task1 evaluations to {out_path}")


if __name__ == "__main__":
    main()
