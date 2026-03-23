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

from task2_prompting import Task2Prompting
from task2_utils import parse_json_response, resolve_selected_polarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 value-action evaluation on repl pairs.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "repl" / "data" / "pairs.csv"),
        help="CSV with paired positive and negative actions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Aisuite model id, e.g. groq:qwen/qwen3-32b or openai:gpt-4o-mini.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to repl/outputs/raw/<model>_task2_raw.csv",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Optional starting row offset before applying limit.",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default=None,
        help="Optional comma-separated country filter.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Optional comma-separated topic filter.",
    )
    return parser.parse_args()


def parse_filter_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def build_output_path(model: str, output: str | None) -> Path:
    if output:
        return Path(output)

    safe_name = model.replace(":", "__").replace("/", "__")
    return ROOT / "repl" / "outputs" / "raw" / f"{safe_name}_task2_raw.csv"


def run_completion(client: ai.Client, model: str, prompt: str, temperature: float) -> tuple[str, float]:
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    duration = time.time() - start
    content = response.choices[0].message.content
    return content, duration


def main() -> None:
    load_dotenv()
    args = parse_args()

    df = pd.read_csv(args.input)

    countries = parse_filter_list(args.countries)
    topics = parse_filter_list(args.topics)
    if countries is not None:
        df = df[df["country"].isin(countries)]
    if topics is not None:
        df = df[df["topic"].isin(topics)]

    if args.offset:
        df = df.iloc[args.offset :]
    if args.limit is not None:
        df = df.head(args.limit)

    df = df.reset_index(drop=False)

    prompting = Task2Prompting()
    client = ai.Client()
    output_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="pairs"):
        for prompt_index in range(8):
            prompt, reverse_order = prompting.generate_prompt(
                country=row["country"],
                topic=row["topic"],
                value=row["value"],
                option1=row["action_pos"],
                option2=row["action_neg"],
                index=prompt_index,
            )

            raw_response, duration = run_completion(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
            )
            parsed = parse_json_response(raw_response) or {}
            selected_polarity = resolve_selected_polarity(parsed.get("action"), reverse_order)

            output_rows.append(
                {
                    "pair_id": int(row["index"]),
                    "country": row["country"],
                    "topic": row["topic"],
                    "value": row["value"],
                    "prompt_index": prompt_index,
                    "reverse_order": reverse_order,
                    "action_pos": row["action_pos"],
                    "action_neg": row["action_neg"],
                    "raw_response": raw_response,
                    "parsed_action": parsed.get("action"),
                    "parsed_explanation": parsed.get("explanation"),
                    "selected_polarity": selected_polarity,
                    "selected_action": (
                        row["action_pos"]
                        if selected_polarity == "positive"
                        else row["action_neg"]
                        if selected_polarity == "negative"
                        else None
                    ),
                    "duration_sec": duration,
                    "model": args.model,
                }
            )

    out_path = build_output_path(args.model, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(out_path, index=False)
    print(f"Wrote {len(output_rows)} task2 evaluations to {out_path}")


if __name__ == "__main__":
    main()
