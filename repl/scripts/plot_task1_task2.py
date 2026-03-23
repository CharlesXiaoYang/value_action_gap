import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Task 1 versus Task 2 comparison figures.")
    parser.add_argument("--comparison", type=str, required=True, help="Comparison CSV path.")
    parser.add_argument("--task1", type=str, required=True, help="Parsed Task 1 CSV path.")
    parser.add_argument("--task2", type=str, required=True, help="Raw Task 2 CSV path.")
    parser.add_argument("--outdir", type=str, default=None, help="Optional figure output directory.")
    return parser.parse_args()


def build_outdir(comparison_path: Path, outdir: str | None) -> Path:
    if outdir:
        return Path(outdir)
    return comparison_path.parents[1] / "figures" / comparison_path.stem


def add_point_labels(ax, frame: pd.DataFrame, xcol: str, ycol: str) -> None:
    for _, row in frame.iterrows():
        ax.annotate(row["value"], (row[xcol], row[ycol]), xytext=(5, 5), textcoords="offset points", fontsize=8)


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison)
    outdir = build_outdir(comparison_path, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmp = pd.read_csv(comparison_path)
    task1 = pd.read_csv(args.task1)
    task2 = pd.read_csv(args.task2)

    if cmp.empty:
        raise ValueError("Comparison file is empty; nothing to plot.")

    plt.style.use("ggplot")

    ordered = cmp.sort_values("positive_choice_rate", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(cmp["endorsement_rate"], cmp["positive_choice_rate"], s=90, color="#0b7285")
    add_point_labels(ax, cmp, "endorsement_rate", "positive_choice_rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set_xlabel("Task 1 endorsement rate (derived from mean Likert score)")
    ax.set_ylabel("Task 2 positive-choice rate")
    ax.set_title("Task 1 vs Task 2 Alignment Scatter")
    fig.tight_layout()
    fig.savefig(outdir / "task1_task2_scatter.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = range(len(ordered))
    ax.barh(y, ordered["endorsement_rate"], height=0.38, label="Task 1 endorsement rate", color="#1f77b4")
    ax.barh([i + 0.4 for i in y], ordered["positive_choice_rate"], height=0.38, label="Task 2 positive-choice rate", color="#ff7f0e")
    ax.set_yticks([i + 0.2 for i in y])
    ax.set_yticklabels(ordered["value"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_title("Task 1 vs Task 2 by Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "task1_task2_bars.png", dpi=200)
    plt.close(fig)

    task1_subset = task1[(task1["parse_success"] == True) & (task1["value"].isin(cmp["value"]))].copy()
    heat1 = task1_subset.pivot_table(index="prompt_index", columns="value", values="likert_score", aggfunc="mean")
    heat1 = heat1[cmp.sort_values("value")["value"].tolist()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(heat1.values, aspect="auto", cmap="YlGnBu", vmin=1, vmax=4)
    ax.set_xticks(range(len(heat1.columns)))
    ax.set_xticklabels(heat1.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat1.index)))
    ax.set_yticklabels(heat1.index)
    ax.set_xlabel("Value")
    ax.set_ylabel("Task 1 prompt index")
    ax.set_title("Task 1 Likert Scores Across Prompt Variants")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Likert score")
    fig.tight_layout()
    fig.savefig(outdir / "task1_prompt_heatmap.png", dpi=200)
    plt.close(fig)

    task2_subset = task2[task2["value"].isin(cmp["value"])].copy()
    task2_subset["positive_numeric"] = task2_subset["selected_polarity"].map({"positive": 1, "negative": 0})
    heat2 = task2_subset.pivot_table(index="prompt_index", columns="value", values="positive_numeric", aggfunc="mean")
    heat2 = heat2[cmp.sort_values("value")["value"].tolist()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(heat2.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat2.columns)))
    ax.set_xticklabels(heat2.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat2.index)))
    ax.set_yticklabels(heat2.index)
    ax.set_xlabel("Value")
    ax.set_ylabel("Task 2 prompt index")
    ax.set_title("Task 2 Positive Choice Rate Across Prompt Variants")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Positive choice rate")
    fig.tight_layout()
    fig.savefig(outdir / "task2_prompt_heatmap.png", dpi=200)
    plt.close(fig)

    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()
