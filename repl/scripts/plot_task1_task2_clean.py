import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaner Task 1 vs Task 2 figures.")
    parser.add_argument("--comparison", type=str, required=True, help="Comparison CSV path.")
    parser.add_argument("--task1", type=str, required=True, help="Parsed Task 1 CSV path.")
    parser.add_argument("--task2", type=str, required=True, help="Raw Task 2 CSV path.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for figures.")
    return parser.parse_args()


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmp = pd.read_csv(args.comparison).sort_values("value").reset_index(drop=True)
    task1 = pd.read_csv(args.task1)
    task2 = pd.read_csv(args.task2)

    plt.style.use("ggplot")

    ordered_gap = cmp.sort_values("gap_abs", ascending=False).reset_index(drop=True)
    ordered_pos = cmp.sort_values("positive_choice_rate", ascending=False).reset_index(drop=True)
    top_gap = ordered_gap.head(15).sort_values("gap_signed")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(cmp["endorsement_rate"], cmp["positive_choice_rate"], s=80, color="#0b7285", alpha=0.85)
    for _, row in ordered_gap.head(12).iterrows():
        ax.annotate(row["value"], (row["endorsement_rate"], row["positive_choice_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Task 1 endorsement rate")
    ax.set_ylabel("Task 2 positive-choice rate")
    ax.set_title("Task 1 vs Task 2 Alignment")
    savefig(fig, outdir / "01_alignment_scatter_clean.png")

    fig, ax = plt.subplots(figsize=(12, 12))
    y = range(len(ordered_pos))
    ax.barh(y, ordered_pos["positive_choice_rate"], color="#ff7f0e", label="Task 2 positive-choice rate")
    ax.barh(y, ordered_pos["endorsement_rate"], color="#1f77b4", alpha=0.7, label="Task 1 endorsement rate")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ordered_pos["value"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_title("All Values Ranked by Task 2 Positive-Choice Rate")
    ax.legend()
    savefig(fig, outdir / "02_all_values_ranked.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#c92a2a" if v < 0 else "#2b8a3e" for v in top_gap["gap_signed"]]
    ax.barh(top_gap["value"], top_gap["gap_signed"], color=colors)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel("Task 2 positive-choice rate minus Task 1 endorsement rate")
    ax.set_title("Largest Value-Action Gaps")
    savefig(fig, outdir / "03_largest_gaps.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    counts = task2["selected_polarity"].fillna("parse_failure").value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    palette = ["#2b8a3e" if x == "positive" else "#c92a2a" if x == "negative" else "#868e96" for x in labels]
    ax.bar(labels, values, color=palette)
    ax.set_ylabel("Count")
    ax.set_title("Task 2 Choice Counts")
    savefig(fig, outdir / "04_choice_counts.png")

    task1_subset = task1[(task1["parse_success"] == True) & (task1["value"].isin(cmp["value"]))].copy()
    task1_heat = task1_subset.pivot_table(index="prompt_index", columns="value", values="likert_score", aggfunc="mean")
    task1_heat = task1_heat[ordered_gap.head(20).sort_values("value")["value"].tolist()]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(task1_heat.values, aspect="auto", cmap="YlGnBu", vmin=1, vmax=4)
    ax.set_xticks(range(len(task1_heat.columns)))
    ax.set_xticklabels(task1_heat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(task1_heat.index)))
    ax.set_yticklabels(task1_heat.index)
    ax.set_xlabel("Top 20 gap values")
    ax.set_ylabel("Task 1 prompt index")
    ax.set_title("Task 1 Scores for Top Gap Values")
    fig.colorbar(im, ax=ax, label="Likert score")
    savefig(fig, outdir / "05_task1_top_gap_heatmap.png")

    task2_subset = task2[task2["value"].isin(cmp["value"])].copy()
    task2_subset["positive_numeric"] = task2_subset["selected_polarity"].map({"positive": 1, "negative": 0})
    task2_heat = task2_subset.pivot_table(index="prompt_index", columns="value", values="positive_numeric", aggfunc="mean")
    task2_heat = task2_heat[ordered_gap.head(20).sort_values("value")["value"].tolist()]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(task2_heat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(task2_heat.columns)))
    ax.set_xticklabels(task2_heat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(task2_heat.index)))
    ax.set_yticklabels(task2_heat.index)
    ax.set_xlabel("Top 20 gap values")
    ax.set_ylabel("Task 2 prompt index")
    ax.set_title("Task 2 Positive Choice Rates for Top Gap Values")
    fig.colorbar(im, ax=ax, label="Positive choice rate")
    savefig(fig, outdir / "06_task2_top_gap_heatmap.png")

    print(f"Wrote cleaner figures to {outdir}")


if __name__ == "__main__":
    main()
