# plot_models.py
# =========================================================
# Compare final aggregated metrics across:
#   - no_constraint
#   - soft_constraint
#   - hard_constraint
#
# Expected input:
#   Results/<budget_tag>/<model>/aggregated/final_table.csv
#
# Example:
#   Results/t2m/no_constraint/aggregated/final_table.csv
#   Results/t2m/soft_constraint/aggregated/final_table.csv
#   Results/t2m/hard_constraint/aggregated/final_table.csv
#
# Output:
#   Results/<budget_tag>/Comparison/
#       - one PNG per metric
#       - combined_summary.csv
# =========================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODELS = ["no_constraint", "soft_constraint", "hard_constraint"]

METRICS = [
    "violations_per_100_steps",
    "success_rate",
    "avg_time_to_failure",
    "mean_episode_length",
    "avg_collisions_per_episode",
    "avg_episode_reward",
]

MODEL_COLORS = {
    "no_constraint": "#1f77b4",   # bleu
    "soft_constraint": "#ff7f0e", # orange
    "hard_constraint": "#2ca02c", # vert
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_final_table(results_root: Path, budget_tag: str, model_name: str) -> pd.DataFrame:
    csv_path = results_root / budget_tag / model_name / "aggregated" / "final_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)


def build_summary_table(results_root: Path, budget_tag: str) -> pd.DataFrame:
    rows = []

    for model_name in MODELS:
        df = load_final_table(results_root, budget_tag, model_name)

        for metric in METRICS:
            sub = df[df["metric"] == metric]
            if sub.empty:
                continue

            mean_val = pd.to_numeric(sub["mean"], errors="coerce").iloc[0]
            std_val = pd.to_numeric(sub["std"], errors="coerce").iloc[0]

            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(mean_val) if pd.notna(mean_val) else 0.0,
                    "std": float(std_val) if pd.notna(std_val) else 0.0,
                }
            )

    return pd.DataFrame(rows)


def save_summary_csv(summary_df: pd.DataFrame, out_dir: Path):
    out_csv = out_dir / "combined_summary.csv"
    summary_df.to_csv(out_csv, index=False)


def add_value_labels(ax, x_positions, means):
    if len(means) == 0:
        return

    max_mean = max(float(m) for m in means) if len(means) > 0 else 0.0
    offset = max(0.02 * max(abs(max_mean), 1.0), 0.02)

    for x, y in zip(x_positions, means):
        ax.text(
            x,
            y + offset,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_metric_bar(summary_df: pd.DataFrame, metric: str, out_dir: Path):
    sub = summary_df[summary_df["metric"] == metric].copy()
    if sub.empty:
        return

    sub["model"] = pd.Categorical(sub["model"], categories=MODELS, ordered=True)
    sub = sub.sort_values("model")

    x = np.arange(len(sub))
    means = sub["mean"].to_numpy(dtype=float)
    stds = sub["std"].to_numpy(dtype=float)
    labels = sub["model"].tolist()
    colors = [MODEL_COLORS[m] for m in labels]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} comparison across models")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    add_value_labels(ax, x, means)

    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}.png", dpi=160)
    plt.close(fig)


def plot_all_metrics(summary_df: pd.DataFrame, out_dir: Path):
    for metric in METRICS:
        plot_metric_bar(summary_df, metric, out_dir)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="Results")
    parser.add_argument("--budget_tag", type=str, required=True)
    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    results_root = Path(args.results_root)
    budget_tag = args.budget_tag

    comparison_dir = results_root / budget_tag / "Comparison"
    ensure_dir(comparison_dir)

    summary_df = build_summary_table(results_root, budget_tag)
    if summary_df.empty:
        raise RuntimeError(
            f"No comparison data found for budget '{budget_tag}'. "
            f"Check aggregated/final_table.csv files."
        )

    save_summary_csv(summary_df, comparison_dir)
    plot_all_metrics(summary_df, comparison_dir)

    print("Comparison plots created in:")
    print(comparison_dir)


if __name__ == "__main__":
    main()
