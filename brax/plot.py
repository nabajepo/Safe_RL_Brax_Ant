# plot.py
# =========================================================
# Compare final aggregated metrics across:
#   - no_constraint
#   - soft_constraint
#   - hard_constraint
#
# Input expected:
#   Results/t2m/<model>/aggregated/final_table.csv
#
# Output:
#   Results/t2m/Comparison/
#       - one PNG per metric
#       - combined_summary.csv
# =========================================================

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RESULTS_ROOT = Path("Results") / "t2m"
MODELS = ["no_constraint", "soft_constraint", "hard_constraint"]
METRICS = [
    "violations_per_100_steps",
    "success_rate",
    "avg_time_to_failure",
    "mean_episode_length",
    "avg_collisions_per_episode",
    "avg_episode_reward",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_final_table(model_name: str) -> pd.DataFrame:
    csv_path = RESULTS_ROOT / model_name / "aggregated" / "final_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def build_summary_table() -> pd.DataFrame:
    rows = []

    for model_name in MODELS:
        df = load_final_table(model_name)

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

    # couleurs différentes
    colors = [
        "#1f77b4",  # bleu
        "#ff7f0e",  # orange
        "#2ca02c",  # vert
    ]

    plt.figure(figsize=(9,6))

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="black"
    )

    plt.xticks(x, labels)
    plt.ylabel(metric)
    plt.title(f"{metric} comparison across models")

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(out_dir / f"{metric}.png", dpi=160)
    plt.close()

def plot_all_metrics(summary_df: pd.DataFrame, out_dir: Path):
    for metric in METRICS:
        plot_metric_bar(summary_df, metric, out_dir)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    comparison_dir = RESULTS_ROOT / "Comparison"
    ensure_dir(comparison_dir)

    summary_df = build_summary_table()
    if summary_df.empty:
        raise RuntimeError("No comparison data found. Check final_table.csv files.")

    save_summary_csv(summary_df, comparison_dir)
    plot_all_metrics(summary_df, comparison_dir)

    print("Comparison plots created in:")
    print(comparison_dir)


if __name__ == "__main__":
    main()