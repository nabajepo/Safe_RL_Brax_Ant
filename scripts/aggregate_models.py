# aggregate_models.py
# =========================================================
# Aggregation entry point for:
#   - no_constraint
#   - soft_constraint
#   - hard_constraint
#
# This script is used AFTER training multiple seeds
# independently, for example with Slurm parallel jobs.
#
# It reads:
#   Results/<budget_tag>/<model_name>/seed_<seed>/...
#
# And creates:
#   Results/<budget_tag>/<model_name>/aggregated/...
#   Results/<budget_tag>/<model_name>/learning_curves/*.png
# =========================================================

import argparse
from pathlib import Path

from scripts.train_pipeline import (
    ensure_dir,
    format_budget_tag,
    aggregate_model_results,
)


VALID_MODELS = ["no_constraint", "soft_constraint", "hard_constraint"]


def parse_args():
    parser = argparse.ArgumentParser()

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=VALID_MODELS,
        help="Model family to aggregate.",
    )

    # -----------------------------------------------------
    # Seeds to aggregate
    # -----------------------------------------------------
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seeds to aggregate, e.g. --seeds 0 1 2 3 4",
    )

    # -----------------------------------------------------
    # Budget / path
    # -----------------------------------------------------
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help="Training budget used for this run, e.g. 2000000",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="Results",
        help="Root results directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name
    budget_tag = format_budget_tag(args.timesteps)
    budget_dir = Path(args.results_root) / budget_tag
    ensure_dir(budget_dir)

    model_budget_dir = budget_dir / model_name
    ensure_dir(model_budget_dir)

    print("\n" + "#" * 90, flush=True)
    print(f"Aggregating results for model: {model_name}", flush=True)
    print(f"budget_tag   : {budget_tag}", flush=True)
    print(f"seeds        : {args.seeds}", flush=True)
    print(f"results_root : {args.results_root}", flush=True)
    print("#" * 90 + "\n", flush=True)

    aggregate_model_results(
        model_name=model_name,
        model_budget_dir=model_budget_dir,
        seeds=args.seeds,
    )

    print(f"\nAggregation completed for model: {model_name}", flush=True)
    print(f"Aggregated results available in: {model_budget_dir / 'aggregated'}", flush=True)
    print(f"Learning-curve plots available in: {model_budget_dir / 'learning_curves'}", flush=True)


if __name__ == "__main__":
    main()