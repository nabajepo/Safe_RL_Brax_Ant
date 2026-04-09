# train_individual_brax_env.py
# =========================================================
# Train only one Brax model family at a time.
#
# This script reuses the same training pipeline as
# train_brax_env.py, but limits execution to one model:
#   - no_constraint
#   - soft_constraint
#   - hard_constraint
# =========================================================

import argparse
from pathlib import Path

import no_constraint_env_brax   # noqa: F401
import soft_constraint_env_brax # noqa: F401
import hard_constraint_env_brax # noqa: F401

from baseline_env_brax import Cfg, ENV_NAME_ORDER
from train_brax_env import (
    ensure_dir,
    format_budget_tag,
    run_single_seed,
    aggregate_model_results,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ENV_NAME_ORDER,
        help="Model family to train: no_constraint / soft_constraint / hard_constraint",
    )

    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--results_root", type=str, default="Results")

    parser.add_argument("--steps_per_env", type=int, default=512)
    parser.add_argument("--ppo_epochs", type=int, default=8)
    parser.add_argument("--minibatch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_every_iters", type=int, default=5)
    parser.add_argument("--eval_eps", type=int, default=50)
    parser.add_argument("--final_eval_eps", type=int, default=200)
    parser.add_argument("--rollouts_per_seed", type=int, default=3)

    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=300)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Cfg(
        num_envs=args.num_envs,
        max_steps=args.max_steps,
    )

    budget_tag = format_budget_tag(args.timesteps)
    budget_dir = Path(args.results_root) / budget_tag
    ensure_dir(budget_dir)

    print("\n" + "#" * 90, flush=True)
    print("Individual training configuration", flush=True)
    print(f"model              : {args.model}", flush=True)
    print(f"timesteps          : {args.timesteps}", flush=True)
    print(f"budget_tag         : {budget_tag}", flush=True)
    print(f"seeds              : {args.seeds}", flush=True)
    print(f"num_envs           : {args.num_envs}", flush=True)
    print(f"max_steps          : {args.max_steps}", flush=True)
    print(f"steps_per_env      : {args.steps_per_env}", flush=True)
    print(f"ppo_epochs         : {args.ppo_epochs}", flush=True)
    print(f"minibatch_size     : {args.minibatch_size}", flush=True)
    print(f"hidden_dim         : {args.hidden_dim}", flush=True)
    print(f"eval_every_iters   : {args.eval_every_iters}", flush=True)
    print(f"eval_eps           : {args.eval_eps}", flush=True)
    print(f"final_eval_eps     : {args.final_eval_eps}", flush=True)
    print(f"rollouts_per_seed  : {args.rollouts_per_seed}", flush=True)
    print(f"results_root       : {args.results_root}", flush=True)
    print("#" * 90 + "\n", flush=True)

    model_budget_dir = budget_dir / args.model
    ensure_dir(model_budget_dir)

    for seed in args.seeds:
        run_single_seed(
            model_name=args.model,
            seed=seed,
            args=args,
            model_budget_dir=model_budget_dir,
            cfg=cfg,
        )

    aggregate_model_results(
        model_name=args.model,
        model_budget_dir=model_budget_dir,
        seeds=args.seeds,
    )

    print("\nIndividual training completed successfully.", flush=True)
    print(f"Results available in: {model_budget_dir}", flush=True)


if __name__ == "__main__":
    main()