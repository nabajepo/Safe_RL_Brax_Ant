# train_model.py
# =========================================================
# Unified training entry point for:
#   - no_constraint
#   - soft_constraint
#   - hard_constraint
#
# This script trains exactly ONE seed.
# It is designed to work well with Slurm parallel jobs.
# =========================================================

import argparse
from pathlib import Path

from baseline_env import Cfg
from scripts.train_pipeline import (
    ensure_dir,
    format_budget_tag,
    run_single_seed,
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
        help="Model family to train.",
    )

    # -----------------------------------------------------
    # Single seed only
    # -----------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Single seed to train.",
    )

    # -----------------------------------------------------
    # Main training budget
    # -----------------------------------------------------
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--results_root", type=str, default="Results")

    # -----------------------------------------------------
    # PPO / rollout settings
    # -----------------------------------------------------
    parser.add_argument("--steps_per_env", type=int, default=64)
    parser.add_argument("--ppo_epochs", type=int, default=8)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=256)

    # -----------------------------------------------------
    # PPO hyperparameters
    # -----------------------------------------------------
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # -----------------------------------------------------
    # Evaluation / checkpoints
    # -----------------------------------------------------
    parser.add_argument("--eval_every_iters", type=int, default=5)
    parser.add_argument("--eval_eps", type=int, default=50)
    parser.add_argument("--final_eval_eps", type=int, default=200)
    parser.add_argument("--rollouts_per_seed", type=int, default=3)
    parser.add_argument("--checkpoint_every_timesteps", type=int, default=5_000_000)
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,  # Default to True for Colab
        help="Automatically resume from latest checkpoint if available",
    )

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    seed = args.seed

    cfg = Cfg(
        num_envs=args.num_envs,
        max_steps=args.max_steps,
    )

    budget_tag = format_budget_tag(args.timesteps)
    budget_dir = Path(args.results_root) / budget_tag
    ensure_dir(budget_dir)

    print("\n" + "#" * 90, flush=True)
    print(f"Training configuration ({model_name})", flush=True)
    print(f"seed                       : {seed}", flush=True)
    print(f"timesteps                  : {args.timesteps}", flush=True)
    print(f"budget_tag                 : {budget_tag}", flush=True)
    print(f"num_envs                   : {args.num_envs}", flush=True)
    print(f"max_steps                  : {args.max_steps}", flush=True)
    print(f"steps_per_env              : {args.steps_per_env}", flush=True)
    print(f"ppo_epochs                 : {args.ppo_epochs}", flush=True)
    print(f"minibatch_size             : {args.minibatch_size}", flush=True)
    print(f"hidden_dim                 : {args.hidden_dim}", flush=True)
    print(f"learning_rate              : {args.learning_rate}", flush=True)
    print(f"gamma                      : {args.gamma}", flush=True)
    print(f"gae_lambda                 : {args.gae_lambda}", flush=True)
    print(f"clip_eps                   : {args.clip_eps}", flush=True)
    print(f"value_coef                 : {args.value_coef}", flush=True)
    print(f"entropy_coef               : {args.entropy_coef}", flush=True)
    print(f"max_grad_norm              : {args.max_grad_norm}", flush=True)
    print(f"eval_every_iters           : {args.eval_every_iters}", flush=True)
    print(f"eval_eps                   : {args.eval_eps}", flush=True)
    print(f"final_eval_eps             : {args.final_eval_eps}", flush=True)
    print(f"rollouts_per_seed          : {args.rollouts_per_seed}", flush=True)
    print(f"checkpoint_every_timesteps : {args.checkpoint_every_timesteps}", flush=True)
    print(f"results_root               : {args.results_root}", flush=True)
    print("#" * 90 + "\n", flush=True)

    model_budget_dir = budget_dir / model_name
    ensure_dir(model_budget_dir)

    run_single_seed(
        model_name=model_name,
        seed=seed,
        args=args,
        model_budget_dir=model_budget_dir,
        cfg=cfg,
        auto_resume=args.auto_resume, 
    )

    print(f"\nTraining completed for model: {model_name} | seed={seed}", flush=True)
    print(f"Results available in: {model_budget_dir / f'seed_{seed}'}", flush=True)

if __name__ == "__main__":
    main()