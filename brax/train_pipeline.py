# train_pipeline.py
# =========================================================
# Shared training pipeline for the CSI4900 Brax project.
#
# This file is used by:
#   - train_no_constraint.py
#   - train_soft_constraint.py
#   - train_hard_constraint.py
#
# Main responsibilities:
#   - training utilities
#   - pure JAX actor-critic network
#   - PPO rollout collection and updates
#   - evaluation helpers
#   - rollout generation and plotting
#   - checkpointing / latest / best model handling
#   - aggregation across seeds
# =========================================================

import csv
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

# Register environments on import
import no_constraint_env  # noqa: F401
import soft_constraint_env  # noqa: F401
import hard_constraint_env  # noqa: F401

from baseline_env import Cfg, get_env_class


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_budget_tag(timesteps: int) -> str:
    if timesteps >= 1_000_000 and timesteps % 1_000_000 == 0:
        return f"t{timesteps // 1_000_000}m"
    if timesteps >= 1000 and timesteps % 1000 == 0:
        return f"t{timesteps // 1000}k"
    return f"t{timesteps}"


def format_checkpoint_tag(timesteps: int) -> str:
    if timesteps >= 1_000_000 and timesteps % 1_000_000 == 0:
        return f"t{timesteps // 1_000_000}m"
    if timesteps >= 1000 and timesteps % 1000 == 0:
        return f"t{timesteps // 1000}k"
    return f"t{timesteps}"


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def sanitize_scalar(metric_name: str, value: Any, fallback_row: Dict[str, Any] | None = None) -> float:
    """
    Convert values to finite float and replace missing values.

    Special rules:
      - avg_time_to_failure -> mean_episode_length if missing
      - mean_episode_length -> max_steps if missing
    General rule:
      - other missing values -> 0.0
    """
    if not _is_missing(value):
        try:
            v = float(value)
            if np.isfinite(v):
                return v
        except Exception:
            pass

    if metric_name == "avg_time_to_failure":
        if fallback_row is not None:
            mel = fallback_row.get("mean_episode_length", 0.0)
            if not _is_missing(mel):
                try:
                    mel_f = float(mel)
                    if np.isfinite(mel_f):
                        return mel_f
                except Exception:
                    pass
        return 0.0

    if metric_name == "mean_episode_length":
        if fallback_row is not None:
            ms = fallback_row.get("max_steps", 0.0)
            if not _is_missing(ms):
                try:
                    ms_f = float(ms)
                    if np.isfinite(ms_f):
                        return ms_f
                except Exception:
                    pass
        return 0.0

    return 0.0


def sanitize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)

    keys_to_force_numeric = [
        "iteration",
        "timesteps",
        "episodes",
        "seed",
        "rollout_id",
        "success_rate",
        "mean_episode_length",
        "avg_episode_reward",
        "violations_per_100_steps",
        "avg_time_to_failure",
        "failure_rate",
        "avg_collisions_per_episode",
        "avg_oob_per_episode",
        "avg_speed_violations_per_episode",
        "avg_falls_per_episode",
        "avg_total_violations_per_episode",
        "success",
        "episode_length",
        "episode_reward",
        "collision_count",
        "oob_count",
        "speed_count",
        "fall_count",
        "total_violations",
        "time_to_failure",
        "x",
        "y",
        "iter_time_sec",
        "train_time_sec",
        "eval_time_sec",
        "elapsed_time_sec",
        "timesteps_per_sec",
        "best_score",
    ]

    if "max_steps" not in out:
        out["max_steps"] = 0.0

    for k in keys_to_force_numeric:
        if k in out:
            out[k] = sanitize_scalar(k, out[k], out)

    int_like = {"iteration", "timesteps", "episodes", "seed", "rollout_id"}
    for k in int_like:
        if k in out:
            out[k] = int(round(float(out[k])))

    return out


def save_dict_rows_to_csv(rows, path):
    if not rows:
        return
    ensure_dir(Path(path).parent)
    clean_rows = [sanitize_record(r) for r in rows]
    keys = list(clean_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(clean_rows)


def save_dict_to_json(data, path):
    ensure_dir(Path(path).parent)
    if isinstance(data, dict):
        data = sanitize_record(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_pickle(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_training_state(path, params, opt_state, meta: Dict[str, Any]):
    payload = {
        "params": params,
        "opt_state": opt_state,
        "meta": meta,
    }
    save_pickle(payload, path)


def load_training_state(path):
    return load_pickle(path)


# =========================================================
# Best model selection
# =========================================================
def compute_selection_score(stats: Dict[str, Any]) -> tuple:
    """
    Hierarchical model selection rule:
      1) higher success_rate is better
      2) lower violations_per_100_steps is better
      3) higher avg_episode_reward is better
    """
    success = float(stats.get("success_rate", 0.0))
    violations = float(stats.get("violations_per_100_steps", 0.0))
    reward = float(stats.get("avg_episode_reward", 0.0))
    return (success, -violations, reward)


def is_better_model(candidate_stats: Dict[str, Any], best_stats: Dict[str, Any] | None) -> bool:
    if best_stats is None:
        return True
    return compute_selection_score(candidate_stats) > compute_selection_score(best_stats)


# =========================================================
# Pure-JAX Actor-Critic Network
# =========================================================
def glorot_init(rng, in_dim, out_dim):
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    w = jax.random.uniform(rng, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return w.astype(jnp.float32), b


def init_mlp_params(rng, obs_dim, action_dim, hidden_dim=256):
    keys = jax.random.split(rng, 6)

    pw1, pb1 = glorot_init(keys[0], obs_dim, hidden_dim)
    pw2, pb2 = glorot_init(keys[1], hidden_dim, hidden_dim)
    pwm, pbm = glorot_init(keys[2], hidden_dim, action_dim)

    vw1, vb1 = glorot_init(keys[3], obs_dim, hidden_dim)
    vw2, vb2 = glorot_init(keys[4], hidden_dim, hidden_dim)
    vwo, vbo = glorot_init(keys[5], hidden_dim, 1)

    params = {
        "policy": {
            "w1": pw1,
            "b1": pb1,
            "w2": pw2,
            "b2": pb2,
            "w_mean": pwm,
            "b_mean": pbm,
            "log_std": jnp.zeros((action_dim,), dtype=jnp.float32),
        },
        "value": {
            "w1": vw1,
            "b1": vb1,
            "w2": vw2,
            "b2": vb2,
            "w_out": vwo,
            "b_out": vbo,
        },
    }
    return params


def mlp_forward(params, obs):
    p = params["policy"]
    px = jnp.tanh(obs @ p["w1"] + p["b1"])
    px = jnp.tanh(px @ p["w2"] + p["b2"])
    mean = px @ p["w_mean"] + p["b_mean"]
    log_std = p["log_std"]

    v = params["value"]
    vx = jnp.tanh(obs @ v["w1"] + v["b1"])
    vx = jnp.tanh(vx @ v["w2"] + v["b2"])
    value = vx @ v["w_out"] + v["b_out"]
    value = value.squeeze(-1)

    return mean, log_std, value


def gaussian_log_prob(action, mean, log_std):
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(
        ((action - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def select_action(params, obs, rng, deterministic=False):
    mean, log_std, value = mlp_forward(params, obs)
    log_std = jnp.clip(log_std, -4.0, 1.0)

    if deterministic:
        action = jnp.clip(mean, -1.0, 1.0)
        log_prob = gaussian_log_prob(action, mean, log_std)
        return action, log_prob, value

    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + std * noise
    action = jnp.clip(action, -1.0, 1.0)
    log_prob = gaussian_log_prob(action, mean, log_std)
    return action, log_prob, value


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    next_value = last_value

    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages.insert(0, gae)
        next_value = values[t]

    advantages = jnp.array(advantages, dtype=jnp.float32)
    returns = advantages + jnp.array(values, dtype=jnp.float32)
    return advantages, returns


def ppo_loss(
    params,
    obs,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    mean, log_std, values = mlp_forward(params, obs)
    log_std = jnp.clip(log_std, -4.0, 1.0)

    log_probs = gaussian_log_prob(actions, mean, log_std)
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    value_loss = jnp.mean((returns - values) ** 2)

    std = jnp.exp(log_std)
    entropy = jnp.mean(
        0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * jnp.e * (std ** 2)), axis=-1)
    )

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total_loss


# =========================================================
# Metrics helpers
# =========================================================
def empty_episode_counters():
    return {
        "collision_count": 0.0,
        "oob_count": 0.0,
        "speed_count": 0.0,
        "fall_count": 0.0,
        "total_violations": 0.0,
        "time_to_failure": np.nan,
        "failed": 0.0,
    }


def update_episode_counters(counters, metrics, current_step):
    collision = float(metrics["collision"][0])
    oob = float(metrics["out_of_bounds"][0])
    speed = float(metrics["speed_violation"][0])
    fall = float(metrics["fall"][0])

    counters["collision_count"] += collision
    counters["oob_count"] += oob
    counters["speed_count"] += speed
    counters["fall_count"] += fall

    violation_now = collision + oob + speed + fall
    counters["total_violations"] += violation_now

    if np.isnan(counters["time_to_failure"]) and violation_now > 0.0:
        counters["time_to_failure"] = float(current_step)
        counters["failed"] = 1.0

    return counters


# =========================================================
# Evaluation and rollout helpers
# =========================================================
def evaluate_model(
    params,
    model_name: str,
    cfg: Cfg,
    n_episodes=100,
    seed=123,
):
    env_cls = get_env_class(model_name)

    successes = []
    episode_lengths = []
    episode_rewards = []

    collisions_per_episode = []
    oob_per_episode = []
    speed_per_episode = []
    fall_per_episode = []
    total_violations_per_episode = []

    violations_per_100_steps_list = []
    time_to_failure_list = []

    for ep in range(n_episodes):
        ep_cfg = Cfg(**cfg.__dict__)
        ep_cfg.num_envs = 1

        env = env_cls(ep_cfg)
        rng_key = jax.random.PRNGKey(seed + ep)
        task_base_seed = (seed + ep) * 100_000 + 1 # fixed constant iter offset for eval
        obs, _ = env.reset(rng_key, base_seed=task_base_seed)

        done = np.array([False])
        ep_reward = 0.0
        last_metrics = None

        counters = empty_episode_counters()
        act_key = jax.random.PRNGKey(seed + ep + 100_000)

        while not bool(done[0]):
            act_key, subkey = jax.random.split(act_key)
            action, _, _ = select_action(
                params=params,
                obs=obs,
                rng=subkey,
                deterministic=True,
            )

            obs, reward, done, metrics = env.step(np.array(action))
            ep_reward += float(np.array(reward)[0])
            last_metrics = metrics

            current_step = int(metrics["steps"][0])
            counters = update_episode_counters(counters, metrics, current_step)

        ep_len = float(last_metrics["steps"][0])

        successes.append(float(last_metrics["success"][0]))
        episode_lengths.append(ep_len)
        episode_rewards.append(float(ep_reward))

        collisions_per_episode.append(counters["collision_count"])
        oob_per_episode.append(counters["oob_count"])
        speed_per_episode.append(counters["speed_count"])
        fall_per_episode.append(counters["fall_count"])
        total_violations_per_episode.append(counters["total_violations"])

        if ep_len > 0:
            violations_per_100_steps_list.append(
                100.0 * counters["total_violations"] / ep_len
            )
        else:
            violations_per_100_steps_list.append(0.0)

        time_to_failure_list.append(counters["time_to_failure"])

    valid_ttf = [x for x in time_to_failure_list if not np.isnan(x)]
    avg_time_to_failure = float(np.mean(valid_ttf)) if valid_ttf else float(np.mean(episode_lengths))

    result = {
        "episodes": int(n_episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "violations_per_100_steps": float(np.mean(violations_per_100_steps_list)) if violations_per_100_steps_list else 0.0,
        "avg_time_to_failure": avg_time_to_failure,
        "failure_rate": float(
            np.mean([0.0 if np.isnan(x) else 1.0 for x in time_to_failure_list])
        ) if time_to_failure_list else 0.0,
        "avg_collisions_per_episode": float(np.mean(collisions_per_episode)) if collisions_per_episode else 0.0,
        "avg_oob_per_episode": float(np.mean(oob_per_episode)) if oob_per_episode else 0.0,
        "avg_speed_violations_per_episode": float(np.mean(speed_per_episode)) if speed_per_episode else 0.0,
        "avg_falls_per_episode": float(np.mean(fall_per_episode)) if fall_per_episode else 0.0,
        "avg_total_violations_per_episode": float(np.mean(total_violations_per_episode)) if total_violations_per_episode else 0.0,
    }

    return sanitize_record(result)


def rollout_episode(
    params,
    model_name: str,
    cfg: Cfg,
    seed: int,
):
    env_cls = get_env_class(model_name)
    roll_cfg = Cfg(**cfg.__dict__)
    roll_cfg.num_envs = 1

    env = env_cls(roll_cfg)
    rng_key = jax.random.PRNGKey(seed)
    task_base_seed = seed * 100_000 + 1 # fixed constant iter offset for rollout
    obs, _ = env.reset(rng_key, base_seed=task_base_seed)

    done = np.array([False])
    last_metrics = None
    total_reward = 0.0

    act_key = jax.random.PRNGKey(seed + 999_999)

    traj_xy = [env.torso_xy_t[0].copy()]
    goal_xy = env.goal[0].copy()
    obs_xy = env.obs_xy[0].copy()
    obs_r = env.obs_r[0].copy()

    counters = empty_episode_counters()

    while not bool(done[0]):
        act_key, subkey = jax.random.split(act_key)
        action, _, _ = select_action(
            params=params,
            obs=obs,
            rng=subkey,
            deterministic=True,
        )

        obs, reward, done, metrics = env.step(np.array(action))
        total_reward += float(np.array(reward)[0])
        last_metrics = metrics

        current_step = int(metrics["steps"][0])
        counters = update_episode_counters(counters, metrics, current_step)

        traj_xy.append(env.torso_xy_t[0].copy())

    traj_xy = np.array(traj_xy, dtype=np.float32)
    ep_len = float(last_metrics["steps"][0])
    violations_per_100_steps = (
        100.0 * counters["total_violations"] / ep_len if ep_len > 0 else 0.0
    )

    rollout_metrics = {
        "success": float(last_metrics["success"][0]),
        "episode_length": ep_len,
        "episode_reward": float(total_reward),
        "collision_count": float(counters["collision_count"]),
        "oob_count": float(counters["oob_count"]),
        "speed_count": float(counters["speed_count"]),
        "fall_count": float(counters["fall_count"]),
        "total_violations": float(counters["total_violations"]),
        "violations_per_100_steps": float(violations_per_100_steps),
        "time_to_failure": (
            float(ep_len) if np.isnan(counters["time_to_failure"])
            else float(counters["time_to_failure"])
        ),
    }

    return {
        "traj_xy": traj_xy,
        "goal_xy": goal_xy,
        "obs_xy": obs_xy,
        "obs_r": obs_r,
        "metrics": sanitize_record(rollout_metrics),
    }


def save_rollout_plot(rollout_data, cfg: Cfg, model_name: str, seed: int, out_png_path):
    traj = rollout_data["traj_xy"]
    goal_xy = rollout_data["goal_xy"]
    obs_xy = rollout_data["obs_xy"]
    obs_r = rollout_data["obs_r"]
    m = rollout_data["metrics"]

    plt.figure(figsize=(8, 8))
    # Trajectory
    plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label="trajectory")

    # Start
    plt.scatter(
        traj[0, 0],
        traj[0, 1],
        marker="s",
        s=140,
        color="tab:blue",
        label="start",
        zorder=6,
    )

    # Goal
    plt.scatter(
        goal_xy[0],
        goal_xy[1],
        marker="*",
        s=420,
        color="orange",
        edgecolor="black",
        label="goal",
        zorder=7,
    )

    # Goal success radius
    goal_circle = plt.Circle(
        (float(goal_xy[0]), float(goal_xy[1])),
        float(cfg.goal_radius),
        fill=False,
        linestyle="-.",
        linewidth=2.5,
        edgecolor="gold",
        zorder=5,
    )
    plt.gca().add_patch(goal_circle)

    end_point = traj[-1]
    plt.scatter(
        end_point[0],
        end_point[1],
        c="red",
        s=100,
        marker="o",
        label="end",
        zorder=5,
    )

    for (cx, cy), r in zip(obs_xy, obs_r):
        body_circle = plt.Circle(
            (float(cx), float(cy)),
            float(r),
            fill=False,
            linewidth=2,
        )
        plt.gca().add_patch(body_circle)

        buffer_circle = plt.Circle(
            (float(cx), float(cy)),
            float(r + cfg.agent_r + cfg.buffer_dist),
            fill=False,
            linestyle="--",
            linewidth=1.5,
        )
        plt.gca().add_patch(buffer_circle)

    # Arena border
    a = cfg.arena_size
    arena_rect = plt.Rectangle(
        (-a, -a),
        2 * a,
        2 * a,
        fill=False,
        linestyle=":",
        linewidth=1.5,
    )
    plt.gca().add_patch(arena_rect)

    plt.title(
        f"Rollout {model_name} | seed={seed} | "
        f"success={int(m['success'])} | "
        f"viol/100={m['violations_per_100_steps']:.2f} | "
        f"coll={m['collision_count']:.0f} | "
        f"oob={m['oob_count']:.0f} | "
        f"speed={m['speed_count']:.0f} | "
        f"fall={m['fall_count']:.0f}"
    )

    plt.xlim(-a - 1.0, a + 1.0)
    plt.ylim(-a - 1.0, a + 1.0)
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close()


# =========================================================
# Plot helpers
# =========================================================
def model_metrics_for_plot():
    return [
        "violations_per_100_steps",
        "success_rate",
        "avg_time_to_failure",
        "mean_episode_length",
        "avg_collisions_per_episode",
        "avg_episode_reward",
    ]


def plot_per_metric_across_seeds(model_name, model_budget_dir, seeds):
    learning_curves_dir = model_budget_dir / "learning_curves"
    ensure_dir(learning_curves_dir)

    metrics = model_metrics_for_plot()

    seed_frames = {}
    for seed in seeds:
        csv_path = model_budget_dir / f"seed_{seed}" / "curves" / "learning_curve.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            seed_frames[seed] = df

    if not seed_frames:
        return

    for metric in metrics:
        plt.figure(figsize=(12, 7))
        plotted_any = False

        for seed, df in seed_frames.items():
            if "iteration" not in df.columns or metric not in df.columns:
                continue

            x = pd.to_numeric(df["iteration"], errors="coerce")
            y = pd.to_numeric(df[metric], errors="coerce")

            if metric == "avg_time_to_failure" and "mean_episode_length" in df.columns:
                mel = pd.to_numeric(df["mean_episode_length"], errors="coerce")
                y = y.fillna(mel)

            y = y.fillna(0.0)
            x = x.fillna(0.0)

            plt.plot(x, y, marker="o", label=f"seed_{seed}")
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Iteration")
        plt.ylabel(metric)
        plt.title(f"{metric} ({model_name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(learning_curves_dir / f"{metric}.png", dpi=160)
        plt.close()


def aggregate_learning_curves(curve_paths, model_name, out_csv_path, out_png_path):
    if not curve_paths:
        return

    frames = [pd.read_csv(p) for p in curve_paths]
    min_len = min(len(df) for df in frames)
    frames = [df.iloc[:min_len].reset_index(drop=True) for df in frames]

    base_iterations = frames[0]["iteration"].to_numpy()
    base_timesteps = frames[0]["timesteps"].to_numpy() if "timesteps" in frames[0].columns else None

    all_metrics = model_metrics_for_plot()

    rows = []
    for i, it in enumerate(base_iterations):
        row = {"iteration": int(it)}

        if base_timesteps is not None:
            row["timesteps"] = int(base_timesteps[i])

        for metric in all_metrics:
            vals = []
            for df in frames:
                if metric in df.columns:
                    v = pd.to_numeric(pd.Series([df.loc[i, metric]]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        vals.append(float(v))

            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
            else:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_std"] = 0.0

        rows.append(sanitize_record(row))

    save_dict_rows_to_csv(rows, out_csv_path)

    df_agg = pd.DataFrame(rows)
    metrics = model_metrics_for_plot()

    plt.figure(figsize=(14, 8))
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df_agg.columns:
            continue

        x = pd.to_numeric(df_agg["iteration"], errors="coerce").fillna(0.0).to_numpy()
        y = pd.to_numeric(df_agg[mean_col], errors="coerce").fillna(0.0).to_numpy()
        s = pd.to_numeric(df_agg[std_col], errors="coerce").fillna(0.0).to_numpy()

        plt.plot(x, y, marker="o", label=f"{metric} (mean)")
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(f"Aggregated Learning Curve ({model_name}) - mean ± std over seeds")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close()


def plot_metrics_vs_seed(final_rows, model_name, out_png_path):
    if not final_rows:
        return

    rows = [sanitize_record(r) for r in final_rows]
    rows = sorted(rows, key=lambda x: x["seed"])
    seeds = [r["seed"] for r in rows]

    plt.figure(figsize=(12, 7))

    if "success_rate" in rows[0]:
        plt.plot(seeds, [r["success_rate"] for r in rows], marker="o", label="success_rate")

    if "violations_per_100_steps" in rows[0]:
        plt.plot(
            seeds,
            [r["violations_per_100_steps"] for r in rows],
            marker="o",
            label="violations_per_100_steps",
        )

    if "avg_collisions_per_episode" in rows[0]:
        plt.plot(
            seeds,
            [r["avg_collisions_per_episode"] for r in rows],
            marker="o",
            label="avg_collisions_per_episode",
        )

    if "mean_episode_length" in rows[0]:
        plt.plot(
            seeds,
            [r["mean_episode_length"] for r in rows],
            marker="o",
            label="mean_episode_length",
        )

    if "avg_episode_reward" in rows[0]:
        plt.plot(
            seeds,
            [r["avg_episode_reward"] for r in rows],
            marker="o",
            label="avg_episode_reward",
        )

    if "avg_time_to_failure" in rows[0]:
        plt.plot(
            seeds,
            [r["avg_time_to_failure"] for r in rows],
            marker="o",
            label="avg_time_to_failure",
        )

    plt.xlabel("Seed")
    plt.ylabel("Value")
    plt.title(f"Performance vs Seed ({model_name})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close()


# =========================================================
# PPO rollout collection
# =========================================================
def collect_rollout(env, params, rng, steps_per_env):
    obs_buf = []
    act_buf = []
    rew_buf = []
    done_buf = []
    logp_buf = []
    val_buf = []

    obs = env._obs()
    num_envs = env.cfg.num_envs
    done_mask = np.zeros((num_envs,), dtype=bool)

    for _ in range(steps_per_env):
        rng, subkey = jax.random.split(rng)

        action, log_prob, value = select_action(
            params=params,
            obs=obs,
            rng=subkey,
            deterministic=False,
        )

        action_np = np.array(action, dtype=np.float32)
        action_np[done_mask] = 0.0

        next_obs, reward, done, _metrics = env.step(action_np)

        reward_np = np.array(reward, dtype=np.float32)
        done_np = np.array(done, dtype=bool)

        reward_np[done_mask] = 0.0
        done_np = np.logical_or(done_np, done_mask)

        obs_buf.append(jnp.nan_to_num(obs))
        act_buf.append(jnp.nan_to_num(jnp.asarray(action_np)))
        rew_buf.append(jnp.nan_to_num(jnp.asarray(reward_np)))
        done_buf.append(jnp.asarray(done_np, dtype=jnp.float32))
        logp_buf.append(jnp.nan_to_num(log_prob))
        val_buf.append(jnp.nan_to_num(value))

        done_mask = np.logical_or(done_mask, done_np)
        obs = next_obs

    _, _, last_values = mlp_forward(params, obs)
    last_values = jnp.nan_to_num(last_values)

    rollout = {
        "obs": jnp.stack(obs_buf, axis=0),
        "actions": jnp.stack(act_buf, axis=0),
        "rewards": jnp.stack(rew_buf, axis=0),
        "dones": jnp.stack(done_buf, axis=0),
        "logp": jnp.stack(logp_buf, axis=0),
        "values": jnp.stack(val_buf, axis=0),
        "last_values": last_values,
        "rng": rng,
    }
    return rollout


def flatten_rollout_for_ppo(rollout, gamma, lam):
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    last_values = rollout["last_values"]

    T, N = rewards.shape

    all_adv = []
    all_ret = []

    for env_i in range(N):
        adv_i, ret_i = compute_gae(
            rewards[:, env_i],
            values[:, env_i],
            dones[:, env_i],
            last_value=last_values[env_i],
            gamma=gamma,
            lam=lam,
        )
        all_adv.append(adv_i)
        all_ret.append(ret_i)

    adv = jnp.concatenate(all_adv, axis=0)
    ret = jnp.concatenate(all_ret, axis=0)

    obs_arr = rollout["obs"].reshape(T * N, -1)
    act_arr = rollout["actions"].reshape(T * N, -1)
    logp_arr = rollout["logp"].reshape(T * N)

    adv = jnp.nan_to_num(adv)
    ret = jnp.nan_to_num(ret)
    obs_arr = jnp.nan_to_num(obs_arr)
    act_arr = jnp.nan_to_num(act_arr)
    logp_arr = jnp.nan_to_num(logp_arr)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return obs_arr, act_arr, logp_arr, adv, ret


def ppo_update(params, opt_state, optimizer, obs_arr, act_arr, logp_arr, adv, ret, rng, args):
    batch_size = obs_arr.shape[0]
    minibatch_size = min(args.minibatch_size, batch_size)

    for _ in range(args.ppo_epochs):
        rng, perm_key = jax.random.split(rng)
        perm = np.array(jax.random.permutation(perm_key, batch_size))

        for start in range(0, batch_size, minibatch_size):
            idx = perm[start:start + minibatch_size]

            mb_obs = obs_arr[idx]
            mb_act = act_arr[idx]
            mb_logp = logp_arr[idx]
            mb_adv = adv[idx]
            mb_ret = ret[idx]

            loss_fn = lambda p: ppo_loss(
                params=p,
                obs=mb_obs,
                actions=mb_act,
                old_log_probs=mb_logp,
                advantages=mb_adv,
                returns=mb_ret,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
            )

            loss, grads = jax.value_and_grad(loss_fn)(params)

            if not np.isfinite(float(loss)):
                continue

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    return params, opt_state, rng


# =========================================================
# Checkpoint helpers
# =========================================================
def save_latest_model(model_dir, params, opt_state, latest_stats, meta):
    save_training_state(
        model_dir / "latest_model.pkl",
        params=params,
        opt_state=opt_state,
        meta=meta,
    )
    save_dict_to_json(latest_stats, model_dir / "latest_model_stats.json")


def save_best_model(model_dir, params, opt_state, best_stats, meta):
    save_training_state(
        model_dir / "best_model.pkl",
        params=params,
        opt_state=opt_state,
        meta=meta,
    )
    save_dict_to_json(best_stats, model_dir / "best_model_stats.json")


def maybe_save_checkpoint(model_dir, params, opt_state, current_stats, meta, current_timesteps, checkpoint_every):
    if checkpoint_every <= 0:
        return

    if current_timesteps % checkpoint_every != 0:
        return

    tag = format_checkpoint_tag(current_timesteps)
    ckpt_dir = model_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    save_training_state(
        ckpt_dir / f"checkpoint_{tag}.pkl",
        params=params,
        opt_state=opt_state,
        meta=meta,
    )
    save_dict_to_json(current_stats, ckpt_dir / f"checkpoint_{tag}_stats.json")


# =========================================================
# Training per seed
# =========================================================
def run_single_seed(model_name, seed, args, model_budget_dir, cfg):
    seed_dir = model_budget_dir / f"seed_{seed}"

    model_dir = seed_dir / "model"
    curves_dir = seed_dir / "curves"
    eval_dir = seed_dir / "eval"
    rollouts_dir = seed_dir / "rollouts"

    for d in [model_dir, curves_dir, eval_dir, rollouts_dir]:
        ensure_dir(d)

    print("\n" + "=" * 80, flush=True)
    print(f"[{model_name}] Starting seed {seed}", flush=True)
    print("=" * 80, flush=True)

    env_cls = get_env_class(model_name)
    env = env_cls(cfg)

    init_rng = jax.random.PRNGKey(seed)
    params = init_mlp_params(
        rng=init_rng,
        obs_dim=env.total_obs_size(),
        action_dim=env.act_size,
        hidden_dim=args.hidden_dim,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate),
    )
    opt_state = optimizer.init(params)

    config_to_save = {
        "model_name": model_name,
        "seed": seed,
        "timesteps": args.timesteps,
        "steps_per_env": args.steps_per_env,
        "num_envs": cfg.num_envs,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "checkpoint_every_timesteps": args.checkpoint_every_timesteps,
        "cfg": cfg.__dict__,
    }
    save_dict_to_json(config_to_save, model_dir / "config.json")

    steps_per_iter = args.steps_per_env * cfg.num_envs
    num_iters = max(1, args.timesteps // steps_per_iter)

    curve_records = []
    rng = jax.random.PRNGKey(seed + 1234)

    best_stats = None
    best_score = None
    train_start_time = time.perf_counter()

    for it in range(1, num_iters + 1):
        iter_start_time = time.perf_counter()

        rng, reset_key = jax.random.split(rng)
        # Deterministic task layout seed: same across all models for same seed+iter
        task_base_seed = seed * 100_000 + it
        env.reset(reset_key, base_seed=task_base_seed)

        rollout = collect_rollout(
            env=env,
            params=params,
            rng=rng,
            steps_per_env=args.steps_per_env,
        )
        rng = rollout["rng"]

        obs_arr, act_arr, logp_arr, adv, ret = flatten_rollout_for_ppo(
            rollout=rollout,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )

        params, opt_state, rng = ppo_update(
            params=params,
            opt_state=opt_state,
            optimizer=optimizer,
            obs_arr=obs_arr,
            act_arr=act_arr,
            logp_arr=logp_arr,
            adv=adv,
            ret=ret,
            rng=rng,
            args=args,
        )

        current_timesteps = int(it * steps_per_iter)
        iter_time_sec = time.perf_counter() - iter_start_time
        elapsed_time_sec = time.perf_counter() - train_start_time
        timesteps_per_sec = float(steps_per_iter / max(iter_time_sec, 1e-9))

        if (it % args.eval_every_iters == 0) or (it == num_iters):
            eval_start_time = time.perf_counter()

            stats = evaluate_model(
                params=params,
                model_name=model_name,
                cfg=cfg,
                n_episodes=args.eval_eps,
                seed=seed * 1000 + current_timesteps,
            )

            eval_time_sec = time.perf_counter() - eval_start_time

            row = {
                "iteration": it,
                "timesteps": current_timesteps,
                "max_steps": args.max_steps,
                "iter_time_sec": iter_time_sec,
                "eval_time_sec": eval_time_sec,
                "elapsed_time_sec": elapsed_time_sec,
                "timesteps_per_sec": timesteps_per_sec,
            }
            row.update(stats)
            row = sanitize_record(row)
            curve_records.append(row)

            current_meta = {
                "model_name": model_name,
                "seed": seed,
                "iteration": it,
                "timesteps": current_timesteps,
            }

            # latest model
            save_latest_model(
                model_dir=model_dir,
                params=params,
                opt_state=opt_state,
                latest_stats=row,
                meta=current_meta,
            )

            # periodic checkpoint
            maybe_save_checkpoint(
                model_dir=model_dir,
                params=params,
                opt_state=opt_state,
                current_stats=row,
                meta=current_meta,
                current_timesteps=current_timesteps,
                checkpoint_every=args.checkpoint_every_timesteps,
            )

            # best model
            if is_better_model(row, best_stats):
                best_stats = dict(row)
                best_score = compute_selection_score(best_stats)
                best_stats["best_score"] = list(best_score)

                save_best_model(
                    model_dir=model_dir,
                    params=params,
                    opt_state=opt_state,
                    best_stats=best_stats,
                    meta=current_meta,
                )

            print(
                f"[Eval {model_name}] "
                f"seed={seed} | "
                f"iter={it}/{num_iters} | "
                f"t={current_timesteps} | "
                f"success={row['success_rate']:.3f} | "
                f"len={row['mean_episode_length']:.1f} | "
                f"viol/100={row['violations_per_100_steps']:.3f} | "
                f"coll/ep={row['avg_collisions_per_episode']:.3f} | "
                f"reward={row['avg_episode_reward']:.3f} | "
                f"ttf={row['avg_time_to_failure']:.3f} | "
                f"speed={row['timesteps_per_sec']:.1f} steps/s",
                flush=True,
            )

    curve_csv_path = curves_dir / "learning_curve.csv"
    save_dict_rows_to_csv(curve_records, curve_csv_path)

    final_stats = evaluate_model(
        params=params,
        model_name=model_name,
        cfg=cfg,
        n_episodes=args.final_eval_eps,
        seed=10_000 + seed,
    )

    final_stats_with_seed = {"seed": seed, "max_steps": args.max_steps}
    final_stats_with_seed.update(final_stats)
    final_stats_with_seed = sanitize_record(final_stats_with_seed)

    save_dict_rows_to_csv([final_stats_with_seed], eval_dir / "final_eval.csv")
    save_dict_to_json(final_stats_with_seed, eval_dir / "final_eval.json")

    print(f"[{model_name}] Seed {seed} final stats: {final_stats_with_seed}", flush=True)

    rollout_rows = []

    for k in range(args.rollouts_per_seed):
        rollout_seed = seed * 1000 + k
        rollout_data = rollout_episode(
            params=params,
            model_name=model_name,
            cfg=cfg,
            seed=rollout_seed,
        )

        rollout_png_path = rollouts_dir / f"rollout_{k}.png"
        save_rollout_plot(
            rollout_data=rollout_data,
            cfg=cfg,
            model_name=model_name,
            seed=rollout_seed,
            out_png_path=rollout_png_path,
        )

        traj = rollout_data["traj_xy"]
        traj_rows = [{"x": float(p[0]), "y": float(p[1])} for p in traj]
        save_dict_rows_to_csv(traj_rows, rollouts_dir / f"rollout_{k}_trajectory.csv")

        row = {"rollout_id": k, "seed": rollout_seed}
        row.update(rollout_data["metrics"])
        rollout_rows.append(sanitize_record(row))

    save_dict_rows_to_csv(rollout_rows, rollouts_dir / "rollout_metrics.csv")

    return final_stats_with_seed


# =========================================================
# Aggregation per model
# =========================================================
def aggregate_model_results(model_name, model_budget_dir, seeds):
    aggregated_dir = model_budget_dir / "aggregated"
    ensure_dir(aggregated_dir)

    all_seed_rows = []
    for seed in seeds:
        eval_csv_path = model_budget_dir / f"seed_{seed}" / "eval" / "final_eval.csv"
        if eval_csv_path.exists():
            rows = pd.read_csv(eval_csv_path).to_dict(orient="records")
            all_seed_rows.extend(rows)

    if not all_seed_rows:
        return

    all_seed_rows = [sanitize_record(r) for r in all_seed_rows]
    save_dict_rows_to_csv(all_seed_rows, aggregated_dir / "all_seed_eval.csv")

    metrics = [
        "violations_per_100_steps",
        "success_rate",
        "avg_time_to_failure",
        "mean_episode_length",
        "avg_collisions_per_episode",
        "avg_episode_reward",
    ]

    df = pd.DataFrame(all_seed_rows)
    final_table = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(series) == 0:
            continue

        final_table.append(
            {
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
            }
        )

    save_dict_rows_to_csv(final_table, aggregated_dir / "final_table.csv")

    curve_paths = []
    for seed in seeds:
        p = model_budget_dir / f"seed_{seed}" / "curves" / "learning_curve.csv"
        if p.exists():
            curve_paths.append(p)

    if curve_paths:
        aggregate_learning_curves(
            curve_paths=curve_paths,
            model_name=model_name,
            out_csv_path=aggregated_dir / "mean_learning_curve.csv",
            out_png_path=aggregated_dir / "mean_learning_curve.png",
        )

    plot_metrics_vs_seed(
        final_rows=all_seed_rows,
        model_name=model_name,
        out_png_path=aggregated_dir / "metrics_vs_seed.png",
    )

    plot_per_metric_across_seeds(
        model_name=model_name,
        model_budget_dir=model_budget_dir,
        seeds=seeds,
    )