# baseline_env.py
# =========================================================
# Shared base environment for the CSI4900 Brax project.
#
# This file contains:
#   - shared config
#   - base Brax + Ant navigation environment
#   - helper functions for:
#       * goal generation
#       * obstacle generation
#       * observation construction
#       * safety checks
#       * metrics
#
# Observation structure:
#   [ant_obs,
#    torso_xy(t-2), torso_xy(t-1), torso_xy(t),
#    torso_vxy,
#    goal_vec_xy,
#    obs1_rel_xy, obs2_rel_xy, obs3_rel_xy,
#    obs1_r, obs2_r, obs3_r]
#
# Notes:
#   - Ant is simulated in 3D by Brax.
#   - The navigation task is defined mainly in 2D (x, y).
#   - The z coordinate is used to detect falling.
#   - Obstacles are logical task obstacles, not physical Brax bodies.
#   - The physical start is the true Brax reset torso position.
#   - Goal and obstacles are sampled at the beginning of each episode
#     and remain fixed during the episode.
# =========================================================

from dataclasses import dataclass, asdict
from typing import Dict, Type, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------
# Compatibility shim:
# some Brax code paths still call jax.tree_map
# but newer JAX expects jax.tree_util.tree_map
# ---------------------------------------------------------
try:
    _ = jax.tree_map
except AttributeError:
    jax.tree_map = jax.tree_util.tree_map

from brax import envs


@dataclass
class Cfg:
    # -----------------------------------------------------
    # Parallel simulation
    # -----------------------------------------------------
    num_envs: int = 2048

    # -----------------------------------------------------
    # Episode
    # -----------------------------------------------------
    max_steps: int = 500

    # -----------------------------------------------------
    # Navigation arena (2D task space)
    # -----------------------------------------------------
    arena_size: float = 6.0
    goal_radius: float = 1.5

    # -----------------------------------------------------
    # Random placement control
    # -----------------------------------------------------
    wall_margin: float = 0.80
    start_goal_min_dist: float = 2.0
    start_goal_max_dist: float = 4.0

    # -----------------------------------------------------
    # Obstacles
    # -----------------------------------------------------
    n_obstacles: int = 3
    obstacle_radius_min: float = 0.25
    obstacle_radius_max: float = 0.45
    obstacle_min_separation: float = 0.70

    # At least one obstacle is placed near the direct path
    # from the Brax start position to the goal.
    path_obstacle_offset: float = 0.8

    # -----------------------------------------------------
    # Start / goal clearance
    # -----------------------------------------------------
    start_clearance_extra: float = 0.55
    goal_clearance_extra: float = 0.65

    # Extra clearance around the goal success region.
    # This makes sure obstacles do not visually/logically
    # invade the success zone around the goal.
    goal_success_clearance_extra: float = 0.25

    # -----------------------------------------------------
    # Safety thresholds
    # -----------------------------------------------------
    buffer_dist: float = 0.60
    v_max: float = 3.00
    fall_threshold: float = 0.25
    agent_r: float = 0.45

    # -----------------------------------------------------
    # Reward settings
    # -----------------------------------------------------
    success_bonus = 40.0
    step_penalty = 0.03
    collision_penalty = 5.0
    oob_penalty = 5.0
    speed_penalty = 0.5
    fall_penalty = 6.0


class BraxAntBase:
    """
    Shared Brax + Ant navigation environment.

    This class wraps the Brax Ant environment and adds a 2D
    navigation task with:
      - Brax-defined physical start
      - random goal
      - random logical obstacles
      - safety-related metrics

    Child classes only need to override `step()` to define
    reward and termination behavior for:
      - no_constraint
      - soft_constraint
      - hard_constraint
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

        # -------------------------------------------------
        # Create the Brax Ant environment
        # -------------------------------------------------
        self.env = envs.get_environment("ant", backend="spring")

        # Vectorized reset / step functions for parallel envs
        self.reset_fn = jax.jit(jax.vmap(self.env.reset))
        self.step_fn = jax.jit(jax.vmap(self.env.step))

        # Native Brax Ant sizes
        self.ant_obs_size = int(self.env.observation_size)
        self.act_size = int(self.env.action_size)

        # Runtime state
        self.state = None
        self.t = None

        # Task state (NumPy side)
        self.goal = None               # shape: (num_envs, 2)
        self.obs_xy = None             # shape: (num_envs, n_obstacles, 2)
        self.obs_r = None              # shape: (num_envs, n_obstacles)

        # Short torso history
        self.torso_xy_tminus2 = None   # shape: (num_envs, 2)
        self.torso_xy_tminus1 = None   # shape: (num_envs, 2)
        self.torso_xy_t = None         # shape: (num_envs, 2)

        # Initial reset only to initialize fields
        init_key = jax.random.PRNGKey(0)
        self.reset(init_key)

    # =====================================================
    # Basic state helpers
    # =====================================================
    def _get_torso_xyz(self) -> np.ndarray:
        """
        Returns the torso position for all parallel environments.

        Shape:
            (num_envs, 3) -> [x, y, z]
        """
        pos = np.array(self.state.pipeline_state.x.pos[:, 0, :], dtype=np.float32)
        return pos

    def _get_torso_xy(self) -> np.ndarray:
        """
        Returns torso (x, y) for all envs.

        Shape:
            (num_envs, 2)
        """
        return self._get_torso_xyz()[:, :2].astype(np.float32)

    def _get_torso_z(self) -> np.ndarray:
        """
        Returns torso z (height) for all envs.

        Shape:
            (num_envs,)
        """
        return self._get_torso_xyz()[:, 2].astype(np.float32)

    def _get_torso_vxy(self) -> np.ndarray:
        """
        Returns torso planar velocity (vx, vy) for all envs.

        Shape:
            (num_envs, 2)
        """
        vel = np.array(self.state.pipeline_state.xd.vel[:, 0, :2], dtype=np.float32)
        return vel

    # =====================================================
    # Random sampling helpers
    # =====================================================
    def _sample_point_in_arena(
        self,
        rng: np.random.Generator,
        wall_margin: float,
    ) -> np.ndarray:
        """
        Samples one random 2D point inside the arena while keeping
        a margin from the walls.
        """
        a = self.cfg.arena_size - wall_margin
        x = float(rng.uniform(-a, a))
        y = float(rng.uniform(-a, a))
        return np.array([x, y], dtype=np.float32)

    def _sample_goal_for_one_env(
        self,
        rng: np.random.Generator,
        start: np.ndarray,
    ) -> np.ndarray:
        """
        Samples one goal using the true Brax reset position as start.

        Conditions:
          - inside arena
          - wall margin respected
          - not too close to start
          - not too far from start
        """
        for _ in range(3000):
            goal = self._sample_point_in_arena(rng, self.cfg.wall_margin)
            d = float(np.linalg.norm(goal - start))

            if d < self.cfg.start_goal_min_dist:
                continue
            if d > self.cfg.start_goal_max_dist:
                continue

            return goal

        # fallback
        fallback = (start + np.array([3.5, 0.0], dtype=np.float32)).astype(np.float32)

        # clamp fallback inside arena margins
        a = self.cfg.arena_size - self.cfg.wall_margin
        fallback[0] = np.clip(fallback[0], -a, a)
        fallback[1] = np.clip(fallback[1], -a, a)
        return fallback

    def _is_goal_valid_against_obstacles(
        self,
        goal: np.ndarray,
        obs_xy: np.ndarray,
        obs_r: np.ndarray,
    ) -> bool:
        """
        Checks that the goal success region does not overlap or get
        too close to any obstacle.

        We want the success circle centered at the goal to remain clear.
        """
        for j in range(len(obs_xy)):
            d = float(np.linalg.norm(goal - obs_xy[j]))
            min_sep = (
                self.cfg.goal_radius
                + float(obs_r[j])
                + self.cfg.goal_success_clearance_extra
            )
            if d <= min_sep:
                return False
        return True

    def _is_obstacle_valid(
        self,
        c: np.ndarray,
        r: float,
        obs_xy: list,
        obs_r: list,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> bool:
        """
        Checks whether a new obstacle candidate is valid.

        Conditions:
          - stays away from walls
          - does not overlap previous obstacles
          - not too close to start
          - not too close to goal center
          - not too close to the goal success zone
        """
        # wall margin
        if (
            abs(float(c[0])) > self.cfg.arena_size - self.cfg.wall_margin
            or abs(float(c[1])) > self.cfg.arena_size - self.cfg.wall_margin
        ):
            return False

        # obstacle-obstacle separation
        for j in range(len(obs_xy)):
            d = float(np.linalg.norm(c - obs_xy[j]))
            min_sep = r + obs_r[j] + self.cfg.obstacle_min_separation
            if d <= min_sep:
                return False

        # keep away from start
        d_start = float(np.linalg.norm(c - start))
        min_start_sep = r + self.cfg.agent_r + self.cfg.start_clearance_extra
        if d_start <= min_start_sep:
            return False

        # keep away from goal center
        d_goal = float(np.linalg.norm(c - goal))
        min_goal_center_sep = r + self.cfg.agent_r + self.cfg.goal_clearance_extra
        if d_goal <= min_goal_center_sep:
            return False

        # keep away from the goal success region
        min_goal_success_sep = (
            r
            + self.cfg.goal_radius
            + self.cfg.goal_success_clearance_extra
        )
        if d_goal <= min_goal_success_sep:
            return False

        return True

    def _sample_path_obstacle_for_one_env(
        self,
        rng: np.random.Generator,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Samples one important obstacle near the direct start->goal path.

        This guarantees that at least one obstacle is relevant for navigation.
        """
        direction = goal - start
        norm = float(np.linalg.norm(direction))

        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32), 0.45

        u = direction / norm
        perp = np.array([-u[1], u[0]], dtype=np.float32)

        for _ in range(2000):
            r = float(
                rng.uniform(
                    self.cfg.obstacle_radius_min,
                    self.cfg.obstacle_radius_max,
                )
            )

            # choose a point along the middle portion of the path
            alpha = float(rng.uniform(0.35, 0.70))
            base = start + alpha * direction

            # apply a random lateral shift
            lateral = float(
                rng.uniform(
                    -self.cfg.path_obstacle_offset,
                    self.cfg.path_obstacle_offset,
                )
            )
            c = (base + lateral * perp).astype(np.float32)

            if self._is_obstacle_valid(c, r, [], [], start, goal):
                return c, r

        # fallback: still try to stay near the path but not inside goal zone
        c = (start + 0.50 * direction).astype(np.float32)
        r = 0.45

        if self._is_obstacle_valid(c, r, [], [], start, goal):
            return c, r

        # final safe fallback
        for _ in range(3000):
            c = self._sample_point_in_arena(rng, self.cfg.wall_margin)
            if self._is_obstacle_valid(c, r, [], [], start, goal):
                return c, r

        return np.array([0.0, 0.0], dtype=np.float32), 0.45

    def _sample_obstacles_for_one_env(
        self,
        rng: np.random.Generator,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples logical obstacles for one environment.

        Strategy:
          - first obstacle: guaranteed relevant obstacle near the path
          - remaining obstacles: random controlled placement
        """
        obs_xy = []
        obs_r = []

        # first obstacle near the direct path
        c0, r0 = self._sample_path_obstacle_for_one_env(rng, start, goal)
        obs_xy.append(c0)
        obs_r.append(r0)

        # remaining controlled random obstacles
        for _ in range(self.cfg.n_obstacles - 1):
            placed = False

            for _ in range(3000):
                r = float(
                    rng.uniform(
                        self.cfg.obstacle_radius_min,
                        self.cfg.obstacle_radius_max,
                    )
                )
                c = self._sample_point_in_arena(rng, self.cfg.wall_margin)

                if self._is_obstacle_valid(c, r, obs_xy, obs_r, start, goal):
                    obs_xy.append(c)
                    obs_r.append(r)
                    placed = True
                    break

            if not placed:
                # fallback for one obstacle
                for _ in range(3000):
                    c = self._sample_point_in_arena(rng, self.cfg.wall_margin)
                    r = 0.45
                    if self._is_obstacle_valid(c, r, obs_xy, obs_r, start, goal):
                        obs_xy.append(c)
                        obs_r.append(r)
                        placed = True
                        break

            if not placed:
                # final fallback
                # pick a neutral but still reasonable value
                obs_xy.append(np.array([0.0, 0.0], dtype=np.float32))
                obs_r.append(0.45)

        obs_xy_arr = np.stack(obs_xy, axis=0).astype(np.float32)
        obs_r_arr = np.array(obs_r, dtype=np.float32)

        return obs_xy_arr, obs_r_arr

    def _sample_task_layout(
        self,
        base_seed: int,
        starts_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples task-level random elements for all parallel environments,
        using the true Brax reset torso positions as starts.

        Returns:
          goal   : (num_envs, 2)
          obs_xy : (num_envs, n_obstacles, 2)
          obs_r  : (num_envs, n_obstacles)
        """
        all_goal = []
        all_obs_xy = []
        all_obs_r = []

        for env_i in range(self.cfg.num_envs):
            rng = np.random.default_rng(base_seed + 10_000 * env_i)

            start_i = starts_xy[env_i].astype(np.float32)

            # sample goal
            goal_i = self._sample_goal_for_one_env(rng, start_i)

            # sample obstacles
            obs_xy_i, obs_r_i = self._sample_obstacles_for_one_env(rng, start_i, goal_i)

            # extra safety pass: ensure goal success region is clean
            # if not, try again a few times
            if not self._is_goal_valid_against_obstacles(goal_i, obs_xy_i, obs_r_i):
                placed_clean = False

                for _ in range(200):
                    goal_i_try = self._sample_goal_for_one_env(rng, start_i)
                    obs_xy_try, obs_r_try = self._sample_obstacles_for_one_env(
                        rng, start_i, goal_i_try
                    )

                    if self._is_goal_valid_against_obstacles(goal_i_try, obs_xy_try, obs_r_try):
                        goal_i = goal_i_try
                        obs_xy_i = obs_xy_try
                        obs_r_i = obs_r_try
                        placed_clean = True
                        break

                if not placed_clean:
                    # keep last valid-enough sample even if imperfect,
                    # but most of the time this second pass should succeed
                    pass

            all_goal.append(goal_i)
            all_obs_xy.append(obs_xy_i)
            all_obs_r.append(obs_r_i)

        return (
            np.stack(all_goal, axis=0).astype(np.float32),
            np.stack(all_obs_xy, axis=0).astype(np.float32),
            np.stack(all_obs_r, axis=0).astype(np.float32),
        )

    # =====================================================
    # Reset
    # =====================================================
    def reset(self, rng_key, base_seed: int | None = None): # accept an optional base_seed override
        """
        Resets all parallel Brax environments and samples a new
        navigation task layout using the true Brax reset torso
        positions as starts.
        """
        keys = jax.random.split(rng_key, self.cfg.num_envs)
        self.state = self.reset_fn(keys)
        self.t = np.zeros((self.cfg.num_envs,), dtype=np.int32)

        # true physical start positions from Brax
        torso_xy = self._get_torso_xy()

        if base_seed is None:
        # fallback for the init call in __init__
            k = np.array(rng_key, dtype=np.uint32)
            base_seed = int(k[0] ^ k[1])

        self.goal, self.obs_xy, self.obs_r = self._sample_task_layout(base_seed, torso_xy)

        # initialize short history from true reset position
        self.torso_xy_tminus2 = torso_xy.copy()
        self.torso_xy_tminus1 = torso_xy.copy()
        self.torso_xy_t = torso_xy.copy()

        return self._obs(), self._metrics()

    # =====================================================
    # Observation
    # =====================================================
    def _obs(self) -> jnp.ndarray:
        """
        Builds the enriched observation.

        Final observation:
          [ant_obs,
           torso_xy_tminus2,
           torso_xy_tminus1,
           torso_xy_t,
           torso_vxy,
           goal_vec_xy,
           obstacle_relative_xy (flattened),
           obstacle_radii]
        """
        ant_obs = self.state.obs

        torso_xy = self.torso_xy_t
        torso_vxy = self._get_torso_vxy()
        goal_vec = (self.goal - torso_xy).astype(np.float32)

        # obstacle positions relative to the current torso position
        obs_rel_xy = (self.obs_xy - torso_xy[:, None, :]).astype(np.float32)
        obs_rel_xy = obs_rel_xy.reshape(self.cfg.num_envs, -1)

        obs_r = self.obs_r.astype(np.float32)

        extra = jnp.asarray(
            np.concatenate(
                [
                    self.torso_xy_tminus2,
                    self.torso_xy_tminus1,
                    self.torso_xy_t,
                    torso_vxy,
                    goal_vec,
                    obs_rel_xy,
                    obs_r,
                ],
                axis=1,
            ),
            dtype=jnp.float32,
        )

        return jnp.concatenate([ant_obs, extra], axis=1)

    def total_obs_size(self) -> int:
        """
        Returns the final enriched observation size.
        """
        return (
            self.ant_obs_size
            + 2 + 2 + 2
            + 2
            + 2
            + self.cfg.n_obstacles * 2
            + self.cfg.n_obstacles
        )

    # =====================================================
    # Geometry / safety helpers
    # =====================================================
    def _dist_to_goal(self) -> np.ndarray:
        """
        Distance from torso XY to the 2D goal.

        Shape:
            (num_envs,)
        """
        d = self.goal - self.torso_xy_t
        return np.linalg.norm(d, axis=1).astype(np.float32)

    def _oob(self) -> np.ndarray:
        """
        Out-of-bounds check in the 2D navigation arena.

        Returns:
            bool array of shape (num_envs,)
        """
        x = self.torso_xy_t[:, 0]
        y = self.torso_xy_t[:, 1]
        return (
            (np.abs(x) > self.cfg.arena_size)
            | (np.abs(y) > self.cfg.arena_size)
        )

    def _obstacle_margins(self) -> np.ndarray:
        """
        Returns the margin to each obstacle for each env.

        margin = distance(torso, obstacle_center) - (obstacle_radius + agent_r)

        Shape:
            (num_envs, n_obstacles)
        """
        dists = np.linalg.norm(
            self.obs_xy - self.torso_xy_t[:, None, :],
            axis=2,
        )
        margins = dists - (self.obs_r + self.cfg.agent_r)
        return margins.astype(np.float32)

    def _min_margin(self) -> np.ndarray:
        """
        Minimum obstacle margin for each env.

        Shape:
            (num_envs,)
        """
        margins = self._obstacle_margins()
        return np.min(margins, axis=1).astype(np.float32)

    def _collision(self) -> np.ndarray:
        """
        Collision occurs when min obstacle margin <= 0.

        Returns:
            bool array of shape (num_envs,)
        """
        return self._min_margin() <= 0.0

    def _buffer_violation(self) -> np.ndarray:
        """
        Buffer violation:
          0 < min_margin < buffer_dist

        Returns:
            bool array of shape (num_envs,)
        """
        mm = self._min_margin()
        return (mm > 0.0) & (mm < self.cfg.buffer_dist)

    def _speed_violation(self) -> np.ndarray:
        """
        Speed violation based on torso planar speed.

        Returns:
            bool array of shape (num_envs,)
        """
        vxy = self._get_torso_vxy()
        speed = np.linalg.norm(vxy, axis=1)
        return speed > self.cfg.v_max

    def _fall(self) -> np.ndarray:
        """
        Falling is detected when torso height z is below the threshold.

        Returns:
            bool array of shape (num_envs,)
        """
        z = self._get_torso_z()
        return z < self.cfg.fall_threshold

    def _success(self) -> np.ndarray:
        """
        Success occurs when the torso reaches the goal radius.

        Returns:
            bool array of shape (num_envs,)
        """
        return self._dist_to_goal() < self.cfg.goal_radius

    # =====================================================
    # Metrics
    # =====================================================
    def _metrics(self, **overrides) -> Dict[str, np.ndarray]:
        """
        Builds the default metrics dictionary for all parallel envs.

        These low-level metrics are still useful internally.
        Final paper/prof metrics can be computed later in train/eval code.
        """
        metrics = {
            "success": self._success().astype(np.float32),
            "dist_to_goal": self._dist_to_goal().astype(np.float32),
            "out_of_bounds": self._oob().astype(np.float32),
            "collision": self._collision().astype(np.float32),
            "buffer_violation": self._buffer_violation().astype(np.float32),
            "speed_violation": self._speed_violation().astype(np.float32),
            "fall": self._fall().astype(np.float32),
            "min_margin": self._min_margin().astype(np.float32),
            "steps": self.t.copy().astype(np.float32),
        }
        metrics.update(overrides)
        return metrics

    # =====================================================
    # Shared reward helper
    # =====================================================
    def _progress_reward(self, dist_before: np.ndarray, dist_after: np.ndarray) -> np.ndarray:
        """
        Positive reward when the agent moves closer to the goal.
        Negative reward when it moves farther away.
        """
        progress = (dist_before - dist_after) / (dist_before + 1e-6)
        return (progress * 5.0).astype(np.float32)

    # =====================================================
    # Shared history update
    # =====================================================
    def _update_torso_history(self):
        """
        Updates torso short history after each Brax step.
        """
        new_xy = self._get_torso_xy()
        self.torso_xy_tminus2 = self.torso_xy_tminus1.copy()
        self.torso_xy_tminus1 = self.torso_xy_t.copy()
        self.torso_xy_t = new_xy.copy()

    # =====================================================
    # Shared step pre/post helpers
    # =====================================================
    def _step_brax(self, action: np.ndarray):
        """
        Applies one Brax step for all environments.

        Parameters
        ----------
        action : np.ndarray or jnp.ndarray
            Shape: (num_envs, act_size)
        """
        action = jnp.asarray(action, dtype=jnp.float32)
        self.state = self.step_fn(self.state, action)
        self._update_torso_history()
        self.t += 1

    # =====================================================
    # Config export helper
    # =====================================================
    def cfg_dict(self) -> dict:
        """
        Returns the config as a serializable dictionary.
        """
        return asdict(self.cfg)

    # =====================================================
    # Abstract step
    # =====================================================
    def step(self, action):
        """
        Child classes must implement:
          - reward definition
          - done definition
          - metrics finalization
        """
        raise NotImplementedError("Child environment must implement step().")


# =========================================================
# Environment registry helpers
# =========================================================
ENV_NAME_ORDER = ["no_constraint", "soft_constraint", "hard_constraint"]
ENV_CLASS_MAP: Dict[str, Type[BraxAntBase]] = {}


def register_env(name: str, cls: Type[BraxAntBase]):
    ENV_CLASS_MAP[name] = cls


def get_env_class(name: str):
    if name not in ENV_CLASS_MAP:
        raise ValueError(f"Unknown environment name: {name}")
    return ENV_CLASS_MAP[name]