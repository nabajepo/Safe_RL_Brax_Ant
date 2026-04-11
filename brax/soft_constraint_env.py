# soft_constraint_env.py
# =========================================================
# Soft-constraint Brax environment.
#
# Main idea:
#   - reward progress toward the goal
#   - apply soft penalties for safety-related violations:
#       * collision
#       * out_of_bounds
#       * speed_violation
#       * fall
#   - episode ends only on:
#       * success
#       * max_steps
#
# Important:
#   - safety violations are penalized
#   - but they do NOT terminate the episode immediately here
#   - this keeps the constraints "soft"
# =========================================================

import numpy as np

from baseline_env import BraxAntBase, register_env


class BraxAntSoftConstraint(BraxAntBase):
    """
    Soft-constraint environment.

    This version adds reward penalties for unsafe behavior, but
    does not terminate immediately on safety violations.
    """

    def step(self, action):
        # -----------------------------------------------
        # Distance before action
        # -----------------------------------------------
        dist_before = self._dist_to_goal()

        # -----------------------------------------------
        # Apply one Brax step
        # -----------------------------------------------
        self._step_brax(action)

        # -----------------------------------------------
        # Distance after action
        # -----------------------------------------------
        dist_after = self._dist_to_goal()

        # -----------------------------------------------
        # Base reward:
        #   stronger progress reward - small step penalty
        # -----------------------------------------------
        progress = self._progress_reward(dist_before, dist_after)
        reward = 3.0 * progress - self.cfg.step_penalty

        # -----------------------------------------------
        # Soft safety checks
        # -----------------------------------------------
        collision = self._collision().astype(np.float32)
        oob = self._oob().astype(np.float32)
        speed_violation = self._speed_violation().astype(np.float32)
        fall = self._fall().astype(np.float32)

        # -----------------------------------------------
        # Apply soft penalties
        # -----------------------------------------------
        reward = reward - collision * self.cfg.collision_penalty
        reward = reward - oob * self.cfg.oob_penalty
        reward = reward - speed_violation * self.cfg.speed_penalty
        reward = reward - fall * self.cfg.fall_penalty

        # -----------------------------------------------
        # Success check
        # -----------------------------------------------
        success = self._success().astype(np.float32)
        reward = reward + success * self.cfg.success_bonus

        # -----------------------------------------------
        # Episode termination:
        # only success or max_steps
        # -----------------------------------------------
        done = (success > 0.0) | (self.t >= self.cfg.max_steps)

        metrics = self._metrics(
            success=success,
            dist_to_goal=dist_after.astype(np.float32),
            out_of_bounds=oob,
            collision=collision,
            speed_violation=speed_violation,
            fall=fall,
        )

        return self._obs(), reward.astype(np.float32), done.astype(bool), metrics


register_env("soft_constraint", BraxAntSoftConstraint)