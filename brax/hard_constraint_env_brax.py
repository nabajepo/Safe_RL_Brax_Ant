# hard_constraint_env_brax.py
# =========================================================
# Hard-constraint Brax environment.
#
# Main idea:
#   - reward progress toward the goal
#   - apply penalties for safety violations:
#       * collision
#       * out_of_bounds
#       * speed_violation
#       * fall
#   - terminate the episode immediately on:
#       * collision
#       * out_of_bounds
#       * speed_violation
#       * fall
#   - success still gives a positive bonus
#
# This version is strict:
#   - stronger progress reward
#   - speed violation is treated as a hard violation again
# =========================================================

import numpy as np

from baseline_env_brax import BraxAntBase, register_env


class BraxAntHardConstraint(BraxAntBase):
    """
    Hard-constraint environment.

    Immediate termination for:
      - collision
      - out_of_bounds
      - speed_violation
      - fall
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
        # Safety checks
        # -----------------------------------------------
        collision = self._collision().astype(np.float32)
        oob = self._oob().astype(np.float32)
        speed_violation = self._speed_violation().astype(np.float32)
        fall = self._fall().astype(np.float32)

        # -----------------------------------------------
        # Penalties
        # -----------------------------------------------
        reward = reward - collision * self.cfg.collision_penalty
        reward = reward - oob * self.cfg.oob_penalty
        reward = reward - speed_violation * self.cfg.speed_penalty
        reward = reward - fall * self.cfg.fall_penalty

        # -----------------------------------------------
        # Success
        # -----------------------------------------------
        success = self._success().astype(np.float32)
        reward = reward + success * self.cfg.success_bonus

        # -----------------------------------------------
        # Hard termination
        #   - success
        #   - collision
        #   - out_of_bounds
        #   - speed_violation
        #   - fall
        #   - max_steps
        # -----------------------------------------------
        done = (
            (success > 0.0)
            | (collision > 0.0)
            | (oob > 0.0)
            | (speed_violation > 0.0)
            | (fall > 0.0)
            | (self.t >= self.cfg.max_steps)
        )

        metrics = self._metrics(
            success=success,
            dist_to_goal=dist_after.astype(np.float32),
            out_of_bounds=oob,
            collision=collision,
            speed_violation=speed_violation,
            fall=fall,
        )

        return self._obs(), reward.astype(np.float32), done.astype(bool), metrics


register_env("hard_constraint", BraxAntHardConstraint)