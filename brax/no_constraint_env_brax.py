# no_constraint_env_brax.py
# =========================================================
# No-constraint Brax environment.
#
# Main idea:
#   - reward progress toward the goal
#   - no explicit safety penalties
#   - no safety-based early termination
#   - episode ends only on:
#       * success
#       * max_steps
#
# Safety events are still measured and logged:
#   - collision
#   - out_of_bounds
#   - speed_violation
#   - fall
# =========================================================

import numpy as np

from baseline_env_brax import BraxAntBase, register_env


class BraxAntNoConstraint(BraxAntBase):
    """
    No-constraint baseline.

    This environment does not penalize safety violations directly.
    It only rewards progress toward the goal, with a small step penalty,
    and gives a success bonus when the goal is reached.
    """

    def step(self, action):
        # -----------------------------------------------
        # Distance before the action
        # -----------------------------------------------
        dist_before = self._dist_to_goal()

        # -----------------------------------------------
        # Apply one Brax step
        # -----------------------------------------------
        self._step_brax(action)

        # -----------------------------------------------
        # Distance after the action
        # -----------------------------------------------
        dist_after = self._dist_to_goal()

        # -----------------------------------------------
        # Base reward:
        #   progress toward goal - small step penalty
        # -----------------------------------------------
        progress = self._progress_reward(dist_before, dist_after)
        reward = progress - self.cfg.step_penalty

        # -----------------------------------------------
        # Success check
        # -----------------------------------------------
        success = self._success().astype(np.float32)
        reward = reward + success * self.cfg.success_bonus

        # -----------------------------------------------
        # No safety-based early termination here
        # Only success or time limit terminate the episode
        # -----------------------------------------------
        done = (success > 0.0) | (self.t >= self.cfg.max_steps)

        # -----------------------------------------------
        # Safety metrics are still tracked for analysis
        # -----------------------------------------------
        collision = self._collision().astype(np.float32)
        oob = self._oob().astype(np.float32)
        speed_violation = self._speed_violation().astype(np.float32)
        fall = self._fall().astype(np.float32)

        metrics = self._metrics(
            success=success,
            dist_to_goal=dist_after.astype(np.float32),
            out_of_bounds=oob,
            collision=collision,
            speed_violation=speed_violation,
            fall=fall,
        )

        return self._obs(), reward.astype(np.float32), done.astype(bool), metrics


register_env("no_constraint", BraxAntNoConstraint)