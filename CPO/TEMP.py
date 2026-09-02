"""
SplitRewardCost - turns the HVAC env's scalar reward into a CMDP.

The env's reward is  -(energy_penalty + comfort_penalty), where comfort is
~450x larger than energy. Summed, comfort IS the reward and energy is noise.
Split apart:

    reward = -energy_penalty          <- the objective CPO maximises
    cost   = 1[comfort_penalty > tau] <- the constraint CPO bounds

Wrapping order matters. RecordEpisodeStatistics must be OUTSIDE this, or the
episode dict is built from the unsplit reward:

    env = SplitRewardCost(env, comfort_threshold=5.0)
    env = RecordEpisodeStatistics(env)
"""
from __future__ import annotations

import gymnasium as gym


class SplitRewardCost(gym.Wrapper):
    """Splits the HVAC scalar reward into an energy objective and a comfort cost.

    Args:
        env: the HVAC environment. Its step() must place 'energy_penalty' and
            'comfort_penalty' in info (see env patch below).
        comfort_threshold: per-step comfort penalty above which the step counts
            as a constraint violation. Chosen from the converged unconstrained
            policy's comfort distribution, not picked a priori.
        cost_mode: 'indicator' counts violating steps (episode cost = number of
            violations, directly interpretable). 'magnitude' accumulates
            max(comfort - tau, 0) instead.
        energy_scale: multiplier on the energy reward. Energy is ~0.04/step, so
            with 288 steps an episode return is ~-11. That is workable, but if
            your value loss stays flat, raise this rather than touching the cost.
    """

    def __init__(self, env: gym.Env, comfort_threshold: float = 5.0,
                 cost_mode: str = "indicator", energy_scale: float = 1.0):
        super().__init__(env)
        if cost_mode not in ("indicator", "magnitude"):
            raise ValueError(f"cost_mode must be 'indicator' or 'magnitude', got {cost_mode}")
        self.comfort_threshold = float(comfort_threshold)
        self.cost_mode = cost_mode
        self.energy_scale = float(energy_scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Penalties are expected as POSITIVE magnitudes in info. Fail loudly
        # rather than silently producing a zero-cost CMDP.
        for key in ("energy_penalty", "comfort_penalty"):
            if key not in info:
                raise KeyError(
                    f"info['{key}'] missing - the HVAC env's step() has not been "
                    "patched to expose its reward components. See env patch."
                )

        energy = abs(float(info["energy_penalty"]))
        comfort = abs(float(info["comfort_penalty"]))

        # --- objective: energy only -----------------------------------------
        new_reward = -self.energy_scale * energy

        # --- constraint: comfort --------------------------------------------
        if self.cost_mode == "indicator":
            cost = float(comfort > self.comfort_threshold)
        else:
            cost = max(comfort - self.comfort_threshold, 0.0)

        info["cost"] = cost
        info["comfort_violation"] = float(comfort > self.comfort_threshold)

        # Consumed by the train loop's per-iteration accumulators. Kept as
        # signed penalties so they read the same as the original reward log.
        info["reward_energy"] = -energy
        info["reward_comfort"] = -comfort
        info["reward_total"] = -(energy + comfort)

        return obs, new_reward, terminated, truncated, info