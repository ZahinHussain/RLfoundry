import gymnasium as gym


class SplitRewardCost(gym.Wrapper):
    """
    Splits BuildingEnv's scalar reward into an energy reward and a comfort cost.

    Prefers the PER-STEP keys added by the step() patch:
        info["energy_penalty"], info["comfort_penalty"]   (both >= 0)

    Falls back to differencing the CUMULATIVE running totals in
    info["reward_breakdown"] if the patch is not applied. That fallback exists
    so an unpatched env still works, but the direct keys are less fragile -
    differencing desyncs if anything resets the env behind this wrapper's back.

    Guarantees, all checked in step():
      - info["cost"] >= 0, larger = worse (the sign CPO's dual solve assumes)
      - info["reward_total"] == the ORIGINAL env reward, untouched
      - reward_energy + reward_comfort == reward_total

    Args:
        cost_scale: multiplier applied to the cost after thresholding.
        reward_mode: "energy_only" gives CPO the energy objective alone (the
            proper CMDP split). "full" passes the original scalar reward
            through unchanged, for a TRPO/PPO baseline on the same wrapper.
        comfort_threshold: per-step comfort penalty above which a step counts
            as a violation. Derive this from the converged unconstrained
            policy's comfort distribution rather than picking it a priori.
        cost_mode:
            "magnitude" - cost = max(|comfort| - threshold, 0). Graded signal,
                easiest for the cost critic to fit. Start here.
            "indicator" - cost = 1 if |comfort| > threshold else 0. Episode
                cost is then a count of violating steps, which is directly
                interpretable but flat and harder to fit.
            "raw"       - cost = |comfort|, no threshold. Reproduces the
                pre-threshold behaviour, for A/B comparison.

    Wrapping order (RecordEpisodeStatistics must be OUTSIDE this one, or the
    episode dict is built from the unsplit reward):

        env = SplitRewardCost(env, comfort_threshold=5.0)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

    Do NOT add NormalizeReward: it rescales the reward but not the cost, which
    breaks the relationship between b and c that the CPO dual solve depends on.
    """

    def __init__(self, env, cost_scale=1.0, reward_mode="energy_only",
                 comfort_threshold=10.0, cost_mode="magnitude",
                 check_consistency=True):
        super().__init__(env)
        assert reward_mode in ("energy_only", "full")
        assert cost_mode in ("magnitude", "indicator", "raw")
        self.cost_scale = float(cost_scale)
        self.reward_mode = reward_mode
        self.comfort_threshold = float(comfort_threshold)
        self.cost_mode = cost_mode
        self.check_consistency = check_consistency
        self._direct = None          # decided on first step()
        self._prev_comfort = 0.0
        self._prev_energy = 0.0

    def _components(self, info):
        """Returns (energy, comfort) for THIS STEP, both <= 0 (reward terms)."""
        if self._direct is None:
            self._direct = "comfort_penalty" in info

        if self._direct:
            # Patched env: positive magnitudes, negate to get reward terms.
            return -float(info["energy_penalty"]), -float(info["comfort_penalty"])

        # Unpatched env: difference the cumulative totals. Read the floats out
        # immediately - reward_breakdown is a LIVE reference to the env's
        # internal dict, not a copy.
        bd = info["reward_breakdown"]
        cum_energy = float(bd["power_consumption"])
        cum_comfort = float(bd["comfort_level"])
        energy = cum_energy - self._prev_energy
        comfort = cum_comfort - self._prev_comfort
        self._prev_energy, self._prev_comfort = cum_energy, cum_comfort
        return energy, comfort

    def _cost_from_comfort(self, comfort):
        """Maps this step's comfort reward term to a non-negative cost."""
        comfort_mag = abs(comfort)

        if self.cost_mode == "magnitude":
            cost = max(comfort_mag - self.comfort_threshold, 0.0)
        elif self.cost_mode == "indicator":
            cost = float(comfort_mag > self.comfort_threshold)
        else:  # "raw"
            cost = comfort_mag

        return cost * self.cost_scale, comfort_mag

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        bd = info.get("reward_breakdown")
        if isinstance(bd, dict):
            self._prev_energy = float(bd.get("power_consumption", 0.0))
            self._prev_comfort = float(bd.get("comfort_level", 0.0))
        else:
            self._prev_energy = self._prev_comfort = 0.0

        info = dict(info)
        info["cost"] = 0.0
        info["comfort_violation"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        energy, comfort = self._components(info)

        if self.check_consistency:
            tol = 1e-6 * max(1.0, abs(float(reward)))
            if abs((energy + comfort) - float(reward)) > tol:
                raise ValueError(
                    f"components do not reconstruct reward: energy={energy} + "
                    f"comfort={comfort} != reward={reward}. If using the "
                    f"cumulative fallback, something reset the env without "
                    f"going through this wrapper."
                )

        cost, comfort_mag = self._cost_from_comfort(comfort)

        info = dict(info)
        info["cost"] = cost
        info["comfort_violation"] = float(comfort_mag > self.comfort_threshold)
        info["reward_total"] = float(reward)
        info["reward_energy"] = energy
        info["reward_comfort"] = comfort

        out_reward = energy if self.reward_mode == "energy_only" else float(reward)
        return obs, out_reward, terminated, truncated, info