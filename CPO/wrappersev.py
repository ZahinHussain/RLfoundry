"""
RLfoundry/Wrappers/split_reward_cost_ev.py

EVChargingEnv (SustainGym) -> CMDP, for the CPO module.

    r(t) = p(t) - cV(t) - cC(t)          original env reward
    reward = p(t) - cC(t)                profit net of emissions   <- MAXIMISE
    cost   = f(cV(t))                    network/power violation   <- CONSTRAIN

WHY cV AND NOT cC:
  cV is a hard physical limit (transformer / line current). Exceeding it trips
  breakers and damages equipment - a genuine safety constraint. cC is an
  externality that belongs in the objective, priced, not bounded. Constraining
  cC instead would just be a carbon tax with extra steps.

CRITICAL - DISABLE ACTION PROJECTION:
  EVChargingEnv can project actions into the convex hull of the feasible set
  before execution. With projection ON, cV(t) ~ 0 always, so the cost
  advantages carry no signal, s = b^T H^-1 b collapses, and CPO.py's
  `if s <= EPS` guard silently returns optim_case=3 - i.e. plain TRPO with a
  flat-zero cost curve that LOOKS like a converged safe policy.
  Construct the env with projection off. Keep a projection-on run as the
  oracle-safe upper baseline.

WRAPPER ORDER:
    FlattenObservation( EVChargingEnv(...) )     <- obs is a Dict space
      -> EVSplitRewardCost(...)
        -> RecordEpisodeStatistics(...)          <- must be OUTERMOST

  Use action_type='continuous' - the CPO Agent has a Gaussian actor.

Episode: 288 steps (24h at 5-min), same as BuildingEnv.
"""

import gymnasium as gym
import numpy as np


class EVSplitRewardCost(gym.Wrapper):
    """
    Guarantees, all checked in step():
      - info["cost"] >= 0, larger = worse (the sign CPO's dual solve assumes)
      - info["reward_total"] == the ORIGINAL env reward, untouched
      - reward_profit + reward_carbon + reward_violation == reward_total

    cost_type:
      "indicator"  cost(t) = 1 if this step violated at all, else 0.
                   J_C in [0, 288], reads as "timesteps with any infrastructure
                   violation per day". Bounded, so c stays O(1) and the dual
                   stays well conditioned. THIS IS THE DEFAULT.
      "magnitude"  cost(t) = |cV(t)| in the env's own units. Smoother gradient
                   for the cost critic, but unbounded - c can grow large and
                   c^2/s > 2*delta pins you in optim_case 0. Secondary /
                   robustness run only, and recalibrate cost_limit for it.
    """

    # Candidate key names, most likely first. The first-step probe reports the
    # keys actually present if none of these match, so a version bump in
    # SustainGym surfaces as a clear error rather than a silent zero cost.
    _PROFIT_KEYS = ("profit", "revenue", "energy_profit")
    _CARBON_KEYS = ("carbon_cost", "emissions_cost", "co2_cost", "carbon")
    _VIOLATION_KEYS = ("excess_charge", "constraint_violation", "violation_cost",
                       "network_violation")

    def __init__(self, env, cost_scale=1.0, cost_type="indicator",
                 reward_mode="profit_minus_carbon", violation_tol=1e-6,
                 breakdown_key="reward_breakdown", breakdown_is_cumulative=False,
                 check_consistency=True):
        super().__init__(env)
        assert reward_mode in ("profit_minus_carbon", "profit_only", "full")
        assert cost_type in ("indicator", "magnitude")
        self.cost_scale = float(cost_scale)
        self.cost_type = cost_type
        self.reward_mode = reward_mode
        self.violation_tol = float(violation_tol)
        self.breakdown_key = breakdown_key
        self.cumulative = bool(breakdown_is_cumulative)
        self.check_consistency = check_consistency

        self._keys = None                       # resolved on first step()
        self._prev = {"p": 0.0, "c": 0.0, "v": 0.0}

    # ------------------------------------------------------------------
    @staticmethod
    def _pick(bd, candidates, what):
        for k in candidates:
            if k in bd:
                return k
        raise KeyError(
            f"could not find the {what} term in the reward breakdown. "
            f"Tried {candidates}. Keys actually present: {sorted(bd.keys())}. "
            f"Add the correct name to the corresponding _KEYS tuple."
        )

    def _resolve_keys(self, bd):
        self._keys = (
            self._pick(bd, self._PROFIT_KEYS, "profit"),
            self._pick(bd, self._CARBON_KEYS, "carbon"),
            self._pick(bd, self._VIOLATION_KEYS, "constraint-violation"),
        )

    def _components(self, info):
        """
        Returns (profit, carbon, violation) as SIGNED REWARD TERMS for this
        step, i.e. they sum to the env reward: profit >= 0, carbon <= 0,
        violation <= 0.
        """
        if self.breakdown_key not in info:
            raise KeyError(
                f"info['{self.breakdown_key}'] missing - EVChargingEnv is not "
                f"emitting a reward breakdown. Keys present: {sorted(info.keys())}"
            )
        bd = info[self.breakdown_key]
        if self._keys is None:
            self._resolve_keys(bd)
        kp, kc, kv = self._keys

        # Read floats out immediately - the breakdown may be a live reference
        # to the env's internal dict, not a copy.
        p_raw, c_raw, v_raw = float(bd[kp]), float(bd[kc]), float(bd[kv])

        if self.cumulative:
            p = p_raw - self._prev["p"]
            c = c_raw - self._prev["c"]
            v = v_raw - self._prev["v"]
            self._prev = {"p": p_raw, "c": c_raw, "v": v_raw}
        else:
            p, c, v = p_raw, c_raw, v_raw

        # The env documents r = p - cV - cC with cV, cC >= 0 magnitudes.
        # Negate to reward-term convention. If the env already stores them
        # signed, check_consistency will fire and tell you.
        return p, -abs(c), -abs(v)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        bd = info.get(self.breakdown_key)
        if self.cumulative and isinstance(bd, dict):
            if self._keys is None:
                self._resolve_keys(bd)
            kp, kc, kv = self._keys
            self._prev = {"p": float(bd.get(kp, 0.0)),
                          "c": float(bd.get(kc, 0.0)),
                          "v": float(bd.get(kv, 0.0))}
        else:
            self._prev = {"p": 0.0, "c": 0.0, "v": 0.0}

        info = dict(info)
        info["cost"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        profit, carbon, violation = self._components(info)

        if self.check_consistency:
            tol = 1e-5 * max(1.0, abs(float(reward)))
            recon = profit + carbon + violation
            if abs(recon - float(reward)) > tol:
                raise ValueError(
                    f"components do not reconstruct reward: profit={profit} + "
                    f"carbon={carbon} + violation={violation} = {recon} != "
                    f"reward={reward}. Most likely causes: the breakdown is "
                    f"CUMULATIVE (pass breakdown_is_cumulative=True), the terms "
                    f"are already stored signed, or the resolved keys "
                    f"{self._keys} are the wrong ones."
                )

        mag = abs(violation)
        if self.cost_type == "indicator":
            cost = float(mag > self.violation_tol)
        else:
            cost = mag
        cost *= self.cost_scale

        if self.reward_mode == "profit_minus_carbon":
            out_reward = profit + carbon
        elif self.reward_mode == "profit_only":
            out_reward = profit
        else:
            out_reward = float(reward)

        info = dict(info)
        info["cost"] = cost
        info["reward_total"] = float(reward)
        info["reward_profit"] = profit
        info["reward_carbon"] = carbon
        info["reward_violation"] = violation
        # aliases so log_cpo_step / the rewards/ panel needs no edit
        info["reward_energy"] = profit
        info["reward_comfort"] = carbon
        return obs, out_reward, terminated, truncated, info


# ----------------------------------------------------------------------
def make_ev_cmdp(env, cost_scale=1.0, cost_type="indicator", **kw):
    """FlattenObservation -> EVSplitRewardCost -> RecordEpisodeStatistics."""
    env = gym.wrappers.FlattenObservation(env)
    env = EVSplitRewardCost(env, cost_scale=cost_scale, cost_type=cost_type, **kw)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def probe_cost_scale(env, n_steps=288, seed=0):
    """
    Run one do-nothing episode and one uniform-random episode, print episodic
    cost for each. Use these to sanity-check cost_limit BEFORE a 500-iteration
    run. Random should violate heavily; zeros should not violate at all.
    """
    out = {}
    for name, sampler in (("zeros", lambda: np.zeros(env.action_space.shape)),
                          ("random", lambda: env.action_space.sample())):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        for _ in range(n_steps):
            obs, r, term, trunc, info = env.step(sampler())
            total += info["cost"]
            if term or trunc:
                break
        out[name] = total
        print(f"[probe] {name:>6}  J_C = {total:.3f}")
    return out