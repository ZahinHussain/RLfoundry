
import copy
import math
import random
import time
import os
import threading
import psutil
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from RLfoundry.CPO.Agent import Agent          # CPO CHANGE: CPO Agent
from RLfoundry.Utils.run_startup import run_startup

_proc = psutil.Process()
_proc.cpu_percent()  # warm-up call - first real reading needs a prior baseline


# ==============================================================================
# UNCHANGED FROM TRPO
# ==============================================================================

def _fisher_vector_product(agent, obs, x, cg_damping):
    """
    Returns (F + cg_damping*I) @ x without ever forming F.

    pi_old is built from the CURRENT params under no_grad. The KL value is
    therefore exactly 0, but its Hessian at that point is the Fisher matrix -
    which is the whole point. This looks like a bug and is not.
    """
    actor_params = tuple(agent.actor.parameters())
    x = x.detach()

    pi_new = agent.actor(obs)
    with torch.no_grad():
        pi_old = agent.actor(obs)

    kl = kl_divergence(pi_old, pi_new).mean()
    kl_grads = torch.autograd.grad(kl, actor_params, create_graph=True)
    flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in kl_grads])

    kl_grad_p = (flat_kl_grad * x).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor_params)
    flat_kl_hessian_p = torch.cat([g.contiguous().view(-1) for g in kl_hessian_p])

    return flat_kl_hessian_p + cg_damping * x


def conjugate_gradient(agent, obs, b, cg_iters, cg_damping, cg_residual_tol=1e-10):
    """Solve Fx = b for x using only Fisher-vector products."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        _Ax = _fisher_vector_product(agent, obs, p, cg_damping)
        alpha = rdotr / (torch.dot(p, _Ax) + 1e-8)
        x += alpha * p
        r -= alpha * _Ax
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < cg_residual_tol:
            break
    return x


def flat_actor_params(agent):
    return torch.cat([p.view(-1) for p in agent.actor.parameters()]).detach()


def update_actor_params(agent, new_flat_params):
    index = 0
    for p in agent.actor.parameters():
        numel = p.numel()
        p.data.copy_(new_flat_params[index:index + numel].view(p.size()))
        index += numel

import math


def cpo_dual_solve(q, r, s, c, delta):


    EPS = 1e-5

    # ---- degenerate guards -------------------------------------------------
    if not all(math.isfinite(v) for v in (q, r, s, c, delta)) or q <= EPS:
        # No usable reward gradient, or NaNs upstream. Do nothing and let the
        # constraint/ panel show optim_case = -1 so it is visible in wandb.
        return dict(optim_case=-1, lam=0.0, nu=0.0, A=0.0, B=0.0,
                    coef_g=0.0, coef_b=0.0)

    if s <= EPS:
        # Cost gradient is ~zero: b carries no directional information, so
        # there is nothing to move along to reduce cost. Fall back to TRPO.
        # If this fires while c > 0 you are violating the budget with no
        # gradient to fix it -- check the cost critic is actually learning
        # (constraint/ panel: cost_explained_variance).
        lam = math.sqrt(q / (2.0 * delta))
        return dict(optim_case=3, lam=lam, nu=0.0, A=q, B=2.0 * delta,
                    coef_g=1.0 / lam, coef_b=0.0)

    # c == 0 exactly makes lam_mid = r / c blow up. Nudge off the boundary.
    if abs(c) < EPS:
        c = EPS if c >= 0.0 else -EPS

    cc_over_s = c * c / s

    # ---- case 0: infeasible ------------------------------------------------
    # No point in the trust region satisfies the constraint. Ignore reward
    # entirely and spend the whole KL budget walking back inside.
    if c > 0.0 and cc_over_s - 2.0 * delta > 0.0:
        return dict(optim_case=0, lam=0.0, nu=0.0, A=0.0, B=0.0,
                    coef_g=0.0, coef_b=-math.sqrt(2.0 * delta / s))

    # ---- case 3: constraint satisfied across the whole trust region --------
    # Nothing to trade off, so this reduces exactly to TRPO. Note that
    # lam = sqrt(q / 2*delta) puts the step on the trust region boundary:
    # 0.5 * x^T H x = 0.5 * (2*delta/q) * q = delta.
    if c < 0.0 and cc_over_s - 2.0 * delta > 0.0:
        lam = math.sqrt(q / (2.0 * delta))
        return dict(optim_case=3, lam=lam, nu=0.0, A=q, B=2.0 * delta,
                    coef_g=1.0 / lam, coef_b=0.0)

    # ---- cases 1 / 2: constraint is active. Solve the 1-D dual in lam. -----
    A = max(q - r * r / s, EPS)           # >= 0 by Cauchy-Schwarz; clamp fp noise
    B = max(2.0 * delta - cc_over_s, EPS)  # > 0 here by the two tests above

    lam_mid = r / c
    lam_a = math.sqrt(A / B)              # optimum on the nu > 0 branch
    lam_b = math.sqrt(q / (2.0 * delta))  # optimum on the nu = 0 branch

    f_a = -math.sqrt(A * B) - r * c / s
    f_b = -math.sqrt(2.0 * q * delta)

    if lam_mid > 0.0:
        f_mid = -0.5 * (q / lam_mid + 2.0 * lam_mid * delta)

        # Project each candidate onto the interval where its branch is valid.
        # If a candidate is projected out, it collapses onto the kink at
        # lam_mid and takes the kink's dual value with it.
        if c < 0.0:
            if lam_a > lam_mid:
                lam_a, f_a = lam_mid, f_mid
            if lam_b < lam_mid:
                lam_b, f_b = lam_mid, f_mid
        else:
            if lam_a < lam_mid:
                lam_a, f_a = lam_mid, f_mid
            if lam_b > lam_mid:
                lam_b, f_b = lam_mid, f_mid

        lam = lam_a if f_a >= f_b else lam_b
    else:
        # lam_mid <= 0 means the kink is not in the feasible range of lam, so
        # only one branch survives and there is nothing to compare.
        #
        # BUGFIX vs the reference implementation: there, the
        # `lam = lam_a if f_a >= f_b else lam_b` line sits OUTSIDE the
        # `if lam_mid > 0` block and silently overwrites this assignment.
        # Keeping it inside the if/else is the fix.
        lam = lam_b if c < 0.0 else lam_a

    lam = max(lam, EPS)
    nu = max(0.0, (lam * c - r) / s)

    # x* = (1/lam) * (H^-1 g  -  nu * H^-1 b)
    return dict(
        optim_case=(1 if c > 0.0 else 2),
        lam=lam,
        nu=nu,
        A=A,
        B=B,
        coef_g=1.0 / lam,
        coef_b=-nu / lam,
    )



# ==============================================================================
# TRAIN
# ==============================================================================

def train(cfg, env, path):
    path = run_startup(path, cfg)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.algo_name,
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "CPO"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    cg_iters = int(cfg.get("cg_iters", 10))
    cg_damping = float(cfg.get("cg_damping", 0.1))
    backtrack_iters = int(cfg.get("backtrack_iters", 10))
    backtrack_coeff = float(cfg.get("backtrack_coeff", 0.5))

    # ===== CPO CHANGE: new config knobs =====
    # cost_limit is PER EPISODE, in the same units the wrapper emits.
    # Sensible starting point: run TRPO first, read its mean episodic cost
    # off wandb, and set cost_limit to ~50-70% of it. A limit above what
    # TRPO already achieves gives you optim_case 3/4 forever and no result.
    cost_limit = float(cfg.get("cost_limit", 25.0))
    # OPTIONAL, borrowed from SapanaChaudhary/PyTorch-CPO: relax the budget
    # geometrically over training (d_k <- d_k * (1 + e_k)). Not in Achiam et
    # al. - it is a curriculum hack. Useful if you find yourself pinned in
    # optim_case 0 from iteration 1. Set to 0.0 (default) for the honest,
    # paper-faithful fixed budget, and SAY WHICH YOU USED in the writeup.
    cost_limit_anneal = float(cfg.get("cost_limit_anneal", 0.0))
    cost_gamma = float(cfg.get("cost_gamma", cfg.gamma))
    cost_gae_lambda = float(cfg.get("cost_gae_lambda", cfg.gae_lambda))
    delta = float(cfg.target_kl)     # trust region radius, named for clarity




    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available() and cfg.cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    agent = Agent(env).to(device)
    # No actor optimizer. The actor is moved by update_actor_params() only.
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=cfg.learning_rate, eps=cfg.eps)
    # ===== CPO CHANGE: second Adam for V_C =====
    optimizer_cost_critic = optim.Adam(agent.cost_critic.parameters(),
                                       lr=cfg.learning_rate, eps=cfg.eps)

    # ---- pre-allocated storage ----
    obs = torch.zeros((cfg.num_steps,) + env.observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)
    clipped_actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)
    logprobs = torch.zeros(cfg.num_steps).to(device)
    rewards = torch.zeros(cfg.num_steps).to(device)
    dones = torch.zeros(cfg.num_steps).to(device)
    values = torch.zeros(cfg.num_steps).to(device)
    # ===== CPO CHANGE: cost storage =====
    costs = torch.zeros(cfg.num_steps).to(device)
    cost_values = torch.zeros(cfg.num_steps).to(device)

    global_step = 0
    next_obs_np, _ = env.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(1).to(device)
    start_time = time.time()

    current_ep_return = 0.0
    current_ep_length = 0
    current_ep_cost = 0.0           # CPO CHANGE
    last_J_C = None                 # CPO CHANGE: carried across iterations
    last_ep_len = None              # CPO CHANGE

    # ========================================================================
    # OUTER LOOP
    # ========================================================================
    for iteration in range(1, cfg.num_iterations + 1):

        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer_critic.param_groups[0]["lr"] = frac * cfg.learning_rate
            # ===== CPO CHANGE: anneal the cost critic too =====
            optimizer_cost_critic.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ===== CPO CHANGE: per-iteration episode accumulators =====
        ep_costs, ep_lens = [], []
        iter_energy = iter_comfort = iter_total = 0.0  # ADD

        # ====================================================================
        # Collect trajectories
        # ====================================================================
        for step in range(0, cfg.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                # ===== CPO CHANGE: both critic heads =====
                action, logprob, _, value, cost_value = agent.get_action_and_values(next_obs)
                values[step] = value.flatten()
                cost_values[step] = cost_value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            action_np = action.detach().cpu().numpy().reshape(-1)
            action_np_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)
            clipped_actions[step] = torch.as_tensor(action_np_clipped, dtype=torch.float32, device=device)

            next_obs_np, reward, terminations, truncations, info = env.step(action_np_clipped)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = float(reward)

            # ===== CPO CHANGE: read the cost the wrapper emitted =====
            # .get() with a 0.0 default would silently give you a zero-cost
            # CMDP if the wrapper is missing. Fail loudly instead.
            if "cost" not in info:
                raise KeyError(
                    "info['cost'] missing - SplitRewardCost wrapper is not "
                    "applied, or is applied OUTSIDE RecordEpisodeStatistics."
                )
            cost = float(info["cost"])
            costs[step] = cost
            iter_energy += float(info["reward_energy"])  # ADD
            iter_comfort += float(info["reward_comfort"])  # ADD
            iter_total += float(info["reward_total"])  # ADD

            current_ep_return += float(reward)
            current_ep_cost += cost
            current_ep_length += 1

            if next_done:
                ep_costs.append(current_ep_cost)
                ep_lens.append(current_ep_length)
                ep_info = info.get("episode")          # capture BEFORE reset

                if getattr(cfg, "debug_episode_prints", False):
                    print(f"[episode end] global_step={global_step} "
                          f"return={current_ep_return:.2f} "
                          f"cost={current_ep_cost:.2f} length={current_ep_length}")

                next_obs_np, _reset_info = env.reset()   # don't shadow `info`

                if cfg.track:
                    ep_return = ep_info["r"] if ep_info is not None else current_ep_return
                    ep_length = ep_info["l"] if ep_info is not None else current_ep_length
                    wandb.log({
                        "charts/episodic_return": ep_return,
                        "charts/episodic_length": ep_length,
                        "charts/episodic_cost": current_ep_cost,
                    }, step=global_step)

                current_ep_return = 0.0
                current_ep_cost = 0.0
                current_ep_length = 0

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

        # ====================================================================
        # GAE - reward (UNCHANGED)
        # ====================================================================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta_t = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                        delta_t + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

            # ================================================================
            # CPO CHANGE: GAE - cost. Identical recursion, cost tensors,
            # cost discount factors.
            # ================================================================
            next_cost_value = agent.get_cost_value(next_obs).flatten()
            cost_advantages = torch.zeros_like(costs).to(device)
            lastgaelam_c = 0

            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues_c = next_cost_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues_c = cost_values[t + 1]

                delta_c = costs[t] + cost_gamma * nextvalues_c * nextnonterminal - cost_values[t]
                cost_advantages[t] = lastgaelam_c = (
                        delta_c + cost_gamma * cost_gae_lambda * nextnonterminal * lastgaelam_c
                )

            cost_returns = cost_advantages + cost_values

        obs_space = env.observation_space
        act_space = env.action_space

        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # ===== CPO CHANGE =====
        b_cost_advantages = cost_advantages.reshape(-1)
        b_cost_returns = cost_returns.reshape(-1)
        b_cost_values = cost_values.reshape(-1)

        # ====================================================================
        # CPO CHANGE: advantage normalisation, done ONCE on the full batch
        # ====================================================================
        # Reward advantages: centre AND scale, as in TRPO.
        # Cost advantages:   CENTRE ONLY. Do NOT divide by std.
        #   The dual solve balances b against the scalar c, which is in raw
        #   cost units. Rescaling b by an arbitrary per-batch std destroys
        #   that relationship and the constraint stops meaning anything.
        #   This is the single most commonly botched line in a CPO port.
        if cfg.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std(correction=0) + 1e-8)
        b_cost_advantages = b_cost_advantages - b_cost_advantages.mean()

        # ====================================================================
        # PHASE A - CRITICS (minibatched, epochs). CPO CHANGE: doubled.
        # ====================================================================
        b_inds = np.arange(cfg.batch_size)
        v_loss = torch.tensor(0.0)
        vc_loss = torch.tensor(0.0)

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb_inds = b_inds[start:start + cfg.minibatch_size]
                mb_obs = b_obs[mb_inds]

                # Actor is not touched here at all, so no shared-graph worry.
                newvalue = agent.get_value(mb_obs).view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)
                optimizer_critic.step()

                new_cost_value = agent.get_cost_value(mb_obs).view(-1)
                vc_loss = 0.5 * ((new_cost_value - b_cost_returns[mb_inds]) ** 2).mean()
                optimizer_cost_critic.zero_grad()
                vc_loss.backward()
                nn.utils.clip_grad_norm_(agent.cost_critic.parameters(), cfg.max_grad_norm)
                optimizer_cost_critic.step()

        # ====================================================================
        # CPO CHANGE: the constraint scalar c
        # ====================================================================
        # J_C is the mean UNDISCOUNTED episodic cost this iteration. Dividing
        # by mean episode length converts a per-episode budget into the
        # per-step units that b lives in - b comes from a .mean() over steps.
        if len(ep_costs) > 0:
            J_C = float(np.mean(ep_costs))
            mean_ep_len = float(np.mean(ep_lens))
            last_J_C, last_ep_len = J_C, mean_ep_len
        elif last_J_C is not None:
            # No episode finished this iteration - reuse the last estimate
            # rather than fabricating one. If this branch fires often your
            # num_steps is too small relative to the Building episode length
            # and c is badly stale; raise num_steps.
            J_C, mean_ep_len = last_J_C, last_ep_len
        else:
            J_C = float(costs.sum().item())
            mean_ep_len = float(cfg.num_steps)

        c = (J_C - cost_limit) / (mean_ep_len + 1e-8)

        # ====================================================================
        # PHASE B - THE CPO STEP. Full batch, ONCE per iteration.
        # ====================================================================
        _, newlogprob, entropy, _, _ = agent.get_action_and_values(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

        actor_params = tuple(agent.actor.parameters())

        # objective: MAXIMISE
        pg_loss = (b_advantages * ratio).mean()
        # constraint surrogate: MINIMISE
        cost_surr = (b_cost_advantages * ratio).mean()

        # retain_graph on the first call - ratio is shared by both.
        g_grads = torch.autograd.grad(pg_loss, actor_params, retain_graph=True)
        g = torch.cat([t.contiguous().view(-1) for t in g_grads]).detach()
        b_grads = torch.autograd.grad(cost_surr, actor_params)
        b_vec = torch.cat([t.contiguous().view(-1) for t in b_grads]).detach()

        # ===== CPO CHANGE: TWO CG solves instead of one =====
        Hinv_g = conjugate_gradient(agent, b_obs, g, cg_iters, cg_damping)
        Hinv_b = conjugate_gradient(agent, b_obs, b_vec, cg_iters, cg_damping)

        q = torch.dot(g, Hinv_g).item()
        r_scalar = torch.dot(g, Hinv_b).item()
        s = torch.dot(b_vec, Hinv_b).item()

        sol = cpo_dual_solve(q, r_scalar, s, c, delta)
        full_step = sol["coef_g"] * Hinv_g + sol["coef_b"] * Hinv_b

        # ====================================================================
        # PHASE B - line search. CPO CHANGE: THREE conditions, not two.
        # ====================================================================
        old_params = flat_actor_params(agent)
        with torch.no_grad():
            _d = agent.actor(b_obs)
            old_dist = Normal(_d.loc.clone(), _d.scale.clone())
            pg_loss_old = pg_loss.item()
            cost_surr_old = cost_surr.item()

        expected_improve = torch.dot(g, full_step).item()
        fraction = 1.0
        backtrack_success = False
        kl = 0.0
        loss_improve = 0.0
        cost_improve = 0.0

        if sol["optim_case"] == -1:
            # nothing to do - leave the actor where it is
            fraction = 0.0
        else:
            for _ in range(backtrack_iters):
                update_actor_params(agent, old_params + fraction * full_step)

                with torch.no_grad():
                    new_dist = agent.actor(b_obs)
                    new_logprob = new_dist.log_prob(b_actions).sum(-1)
                    new_ratio = (new_logprob - b_logprobs).exp()
                    new_pg_loss = (b_advantages * new_ratio).mean().item()
                    new_cost_surr = (b_cost_advantages * new_ratio).mean().item()
                    kl = kl_divergence(old_dist, new_dist).mean().item()

                loss_improve = new_pg_loss - pg_loss_old
                cost_improve = new_cost_surr - cost_surr_old

                # 1. trust region respected
                cond_kl = kl <= delta
                # 2. reward surrogate improved.
                #    Waived in cases 0 and 1 - when you are violating the
                #    constraint, giving up reward to get back inside the
                #    budget is the CORRECT behaviour, not a failed step.
                cond_improve = (loss_improve > 0) or (sol["optim_case"] <= 1)
                # 3. constraint respected. If already violating (c > 0) then
                #    max(-c, 0) == 0, so the cost surrogate must strictly not
                #    increase. If feasible (c < 0), -c is the remaining slack
                #    and the step may spend up to that much of it.
                cond_cost = cost_improve <= max(-c, 0.0)

                if cond_kl and cond_improve and cond_cost:
                    backtrack_success = True
                    break
                fraction *= backtrack_coeff

            if not backtrack_success:
                update_actor_params(agent, old_params)
                fraction = 0.0
        sps = int(global_step / (time.time() - start_time))
        print(f"[iter {iteration:>4d}]"
              f"  energy/step {iter_energy / cfg.num_steps:>11.5f}"
              f"  comfort/step {iter_comfort / cfg.num_steps:>11.5f}"
              f"  ep_cost {J_C:>10.1f} / {cost_limit:>8.1f}")

        # ====================================================================
        # LOGGING
        # ====================================================================
        if cfg.track:
            _n_eps = cfg.num_steps / max(mean_ep_len, 1.0)
            _denom = abs(iter_energy) + abs(iter_comfort) + 1e-8
            _violation = max(0.0, J_C - cost_limit)
            # single scalar: episodic return minus a penalty on budget overrun.
            # reward and cost are in different units, so score_penalty is an
            # exchange rate you pick, not something the CMDP gives you.
            _combined = (iter_total / _n_eps)

            wandb.log({
                "rewards/ep_energy": iter_energy / _n_eps,
                "rewards/ep_comfort": iter_comfort / _n_eps,
                "rewards/ep_total": iter_total / _n_eps,
                "rewards/step_energy": iter_energy / cfg.num_steps,
                "rewards/step_comfort": iter_comfort / cfg.num_steps,
                "rewards/energy_share": abs(iter_energy) / _denom,
                "rewards/comfort_share": abs(iter_comfort) / _denom,
                # ---- combined reward + safety views ----
                "charts/combined_score": _combined,
                "charts/constraint_violation": _violation,
                "losses/critic_combined": (v_loss + vc_loss).item(),
            }, step=global_step)
            log_cpo_step(
                global_step, env, agent, rewards, costs,
                b_values, b_returns, b_cost_values, b_cost_returns,
                pg_loss, v_loss, vc_loss, cost_surr, entropy,
                old_approx_kl, approx_kl, kl,
              fraction, expected_improve, clipfrac,
                sol, c, J_C, cost_limit, loss_improve, cost_improve,
                clipped_actions, obs, optimizer_critic, optimizer_cost_critic, device,
            )

        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        now = time.time()
        if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
            save_checkpoint(path, global_step, agent, optimizer_critic, optimizer_cost_critic)
            last_numpy_save_t = now

        # OPTIONAL cost-limit annealing (see cost_limit_anneal above)
        cost_limit = cost_limit * (1.0 + cost_limit_anneal)

    save_checkpoint(path, global_step, agent, optimizer_critic, optimizer_cost_critic)
    wandb.finish()
    env.close()


# ==============================================================================
# Logging subroutine - TRPO's, plus a constraint/ panel group
# ==============================================================================

def log_cpo_step(
    global_step, env, agent, rewards, costs,
    b_values, b_returns, b_cost_values, b_cost_returns,
    pg_loss, v_loss, vc_loss, cost_surr, entropy,
    old_approx_kl, approx_kl, kl,
    backtrack_fraction, expected_improve, clipfrac,
    sol, c, J_C, cost_limit, loss_improve, cost_improve,
    clipped_actions, obs, optimizer_critic, optimizer_cost_critic, device,
):
    log_dict = {}

    # --- 1. REWARDS + COSTS ---
    log_dict["rewards/mean"] = rewards.mean().item()
    log_dict["costs/mean"] = costs.mean().item()          # CPO CHANGE
    log_dict["costs/max"] = costs.max().item()            # CPO CHANGE
    log_dict["costs/frac_nonzero"] = (costs > 0).float().mean().item()  # CPO CHANGE

    # --- 2. LOSSES ---
    def _ev(pred, true):
        var_y = np.var(true)
        return float("nan") if var_y == 0 else 1 - np.var(true - pred) / var_y

    explained_var = _ev(b_values.cpu().numpy(), b_returns.cpu().numpy())
    cost_explained_var = _ev(b_cost_values.cpu().numpy(), b_cost_returns.cpu().numpy())

    log_dict.update({
        "losses/policy_loss": pg_loss.item(),
        "losses/value_loss": v_loss.item(),
        "losses/cost_value_loss": vc_loss.item(),            # CPO CHANGE
        "losses/cost_surrogate": cost_surr.item(),           # CPO CHANGE
        "losses/entropy": entropy.mean().item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/line_search_kl": kl,
        "losses/clipfrac": clipfrac,
        "losses/explained_variance": explained_var,
        "losses/cost_explained_variance": cost_explained_var,  # CPO CHANGE
        "losses/backtrack_fraction": backtrack_fraction,
        "losses/expected_improve": expected_improve,
        "losses/loss_improve": loss_improve,                 # CPO CHANGE
        "losses/cost_improve": cost_improve,                 # CPO CHANGE
    })

    # ===== CPO CHANGE: the constraint panel. Read this group FIRST when
    # a run misbehaves - it tells you which branch of the dual you are in. =====
    log_dict.update({
        "constraint/ep_cost": J_C,
        "constraint/cost_limit": cost_limit,
        "constraint/c": c,
        "constraint/violation": max(0.0, J_C - cost_limit),
        "constraint/optim_case": sol["optim_case"],
        "constraint/lambda": sol["lam"],
        "constraint/nu": sol["nu"],
        "constraint/A": sol["A"],
        "constraint/B": sol["B"],
        "constraint/coef_g": sol["coef_g"],
        "constraint/coef_b": sol["coef_b"],
    })

    # --- 3/4. ACTIONS (UNCHANGED FROM TRPO) ---
    actions_np = clipped_actions.detach().cpu().numpy()
    actions_flat = actions_np.reshape(actions_np.shape[0], -1)
    action_dim_means = actions_flat.mean(axis=0)
    action_std_per_dim = torch.exp(agent.actor.actor_logstd).detach().cpu().numpy().flatten()

    log_dict["actions_mean/aggregate"] = float(actions_np.mean())
    for dim_idx in range(actions_flat.shape[-1]):
        log_dict[f"actions_mean/dim{dim_idx}"] = float(action_dim_means[dim_idx])

    log_dict.update({
        "actions_std/aggregate": float(action_std_per_dim.mean()),
        "actions_std/min": float(action_std_per_dim.min()),
        "actions_std/max": float(action_std_per_dim.max()),
        "actions_std/frac_at_clip_bounds": float(
            np.mean(
                (actions_np <= env.action_space.low + 1e-3)
                | (actions_np >= env.action_space.high - 1e-3)
            )
        ),
    })
    for dim_idx, std_val in enumerate(action_std_per_dim):
        log_dict[f"actions_std/dim{dim_idx}"] = float(std_val)

    # --- 5/6. OBSERVATIONS (UNCHANGED FROM TRPO) ---
    obs_np = obs.detach().cpu().numpy()
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)
    obs_dim_means = obs_flat.mean(axis=0)
    obs_dim_stds = obs_flat.std(axis=0)

    log_dict["observations_mean/aggregate"] = float(obs_np.mean())
    for dim_idx in range(obs_flat.shape[-1]):
        log_dict[f"observations_mean/dim{dim_idx}"] = float(obs_dim_means[dim_idx])

    log_dict["observations_std/aggregate"] = float(obs_np.std())
    for dim_idx in range(obs_flat.shape[-1]):
        log_dict[f"observations_std/dim{dim_idx}"] = float(obs_dim_stds[dim_idx])

    log_dict.update({
        "debug/action_min": float(actions_np.min()),
        "debug/action_max": float(actions_np.max()),
        "debug/action_mean": float(actions_np.mean()),
    })

    # --- 8. CHARTS ---
    log_dict.update({
        "charts/critic_learning_rate": optimizer_critic.param_groups[0]["lr"],
        "charts/cost_critic_learning_rate": optimizer_cost_critic.param_groups[0]["lr"],
        "charts/actor_learning_rate": backtrack_fraction,
    })

    # --- CPU / RAM / GPU (UNCHANGED FROM TRPO) ---
    per_core = psutil.cpu_percent(percpu=True)
    log_dict.update({
        "cpu/proc_cpu_percent": _proc.cpu_percent(),
        "cpu/proc_num_threads": _proc.num_threads(),
        "cpu/sys_cpu_percent": psutil.cpu_percent(),
        "cpu/core_max_percent": max(per_core),
        "cpu/core_mean_percent": sum(per_core) / len(per_core),
        "cpu/cores_active_count": sum(1 for c_ in per_core if c_ > 20.0),
        "cpu/num_cores": len(per_core),
    })
    for core_idx, core_pct in enumerate(per_core):
        log_dict[f"cpu_percore/core{core_idx}"] = core_pct

    log_dict.update({
        "ram/proc_used_gb": _proc.memory_info().rss / (1024 ** 3),
        "ram/sys_percent": psutil.virtual_memory().percent,
    })

    try:
        io = _proc.io_counters()
        log_dict.update({
            "ram/proc_read_bytes": io.read_bytes,
            "ram/proc_write_bytes": io.write_bytes,
        })
    except (AttributeError, NotImplementedError):
        pass  # io_counters not available on macOS

    if device.type == "cuda":
        log_dict.update({
            "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
            "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        })
        try:
            log_dict["gpu/utilization_percent"] = torch.cuda.utilization(device)
        except Exception:
            pass  # pynvml not available on this driver setup

    wandb.log(log_dict, step=global_step)


# ==============================================================================
# Checkpointing - TRPO's, with the cost critic optimizer added
# ==============================================================================

def clone_optimizer_state(optimizer_state: dict) -> dict:
    cloned = {"param_groups": copy.deepcopy(optimizer_state["param_groups"]), "state": {}}
    for param_id, state in optimizer_state["state"].items():
        cloned["state"][param_id] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in state.items()
        }
    return cloned


def write_checkpoint(step_dir, model_dir, global_step, agent_state,
                     optimizer_state, cost_optimizer_state, arrays):
    try:
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"cpo_step_{global_step}.pt")
        model_tmp = model_path + ".tmp"
        torch.save({
            "agent_state_dict": agent_state,
            "optimizer_state_dict": optimizer_state,
            # ===== CPO CHANGE =====
            "cost_optimizer_state_dict": cost_optimizer_state,
            "global_step": global_step,
        }, model_tmp)
        os.replace(model_tmp, model_path)

        print(f"[checkpoint] step {global_step} -> {step_dir}, {model_path}")

    except Exception as e:
        print(f"[checkpoint] FAILED at step {global_step}: {e}")


def save_checkpoint(
        save_dir: str,
        global_step: int,
        agent,
        optimizer_critic,
        optimizer_cost_critic,          # CPO CHANGE: new positional arg
        block: bool = False,
        **arrays: np.ndarray,
) -> None:
    """
    The actor still has no optimizer - it is moved by the CG/dual/line-search
    step. agent_state covers actor, critic AND cost_critic automatically.
    """
    agent_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    optimizer_state = clone_optimizer_state(optimizer_critic.state_dict())
    cost_optimizer_state = clone_optimizer_state(optimizer_cost_critic.state_dict())
    arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}

    model_dir = os.path.join(save_dir, "models")

    t = threading.Thread(
        target=write_checkpoint,
        args=(save_dir, model_dir, global_step, agent_state,
              optimizer_state, cost_optimizer_state, arrays),
        daemon=True,
    )
    t.start()
    if block:
        t.join()


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    # Same pre-existing gap as PPO/TD3/SAC/TRPO.
    raise NotImplementedError(
        "main() must construct `env` and `path` before calling train(cfg, env, path). "
        "The env MUST be wrapped: SplitRewardCost inside RecordEpisodeStatistics."
    )


if __name__ == "__main__":
    main()