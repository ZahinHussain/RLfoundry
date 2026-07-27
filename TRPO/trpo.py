import copy
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
from torch.distributions.kl import kl_divergence

from RLfoundry.TRPO import Agent
from RLfoundry.Utils.run_startup import run_startup

_proc = psutil.Process()
_proc.cpu_percent()  # warm-up call — first real reading needs a prior baseline


# ==============================================================================
# TRPO core math: Fisher-vector product, conjugate gradient, param (un)flattening
# ==============================================================================
#
# NOTE ON A DELIBERATE CHANGE FROM THE REFERENCE IMPLEMENTATION:
# The reference (CleanRL PR #435) recomputes the CG + line-search step once
# per minibatch, inside the update_epochs/minibatch loop. That breaks the
# trust-region guarantee TRPO is built on: the KL constraint bounds the step
# against the *data it was computed on*, so re-deriving a fresh natural-
# gradient step for every minibatch and applying it in sequence is closer to
# repeated first-order updates than to a single trust-region step. Standard
# TRPO takes ONE step per iteration using the full rollout batch. That's what
# this file does — the actor is updated once per iteration below; only the
# critic is trained across minibatch epochs, same as your PPO file.

def _fisher_vector_product(agent, obs, x, cg_damping):
    actor_params = tuple(agent.actor_parameters())
    pi_new = agent.get_distribution(obs)
    with torch.no_grad():
        pi_old = agent.get_distribution(obs)
    kl = kl_divergence(pi_old, pi_new).mean()
    kl_grads = torch.autograd.grad(kl, actor_params, create_graph=True)
    flat_kl_grad = torch.cat([grad.contiguous().view(-1) for grad in kl_grads])
    kl_grad_p = (flat_kl_grad * x).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor_params)
    flat_kl_hessian_p = torch.cat([grad.contiguous().view(-1) for grad in kl_hessian_p])

    # damping term stabilizes the CG solve — see
    # https://www2.maths.lth.se/matematiklth/vision/publdb/reports/pdf/byrod-eccv-10.pdf
    return flat_kl_hessian_p + cg_damping * x


def conjugate_gradient(agent, obs, b, cg_iters, cg_damping, cg_residual_tol=1e-10):
    """
    Solves Fx = b for x without forming the Fisher information matrix F
    explicitly — only the Fisher-vector product is needed. See
    https://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
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
    return torch.cat([p.view(-1) for p in agent.actor_parameters()]).detach()


def update_actor_params(agent, new_flat_params):
    index = 0
    for p in agent.actor_parameters():
        numel = p.numel()
        p.data.copy_(new_flat_params[index:index + numel].view(p.size()))
        index += numel


# ==============================================================================
# Training loop
# ==============================================================================

def train(cfg, env, path):
    path = run_startup(path, cfg)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.algo_name,
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "trpo"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    # TRY NOT TO MODIFY: seeding
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
    optimizer_critic = optim.Adam(agent.critic_parameters(), lr=cfg.learning_rate, eps=cfg.eps)

    # ---- pre-allocated storage for one rollout's worth of data ----
    obs = torch.zeros((cfg.num_steps,) + env.observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)
    clipped_actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)
    logprobs = torch.zeros(cfg.num_steps).to(device)
    rewards = torch.zeros(cfg.num_steps).to(device)
    dones = torch.zeros(cfg.num_steps).to(device)
    values = torch.zeros(cfg.num_steps).to(device)

    global_step = 0
    next_obs_np, _ = env.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(1).to(device)
    start_time = time.time()

    current_ep_return = 0.0
    current_ep_length = 0

    # ========================================================================
    # OUTER LOOP
    # ========================================================================
    for iteration in range(1, cfg.num_iterations + 1):

        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer_critic.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ====================================================================
        # Collect trajectories
        # ====================================================================
        for step in range(0, cfg.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            action_np = action.detach().cpu().numpy().reshape(-1)
            action_np_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)
            clipped_actions[step] = torch.as_tensor(action_np_clipped, dtype=torch.float32, device=device)

            next_obs_np, reward, terminations, truncations, info = env.step(action_np_clipped)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = float(reward)

            current_ep_return += float(reward)
            current_ep_length += 1

            if next_done:
                if getattr(cfg, "debug_episode_prints", False):
                    print(
                        f"[episode end] global_step={global_step} return={current_ep_return:.2f} length={current_ep_length}")

                next_obs_np, info = env.reset()

                if cfg.track:
                    ep_return = info["episode"]["r"] if "episode" in info else current_ep_return
                    ep_length = info["episode"]["l"] if "episode" in info else current_ep_length
                    wandb.log({
                        "charts/episodic_return": ep_return,
                        "charts/episodic_length": ep_length,
                    }, step=global_step)

                current_ep_return = 0.0
                current_ep_length = 0

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

        # ====================================================================
        # GAE (Generalized Advantage Estimation)
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

                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                        delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        obs_space = env.observation_space
        act_space = env.action_space

        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if cfg.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(correction=0) + 1e-8)

        # ====================================================================
        # 1. Critic update — plain Adam regression, minibatched over epochs
        # ====================================================================
        b_inds = np.arange(cfg.batch_size)
        v_loss = torch.tensor(0.0)
        for epoch in range(cfg.critic_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic_parameters(), cfg.max_grad_norm)
                optimizer_critic.step()

        # ====================================================================
        # 2. Actor update — single full-batch TRPO trust-region step
        # ====================================================================
        _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()

        pg_loss = (b_advantages * ratio).mean()
        pg_grad = torch.autograd.grad(pg_loss, tuple(agent.actor_parameters()))
        flat_pg_grad = torch.cat([grad.contiguous().view(-1) for grad in pg_grad])

        step_dir = conjugate_gradient(agent, b_obs, flat_pg_grad, cfg.cg_iters, cfg.cg_damping)
        shs = torch.dot(step_dir, _fisher_vector_product(agent, b_obs, step_dir, cfg.cg_damping))
        step_size = torch.sqrt(2 * cfg.target_kl / (shs + 1e-8))
        full_step = step_dir * step_size

        # Backtracking line search: shrink the step until both the KL bound
        # and a genuine surrogate-objective improvement hold.
        old_params = flat_actor_params(agent)
        old_agent = copy.deepcopy(agent)
        expected_improve = (flat_pg_grad * full_step).sum().item()
        fraction = 1.0
        backtrack_success = False
        for _ in range(cfg.backtrack_iters):
            new_params = old_params + fraction * full_step
            update_actor_params(agent, new_params)

            with torch.no_grad():
                new_dist = agent.get_distribution(b_obs)
                new_logprob = new_dist.log_prob(b_actions).sum(-1)
                new_ratio = (new_logprob - b_logprobs).exp()
                new_pg_loss = (b_advantages * new_ratio).mean()
                loss_improve = (new_pg_loss - pg_loss).item()
                kl = kl_divergence(old_agent.get_distribution(b_obs), new_dist).mean().item()

            if kl <= cfg.target_kl and loss_improve > 0:
                backtrack_success = True
                break
            fraction *= cfg.backtrack_coeff

        if not backtrack_success:
            update_actor_params(agent, old_params)
            fraction = 0.0

        # ====================================================================
        # LOGGING TO WANDB
        # ====================================================================
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)

        if cfg.track:
            log_trpo_step(
                global_step, env, agent, rewards,
                b_values, b_returns, pg_loss, v_loss, entropy,
                old_approx_kl, approx_kl, kl, fraction, expected_improve,
                clipped_actions, obs, optimizer_critic, device,
            )

        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        now = time.time()
        if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
            save_checkpoint(path, global_step, agent, optimizer_critic)
            last_numpy_save_t = now

    save_checkpoint(path, global_step, agent, optimizer_critic)
    wandb.finish()
    env.close()


# ==============================================================================
# Logging subroutine
# ==============================================================================

def log_trpo_step(
    global_step, env, agent, rewards,
    b_values, b_returns, pg_loss, v_loss, entropy,
    old_approx_kl, approx_kl, kl, backtrack_fraction, expected_improve,
    clipped_actions, obs, optimizer_critic, device,
):
    log_dict = {}

    # --- 1. REWARDS ---
    log_dict["rewards/mean"] = rewards.mean().item()

    # --- 2. LOSSES & OPTIMIZATION ---
    y_pred = b_values.cpu().numpy()
    y_true = b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    log_dict.update({
        "losses/policy_loss": pg_loss.item(),
        "losses/value_loss": v_loss.item(),
        "losses/entropy": entropy.mean().item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/line_search_kl": kl,
        "losses/explained_variance": explained_var,
        "losses/backtrack_fraction": backtrack_fraction,
        "losses/expected_improve": expected_improve,
    })

    # --- 3. ACTIONS: mean (aggregate + per-dim) ---
    actions_np = clipped_actions.detach().cpu().numpy()
    actions_flat = actions_np.reshape(actions_np.shape[0], -1)
    action_dim_means = actions_flat.mean(axis=0)
    action_std_per_dim = torch.exp(agent.actor_logstd).detach().cpu().numpy().flatten()

    log_dict["actions_mean/aggregate"] = float(actions_np.mean())
    for dim_idx in range(actions_flat.shape[-1]):
        log_dict[f"actions_mean/dim{dim_idx}"] = float(action_dim_means[dim_idx])

    # --- 4. ACTIONS: std (aggregate + per-dim) ---
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

    # --- 5. OBSERVATIONS: mean (aggregate + per-dim) ---
    obs_np = obs.detach().cpu().numpy()
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)
    obs_dim_means = obs_flat.mean(axis=0)
    obs_dim_stds = obs_flat.std(axis=0)

    log_dict["observations_mean/aggregate"] = float(obs_np.mean())
    for dim_idx in range(obs_flat.shape[-1]):
        log_dict[f"observations_mean/dim{dim_idx}"] = float(obs_dim_means[dim_idx])

    # --- 6. OBSERVATIONS: std (aggregate + per-dim) ---
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
    })

    # --- CPU (separate panel group) ---
    per_core = psutil.cpu_percent(percpu=True)

    log_dict.update({
        "cpu/proc_cpu_percent": _proc.cpu_percent(),
        "cpu/proc_num_threads": _proc.num_threads(),
        "cpu/sys_cpu_percent": psutil.cpu_percent(),
        "cpu/core_max_percent": max(per_core),
        "cpu/core_mean_percent": sum(per_core) / len(per_core),
        "cpu/cores_active_count": sum(1 for c in per_core if c > 20.0),
        "cpu/num_cores": len(per_core),
    })
    for core_idx, core_pct in enumerate(per_core):
        log_dict[f"cpu_percore/core{core_idx}"] = core_pct

    # --- RAM (separate panel group) ---
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

    # --- GPU (separate panel group) ---
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
# Checkpointing — identical pattern to the PPO file: clone everything before
# handing off to a background thread so the main loop never races the writer.
# ==============================================================================

def clone_optimizer_state(optimizer_state: dict) -> dict:
    cloned = {"param_groups": copy.deepcopy(optimizer_state["param_groups"]), "state": {}}
    for param_id, state in optimizer_state["state"].items():
        cloned["state"][param_id] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in state.items()
        }
    return cloned


def write_checkpoint(step_dir, model_dir, global_step, agent_state, optimizer_state, arrays):
    try:
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"trpo_step_{global_step}.pt")
        model_tmp = model_path + ".tmp"
        torch.save({
            "agent_state_dict": agent_state,
            "optimizer_state_dict": optimizer_state,
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
        block: bool = False,
        **arrays: np.ndarray,
) -> None:
    """
    Note: only the critic optimizer has state worth checkpointing (the
    actor has no optimizer — it's updated via the manual CG/line-search
    step), but agent_state covers both actor and critic weights.
    """
    agent_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    optimizer_state = clone_optimizer_state(optimizer_critic.state_dict())
    arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}

    model_dir = os.path.join(save_dir, "models")

    t = threading.Thread(
        target=write_checkpoint,
        args=(save_dir, model_dir, global_step, agent_state, optimizer_state, arrays),
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
    # NOTE: same pre-existing gap as PPO/TD3/SAC — train() needs (cfg, env, path),
    # main() only has cfg. Needs env construction + path wired in from
    # wherever your other entry points build these.
    raise NotImplementedError(
        "main() must construct `env` and `path` before calling train(cfg, env, path)."
    )


if __name__ == "__main__":
    main()