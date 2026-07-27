import copy
import random
import time
import os
import threading

import hydra
import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from RLfoundry.PPO.AgentDiscrete import Agent  # Ensure this is updated to Categorical!
from RLfoundry.Utils.run_startup import run_startup
from RLfoundry.Utils.checkpoint import create_folder_checkpoint, create_timestamp_folder

_proc = psutil.Process()
_proc.cpu_percent()  # w
def train(cfg, env, path):
    path = run_startup(path, cfg)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=os.path.basename(path),
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "ppo"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=cfg.eps)

    # ---- pre-allocated storage for one rollout's worth of data ----
    obs = torch.zeros((cfg.num_steps,) + env.observation_space.shape).to(device)

    # DISCRETE CHANGE: Actions are scalar integers, not continuous vectors
    actions = torch.zeros(cfg.num_steps, dtype=torch.long).to(device)

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

    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(0, cfg.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # DISCRETE CHANGE: No clipping needed. Just extract the integer action.
            next_obs_np, reward, terminations, truncations, info = env.step(action)
            done = np.logical_or(terminations, truncations)

            if done:
                next_obs_np, info = env.reset()
            current_ep_return += float(reward)
            current_ep_length += 1
            if done:
                if getattr(cfg, "debug_episode_prints", False):
                    print(
                        f"[episode end] global_step={global_step} return={current_ep_return:.2f} length={current_ep_length}")
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
            next_done = torch.as_tensor(float(done), dtype=torch.float32, device=device)

        # GAE Calculation (Remains exactly the same)
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

        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)  # Flattened 1D tensor of integers
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO-Clip update
        b_inds = np.arange(cfg.batch_size)
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # DISCRETE CHANGE: Cast actions to long for the Categorical distribution
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(correction=0) + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)

        # ====================================================================
        # LOGGING TO WANDB
        # BUG FIX: this block previously ran unconditionally with no
        # `cfg.track` guard, unlike every other algorithm file. Wrapped to
        # match convention — if cfg.track=False, wandb.log should never be
        # called at all.
        # ====================================================================
        if cfg.track:
            log_ppo_step(
                global_step, env, rewards,
                b_values, b_returns, pg_loss, v_loss, entropy_loss,
                old_approx_kl, approx_kl, b_actions, obs,
                optimizer, device, sps, grad_norm,
            )

        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        now = time.time()
        if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
            save_checkpoint(
                path,
                global_step,
                agent,
                optimizer,
            )
            last_numpy_save_t = now

        # Final save
    save_checkpoint(
        path,
        global_step,
        agent,
        optimizer,
    )

    wandb.finish()

    env.close()


# ==============================================================================
# Logging subroutine
# ==============================================================================

def log_ppo_step(
    global_step,
    env,
    rewards,
    b_values,
    b_returns,
    pg_loss,
    v_loss,
    entropy_loss,
    old_approx_kl,
    approx_kl,
    actions,
    obs,
    optimizer,
    device,
    sps,
    grad_norm,
):
    log_dict = {}

    # ------------------------------------------------------------------
    # 1. REWARDS
    # ------------------------------------------------------------------
    log_dict["rewards/mean"] = rewards.mean().item()

    # ------------------------------------------------------------------
    # 2. LOSSES
    # ------------------------------------------------------------------
    y_pred = b_values.detach().cpu().numpy()
    y_true = b_returns.detach().cpu().numpy()
    var_y = np.var(y_true)
    explained_var = (
        float("nan")
        if var_y == 0
        else 1 - np.var(y_true - y_pred) / var_y
    )

    log_dict.update({
        "losses/policy_loss": pg_loss.item(),
        "losses/value_loss": v_loss.item(),
        "losses/entropy": entropy_loss.item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/explained_variance": explained_var,
    })

    # ------------------------------------------------------------------
    # 3. ACTIONS (Discrete)
    # ------------------------------------------------------------------
    actions_np = actions.detach().cpu().numpy().flatten().astype(np.int64)
    log_dict["actions_mean/aggregate"] = float(actions_np.mean())
    log_dict["actions_std/aggregate"] = float(actions_np.std())

    num_actions = env.action_space.n
    action_counts = np.bincount(actions_np, minlength=num_actions)
    action_probs = action_counts / max(action_counts.sum(), 1)

    for a_idx in range(num_actions):
        log_dict[f"actions_mean/action_{a_idx}_frac"] = float(action_probs[a_idx])

    # ------------------------------------------------------------------
    # 4. OBSERVATIONS
    # ------------------------------------------------------------------
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

    obs_has_nan_or_inf = bool(
        np.isnan(obs_np).any() or np.isinf(obs_np).any()
    )

    # ------------------------------------------------------------------
    # 5. TRAINING
    # ------------------------------------------------------------------
    log_dict.update({
        "system/SPS": sps,
        "system/grad_norm": (
            grad_norm.item()
            if torch.is_tensor(grad_norm)
            else float(grad_norm)
        ),
        "system/obs_has_nan_or_inf": int(obs_has_nan_or_inf),
        "charts/learning_rate": optimizer.param_groups[0]["lr"],
    })

    # ------------------------------------------------------------------
    # 6. CPU
    # ------------------------------------------------------------------
    per_core = psutil.cpu_percent(percpu=True)

    log_dict.update({
        "cpu/proc_cpu_percent": _proc.cpu_percent(),
        "cpu/proc_num_threads": _proc.num_threads(),
        "cpu/sys_cpu_percent": psutil.cpu_percent(),
        "cpu/core_max_percent": max(per_core),
        "cpu/core_mean_percent": sum(per_core) / len(per_core),
        "cpu/cores_active_count": sum(c > 20.0 for c in per_core),
        "cpu/num_cores": len(per_core),
    })

    for core_idx, core_pct in enumerate(per_core):
        log_dict[f"cpu_percore/core{core_idx}"] = core_pct

    # ------------------------------------------------------------------
    # 7. RAM
    # ------------------------------------------------------------------
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
        pass

    # ------------------------------------------------------------------
    # 8. GPU
    # ------------------------------------------------------------------
    if device.type == "cuda":
        log_dict.update({
            "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
            "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        })

        try:
            log_dict["gpu/utilization_percent"] = torch.cuda.utilization(device)
        except Exception:
            pass

    wandb.log(log_dict, step=global_step)


def clone_optimizer_state(optimizer_state: dict) -> dict:
    """Deep-clone all tensors inside an optimizer state_dict so the
    background writer thread doesn't race with optimizer.step()."""
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

        # --- rollout arrays (atomic write) ---

        # --- model + optimizer (atomic write) ---
        model_path = os.path.join(model_dir, f"ppo_step_{global_step}.pt")
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
        optimizer,
        block: bool = False,
        **arrays: np.ndarray,
) -> None:
    """
    Save rollout arrays and model checkpoint asynchronously.

    Everything is cloned/copied BEFORE the thread starts — the main thread
    keeps training (mutating agent/optimizer/rollout buffers in place)
    while the background thread serializes, so anything not copied here
    is a race condition. Writes are atomic (tmp file + os.replace) so a
    crash mid-write can't leave a corrupted checkpoint on disk.
    """
    agent_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    optimizer_state = clone_optimizer_state(optimizer.state_dict())
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
    # NOTE: same pre-existing gap as TD3/SAC — train() needs (cfg, env, path),
    # main() only has cfg. Needs env construction + path wired in from
    # wherever your other entry points build these.
    raise NotImplementedError(
        "main() must construct `env` and `path` before calling train(cfg, env, path)."
    )


if __name__ == "__main__":
    main()