import copy
import random
import time
import os
import threading

import hydra
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from gymnasium.wrappers import FlattenObservation
from RLfoundry.RPO.Agent import Agent
from RLfoundry.Utils.run_startup import run_startup

from RLfoundry.Utils.checkpoint import create_folder_checkpoint, create_timestamp_folder


def train(cfg, env, path):
    # run startup
    path = run_startup(path, cfg)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=os.path.basename(path),
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "rpo"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ==================================================
    # PPO TRAINING CODE GOES HERE
    # ==================================================

    # Cude and device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    agent = Agent(env, cfg.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=cfg.eps)

    # ---- pre-allocated storage for one rollout's worth of data ----
    obs = torch.zeros((cfg.num_steps,) + env.observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)

    # Buffer to store the CLIPPED actions for accurate Wandb logging
    clipped_actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)

    logprobs = torch.zeros(cfg.num_steps).to(device)
    rewards = torch.zeros(cfg.num_steps).to(device)
    dones = torch.zeros(cfg.num_steps).to(device)
    values = torch.zeros(cfg.num_steps).to(device)

    global_step = 0
    next_obs_np, _ = env.reset(seed=cfg.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.zeros(1).to(device)
    start_time = time.time()

    current_ep_return = 0.0
    current_ep_length = 0

    # ========================================================================
    # [LINE 2] OUTER LOOP ? one iteration = one full "collect, then update" cycle
    # ========================================================================
    for iteration in range(1, cfg.num_iterations + 1):

        if cfg.anneal_lr:
            # [EXTRA] linearly shrink the learning rate as training progresses
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ====================================================================
        # [LINE 3] "Collect set of trajectories D_k by running policy pi_k"
        # ====================================================================

        for step in range(0, cfg.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # no_grad: we are NOT training right now, just acting in the
            # environment and recording what happens. Gradients only get
            # computed later, during the update phase.
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            # Store raw action for PPO importance sampling ratio
            actions[step] = action
            logprobs[step] = logprob

            # step the actual environment forward
            action_np = action.detach().cpu().numpy().reshape(-1)  # always gives 1D (act_dim,)
            action_np_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)

            # Store clipped action for Wandb logging
            clipped_actions[step] = torch.as_tensor(action_np_clipped, dtype=torch.float32, device=device)

            next_obs_np, reward, terminations, truncations, info = env.step(action_np_clipped)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = float(reward)

            current_ep_return += float(reward)
            current_ep_length += 1

            # "do the action until it all gone" -> reset when episode ends and keep going
            if next_done:
                if getattr(cfg, "debug_episode_prints", False):
                    print(
                        f"[episode end] global_step={global_step} return={current_ep_return:.2f} length={current_ep_length}")

                # Reset environment to continue taking actions in the next episode
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
        # [LINE 4 + 5] "Compute rewards-to-go" + "Compute advantage estimates"
        #
        # NOTE: this implementation computes them TOGETHER via GAE, which is
        # a deviation from the pseudocode worth naming explicitly in your
        # dissertation. Spinning Up's pseudocode treats rewards-to-go as a
        # Monte Carlo target computed independently of advantages. CleanRL
        # (like most modern PPO code) instead computes GAE advantages first,
        # THEN derives the value-function target as returns = advantages + values.
        # This is lower-variance and is the de facto standard, but it's
        # technically GAE-PPO, not literal vanilla PPO-Clip.
        # ====================================================================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            # walk BACKWARD through the rollout ? this is the standard GAE
            # recursive computation, equivalent to (but far cheaper than)
            # summing an infinite series of discounted TD errors
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # TD error / "surprise" at this timestep:
                # actual reward + discounted next value, minus what we predicted
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                # GAE: exponentially-weighted sum of TD errors going forward,
                # computed efficiently backward
                advantages[t] = lastgaelam = (
                        delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )

            # [LINE 4] the value function's regression TARGET ? this stands
            # in for "rewards-to-go" in the pseudocode
            returns = advantages + values

        obs_space = env.observation_space
        act_space = env.action_space

        # ---- pre-allocated storage for one rollout's worth of data ----
        # [LINE 3] this is where D_k actually lives, as plain tensors rather
        # than a list of trajectory objects ? much faster on GPU

        # flatten (num_steps, num_env, ...) -> (num_steps * num_env, ...)
        # so all parallel environments' data gets mixed together into one
        # big pool for minibatching
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ====================================================================
        # [LINE 6 + 7] "Update the policy by maximizing PPO-Clip objective"
        #              "Fit value function by regression on MSE"
        #
        # These happen TOGETHER, in the same loop, on the same minibatches,
        # combined into one scalar loss per minibatch. The pseudocode lists
        # them as two separate steps, but in practice they share one
        # optimizer and one backward pass for efficiency.
        # ====================================================================
        b_inds = np.arange(cfg.batch_size)
        for epoch in range(cfg.update_epochs):
            # reshuffle which data goes in which minibatch every epoch ?
            # standard SGD practice, prevents the network from learning
            # spurious orderings
            np.random.shuffle(b_inds)

            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # re-run the CURRENT (possibly already-updated-this-epoch)
                # policy on the OLD states/actions, to get a fresh log-prob
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # ---- THE CORE PPO RATIO ----
                # log(pi_new) - log(pi_old), exponentiated -> pi_new/pi_old
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    # FIX: Added correction=0 to prevent NaN when minibatch size is 1
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(correction=0) + 1e-8)
                # ---- [LINE 6] THE PPO-CLIP OBJECTIVE ITSELF ----
                # Spinning Up's formula:
                #   L = min( ratio * A_hat,  clip(ratio, 1-eps, 1+eps) * A_hat )
                # We want to MAXIMIZE this, but PyTorch optimizers MINIMIZE,
                # so everything below is negated (hence -mb_advantages and
                # torch.max instead of torch.min ? negating min(a,b) gives
                # max(-a,-b))
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ---- [LINE 8] VALUE FUNCTION MSE REGRESSION ----
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    # [EXTRA] clipped value loss ? not in pseudocode, but
                    # mirrors the policy clip: don't let the value estimate
                    # move too far from its old value in one update
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    # plain MSE ? this is the literal pseudocode line 8
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # [EXTRA] entropy bonus ? encourages exploration, not in
                # pseudocode at all
                entropy_loss = entropy.mean()

                # ---- COMBINE EVERYTHING INTO ONE LOSS ----
                # policy loss MINUS entropy bonus (we want to ENCOURAGE
                # entropy, hence the minus sign on a term we're minimizing)
                # PLUS value loss weighted by vf_coef
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # [EXTRA] clip gradients so no single update is too violent
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            # [EXTRA] optional early stopping if policy is changing too fast

            ''''

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break
                '''

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)

        # ====================================================================
        # LOGGING TO WANDB (Organized by category for legibility)
        # ====================================================================

        log_dict = {}

        # --- 1. REWARDS --S    -
        log_dict["rewards/mean"] = rewards.mean().item()

        # --- 2. LOSSES & OPTIMIZATION ---
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_dict.update({
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/explained_variance": explained_var,
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

        # --- 7. SYSTEM (throughput + health diagnostics) ---
        obs_has_nan_or_inf = bool(np.isnan(obs_np).any() or np.isinf(obs_np).any())
        log_dict.update({
            "system/SPS": sps,
            "system/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
            "system/obs_has_nan_or_inf": int(obs_has_nan_or_inf),
        })

        # --- 8. CHARTS ---
        log_dict.update({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
        })

        wandb.log(log_dict, step=global_step)

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
        model_path = os.path.join(model_dir, f"rpo_step_{global_step}.pt")
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

    Everything is cloned/copied BEFORE the thread starts ? the main thread
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
    train(cfg)


if __name__ == "__main__":
    main()