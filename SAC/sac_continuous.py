# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import copy
import os
import random
import threading
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from RLfoundry.Utils.Buffer.Buffer import ReplayBuffer
from SACNetwork import SoftQNetwork, Actor

from RLfoundry.Utils.run_startup import run_startup


# ==============================================================================
# Main training loop
# ==============================================================================

def train(cfg, env, path):
    # ------------------------------------------------------------------
    # Config + run naming
    # ------------------------------------------------------------------

    # run startup
    path = run_startup(path, cfg)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=os.path.basename(path),
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "sac"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # ------------------------------------------------------------------
    # Networks
    # Actor + twin Q-networks + target networks (SAC uses two Q-networks
    # to reduce overestimation bias; targets are updated via Polyak averaging)
    # ------------------------------------------------------------------
    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Separate optimizers: one for both Q-networks, one for the actor
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.policy_lr)

    # ------------------------------------------------------------------
    # Automatic entropy tuning
    # Target entropy = -|A| (heuristic from Haarnoja et al. 2018)
    # log_alpha is learned; alpha = exp(log_alpha) stays positive
    # ------------------------------------------------------------------
    if cfg.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(env.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=cfg.q_lr)
    else:
        log_alpha = None
        a_optimizer = None
        alpha = cfg.alpha

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    env.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        cfg.buffer_size,
        env.single_observation_space,
        env.single_action_space,
        device,
        n_envs=cfg.num_envs,
        handle_timeout_termination=False,
    )

    # ------------------------------------------------------------------
    # Checkpoint threading state
    # ------------------------------------------------------------------

    # Metric history lists (saved as numpy arrays at each checkpoint)
    qf1_loss_history = []
    actor_loss_history = []
    episodic_return_history = []

    start_time = time.time()

    # ==================================================================
    # Main loop
    # ==================================================================
    obs, _ = env.reset(seed=cfg.seed)

    for global_step in range(cfg.total_timesteps):

        # --------------------------------------------------------------
        # Action selection
        # Before learning starts: random exploration
        # After: sample from the stochastic actor (reparameterisation trick)
        # --------------------------------------------------------------
        if global_step < cfg.learning_starts:
            actions = np.array([env.single_action_space.sample()
                                for _ in range(env.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # --------------------------------------------------------------
        # Step environment
        # --------------------------------------------------------------
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Log episodic returns when an episode finishes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_return = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    print(f"global_step={global_step}, episodic_return={ep_return}")

                    episodic_return_history.append(ep_return)

                    if cfg.track:
                        wandb.log({
                            "charts/episodic_return": ep_return,
                            "charts/episodic_length": ep_length,
                        }, step=global_step)
                    break

        # Handle truncation: use `final_observation` so the bootstrap target
        # is not poisoned by the reset observation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # ==============================================================
        # Training updates (only after replay buffer has enough data)
        # ==============================================================
        if global_step > cfg.learning_starts:

            data = rb.sample(cfg.batch_size)

            # ----------------------------------------------------------
            # Critic update
            # Compute SAC Bellman target:
            #   y = r + γ(1-d) * [min_Q(s', ã') - α * log π(ã'|s')]
            # where ã' ~ π(·|s')
            # ----------------------------------------------------------
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                # Take the pessimistic (min) estimate to reduce overestimation
                min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                        (1 - data.dones.flatten()) * cfg.gamma * min_qf_next_target.view(-1)
                )

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # ----------------------------------------------------------
            # Actor + alpha update (TD3-style delayed policy update)
            # Actor maximises E[min_Q(s, ã) - α * log π(ã|s)]
            # Alpha minimises E[-log_α * (log π + H̄)]  where H̄ = target_entropy
            # ----------------------------------------------------------
            if global_step % cfg.policy_frequency == 0:
                # Repeat actor update `policy_frequency` times to compensate
                # for the delay (keeps actor/critic update ratio ~1:1)
                for _ in range(cfg.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Entropy-regularised actor loss
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if cfg.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                                -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                qf1_loss_history.append(qf1_loss.item())
                actor_loss_history.append(actor_loss.item())

                if cfg.track:
                    log_dict = {}

                    # --- 1. REWARDS ---
                    log_dict["rewards/mean"] = data.rewards.mean().item()

                    # --- 2. LOSSES & OPTIMIZATION ---
                    y_pred = qf1_a_values.detach().cpu().numpy()
                    y_true = next_q_value.detach().cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    log_dict.update({
                        "losses/actor_loss": actor_loss.item(),
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/qf2_loss": qf2_loss.item(),
                        "losses/qf_loss": qf_loss.item(),
                        "losses/alpha": alpha,
                        "losses/entropy": (-log_pi).mean().item(),
                        "losses/explained_variance": explained_var,
                    })
                    if cfg.autotune:
                        log_dict["losses/alpha_loss"] = alpha_loss.item()

                    # --- 3. ACTIONS: mean (aggregate + per-dim) ---
                    actions_np = pi.detach().cpu().numpy()
                    actions_flat = actions_np.reshape(actions_np.shape[0], -1)
                    action_dim_means = actions_flat.mean(axis=0)
                    action_std_per_dim = actions_flat.std(axis=0)

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
                                (actions_np <= env.single_action_space.low + 1e-3)
                                | (actions_np >= env.single_action_space.high - 1e-3)
                            )
                        ),
                    })
                    for dim_idx, std_val in enumerate(action_std_per_dim):
                        log_dict[f"actions_std/dim{dim_idx}"] = float(std_val)

                    # --- 5. OBSERVATIONS: mean (aggregate + per-dim) ---
                    obs_np = data.observations.detach().cpu().numpy()
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
                        "system/SPS": int(global_step / (time.time() - start_time)),
                        "system/obs_has_nan_or_inf": int(obs_has_nan_or_inf),
                    })

                    # --- 8. CHARTS ---
                    log_dict.update({
                        "charts/learning_rate_actor": actor_optimizer.param_groups[0]["lr"],
                        "charts/learning_rate_q": q_optimizer.param_groups[0]["lr"],
                    })

                    wandb.log(log_dict, step=global_step)

            # ----------------------------------------------------------
            # Target network Polyak update
            # θ_target ← τ * θ + (1 - τ) * θ_target
            # ----------------------------------------------------------
            if global_step % cfg.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        cfg.tau * param.data + (1 - cfg.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        cfg.tau * param.data + (1 - cfg.tau) * target_param.data
                    )

            # ----------------------------------------------------------
            # Async checkpoint (threaded, deep-copy before dispatch)
            # ----------------------------------------------------------
            now = time.time()
            if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
                save_checkpoint(
                    path,
                    global_step=cfg.total_timesteps,
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    qf1_target=qf1_target,
                    qf2_target=qf2_target,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    log_alpha=log_alpha,
                    a_optimizer=a_optimizer,
                )

                last_numpy_save_t = now

            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps)

            # ------------------------------------------------------------------
            # Final checkpoint — wait for any background thread to finish first
            # ------------------------------------------------------------------

            path = path + "Final"
            save_checkpoint(
                path,
                global_step=cfg.total_timesteps,
                actor=actor,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                log_alpha=log_alpha,
                a_optimizer=a_optimizer,
            )

        env.close()

    wandb.finish()


# ==============================================================================
# Entry point
# ==============================================================================

def save_checkpoint(
        save_dir: str,
        global_step: int,
        actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        actor_optimizer,
        q_optimizer,
        log_alpha=None,
        a_optimizer=None,
        **arrays: np.ndarray,
) -> None:
    # ------------------------------------------------------------------
    # Numpy arrays (loss histories, episodic returns, etc.)
    # ------------------------------------------------------------------
    if arrays:
        step_dir = os.path.join(save_dir, "numpy", f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        for name, arr in arrays.items():
            np.save(os.path.join(step_dir, f"{name}.npy"), arr)
        print(f"[numpy checkpoint] saved {list(arrays.keys())} -> {step_dir}")

    # ------------------------------------------------------------------
    # Model checkpoint
    # ------------------------------------------------------------------
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"sac_step_{global_step}.pt")

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "log_alpha": log_alpha.item() if log_alpha is not None else None,
        "alpha_optimizer_state_dict": a_optimizer.state_dict() if a_optimizer is not None else None,
        "global_step": global_step,
    }

    torch.save(checkpoint, model_path)
    print(f"[model checkpoint] saved -> {model_path}")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()