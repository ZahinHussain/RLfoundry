# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import wandb

import torch.optim as optim
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf

from RLfoundry.TD3.TD3Network import Actor, QNetwork
from RLfoundry.Utils.Buffer.Buffer import ReplayBuffer

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
        job_type=getattr(cfg, "algo_name", "td3"),
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

    if torch.cuda.is_available() and cfg.cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Networks
    # Actor + twin Q-networks + target networks (TD3 uses two Q-networks
    # to reduce overestimation bias; targets are updated via Polyak averaging.
    # TD3 also maintains a target actor for Target Policy Smoothing)
    # ------------------------------------------------------------------
    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf2 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    qf2_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Separate optimizers: one for both Q-networks, one for the actor
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.learning_rate)

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        cfg.buffer_size,
        env.observation_space,
        env.action_space,
        device,
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
        # After: deterministic actor + Gaussian exploration noise
        # --------------------------------------------------------------
        if global_step < cfg.learning_starts:
            actions = env.action_space.sample()

        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).unsqueeze(0).to(device))
                actions += torch.normal(0, actor.action_scale * cfg.exploration_noise)
                actions = actions.cpu().numpy().squeeze(0).clip(
                    env.action_space.low, env.action_space.high
                )

        # --------------------------------------------------------------
        # Step environment
        # --------------------------------------------------------------
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Log episodic returns when an episode finishes
        if "episode" in infos:
            ep_return = infos["episode"]["r"]
            ep_length = infos["episode"]["l"]
            print(f"global_step={global_step}, episodic_return={ep_return}")
            episodic_return_history.append(ep_return)

            if cfg.track:
                wandb.log({
                    "charts/episodic_return": ep_return,
                    "charts/episodic_length": ep_length,
                }, step=global_step)

        # Handle truncation: use `final_observation` so the bootstrap target
        # is not poisoned by the reset observation
        done = terminations or truncations

        real_next_obs = next_obs.copy()

        rb.add(
            obs[np.newaxis],
            real_next_obs[np.newaxis],
            actions[np.newaxis],
            np.array([rewards]),
            np.array([done]),
            [infos],
        )

        if done:
            next_obs, _ = env.reset()

        obs = next_obs
        # ==============================================================
        # Training updates (only after replay buffer has enough data)
        # ==============================================================
        if global_step > cfg.learning_starts:

            data = rb.sample(cfg.batch_size)

            # ----------------------------------------------------------
            # Critic update
            # Compute TD3 Bellman target with Target Policy Smoothing:
            #   ã' = clip(π_target(s') + clip(ε, -c, c), a_low, a_high)
            #   y  = r + γ(1-d) * min_Q_target(s', ã')
            # Clipped noise prevents the policy from exploiting Q-function
            # errors at specific actions
            # ----------------------------------------------------------
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * cfg.policy_noise
                ).clamp(-cfg.noise_clip, cfg.noise_clip) * target_actor.action_scale

                action_low = torch.tensor(env.action_space.low, device=device)
                action_high = torch.tensor(env.action_space.high, device=device)
                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(action_low, action_high)

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
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
            # Actor + target network update (TD3-style delayed policy update)
            # Actor maximises E[Q1(s, π(s))]
            # Delayed to keep actor/critic update ratio stable
            # ----------------------------------------------------------
            if global_step % cfg.policy_frequency == 0:
                actor_actions = actor(data.observations)
                actor_loss = -qf1(data.observations, actor_actions).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Polyak update for actor and both critic targets
                # θ_target ← τ * θ + (1 - τ) * θ_target
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(
                        cfg.tau * param.data + (1 - cfg.tau) * target_param.data
                    )
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        cfg.tau * param.data + (1 - cfg.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        cfg.tau * param.data + (1 - cfg.tau) * target_param.data
                    )

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
                        "losses/explained_variance": explained_var,
                    })

                    # --- 3. ACTIONS: mean (aggregate + per-dim) ---
                    # TD3's actor is deterministic, so there's no learned std
                    # parameter like PPO's actor_logstd; per-dim std here is
                    # the empirical spread of the actor's outputs across the batch.
                    actions_np = actor_actions.detach().cpu().numpy()
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
                                (actions_np <= env.action_space.low + 1e-3)
                                | (actions_np >= env.action_space.high - 1e-3)
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
            # Async checkpoint (threaded, deep-copy before dispatch)
            # ----------------------------------------------------------
            now = time.time()
            if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
                save_checkpoint(
                    path,
                    global_step=global_step,
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    qf1_target=qf1_target,
                    qf2_target=qf2_target,
                    target_actor=target_actor,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    qf1_loss_history=np.array(qf1_loss_history),
                    actor_loss_history=np.array(actor_loss_history),
                    episodic_return_history=np.array(episodic_return_history),
                )

                last_numpy_save_t = now

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)

    # ------------------------------------------------------------------
    # Final checkpoint — wait for any background thread to finish first
    # ------------------------------------------------------------------
    save_checkpoint(
        path,
        global_step=cfg.total_timesteps,
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        target_actor=target_actor,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        qf1_loss_history=np.array(qf1_loss_history),
        actor_loss_history=np.array(actor_loss_history),
        episodic_return_history=np.array(episodic_return_history),
    )

    wandb.finish()

    env.close()


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
        target_actor,
        actor_optimizer,
        q_optimizer,
        **arrays: np.ndarray,
) -> None:
    # ------------------------------------------------------------------
    # Numpy arrays (loss histories, episodic returns, etc.)
    # ------------------------------------------------------------------
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

    model_path = os.path.join(model_dir, f"td3_step_{global_step}.pt")

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "target_actor_state_dict": target_actor.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
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