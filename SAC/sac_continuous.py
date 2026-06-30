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

import torch.optim as optim
import torch.nn.functional as F
import wandb

from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


from RLfoundry.SAC.SACNetwork import Actor, SoftQNetwork
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
            torch.Tensor(env.action_space.shape).to(device)
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
        # After: sample from the stochastic actor (reparameterisation trick)
        # --------------------------------------------------------------
        if global_step < cfg.learning_starts:
            actions = env.action_space.sample()

        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            actions = actions.detach().cpu().numpy().squeeze(0)

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
                    break

        # Handle truncation: use `final_observation` so the bootstrap target
        # is not poisoned by the reset observation

        real_next_obs = next_obs.copy()


        rb.add(
            obs[np.newaxis],  # (obs_dim,) -> (1, obs_dim)
            real_next_obs[np.newaxis],  # same
            actions[np.newaxis],  # (act_dim,) -> (1, act_dim)
            np.array([rewards]),
            np.array([terminations]),
            [infos],  # dict -> [dict]
        )
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
                    global_step=global_step,
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    qf1_target=qf1_target,
                    qf2_target=qf2_target,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    log_alpha=log_alpha,
                    a_optimizer=a_optimizer,
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
        path + "Final",
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
        qf1_loss_history=np.array(qf1_loss_history),
        actor_loss_history=np.array(actor_loss_history),
        episodic_return_history=np.array(episodic_return_history),
    )

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
        actor_optimizer,
        q_optimizer,
        log_alpha=None,
        a_optimizer=None,
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
