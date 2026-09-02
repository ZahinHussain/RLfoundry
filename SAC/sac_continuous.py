# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import psutil
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from RLfoundry.Utils.Buffer.Buffer import ReplayBuffer

from RLfoundry.SAC.SACNetwork import Actor, SoftQNetwork




from RLfoundry.Utils.transfer_sac import (
    load_pretrained_sac, apply_freeze_sac, evaluate_sac,
    trainable_params_multi, count_params,
)

from RLfoundry.Utils.run_startup import run_startup


# ==============================================================================
# Module-level state for CPU/RAM monitoring
# ==============================================================================

_proc = psutil.Process()
_proc.cpu_percent()  # warm-up call — first real reading needs a prior baseline


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
        name=cfg.algo_name,
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "ppo"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    if torch.cuda.is_available() and cfg.cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # Cude and device setup
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)

    skipped, ckpt_log_alpha = [], None
    if cfg.transfer_mode != "scratch":
        assert cfg.checkpoint_path, "transfer_mode != scratch requires checkpoint_path"
        _, skipped, ckpt_log_alpha = load_pretrained_sac(
            actor, qf1, qf2, qf1_target, qf2_target,
            cfg.checkpoint_path, device,
            load_critic=cfg.load_critic,
            load_alpha=cfg.get("load_alpha", False),
        )
    else:
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

    if cfg.transfer_mode == "freeze":
        apply_freeze_sac(actor, qf1, qf2, freeze_critic=cfg.load_critic)

    if cfg.track:
        trainable, total = count_params(actor, qf1, qf2)
        wandb.summary["transfer/mode"] = cfg.transfer_mode
        wandb.summary["transfer/skipped_keys"] = skipped
        wandb.summary["transfer/trainable_params"] = trainable
        wandb.summary["transfer/total_params"] = total

    if cfg.transfer_mode == "zeroshot":
        all_rets, all_lens = [], []
        for it in range(10):
            rets, lens = evaluate_sac(actor, env, cfg, device, cfg.eval_episodes)
            all_rets.extend(rets)
            all_lens.extend(lens)

            mean_ret = float(np.mean(rets))
            global_step = (it + 1) * cfg.eval_episodes
            print(f"[zeroshot] iter={it + 1}/10 return={mean_ret:.2f}", flush=True)

            if cfg.track:
                wandb.log({
                    "charts/episodic_return": mean_ret,
                    "charts/episodic_length": float(np.mean(lens)),
                }, step=global_step)

        print(f"[zeroshot] overall return {np.mean(all_rets):.2f} "
              f"+/- {np.std(all_rets):.2f} over {len(all_rets)} episodes")

        if cfg.track:
            wandb.finish()
        env.close()
        return


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
    # Metric history lists (saved as numpy arrays at each checkpoint)
    # ------------------------------------------------------------------
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
        # NOTE: no `break` here — training must continue past episode boundaries.
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

        # ----------------------------------------------------------------
        # Single (non-vectorized) env: `next_obs` from env.step() IS already
        # the true final observation on truncation — Gymnasium only injects
        # "final_observation" for vector envs with autoreset. No lookup needed.
        # ----------------------------------------------------------------
        real_next_obs = next_obs.copy()

        rb.add(
            obs[np.newaxis],  # (obs_dim,) -> (1, obs_dim)
            real_next_obs[np.newaxis],  # same
            actions[np.newaxis],  # (act_dim,) -> (1, act_dim)
            np.array([rewards]),
            np.array([terminations]),
            [infos],  # dict -> [dict]
        )

        # Reset on episode end, otherwise carry the obs forward
        if terminations or truncations:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # ==============================================================
        # Training updates (only after replay buffer has enough data)
        # ==============================================================
        if global_step > cfg.learning_starts:

            data = rb.sample(cfg.batch_size)

            # ----------------------------------------------------------
            # Critic update
            # Compute SAC Bellman target:
            #   y = r + gamma*(1-d) * [min_Q(s', a') - alpha * log pi(a'|s')]
            # where a' ~ pi(.|s')
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
            # Actor maximises E[min_Q(s, a) - alpha * log pi(a|s)]
            # Alpha minimises E[-log_alpha * (log pi + H_target)]
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

                    # BUG FIX: alpha_loss must exist even when cfg.autotune is
                    # False, since log_sac_step is always called below and
                    # always expects an alpha_loss argument (it only calls
                    # .item() on it when autotune=True, but the Python name
                    # itself has to exist first or you get a NameError before
                    # the function is even entered).
                    alpha_loss = None

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
                    # BUG FIX (was two bugs):
                    #   1. `autotune` doesn't exist as a bare name in train() —
                    #      only cfg.autotune does. Pass cfg.autotune instead.
                    #   2. `device` was missing entirely — log_sac_step needs
                    #      it for the GPU block and the call would raise
                    #      TypeError (missing positional arg) without it.
                    log_sac_step(
                        global_step, env, data, pi,
                        qf1_a_values, next_q_value,
                        actor_loss, qf1_loss, qf2_loss, qf_loss,
                        alpha, log_pi, cfg.autotune, alpha_loss,
                        actor_optimizer, q_optimizer, device,
                    )

            # ----------------------------------------------------------
            # Target network Polyak update
            # theta_target <- tau * theta + (1 - tau) * theta_target
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
            # Async checkpoint (periodic, based on wall-clock interval)
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

    # ======================================================================
    # Final checkpoint — runs ONCE, after the loop finishes
    # ======================================================================
    save_checkpoint(
        os.path.join(path, "Final"),
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
    wandb.finish()


# ==============================================================================
# Checkpoint saving
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


# ==============================================================================
# Logging subroutine
# ==============================================================================

def log_sac_step(
    global_step, env, data, pi,
    qf1_a_values, next_q_value,
    actor_loss, qf1_loss, qf2_loss, qf_loss,
    alpha, log_pi, autotune, alpha_loss,
    actor_optimizer, q_optimizer, device
):
    log_dict = {}

    log_dict["rewards/mean"] = data.rewards.mean().item()

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
    # BUG FIX: alpha_loss is None when cfg.autotune is False (see train()) —
    # guard on autotune AND alpha_loss not being None before calling .item(),
    # otherwise this throws AttributeError: 'NoneType' object has no attribute 'item'
    if autotune and alpha_loss is not None:
        log_dict["losses/alpha_loss"] = alpha_loss.item()

    actions_np = pi.detach().cpu().numpy()
    actions_flat = actions_np.reshape(actions_np.shape[0], -1)
    action_dim_means = actions_flat.mean(axis=0)
    action_std_per_dim = actions_flat.std(axis=0)

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

    obs_np = data.observations.detach().cpu().numpy()
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
        "charts/learning_rate_actor": actor_optimizer.param_groups[0]["lr"],
        "charts/learning_rate_q": q_optimizer.param_groups[0]["lr"],
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


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    # NOTE: same pre-existing gap as TD3 — train() needs (cfg, env, path),
    # main() only has cfg. Needs env construction + path wired in from
    # wherever your other entry points build these.
    raise NotImplementedError(
        "main() must construct `env` and `path` before calling train(cfg, env, path)."
    )


if __name__ == "__main__":
    main()