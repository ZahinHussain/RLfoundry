# Discrete-action SAC. Adapted from the continuous training file.
# Reference: Christodoulou (2019), "Soft Actor-Critic for Discrete Action
# Settings" (arXiv:1910.07207) and CleanRL's sac_atari.py.
#
# LOGGING NOTE: the wandb schema below is deliberately key-compatible with
# the PPO file so runs can be overlaid on the same panels. Shared keys:
#   charts/episodic_return, charts/episodic_length, charts/learning_rate
#   rewards/mean
#   losses/policy_loss, losses/value_loss, losses/entropy,
#   losses/explained_variance
#   actions_mean/aggregate, actions_std/aggregate,
#   actions_mean/action_{i}_frac
#   observations_mean/*, observations_std/*
#   system/SPS, system/grad_norm, system/obs_has_nan_or_inf
#   cpu/*, cpu_percore/*, ram/*, gpu/*
# SAC-only metrics keep their own namespaces (losses/qf*_loss, losses/alpha*,
# actions_probs/*) so they never collide with a PPO panel.
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
from RLfoundry.SAC.SACDiscreteNetwork import Actor, SoftQNetwork

from RLfoundry.Utils.run_startup import run_startup


# ==============================================================================
# Module-level state for CPU/RAM monitoring
# ==============================================================================

_proc = psutil.Process()
_proc.cpu_percent()  # warm-up call - first real reading needs a prior baseline


# ==============================================================================
# Main training loop
# ==============================================================================

def train(cfg, env, path):
    # ------------------------------------------------------------------
    # Config + run naming
    # ------------------------------------------------------------------

    # run startup
    path = run_startup(path, cfg)

    # Fail loudly rather than producing silent garbage if this file is ever
    # pointed at a Box env - the losses below assume a finite action set.
    assert isinstance(env.action_space, gym.spaces.Discrete), (
        "sac_discrete only supports gym.spaces.Discrete action spaces; "
        f"got {type(env.action_space).__name__}. Use the continuous SAC file."
    )
    n_actions = int(env.action_space.n)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.algo_name,
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "sac_discrete"),
        mode="offline" if cfg.get("wandb_offline", False) else "online",
    )

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    # PPO logs once per iteration (every cfg.num_steps env steps); this file
    # would otherwise log on every update, which is far denser. 0 keeps the
    # old behaviour (log on every policy update).
    log_interval = int(cfg.get("log_interval", 0))

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
    # Actor + twin Q-networks + target networks. Same structure as the
    # continuous case, but each Q-network now maps S -> R^|A| rather than
    # (S, A) -> R, so the action is no longer a network input.
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
    # Discrete target entropy = scale * log|A|, i.e. a fixed fraction of the
    # entropy of the uniform policy (Christodoulou 2019 uses 0.98). The
    # continuous heuristic -dim(A) does not apply: env.action_space.shape is
    # () for a Discrete space, so -prod(shape) would silently evaluate to -1.
    # ------------------------------------------------------------------
    if cfg.autotune:
        target_entropy_scale = float(cfg.get("target_entropy_scale", 0.89))
        target_entropy = -target_entropy_scale * float(np.log(1.0 / n_actions))
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

    # Manual episode accounting. Mirrors the PPO file: if the env is not
    # wrapped in RecordEpisodeStatistics there is no infos["episode"], and
    # charts/episodic_return would silently never appear - which is exactly
    # the panel the two runs need to share.
    current_ep_return = 0.0
    current_ep_length = 0

    # ==================================================================
    # Main loop
    # ==================================================================
    obs, _ = env.reset(seed=cfg.seed)

    for global_step in range(cfg.total_timesteps):

        # --------------------------------------------------------------
        # Action selection
        # Before learning starts: random exploration
        # After: sample from the categorical actor
        # --------------------------------------------------------------
        if global_step < cfg.learning_starts:
            actions = int(env.action_space.sample())
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
            # (1,) int64 tensor -> plain Python int for env.step()
            actions = int(actions.item())

        # --------------------------------------------------------------
        # Step environment
        # --------------------------------------------------------------
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        current_ep_return += float(rewards)
        current_ep_length += 1

        # ----------------------------------------------------------------
        # Single (non-vectorized) env: `next_obs` from env.step() IS already
        # the true final observation on truncation - Gymnasium only injects
        # "final_observation" for vector envs with autoreset. No lookup needed.
        # ----------------------------------------------------------------
        real_next_obs = next_obs.copy()

        rb.add(
            obs[np.newaxis],  # (obs_dim,) -> (1, obs_dim)
            real_next_obs[np.newaxis],  # same
            np.array([[actions]]),  # int -> (1, 1) action index
            np.array([rewards]),
            np.array([terminations]),
            [infos],  # dict -> [dict]
        )

        # --------------------------------------------------------------
        # Episode end: log returns, then reset. Otherwise carry obs forward.
        # NOTE: no `break` here - training must continue past episode ends.
        # --------------------------------------------------------------
        if terminations or truncations:
            ep_return = infos["episode"]["r"] if "episode" in infos else current_ep_return
            ep_length = infos["episode"]["l"] if "episode" in infos else current_ep_length

            print(f"global_step={global_step}, episodic_return={ep_return}")
            episodic_return_history.append(ep_return)

            if cfg.track:
                wandb.log({
                    "charts/episodic_return": ep_return,
                    "charts/episodic_length": ep_length,
                }, step=global_step)

            current_ep_return = 0.0
            current_ep_length = 0

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
            # Discrete Bellman target - the expectation over next actions is
            # computed EXACTLY rather than from a single sampled a':
            #   V(s') = sum_a pi(a|s') * [min_Q(s',a) - alpha * log pi(a|s')]
            #   y     = r + gamma * (1 - d) * V(s')
            # ----------------------------------------------------------
            with torch.no_grad():
                _, next_state_log_pi, next_state_action_probs = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations)
                qf2_next_target = qf2_target(data.next_observations)
                # Pessimistic (min) estimate, weighted by the policy
                min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                )
                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = data.rewards.flatten() + (
                        (1 - data.dones.flatten()) * cfg.gamma * min_qf_next_target
                )

            # Q-values for the actions actually taken: index into the |A|
            # outputs instead of feeding the action through the network.
            qf1_a_values = qf1(data.observations).gather(1, data.actions.long()).view(-1)
            qf2_a_values = qf2(data.observations).gather(1, data.actions.long()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            # max_norm=inf measures the norm without clipping anything, so
            # system/grad_norm is comparable to PPO's without changing SAC's
            # update rule (PPO clips at cfg.max_grad_norm; SAC does not).
            q_grad_norm = nn.utils.clip_grad_norm_(
                list(qf1.parameters()) + list(qf2.parameters()), float("inf")
            )
            q_optimizer.step()

            # ----------------------------------------------------------
            # Actor + alpha update (delayed update kept for consistency
            # with the continuous file; the discrete SAC paper updates
            # every step, so policy_frequency=1 is also defensible here)
            #
            # Actor minimises sum_a pi(a|s) * [alpha * log pi(a|s) - min_Q(s,a)]
            # The Q-values are treated as constants: with a discrete action
            # there is no gradient path from the actor through the critic,
            # so the policy gradient comes from the explicit sum instead.
            # ----------------------------------------------------------
            if global_step % cfg.policy_frequency == 0:
                for _ in range(cfg.policy_frequency):
                    _, log_pi, action_probs = actor.get_action(data.observations)
                    with torch.no_grad():
                        qf1_values = qf1(data.observations)
                        qf2_values = qf2(data.observations)
                        min_qf_values = torch.min(qf1_values, qf2_values)

                    actor_loss = (
                            action_probs * ((alpha * log_pi) - min_qf_values)
                    ).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        actor.parameters(), float("inf")
                    )
                    actor_optimizer.step()

                    # alpha_loss must exist even when cfg.autotune is False,
                    # since log_sac_step is always called below and always
                    # expects an alpha_loss argument.
                    alpha_loss = None

                    if cfg.autotune:
                        # No second get_action call needed - log_pi and
                        # action_probs above already cover every action, and
                        # both are detached here so the freed actor graph is
                        # never touched.
                        alpha_loss = (
                                action_probs.detach()
                                * (-log_alpha.exp() * (log_pi + target_entropy).detach())
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                qf1_loss_history.append(qf1_loss.item())
                actor_loss_history.append(actor_loss.item())

                sps = int(global_step / (time.time() - start_time))

                should_log = log_interval <= 0 or global_step % log_interval == 0
                if cfg.track and should_log:
                    log_sac_step(
                        global_step, env, data, action_probs,
                        qf1_a_values, next_q_value,
                        actor_loss, qf1_loss, qf2_loss, qf_loss,
                        alpha, log_pi, cfg.autotune, alpha_loss,
                        actor_optimizer, q_optimizer, device,
                        sps=sps,
                        actor_grad_norm=actor_grad_norm,
                        q_grad_norm=q_grad_norm,
                        log_per_dim_obs=bool(cfg.get("log_per_dim_obs", True)),
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

            print("SPS:", int(global_step / (time.time() - start_time)))

    # ======================================================================
    # Final checkpoint - runs ONCE, after the loop finishes
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

    # Distinct filename prefix: these state dicts are NOT loadable by the
    # continuous SAC file (different layer shapes and head names).
    model_path = os.path.join(model_dir, f"sac_discrete_step_{global_step}.pt")

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
#
# Section numbering mirrors log_ppo_step so the two are diffable.
# ==============================================================================

def log_sac_step(
    global_step, env, data, action_probs,
    qf1_a_values, next_q_value,
    actor_loss, qf1_loss, qf2_loss, qf_loss,
    alpha, log_pi, autotune, alpha_loss,
    actor_optimizer, q_optimizer, device,
    sps=None,
    actor_grad_norm=None,
    q_grad_norm=None,
    log_per_dim_obs=True,
):
    log_dict = {}

    # ------------------------------------------------------------------
    # 1. REWARDS
    # ------------------------------------------------------------------
    log_dict["rewards/mean"] = data.rewards.mean().item()

    # ------------------------------------------------------------------
    # 2. LOSSES
    # ------------------------------------------------------------------
    y_pred = qf1_a_values.detach().cpu().numpy()
    y_true = next_q_value.detach().cpu().numpy()
    var_y = np.var(y_true)
    explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # Entropy must be weighted by pi: log_pi now holds log-probs for ALL
    # actions, so the continuous version's (-log_pi).mean() would average
    # over the whole action set instead of taking the policy expectation.
    # This is the same quantity PPO logs as losses/entropy (nats).
    entropy = -(action_probs * log_pi).sum(dim=1)

    # PPO-comparable aliases. policy_loss <- actor_loss; value_loss <-
    # qf1_loss (single-critic MSE, closest in scale to PPO's 0.5*MSE;
    # qf_loss is the sum over both critics and so runs ~2x higher).
    log_dict.update({
        "losses/policy_loss": actor_loss.item(),
        "losses/value_loss": qf1_loss.item(),
        "losses/entropy": entropy.mean().item(),
        "losses/explained_variance": explained_var,
    })

    # SAC-native losses (no PPO counterpart, own keys)
    log_dict.update({
        "losses/actor_loss": actor_loss.item(),
        "losses/qf1_loss": qf1_loss.item(),
        "losses/qf2_loss": qf2_loss.item(),
        "losses/qf_loss": qf_loss.item(),
        "losses/alpha": alpha,
    })
    if autotune and alpha_loss is not None:
        log_dict["losses/alpha_loss"] = alpha_loss.item()

    # ------------------------------------------------------------------
    # 3. ACTIONS (Discrete)
    # Same computation as PPO: empirical stats over the integer actions in
    # the batch. Caveat when reading the panel - PPO's batch is the current
    # on-policy rollout, this one is a replay sample, so it mixes actions
    # from older policies and lags PPO's version of the same curve.
    # ------------------------------------------------------------------
    actions_np = data.actions.detach().cpu().numpy().flatten().astype(np.int64)
    log_dict["actions_mean/aggregate"] = float(actions_np.mean())
    log_dict["actions_std/aggregate"] = float(actions_np.std())

    num_actions = int(env.action_space.n)
    action_counts = np.bincount(actions_np, minlength=num_actions)
    action_frac = action_counts / max(action_counts.sum(), 1)

    for a_idx in range(num_actions):
        log_dict[f"actions_mean/action_{a_idx}_frac"] = float(action_frac[a_idx])

    # ------------------------------------------------------------------
    # 3b. ACTION DISTRIBUTION (SAC-only, separate namespace)
    # PPO only sees sampled actions; here the full categorical is available,
    # so these go under actions_probs/ rather than overwriting the shared
    # actions_mean/ and actions_std/ keys above with a different quantity.
    # ------------------------------------------------------------------
    probs_np = action_probs.detach().cpu().numpy()
    prob_means = probs_np.mean(axis=0)
    prob_stds = probs_np.std(axis=0)
    max_probs = probs_np.max(axis=1)

    log_dict["actions_probs/entropy"] = float(entropy.mean().item())
    log_dict["actions_probs/entropy_uniform"] = float(np.log(probs_np.shape[-1]))
    for a_idx in range(probs_np.shape[-1]):
        log_dict[f"actions_probs/prob_a{a_idx}"] = float(prob_means[a_idx])

    log_dict.update({
        # Spread of each action's probability across the batch: near zero
        # means the policy is state-independent (collapsed).
        "actions_probs/std_aggregate": float(prob_stds.mean()),
        "actions_probs/std_min": float(prob_stds.min()),
        "actions_probs/std_max": float(prob_stds.max()),
        # Determinism proxy - the discrete stand-in for frac_at_clip_bounds.
        "actions_probs/max_prob_mean": float(max_probs.mean()),
        "actions_probs/frac_near_deterministic": float(np.mean(max_probs > 0.99)),
    })
    for a_idx, std_val in enumerate(prob_stds):
        log_dict[f"actions_probs/std_a{a_idx}"] = float(std_val)

    # ------------------------------------------------------------------
    # 4. OBSERVATIONS
    # ------------------------------------------------------------------
    obs_np = data.observations.detach().cpu().numpy()
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)

    obs_dim_means = obs_flat.mean(axis=0)
    obs_dim_stds = obs_flat.std(axis=0)

    log_dict["observations_mean/aggregate"] = float(obs_np.mean())

    # Per-dim series, same as PPO. On by default so the two runs share every
    # observations_mean/dim* and observations_std/dim* panel. A one-hot
    # tabular env writes two scalars per obs dimension per log call, so on a
    # 716-dim obs this is ~1.4k series - set log_per_dim_obs=false in the
    # config if a run gets too heavy (PPO would need the same treatment).
    if log_per_dim_obs:
        for dim_idx in range(obs_flat.shape[-1]):
            log_dict[f"observations_mean/dim{dim_idx}"] = float(obs_dim_means[dim_idx])

    log_dict["observations_std/aggregate"] = float(obs_np.std())

    if log_per_dim_obs:
        for dim_idx in range(obs_flat.shape[-1]):
            log_dict[f"observations_std/dim{dim_idx}"] = float(obs_dim_stds[dim_idx])

    obs_has_nan_or_inf = bool(np.isnan(obs_np).any() or np.isinf(obs_np).any())

    # ------------------------------------------------------------------
    # 5. TRAINING
    # system/grad_norm is the actor's norm, so it lines up with PPO's
    # (which is the norm over the whole actor-critic agent, policy included).
    # The critic norm is kept alongside under its own key.
    # ------------------------------------------------------------------
    def _as_float(x):
        if x is None:
            return None
        return x.item() if torch.is_tensor(x) else float(x)

    if sps is not None:
        log_dict["system/SPS"] = int(sps)
    if actor_grad_norm is not None:
        log_dict["system/grad_norm"] = _as_float(actor_grad_norm)
        log_dict["system/grad_norm_actor"] = _as_float(actor_grad_norm)
    if q_grad_norm is not None:
        log_dict["system/grad_norm_q"] = _as_float(q_grad_norm)

    log_dict["system/obs_has_nan_or_inf"] = int(obs_has_nan_or_inf)

    # PPO has a single optimizer, so charts/learning_rate is the shared key;
    # the two SAC-specific rates keep their existing names.
    log_dict.update({
        "charts/learning_rate": actor_optimizer.param_groups[0]["lr"],
        "charts/learning_rate_actor": actor_optimizer.param_groups[0]["lr"],
        "charts/learning_rate_q": q_optimizer.param_groups[0]["lr"],
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
        "cpu/cores_active_count": sum(1 for c in per_core if c > 20.0),
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
        pass  # io_counters not available on macOS

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
            pass  # pynvml not available on this driver setup

    wandb.log(log_dict, step=global_step)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    # NOTE: same pre-existing gap as TD3 and continuous SAC - train() needs
    # (cfg, env, path), main() only has cfg. Needs env construction + path
    # wired in from wherever your other entry points build these.
    raise NotImplementedError(
        "main() must construct `env` and `path` before calling train(cfg, env, path)."
    )


if __name__ == "__main__":
    main()