

import os
import random
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import OmegaConf

from RLfoundry.DQN.network import Network
from RLfoundry.DQN.replay_buffer import ReplayBuffer

from RLfoundry.Utils.run_startup import run_startup

_proc = psutil.Process()
_proc.cpu_percent()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def train(cfg, env, path):
    # ------------------------------------------------------------------
    # Config + run naming
    # ------------------------------------------------------------------
    path = run_startup(path, cfg)

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.algo_name,
        group=cfg.env_id,
        job_type=getattr(cfg, "algo_name", "dqn"),
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
    # Networks (plain Q-network, no dueling/noisy/distributional head)
    # ------------------------------------------------------------------
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    obs_dim = obs_shape[0]

    policy_net = Network(in_dim=obs_dim, out_dim=num_actions).to(device)
    target_net = Network(in_dim=obs_dim, out_dim=num_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------------
    # Plain uniform replay buffer (no PER, no N-step)
    # ------------------------------------------------------------------
    rb = ReplayBuffer(
        obs_dim=obs_shape[0],
        size=cfg.buffer_size,
        batch_size=cfg.batch_size,
    )

    # ------------------------------------------------------------------
    # Metric history lists (saved as numpy arrays at each checkpoint)
    # ------------------------------------------------------------------
    loss_history = []
    episodic_return_history = []
    learn_steps = 0

    start_time = time.time()

    # ==================================================================
    # Main loop
    # ==================================================================
    obs, _ = env.reset(seed=cfg.seed)
    obs = np.array(obs, dtype=np.float32, copy=False)

    for global_step in range(cfg.total_timesteps):

        # --------------------------------------------------------------
        # Epsilon-greedy action selection
        # --------------------------------------------------------------
        epsilon = linear_schedule(
            cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = policy_net(obs_t)
                action = q_vals.argmax(1).item()

        # --------------------------------------------------------------
        # Step environment
        # --------------------------------------------------------------
        next_obs, reward, terminated, truncated, infos = env.step(action)
        next_obs = np.array(next_obs, dtype=np.float32, copy=False)
        done = terminated or truncated

        if "episode" in infos:
            ep_return = infos["episode"]["r"]
            ep_length = infos["episode"]["l"]
            print(f"global_step={global_step}, episodic_return={ep_return}")
            episodic_return_history.append(ep_return)

            if cfg.track:
                wandb.log({
                    "charts/episodic_return": ep_return,
                    "charts/episodic_length": ep_length,
                    "charts/epsilon": epsilon,
                }, step=global_step)

        rb.store(obs, action, reward, next_obs, done)

        if done:
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32, copy=False)
        else:
            obs = next_obs

        # ==============================================================
        # Training update
        # ==============================================================
        if global_step > cfg.learning_starts and len(rb) >= cfg.batch_size:



            states, actions, rewards, next_states, dones = rb.sample_batch(cfg.batch_size)



            states      = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t   = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
            rewards_t   = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones_t     = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

            # Standard 1-step Bellman target (no double-Q, no distributional)
            with torch.no_grad():
                next_q = target_net(next_states)
                next_q_max, _ = next_q.max(dim=1, keepdim=True)
                target = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q_max

            q_all = policy_net(states)
            q_sa = q_all.gather(1, actions_t)

            loss = F.smooth_l1_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            td = (target - q_sa).detach().abs().cpu().numpy()

            loss_history.append(loss.item())
            learn_steps += 1

            # ----------------------------------------------------------
            # Target network update
            # ----------------------------------------------------------
            if cfg.target_update_type == "soft":
                for name, param in policy_net.state_dict().items():
                    t = target_net.state_dict()[name]
                    target_net.state_dict()[name].copy_(
                        param * cfg.tau + t * (1 - cfg.tau)
                    )
            elif cfg.target_update_type == "hard":
                if learn_steps % cfg.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # ----------------------------------------------------------
            # wandb logging — 8 panel groups
            # ----------------------------------------------------------

            if cfg.track:
                log_dqn_step(
                    global_step, num_actions, rewards_t, loss, td,
                    actions_t, states, epsilon, rb, optimizer, device,
                )

        # ----------------------------------------------------------------
        # Async checkpoint (periodic, based on wall-clock interval)
        # ----------------------------------------------------------------
        now = time.time()
        if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
            save_checkpoint(
                path,
                global_step=global_step,
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                loss_history=np.array(loss_history),
                episodic_return_history=np.array(episodic_return_history),
            )
            last_numpy_save_t = now

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)

    # ======================================================================
    # Final checkpoint
    # ======================================================================
    save_checkpoint(
        os.path.join(path, "Final"),
        global_step=cfg.total_timesteps,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        loss_history=np.array(loss_history),
        episodic_return_history=np.array(episodic_return_history),
    )

    env.close()
    wandb.finish()

def log_dqn_step(
    global_step, num_actions, rewards_t, loss, td,
    actions_t, states, epsilon, rb, optimizer, device,
):
    log_dict = {}

    # --- 1. REWARDS ---
    log_dict["rewards/mean"] = float(rewards_t.mean().item())

    # --- 2. LOSSES ---
    log_dict.update({
        "losses/dqn_loss": loss.item(),
        "losses/td_error_mean": float(td.mean()),
        "losses/td_error_max": float(td.max()),
    })

    # --- 3/4. ACTIONS: mean + std (discrete -> action-frequency style) ---
    actions_np = actions_t.detach().cpu().numpy().flatten()
    log_dict["actions_mean/aggregate"] = float(actions_np.mean())
    log_dict["actions_std/aggregate"] = float(actions_np.std())
    for a_idx in range(num_actions):
        log_dict[f"actions_mean/action_{a_idx}_frac"] = float(np.mean(actions_np == a_idx))

    # --- 5/6. OBSERVATIONS: mean + std (aggregate + per-dim) ---
    obs_np = states.detach().cpu().numpy()
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)
    obs_dim_means = obs_flat.mean(axis=0)
    obs_dim_stds = obs_flat.std(axis=0)

    log_dict["observations_mean/aggregate"] = float(obs_flat.mean())
    log_dict["observations_std/aggregate"] = float(obs_flat.std())
    for dim_idx in range(obs_flat.shape[-1]):
        log_dict[f"observations_mean/dim{dim_idx}"] = float(obs_dim_means[dim_idx])
        log_dict[f"observations_std/dim{dim_idx}"] = float(obs_dim_stds[dim_idx])

    # --- 8. CHARTS ---
    log_dict.update({
        "charts/learning_rate": optimizer.param_groups[0]["lr"],
        "charts/epsilon": epsilon,
        "charts/buffer_size": len(rb),
    })

    # --- 8. CHARTS ---
    log_dict.update({
        "charts/learning_rate": optimizer.param_groups[0]["lr"],
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


def save_checkpoint(save_dir, global_step, policy_net, target_net, optimizer, **arrays):
    if arrays:
        step_dir = os.path.join(save_dir, "numpy", f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        for name, arr in arrays.items():
            np.save(os.path.join(step_dir, f"{name}.npy"), arr)
        print(f"[numpy checkpoint] saved {list(arrays.keys())} -> {step_dir}")

    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"dqn_step_{global_step}.pt")

    checkpoint = {
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }

    torch.save(checkpoint, model_path)
    print(f"[model checkpoint] saved -> {model_path}")