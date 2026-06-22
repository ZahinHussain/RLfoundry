import random
import time
import os

import hydra
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from RLFoundry.PPO import Agent
from RLFoundry.env.env import make_vec_env
from RLFoundry.Utils.checkpoint import create_folder_checkpoint, create_timestamp_folder

# ============================================================
# How often (in seconds) to flush rollout data to disk
# ============================================================
SAVE_INTERVAL_SECONDS = 30 * 60  # 30 minutes


def save_numpy_checkpoint(save_dir: str, global_step: int, **arrays: np.ndarray) -> None:
    """
    Save a dict of named numpy arrays to <save_dir>/numpy/<global_step>/.
    Each array is stored as its own .npy file.

    Args:
        save_dir:    root checkpoint directory for this run
        global_step: current env step (used as the sub-folder name)
        **arrays:    keyword args of the form array_name=np.ndarray
    """
    step_dir = os.path.join(save_dir, "numpy", f"step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)

    for name, arr in arrays.items():
        np.save(os.path.join(step_dir, f"{name}.npy"), arr)

    print(f"[numpy checkpoint] saved {list(arrays.keys())} → {step_dir}")


def train(cfg, path):

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    create_folder_checkpoint(path, cfg.exp_name)

    # Root directory for all checkpoints belonging to this run
    run_checkpoint_dir = os.path.join(path, cfg.exp_name)

    run_name = f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env setup
    envs = make_vec_env(cfg.gym_id, cfg.num_envs, cfg.gamma, cfg.seed)
    assert isinstance(envs.single_action_space, gym.spaces.Box), \
        "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ------------------------------------------------------------------ #
    # Pre-allocated rollout buffers                                        #
    # ------------------------------------------------------------------ #
    obs       = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
    actions   = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape).to(device)
    logprobs  = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards   = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones     = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values    = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # These are populated after GAE, referenced outside the loop for the
    # final numpy save — initialise to None so the save guard works.
    returns    = None
    advantages = None

    global_step = 0
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs  = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    start_time         = time.time()
    last_numpy_save_t  = time.time()   # tracks when we last wrote numpy arrays

    # ================================================================== #
    # OUTER LOOP                                                           #
    # ================================================================== #
    for iteration in range(1, cfg.num_iterations + 1):

        # ---- learning-rate annealing ---------------------------------- #
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ================================================================ #
        # COLLECT ROLLOUT                                                   #
        # ================================================================ #
        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step]   = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step]  = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs  = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # ================================================================ #
        # GAE — compute advantages and returns                             #
        # ================================================================ #
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues      = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues      = values[t + 1]

                delta          = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t]  = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # ================================================================ #
        # FLATTEN buffers for minibatching                                  #
        # ================================================================ #
        b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs   = logprobs.reshape(-1)
        b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values.reshape(-1)

        # ================================================================ #
        # UPDATE — PPO-Clip + value regression                             #
        # ================================================================ #
        b_inds = np.arange(cfg.batch_size)
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end     = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio    = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(correction=0) + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped        = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss         = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss         = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # ================================================================ #
        # NUMPY CHECKPOINT — every 30 minutes                              #
        # ================================================================ #
        now = time.time()
        if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
            save_numpy_checkpoint(
                run_checkpoint_dir,
                global_step,
                # Raw rollout buffers (shape: num_steps × num_envs × ...)
                obs        = obs.cpu().numpy(),
                actions    = actions.cpu().numpy(),
                logprobs   = logprobs.cpu().numpy(),
                rewards    = rewards.cpu().numpy(),
                dones      = dones.cpu().numpy(),
                values     = values.cpu().numpy(),
                # Post-GAE targets (same shape)
                returns    = returns.cpu().numpy(),
                advantages = advantages.cpu().numpy(),
            )
            last_numpy_save_t = now

        # ================================================================ #
        # LOGGING                                                           #
        # ================================================================ #
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss",    v_loss.item(),       global_step)
        writer.add_scalar("losses/policy_loss",   pg_loss.item(),      global_step)
        writer.add_scalar("losses/entropy",       entropy_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    # ------------------------------------------------------------------ #
    # Final numpy save on run completion                                   #
    # ------------------------------------------------------------------ #
    if returns is not None and advantages is not None:
        save_numpy_checkpoint(
            run_checkpoint_dir,
            global_step,
            obs        = obs.cpu().numpy(),
            actions    = actions.cpu().numpy(),
            logprobs   = logprobs.cpu().numpy(),
            rewards    = rewards.cpu().numpy(),
            dones      = dones.cpu().numpy(),
            values     = values.cpu().numpy(),
            returns    = returns.cpu().numpy(),
            advantages = advantages.cpu().numpy(),
        )

    envs.close()
    writer.close()


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    path = os.getcwd()
    train(cfg, path)


if __name__ == "__main__":
    main()