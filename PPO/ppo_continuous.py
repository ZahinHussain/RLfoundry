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

from gymnasium.wrappers import FlattenObservation
from RLfoundry.PPO.Agent import Agent
from RLfoundry.Utils.run_startup import run_startup

from RLfoundry.Utils.checkpoint import create_folder_checkpoint, create_timestamp_folder


def train(cfg, env, path):
    # run startup
    path = run_startup(path, cfg)

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

  
    else:
        device = torch.device("cpu")
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=cfg.eps)

    # ---- pre-allocated storage for one rollout's worth of data ----
    obs = torch.zeros((cfg.num_steps,) + env.observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps,) + env.action_space.shape).to(device)
    logprobs = torch.zeros(cfg.num_steps).to(device)
    rewards = torch.zeros(cfg.num_steps).to(device)
    dones = torch.zeros(cfg.num_steps).to(device)
    values = torch.zeros(cfg.num_steps).to(device)

    global_step = 0
    next_obs_np, _ = env.reset(seed=cfg.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.zeros(1).to(device)
    start_time = time.time()

    # ========================================================================
    # [LINE 2] OUTER LOOP — one iteration = one full "collect, then update" cycle
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
            actions[step] = action
            logprobs[step] = logprob

            # step the actual environment forward

            action_np = action.detach().cpu().numpy().reshape(-1)  # always gives 1D (act_dim,)
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_obs, reward, terminations, truncations, info = env.step(action_np)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            if next_done:
                # episode ended -> manually reset since this is a single
                # (non-vectorized) env and Gymnasium does NOT auto-reset


                next_obs, info = env.reset()

            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)

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

            # walk BACKWARD through the rollout — this is the standard GAE
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

            # [LINE 4] the value function's regression TARGET — this stands
            # in for "rewards-to-go" in the pseudocode
            returns = advantages + values

        obs_space = env.observation_space
        act_space = env.action_space

        # ---- pre-allocated storage for one rollout's worth of data ----
        # [LINE 3] this is where D_k actually lives, as plain tensors rather
        # than a list of trajectory objects — much faster on GPU

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
            # reshuffle which data goes in which minibatch every epoch —
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

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    # FIX: Added correction=0 to prevent NaN when minibatch size is 1
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(correction=0) + 1e-8)
                # ---- [LINE 6] THE PPO-CLIP OBJECTIVE ITSELF ----
                # Spinning Up's formula:
                #   L = min( ratio * A_hat,  clip(ratio, 1-eps, 1+eps) * A_hat )
                # We want to MAXIMIZE this, but PyTorch optimizers MINIMIZE,
                # so everything below is negated (hence -mb_advantages and
                # torch.max instead of torch.min — negating min(a,b) gives
                # max(-a,-b))
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ---- [LINE 8] VALUE FUNCTION MSE REGRESSION ----
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    # [EXTRA] clipped value loss — not in pseudocode, but
                    # mirrors the policy clip: don't let the value estimate
                    # move too far from its old value in one update
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    # plain MSE — this is the literal pseudocode line 8
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # [EXTRA] entropy bonus — encourages exploration, not in
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
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            # [EXTRA] optional early stopping if policy is changing too fast

            ''''

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break
                '''

        now = time.time()
        if now - last_numpy_save_t >= SAVE_INTERVAL_SECONDS:
            save_checkpoint(
                path,
                global_step,
                agent,
                optimizer,
                obs=obs.cpu().numpy(),
                actions=actions.cpu().numpy(),
                logprobs=logprobs.cpu().numpy(),
                rewards=rewards.cpu().numpy(),
                dones=dones.cpu().numpy(),
                values=values.cpu().numpy(),
                returns=returns.cpu().numpy(),
                advantages=advantages.cpu().numpy(),
            )
            last_numpy_save_t = now

        print("SPS:", int(global_step / (time.time() - start_time)))

    save_checkpoint(
        path,
        global_step,
        agent,
        optimizer,
        obs=obs.cpu().numpy(),
        actions=actions.cpu().numpy(),
        logprobs=logprobs.cpu().numpy(),
        rewards=rewards.cpu().numpy(),
        dones=dones.cpu().numpy(),
        values=values.cpu().numpy(),
        returns=returns.cpu().numpy(),
        advantages=advantages.cpu().numpy(),
    )

    env.close()


def save_checkpoint(
        save_dir: str,
        global_step: int,
        agent,
        optimizer,
        **arrays: np.ndarray,
) -> None:
    """
    Save rollout arrays and model checkpoint.
    """

    # Save numpy arrays
    step_dir = os.path.join(save_dir, "numpy", f"step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)

    for name, arr in arrays.items():
        np.save(os.path.join(step_dir, f"{name}.npy"), arr)

    print(f"[numpy checkpoint] saved {list(arrays.keys())} -> {step_dir}")

    # Save model
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f"ppo_step_{global_step}.pt"
    )

    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
