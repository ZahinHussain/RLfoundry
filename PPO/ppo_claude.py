"""
============================================================================
PPO-CLIP, FULLY ANNOTATED — mapped line-by-line to Spinning Up Algorithm 1
============================================================================

Original pseudocode for reference:

  1. Input: initial policy params θ0, initial value function params φ0
  2. for k = 0,1,2,... do
  3.     Collect trajectories D_k = {τ_i} by running policy π_k = π(θ_k)
  4.     Compute rewards-to-go R_hat
  5.     Compute advantage estimates A_hat (any method, using value fn V_φk)
  6.     Update policy by maximizing PPO-Clip objective (via Adam/SGD)
  7.     Fit value function by regression on MSE (via gradient descent)
  8. end for

Every section below is tagged [LINE X] so you can see exactly where in the
loop you are. Sections that DON'T appear in the pseudocode at all (but exist
because real implementations need them) are tagged [EXTRA] — these are the
things worth flagging explicitly in your dissertation as implementation
details rather than core algorithm.
============================================================================
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# CONFIG — every hyperparameter the algorithm needs, with WHY it's there
# ============================================================================
@dataclass
class Args:
    seed: int = 1
    # reproducibility: same seed -> same random numbers -> same run, useful
    # for debugging and for honest comparisons between algorithm variants

    cuda: bool = True
    # run on GPU if available — PPO doesn't strictly need it for small MLPs,
    # but you have a 4070 Super sitting there, may as well use it

    env_id: str = "HalfCheetah-v4"
    # which Gym environment to train on. In RLFoundry this is where you'd
    # swap in ICU-Sepsis, pymgrid, RWARE, etc.

    total_timesteps: int = 1_000_000
    # how many environment steps to run before stopping, across ALL parallel
    # envs combined

    learning_rate: float = 3e-4
    # Adam's step size. 3e-4 is the famous "Karpathy constant" — works
    # surprisingly often as a default for Adam on RL/NLP problems

    num_envs: int = 1
    # how many copies of the environment run in parallel (vectorized).
    # More envs = more diverse data per "rollout", at the cost of CPU/RAM

    num_steps: int = 2048
    # [LINE 3] how many steps EACH environment takes before we stop collecting
    # and do an update. This is the size of one "rollout" / trajectory batch.

    anneal_lr: bool = True
    # linearly decay the learning rate to 0 over training — common trick to
    # stabilize learning in late training when you're near a good policy

    gamma: float = 0.99
    # discount factor. How much future reward is worth vs immediate reward.
    # 0.99 means a reward 100 steps away is still worth ~37% of its face value

    gae_lambda: float = 0.95
    # [LINE 5] controls the bias-variance tradeoff in advantage estimation.
    # lambda=0 -> pure TD (low variance, high bias)
    # lambda=1 -> pure Monte Carlo (high variance, low bias)
    # 0.95 is the standard middle ground

    num_minibatches: int = 32
    # [LINE 6/7] split each big rollout batch into this many smaller chunks
    # for gradient updates (full-batch gradient descent is unstable/slow)

    update_epochs: int = 10
    # [LINE 6/7] how many times we reuse the SAME collected data for gradient
    # updates before throwing it away and collecting fresh data. This is
    # PPO's main sample-efficiency trick vs vanilla policy gradient (which
    # can only use each batch of data once)

    norm_adv: bool = True
    # [LINE 5, extra] normalize advantages to mean 0, std 1 within each
    # minibatch. Pure implementation detail for gradient stability, not in
    # the original paper's math

    clip_coef: float = 0.2
    # [LINE 6] epsilon in the clipped objective. How far the new policy is
    # allowed to drift from the old policy in one update, in probability-
    # ratio terms. 0.2 means ratio is clamped to [0.8, 1.2]

    clip_vloss: bool = True
    # [EXTRA] also clip the value function update, mirroring the policy
    # clip. NOT in Spinning Up pseudocode (line 8 just says "MSE regression")
    # but is in the original PPO paper's code and is now standard practice

    ent_coef: float = 0.0
    # [EXTRA] entropy bonus coefficient. Encourages exploration by rewarding
    # the policy for staying "spread out" / uncertain. Pseudocode line 6
    # only mentions the clip objective — entropy is a common addition.
    # Default 0.0 here means it's OFF unless you turn it on.

    vf_coef: float = 0.5
    # [LINE 8] how much the value loss contributes to the combined loss,
    # relative to the policy loss

    max_grad_norm: float = 0.5
    # [EXTRA] gradient clipping — caps how big a single update step can be,
    # prevents one bad batch from destroying the policy. Not in pseudocode.

    target_kl: float = None
    # [EXTRA] optional early-stopping: if the policy changes too much
    # (measured by KL divergence) during the update_epochs loop, stop early.
    # Pure implementation safety valve.

    # computed automatically, not set by hand:
    batch_size: int = 0       # num_envs * num_steps
    minibatch_size: int = 0   # batch_size // num_minibatches
    num_iterations: int = 0   # total_timesteps // batch_size


# ============================================================================
# ENVIRONMENT SETUP — wrapping the raw Gym env with preprocessing
# ============================================================================
def make_env(env_id, idx, gamma):
    """
    [LINE 3 setup] Builds one environment instance with a stack of wrappers.
    None of this is "PPO" per se — it's standard RL preprocessing that makes
    PPO (or any algorithm) train more reliably.
    """
    def thunk():
        env = gym.make(env_id)

        # Flatten dict/tuple observations into a single vector — needed
        # because the policy network expects a flat input
        env = gym.wrappers.FlattenObservation(env)

        # Track episode return/length automatically so we can log it
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Clip actions to the env's valid range BEFORE sending them in —
        # the policy is a Gaussian, which can technically sample values
        # outside [action_low, action_high]
        env = gym.wrappers.ClipAction(env)

        # Running mean/std normalization of observations. Without this,
        # if one observation dimension is in range [0, 1000] and another
        # is in [0, 1], the network effectively ignores the small one.
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )

        # Same idea but for rewards — keeps reward scale roughly consistent
        # so the value function doesn't have to learn wildly different
        # magnitudes across environments
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    [EXTRA] Orthogonal weight initialization instead of PyTorch's default.
    Empirically, this specific init scheme (from the original PPO paper)
    makes training noticeably more stable. Nothing to do with the PPO
    *algorithm* — it's a neural network training detail.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ============================================================================
# THE AGENT — [LINE 1] theta_0 (policy) and phi_0 (value function)
# ============================================================================
class Agent(nn.Module):
    """
    Two completely separate MLPs:
      - self.critic     = V_phi(s)      -> a single number, "how good is this state"
      - self.actor_mean  = mu_theta(s)   -> the MEAN of a Gaussian action distribution

    Why separate networks rather than one shared trunk? Simpler to reason
    about and tune independently — shared-trunk is an optimization some
    implementations use, but isn't required by the algorithm.
    """
    def __init__(self, envs):
        super().__init__()

        # ---- VALUE FUNCTION: phi_0 ----
        # Maps state -> scalar value estimate. 2 hidden layers of 64 units,
        # Tanh activations (smoother gradients than ReLU for small RL nets).
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # final layer: small std, single output
        )

        # ---- POLICY MEAN: theta_0 (part 1) ----
        # Same shape, but outputs one number PER ACTION DIMENSION (the mean
        # of each action's Gaussian distribution)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            # std=0.01: start with TINY action means near zero so the
            # initial policy doesn't do anything wild before learning starts
        )

        # ---- POLICY STD: theta_0 (part 2) ----
        # The standard deviation of the action distribution is a SEPARATE
        # learnable parameter, NOT a function of state. It's a single vector
        # shared across all states. This is a common simplification —
        # "state-independent" exploration noise.
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """Just run the critic. Used for bootstrapping at the end of a rollout."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        [LINE 3] Core policy-execution step. Given a state x:
          1. compute the Gaussian distribution over actions
          2. sample an action (or evaluate the log-prob of a GIVEN action,
             when re-used later during the update step)
          3. also return the value estimate, since we need both at once
             during the rollout
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)  # exp() keeps std positive
        probs = Normal(action_mean, action_std)

        if action is None:
            # We're COLLECTING data: sample a fresh action from the policy
            action = probs.sample()
        else:
            # We're UPDATING the policy: action was already taken earlier,
            # we just need this distribution's log-prob of THAT action
            # (this is how ratio = pi_new/pi_old gets computed later)
            pass

        return (
            action,
            probs.log_prob(action).sum(1),   # log pi_theta(a|s), summed over action dims
            probs.entropy().sum(1),          # entropy bonus term [EXTRA]
            self.critic(x),                  # V_phi(s)
        )


# ============================================================================
# MAIN TRAINING LOOP — [LINE 2] "for k = 0, 1, 2, ... do"
# ============================================================================
def train():
    args = Args()
    args.env_id = "Pendulum-v1"   # built-in, no extra deps, just to prove the loop runs
    args.total_timesteps = 4096
    args.num_steps = 256
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.gamma) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    # ONE optimizer updates BOTH the actor and critic networks together —
    # this is why the combined loss later sums policy loss + value loss
    # into a single scalar before calling .backward()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ---- pre-allocated storage for one rollout's worth of data ----
    # [LINE 3] this is where D_k actually lives, as plain tensors rather
    # than a list of trajectory objects — much faster on GPU
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # ========================================================================
    # [LINE 2] OUTER LOOP — one iteration = one full "collect, then update" cycle
    # ========================================================================
    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            # [EXTRA] linearly shrink the learning rate as training progresses
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ====================================================================
        # [LINE 3] "Collect set of trajectories D_k by running policy pi_k"
        # ====================================================================
        for step in range(0, args.num_steps):
            global_step += args.num_envs
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
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

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
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # TD error / "surprise" at this timestep:
                # actual reward + discounted next value, minus what we predicted
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]

                # GAE: exponentially-weighted sum of TD errors going forward,
                # computed efficiently backward
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )

            # [LINE 4] the value function's regression TARGET — this stands
            # in for "rewards-to-go" in the pseudocode
            returns = advantages + values

        # flatten (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        # so all parallel environments' data gets mixed together into one
        # big pool for minibatching
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            # reshuffle which data goes in which minibatch every epoch —
            # standard SGD practice, prevents the network from learning
            # spurious orderings
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
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
                    approx_kl = ((ratio - 1) - logratio).mean()  # [EXTRA] for target_kl early stop

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ---- [LINE 6] THE PPO-CLIP OBJECTIVE ITSELF ----
                # Spinning Up's formula:
                #   L = min( ratio * A_hat,  clip(ratio, 1-eps, 1+eps) * A_hat )
                # We want to MAXIMIZE this, but PyTorch optimizers MINIMIZE,
                # so everything below is negated (hence -mb_advantages and
                # torch.max instead of torch.min — negating min(a,b) gives
                # max(-a,-b))
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ---- [LINE 8] VALUE FUNCTION MSE REGRESSION ----
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # [EXTRA] clipped value loss — not in pseudocode, but
                    # mirrors the policy clip: don't let the value estimate
                    # move too far from its old value in one update
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # [EXTRA] clip gradients so no single update is too violent
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # [EXTRA] optional early stopping if policy is changing too fast
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        print(f"iteration {iteration}/{args.num_iterations} done, global_step={global_step}")

    envs.close()


if __name__ == "__main__":
    train()