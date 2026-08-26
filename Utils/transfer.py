"""
Transfer-learning helpers for RLfoundry PPO.

Drop this in RLfoundry/Utils/transfer.py and import from the transfer PPO
training file:

    from RLfoundry.Utils.transfer import (
        load_pretrained, apply_freeze, trainable_params, evaluate,
    )

Supports four conditions via cfg.transfer_mode:
    scratch   - random init, normal training (the baseline)
    zeroshot  - load checkpoint, no gradient steps, evaluate only
    finetune  - load checkpoint, train everything
    freeze    - load checkpoint, train only the final actor layer (+ critic)
"""

import numpy as np
import torch
import torch.nn as nn

_STATE_KEYS = ("agent", "agent_state_dict", "model", "model_state_dict", "state_dict")


def _extract_agent_sd(ckpt):
    """Pull the Agent weights out of whatever save_checkpoint() wrote."""
    if not isinstance(ckpt, dict):
        raise TypeError(f"unexpected checkpoint type: {type(ckpt)}")
    for k in _STATE_KEYS:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # already a bare state_dict
    if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError(
        f"could not find agent weights in checkpoint. top-level keys: {list(ckpt)}"
    )


def load_pretrained(agent, path, device, load_critic=True, strict_dims=False):
    """Load source-task weights into `agent`, skipping any tensor whose name or
    shape does not match (i.e. different obs_dim / act_dim between tasks).

    Returns (raw_checkpoint, skipped_keys). Always check skipped_keys: if the
    first trunk layer is in there, almost nothing transferred.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    src = _extract_agent_sd(ckpt)

    if not load_critic:
        src = {k: v for k, v in src.items() if not k.startswith("critic.")}

    tgt = agent.state_dict()
    ok = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    skipped = sorted(set(src) - set(ok))

    if strict_dims and skipped:
        raise RuntimeError(f"[transfer] dim mismatch on: {skipped}")

    agent.load_state_dict(ok, strict=False)

    print(f"[transfer] loaded {len(ok)}/{len(src)} tensors from {path}")
    if skipped:
        print(f"[transfer] SKIPPED (name/shape mismatch, randomly init'd): {skipped}")
    return ckpt, skipped


def apply_freeze(agent, train_critic=True, train_logstd=True):
    """Freeze everything except the final actor layer.

    The critic stays trainable by default: the value function is specific to the
    target task's reward scale, so a frozen critic gives systematically wrong
    advantages and the comparison stops being about the policy.
    """
    for p in agent.parameters():
        p.requires_grad_(False)

    last_linear = [m for m in agent.actor_mean if isinstance(m, nn.Linear)][-1]
    last_linear.requires_grad_(True)

    if train_logstd:
        agent.actor_logstd.requires_grad_(True)
    if train_critic:
        agent.critic.requires_grad_(True)

    frozen = [n for n, p in agent.named_parameters() if not p.requires_grad]
    print(f"[transfer] frozen tensors: {frozen}")


def trainable_params(agent):
    return [p for p in agent.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(agent, env, cfg, device, n_episodes=30, deterministic=True):
    """Deterministic rollouts using the actor mean. Used for the zero-shot
    condition, and should also be called at the end of the other three so every
    condition is scored the same way.
    """
    returns, lengths = [], []
    for ep in range(n_episodes):
        obs_np, _ = env.reset(seed=int(cfg.seed) + 10_000 + ep)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            if deterministic:
                action = agent.actor_mean(obs_t.unsqueeze(0)).squeeze(0)
            else:
                action, _, _, _ = agent.get_action_and_value(obs_t)
            action_np = action.cpu().numpy().reshape(-1)
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            obs_np, reward, term, trunc, _ = env.step(action_np)
            ep_ret += float(reward)
            ep_len += 1
            done = bool(term or trunc)

        returns.append(ep_ret)
        lengths.append(ep_len)

    returns = np.asarray(returns)
    print(
        f"[eval] n={n_episodes} mean={returns.mean():.2f} "
        f"median={np.median(returns):.2f} std={returns.std():.2f}"
    )
    return returns, np.asarray(lengths)
