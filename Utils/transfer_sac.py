"""
SAC-specific transfer utilities.

Mirrors RLfoundry.Utils.transfer, which was written against the single-module
PPO `Agent` and a {"agent_state_dict": ...} checkpoint. SAC has three separate
loadable modules (actor, qf1, qf2) plus two targets that must be re-synced
AFTER loading, so it needs its own loader.

Checkpoint layout assumed (matches save_checkpoint in sac_continuous_action.py):
    actor_state_dict, qf1_state_dict, qf2_state_dict,
    qf1_target_state_dict, qf2_target_state_dict, log_alpha, global_step
"""

import numpy as np
import torch


# ==============================================================================
# Layer names kept trainable under transfer_mode="freeze"
# ("freeze all but last layer" — the partial condition)
# Adjust these if SACNetwork.py uses different attribute names.
# ==============================================================================
ACTOR_HEAD_PREFIXES = ("fc_mean", "fc_logstd")
CRITIC_HEAD_PREFIXES = ("fc3",)


# ==============================================================================
# Loading
# ==============================================================================

def _filter_by_shape(module, source_sd):
    """Keep only keys that exist in `module` with a matching shape.

    Returns (filtered_sd, dropped_keys). Shape mismatches happen when the
    source and target tasks differ in obs/action dimensionality — those layers
    stay at their random init rather than crashing the load.
    """
    target_sd = module.state_dict()
    filtered, dropped = {}, []

    for k, v in source_sd.items():
        if k not in target_sd:
            dropped.append(f"{k} <not in target module>")
        elif target_sd[k].shape != v.shape:
            dropped.append(f"{k} <shape {tuple(v.shape)} != {tuple(target_sd[k].shape)}>")
        else:
            filtered[k] = v

    return filtered, dropped


def load_pretrained_sac(
    actor,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    checkpoint_path,
    device,
    load_critic=True,
    load_alpha=False,
):
    """Load a SAC checkpoint into the actor and (optionally) the twin critics.

    Target networks are always re-synced from the freshly loaded online critics,
    so this MUST be called before any manual `qfN_target.load_state_dict(...)`
    in the training script — otherwise the targets keep their random init and
    the Bellman bootstrap starts from noise.

    Returns (global_step, skipped_keys, log_alpha_value).
    `log_alpha_value` is None unless load_alpha=True and the checkpoint has one.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    modules = [("actor", actor)]
    if load_critic:
        modules += [("qf1", qf1), ("qf2", qf2)]

    skipped = []
    for name, module in modules:
        source_sd = ckpt.get(f"{name}_state_dict")
        if source_sd is None:
            skipped.append(f"{name}: <no '{name}_state_dict' key in checkpoint>")
            continue

        filtered, dropped = _filter_by_shape(module, source_sd)
        skipped.extend(f"{name}.{k}" for k in dropped)

        missing, unexpected = module.load_state_dict(filtered, strict=False)
        skipped.extend(f"{name}.{k} <missing, left at init>" for k in missing)
        skipped.extend(f"{name}.{k} <unexpected>" for k in unexpected)

        loaded = len(filtered)
        print(f"[transfer] {name}: loaded {loaded}/{len(module.state_dict())} tensors")

    # Targets always follow the online critics after loading.
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    print("[transfer] qf1_target/qf2_target re-synced from loaded critics")

    log_alpha_value = ckpt.get("log_alpha") if load_alpha else None
    if load_alpha and log_alpha_value is None:
        skipped.append("log_alpha: <not in checkpoint, using fresh init>")

    return ckpt.get("global_step"), skipped, log_alpha_value


# ==============================================================================
# Freezing (partial transfer condition)
# ==============================================================================

def apply_freeze_sac(actor, qf1=None, qf2=None, freeze_critic=True):
    """Freeze everything except the final layer of each module.

    Critics are only frozen if they were actually loaded — freezing a randomly
    initialised critic trunk would cripple the run rather than test transfer.
    """
    frozen = []

    for name, p in actor.named_parameters():
        if not name.startswith(ACTOR_HEAD_PREFIXES):
            p.requires_grad_(False)
            frozen.append(f"actor.{name}")

    if freeze_critic:
        for label, net in (("qf1", qf1), ("qf2", qf2)):
            if net is None:
                continue
            for name, p in net.named_parameters():
                if not name.startswith(CRITIC_HEAD_PREFIXES):
                    p.requires_grad_(False)
                    frozen.append(f"{label}.{name}")

    print(f"[transfer] froze {len(frozen)} parameter tensors")
    return frozen


# ==============================================================================
# Parameter helpers (multi-module versions)
# ==============================================================================

def trainable_params_multi(*modules):
    return [p for m in modules if m is not None
            for p in m.parameters() if p.requires_grad]


def count_params(*modules):
    """Returns (trainable, total) across all given modules."""
    total = sum(p.numel() for m in modules if m is not None for p in m.parameters())
    trainable = sum(p.numel() for p in trainable_params_multi(*modules))
    return trainable, total


# ==============================================================================
# Evaluation
# ==============================================================================

@torch.no_grad()
def evaluate_sac(actor, env, cfg, device, n_episodes, deterministic=True):
    """Roll out `n_episodes` full episodes and return (returns, lengths).

    Uses the deterministic action (the tanh-squashed mean, third return value
    of Actor.get_action) rather than a sample. A sampled policy adds exploration
    noise that makes the zero-shot condition look worse than the transferred
    policy actually is, which would bias the whole comparison.
    """
    max_steps = getattr(cfg, "eval_max_steps", None)
    rets, lens = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + 10_000 + ep)
        ep_return, ep_length, done = 0.0, 0, False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            sampled, _, mean = actor.get_action(obs_t)
            action = (mean if deterministic else sampled)
            action = action.cpu().numpy().squeeze(0)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminations, truncations, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = bool(terminations or truncations)

            if max_steps is not None and ep_length >= max_steps:
                break

        rets.append(ep_return)
        lens.append(ep_length)

    return rets, lens
