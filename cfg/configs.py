"""
RLFoundry structured configs — flat, no shared base class.

Each algo's dataclass is fully self-contained (mirrors the original YAML
field-for-field, bugs fixed). Load one, get back an OmegaConf DictConfig,
pass it straight into your existing train(cfg, env, path) — cfg behaves
exactly like the cfg Hydra was already handing you from YAML, so nothing
downstream needs to change.
"""

from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore


# ============================================================================
# PPO
# ============================================================================

@dataclass
class PPOConfig:
    exp_name: str = "ppo"
    algo_name: str = "ppo"
    env_id: str = "default"          # set per-environment at load time

    learning_rate: float = 3e-4
    seed: int = 67
    gamma: float = 0.99
    total_timesteps: int = 1_024_000
    torch_deterministic: bool = True

    track: bool = True
    wandb_project_name: str = "default"
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False

    num_steps: int = 2048
    num_envs: int = 1
    num_iterations: int = 500
    anneal_lr: bool = True
    gae_lambda: float = 0.95
    batch_size: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 128
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    interval_save: int = 30
    eps: float = 1e-5


# ============================================================================
# RPO
# ============================================================================

@dataclass
class RPOConfig:
    exp_name: str = "rpo"
    algo_name: str = "rpo"
    env_id: str = "SustainGym-BuildingHvac-v0"

    learning_rate: float = 3e-4
    seed: int = 67
    gamma: float = 0.99
    total_timesteps: int = 1_024_000
    torch_deterministic: bool = True

    track: bool = True
    wandb_project_name: str = "SustainGym-BuildingHvac-v0-rpo"
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False

    num_steps: int = 2048
    num_envs: int = 1
    num_iterations: int = 500
    anneal_lr: bool = True
    gae_lambda: float = 0.95
    batch_size: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 128
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    interval_save: int = 30
    eps: float = 1e-5
    rpo_alpha: float = 0.5
    cuda: bool = True


# ============================================================================
# SAC
# ============================================================================

@dataclass
class SACConfig:
    exp_name: str = "sac"
    algo_name: str = "sac"
    env_id: str = "default"

    seed: int = 67
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = True
    wandb_project_name: str = "default"
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False
    capture_video: bool = False

    num_envs: int = 1
    total_timesteps: int = 1_024_000
    learning_starts: int = 5000
    batch_size: int = 256

    buffer_size: int = 1_000_000

    gamma: float = 0.99
    tau: float = 0.005

    policy_lr: float = 3e-4
    q_lr: float = 1e-3

    policy_frequency: int = 2
    target_network_frequency: int = 1

    alpha: float = 0.2
    autotune: bool = True

    interval_save: int = 30


# ============================================================================
# TD3
# ============================================================================

@dataclass
class TD3Config:
    exp_name: str = "td3"
    algo_name: str = "td3"
    seed: int = 67                    # was 1 in source — flagged as inconsistent, fix here if deliberate
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = True                # was False in source — see chat
    wandb_project_name: str = "default"  # was "RLFoundry" in source — see chat
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "default"           # was "Hopper-v4" (stale CleanRL default) — see chat
    num_envs: int = 1

    total_timesteps: int = 1_024_000  # was 1_000_000 — aligned with others; override if intentional
    learning_rate: float = 3e-4
    learning_starts: int = 25000
    batch_size: int = 256

    buffer_size: int = 1_000_000

    gamma: float = 0.99
    tau: float = 0.005

    policy_noise: float = 0.2
    exploration_noise: float = 0.1
    policy_frequency: int = 2
    noise_clip: float = 0.5

    interval_save: int = 5


# ============================================================================
# DQN
# ============================================================================
# NOTE: built from standard (Rainbow) DQN / CleanRL-style fields to match the
# shape of your other four configs — I don't have your actual dqn.yaml.
# Check the DQN-specific block against your real file and override in the
# load_cfg() call (or edit the defaults here) if anything's off.

@dataclass
class DQNConfig:
    exp_name: str = "dqn"
    algo_name: str = "dqn"
    env_id: str = "default"           # set per-environment at load time
    gym_id: str = "default"           # legacy field seen in your script — kept for compatibility

    seed: int = 67
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = True
    wandb_project_name: str = "default"
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False
    capture_video: bool = False

    num_envs: int = 1
    total_timesteps: int = 1_024_000
    learning_rate: float = 2.5e-4

    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    train_frequency: int = 4
    batch_size: int = 128

    gamma: float = 0.99
    tau: float = 1.0                  # 1.0 = hard target update
    target_network_frequency: int = 500

    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5

    interval_save: int = 30

# ============================================================================
# Loader — this is the part that plugs into your existing code
# ============================================================================

_CONFIGS = {
    "ppo": PPOConfig,
    "rpo": RPOConfig,
    "sac": SACConfig,
    "td3": TD3Config,
    "dqn": DQNConfig,

}


def load_cfg(algo_name: str, **overrides) -> DictConfig:
    """
    Build a DictConfig from a dataclass's defaults, with optional overrides.
    Returns the same DictConfig type Hydra would've handed train(cfg, env, path)
    from a YAML file — no downstream code needs to change.

    Example:
        cfg = load_cfg("ppo", env_id="BipedalWalker-v3", seed=42)
        train(cfg, env, path)
    """
    if algo_name not in _CONFIGS:
        raise ValueError(f"Unknown algo '{algo_name}'. Choose from {list(_CONFIGS)}")

    cfg = OmegaConf.structured(_CONFIGS[algo_name]())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


if __name__ == "__main__":
    # No YAML, no CLI, no @hydra.main — just call your train() directly.
    cfg = load_cfg("ppo", env_id="BipedalWalker-v3", seed=42)
    print(OmegaConf.to_yaml(cfg))
    assert cfg.target_kl is None
    assert isinstance(cfg.gamma, float) and cfg.gamma == 0.99
    print("cfg.env_id ->", cfg.env_id, " cfg.seed ->", cfg.seed)

    # Equivalent for every other algo:
    # cfg = load_cfg("sac", env_id="ICU-Sepsis-v0")
    # cfg = load_cfg("td3", env_id="RWARE-v0", total_timesteps=2_000_000)