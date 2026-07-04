import time
import os
import torch
import torch.nn as nn
import torch.optim as optim

from RLfoundry.Utils.run_startup import run_startup
from RLfoundry.DQN.rainbowdqn_agent import DQN
import numpy as np


def train(cfg, env, path):
    path = run_startup(path, cfg)

    config = {
        "batch_size": getattr(cfg, "batch_size", 32),
        "gamma": getattr(cfg, "gamma", 0.99),
        "eps_start": getattr(cfg, "eps_start", 1.0),
        "eps_end": getattr(cfg, "eps_end", 0.01),
        "eps_decay": getattr(cfg, "eps_decay", 0.995),
        "tau": getattr(cfg, "tau", 0.005),
        "lr": getattr(cfg, "learning_rate", 3e-4),
        "env_name": env.spec.id,
    }

    agent = DQN(config)
    agent.train(max_frames=cfg.total_timesteps, verbose=True, run_name=cfg.exp_name)

    # Cude and device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")


    else:
        device = torch.device("cpu")
    agent = Agent(env).to(device)
