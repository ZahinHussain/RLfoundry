import time

from RLfoundry.Utils.run_startup import run_startup
import gymnasium as gym
from  gymnasium.wrappers import AtariPreprocessing ,FrameStackObservation
import torch
import numpy as np
from distributed import C51Distribution
from network import C51NoisyDuelingCNN, C51NoisyDuelingMLP
from datetime import datetime

from replay_buffer import NStepPrioritizedReplayBuffer
import time
import os
from datetime import datetime

import torch.optim as optim
from matplotlib import pyplot as plt



def train(cfg,env,path):

    # run startup
    path = run_startup(path, cfg)

    SAVE_INTERVAL_SECONDS = int(cfg.interval_save) * 60
    last_numpy_save_t = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")