from RLFoundry.PPO import PPO_Continuous as ppo

from RLFoundry.env.env import make_env
import os
import gymnasium as gym
print(gym.__version__)
from omegaconf import OmegaConf


cfg = OmegaConf.load("ppo.yaml")

cfg.exp_name = "BipedalWalker"
cfg.learning_rate = 3e-4
cfg.seed = 42
cfg.total_timesteps = 10_000
cfg.gym_id ="BipedalWalker-v3"



path = os.path.dirname(os.path.abspath(__file__))

ppo.train(cfg,path)



