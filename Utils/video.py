"""
render_bipedal.py
-----------------
Loads a saved PPO checkpoint and renders BipedalWalker-v3 to an MP4.

Usage:
    python render_bipedal.py

Edit the CONFIG block below to match your setup.
"""

import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "BipedalWalker/2026-06-22_17-03-31/models/ppo_step_204800.pt"  # ← path to your .pt file
OUTPUT_DIR      = "renders"                           # folder for the MP4
NUM_EPISODES    = 3                                   # how many episodes to record
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─── AGENT ────────────────────────────────────────────────────────────────────
# Paste your Agent class here (or import it from your training file).
# It must have a get_action(obs) method or equivalent.
#
# Example stub — replace with your real class:

import torch.nn as nn
from torch.distributions import Normal

class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_action(self, obs):
        mean = self.actor_mean(obs)
        std  = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist.mean  # deterministic at eval time

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build env with video recording wrapper
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=OUTPUT_DIR,
        name_prefix="bipedal",
        episode_trigger=lambda ep: True,   # record every episode
    )

    obs_dim = env.observation_space.shape[0]   # 24
    act_dim = env.action_space.shape[0]        # 4

    # Load checkpoint
    agent = Agent(obs_dim, act_dim).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    print(f"Loaded checkpoint from step {checkpoint.get('global_step', '?')}")
    print(f"Running {NUM_EPISODES} episode(s)...")

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_t).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = ep_return  # keep mypy happy
            done = terminated or truncated
            ep_return += reward
            steps += 1

        print(f"  Episode {ep + 1}: return={ep_return:.1f}, steps={steps}")

    env.close()
    print(f"\nVideos saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Open with: vlc renders/bipedal-episode-0.mp4")


if __name__ == "__main__":
    main()