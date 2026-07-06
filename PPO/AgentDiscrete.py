import torch
import torch.nn as nn
import numpy as np
# CHANGE 1: Import Categorical instead of Normal
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]  # e.g. 4 for CartPole


        act_dim = env.action_space[0].n


        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # CHANGE 3: Single actor network that outputs raw logits for each action choice
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        # CHANGE 4: Removed self.actor_logstd (not needed for discrete)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # CHANGE 5: Use Categorical distribution with logits
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()


        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x)


        if was_1d:
            action = action.squeeze(0)
            value = value.squeeze(0)

        return action, logprob, entropy, value