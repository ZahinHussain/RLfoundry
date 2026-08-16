import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    actor_mean + actor_logstd in ONE module, so actor.parameters() yields
    exactly theta and nothing else - same as CleanRL's Actor. That is the
    whole reason this is a separate class: CG, the Fisher-vector product and
    update_actor_params() all operate on a flat vector of theta, and a stray
    critic weight in that vector would get a trust-region step applied to it.
    """

    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        # (act_dim,) not (1, act_dim): single env passes obs unbatched, so
        # action_mean is 1D and expand_as from a 2D source raises. 1D expands
        # against both (act_dim,) and (N, act_dim).
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # sum(-1) not sum(1): unbatched obs has no dim 1.
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.critic(x)

    def get_value(self, x):
        return self.critic(x)


class Agent(nn.Module):
    """
    Thin container so train() keeps one object and one state_dict.
    agent.actor.parameters() -> theta
    agent.critic.parameters() -> phi
    agent.actor(x) -> Normal
    """

    def __init__(self, env):
        super().__init__()
        self.actor = Actor(env)
        self.critic = Critic(env)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action, logprob, entropy = self.actor.get_action(x, action)
        return action, logprob, entropy, self.critic(x)