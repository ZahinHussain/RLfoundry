import gymnasium as gym
import numpy as np
import inspect


def make_env(env_id, gamma, seed=None):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space  # required in Gym 1.x
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10)
        )

        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

        print(inspect.signature(gym.wrappers.TransformObservation))
        print(inspect.signature(gym.wrappers.TransformReward))

        return env



    return thunk


def make_vec_env(env_id, num_envs, gamma, seed):
    return gym.vector.SyncVectorEnv(
        [
            make_env(env_id=env_id, gamma=gamma, seed=seed + i)
            for i in range(num_envs)
        ]
    )