
import gymnasium as gym






def apply_wrappers(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    return env