import numpy as np
import gymnasium as gym
from gymnasium.core import ObservationWrapper, ActionWrapper


class Float32ActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(Float32ActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.float32(self.env.action_space.low),
            high=np.float32(self.env.action_space.high),
            shape=self.env.action_space.shape,
            dtype=np.float32
        )

    def action(self, act):
        return act.astype(np.float32)


class Float32ObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(Float32ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.float32(self.env.observation_space.low),
            high=np.float32(self.env.observation_space.high),
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32)
