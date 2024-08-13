"""
Some self-defined wrappers for the MiniWorld environment.
"""

import gymnasium as gym

from gymnasium.core import ObservationWrapper, ActionWrapper

import numpy as np


class RGBImgObsRevChannelWrapper(ObservationWrapper):
    """
    The wrapper to reverse the shape of observation from (h,w,c) to (c,h,w).
    """

    def __init__(self, env):
        super(RGBImgObsRevChannelWrapper, self).__init__(env)
        # define the new observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1]),
            dtype=np.uint8
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class MiniWorldFloatObs(ObservationWrapper):
    """
    The wrapper to change the data type of the observation space to float32.
    """

    def __init__(self, env):
        super(MiniWorldFloatObs, self).__init__(env)
        # define the new observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return observation / 255.0


class RemovePickUpActionWrapper(ActionWrapper):
    """
    The wrapper to remove the pickup action from the action space.
    """

    def __init__(self, env):
        super(RemovePickUpActionWrapper, self).__init__(env)
        # define the new action space
        self.action_space = gym.spaces.Discrete(env.action_space.n - 1)

    def action(self, act):
        return act