import gymnasium as gym

import numpy as np

import RLEnvs.MyMiniGrid.MyKeyDoorEnvs
import RLEnvs.MyMiniWorld.pickupobjects

from minigrid.wrappers import *

from RLEnvs.MyMiniGrid.Wrappers import NormalRevChannelWrapper, FloatObservationWrapper, MovetoFourDirectionsWrapper, \
    AutoPickUpKeyOpenDoorOne, RemoveUnusedKeyDoorActionWrapper
from RLEnvs.MyMiniWorld.Wrappers import RGBImgObsRevChannelWrapper, RemovePickUpActionWrapper

import safety_gymnasium
from safety_gymnasium.wrappers.gymnasium_conversion import SafetyGymnasium2Gymnasium
from safety_gymnasium.wrappers.self_defined import Float32ActionWrapper, Float32ObservationWrapper


def minigrid_env_maker(env_id, seed=1, render=False):
    """
    Make the MiniGrid environment.
    The agent will move to four directions, the key and door are automatically handled.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = FullyObsWrapper(env)  # Fully observable gridworld instead of the agent view
    env = NormalRevChannelWrapper(env)  # change the channel order to [channel, width, height]
    # change the date type of the observation space to float32
    env = FloatObservationWrapper(env)
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field

    # env = RGBImgObsWrapper(env, tile_size=8)  # use fully observable RGB image as observation
    # env = RGBImgPartialObsWrapper(env, tile_size=8)  # use partially observable RGB image as observation
    # env = SymbolicObsWrapper(env)  # fully observable grid with symbolic state representations (not RGB image)
    # env = ViewSizeWrapper(env,agent_view_size=7)    # set the view size of the agent

    # env = AgentLocation(env)  # add the agent location to the `info` with the key `agent_loc`
    env = MovetoFourDirectionsWrapper(env)  # change the action space to make the agent move to four directions directly
    env = AutoPickUpKeyOpenDoorOne(env)  # make the agent to automatically pick up the key once it steps on it
    # env = RemoveUnusedKeyDoorActionWrapper(env)

    return env


def miniworld_env_maker(env_id, seed=1, render=False):
    """
    Make the MiniWorld environment.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    # reverse the channel order to [channel, height, width]
    env = RGBImgObsRevChannelWrapper(env)
    # env = MiniWorldFloatObs(env)  # change the data type to float32, and normalize the observation to [0, 1]
    env = RemovePickUpActionWrapper(env)  # remove the pickup action from the action space

    return env


def car_navigation_env_maker(env_id, config, seed=1, render=False):
    """
    Make the Car Navigation environment.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    # env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    if render:
        env = safety_gymnasium.make(env_id, render_mode="human", config={"goal_xy": config["goal_xy"],
                                                                         "reward_type": "sparse",
                                                                         "agent_xy": config["agent_xy"],
                                                                         "agent_rot": 1})
    else:
        env = safety_gymnasium.make(env_id, config={"goal_xy": config["goal_xy"],
                                                    "reward_type": "sparse",
                                                    "agent_xy": config["agent_xy"],
                                                    "agent_rot": 1})

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = SafetyGymnasium2Gymnasium(env)
    env = Float32ActionWrapper(env)
    env = Float32ObservationWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
