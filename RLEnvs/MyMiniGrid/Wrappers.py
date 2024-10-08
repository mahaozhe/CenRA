"""
Some self-defined wrappers for the MiniGrid environment.
"""

import gymnasium as gym

from gymnasium.core import ObservationWrapper, ActionWrapper

import numpy as np


class RemoveUnusedKeyDoorActionWrapper(ActionWrapper):
    """
    The wrapper to remove the unused actions (original action 4 and 6),
    make the original action 5 as the new action 4.
    """

    def __init__(self, env):
        super(RemoveUnusedKeyDoorActionWrapper, self).__init__(env)
        env.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        # action 4 - open the door
        if action == 4:
            action = 5

        return action


class AgentLocation(gym.core.Wrapper):
    """
    The wrapper to indicate the location of the agent in the `info`.
    """

    def __init__(self, env):
        super(AgentLocation, self).__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        agent_loc = self.unwrapped.agent_pos

        info["agent_loc"] = tuple(agent_loc)

        return observation, reward, terminated, truncated, info


class MovetoFourDirectionsWrapper(gym.core.Wrapper):
    """
    The wrapper to modify the action space to `Discrete(4)`,
    making the agent only moves to four directions for one step:

    original actions:
    * 0 - turn left
    * 1 - turn right
    * 2 - move forward

    new mapped actions:
    * 0 - move forward
        2*
    * 1 - move to left
        0 2* 1
    * 2 - move to right
        1 2* 0
    * 3 - move backward
        0 0 2* 1 1

    Note: after stepping original action 2, need to record the `reward` and to check `done`.
    """

    def __init__(self, env):
        super(MovetoFourDirectionsWrapper, self).__init__(env)
        env.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        # action 0 - move forward
        if action == 0:
            return self.env.step(2)

        # action 1 - move to left
        if action == 1:
            _, rewards, _, _, _ = self.env.step(0)

            o, r, te, tr, i = self.env.step(2)
            rewards += r

            self.unwrapped.step_count -= 1
            if te or tr:
                return o, rewards, te, tr, i

            o, r, te, tr, i = self.env.step(1)
            rewards += r

            self.unwrapped.step_count -= 1
            return o, rewards, te, tr, i

        # action 2 - move to right
        if action == 2:
            _, rewards, _, _, _ = self.env.step(1)

            o, r, te, tr, i = self.env.step(2)
            rewards += r

            self.unwrapped.step_count -= 1
            if te or tr:
                return o, rewards, te, tr, i

            o, r, te, tr, i = self.env.step(0)
            rewards += r

            self.unwrapped.step_count -= 1
            return o, rewards, te, tr, i

        # action 3 - move backward
        if action == 3:
            _, rewards, _, _, _ = self.env.step(0)

            _, r, _, _, _ = self.env.step(0)
            rewards += r

            o, r, te, tr, i = self.env.step(2)
            rewards += r

            self.unwrapped.step_count -= 2
            if te or tr:
                return o, rewards, te, tr, i

            _, r, _, _, _ = self.env.step(1)
            rewards += r

            _, r, _, _, _ = self.env.step(1)
            rewards += r

            self.unwrapped.step_count -= 2
            return o, rewards, te, tr, i


class AutoPickUpKeyOpenDoorOne(gym.core.Wrapper):
    """
    The wrapper to make the agent to automatically pick up the key once it steps on it.
    Once the agent picks up the key, the door will be set to open for it to pass through.
    Assume there is only one key and one door in the environment.
    """

    def __init__(self, env):
        super(AutoPickUpKeyOpenDoorOne, self).__init__(env)

        self.key_picked = False

    def reset(self, **kwargs):
        self.key_picked = False

        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped

        agent_loc = env.agent_pos

        # get the key and the door locations
        key_locs = env.key_locs
        door_locs = env.door_locs

        # check if the agent steps on the key, if so, auto-pick up the key, and open the door
        if not self.key_picked and agent_loc == key_locs:
            self.key_picked = True
            self.grid.set(key_locs[0], key_locs[1], None)
            door_obj = env.grid.get(door_locs[0], door_locs[1])
            door_obj.is_locked = False
            door_obj.is_open = True

        return obs, reward, terminated, truncated, info


class AutoPickUpKeyOpenDoor(gym.core.Wrapper):
    """
    The wrapper to make the agent to automatically pick up the key once it steps on it.
    Once the agent picks up the key, the door will be set to open for it to pass through.
    """

    def __init__(self, env):
        super(AutoPickUpKeyOpenDoor, self).__init__(env)

        self.key_num = len(env.unwrapped.key_locs)
        self.key_picked = [False] * self.key_num

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.key_picked = [False] * self.key_num

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped

        agent_loc = env.agent_pos

        # get the key and the door locations
        key_locs = env.key_locs
        door_locs = env.door_locs

        # check if the agent steps on the key, if so, auto-pick up the key, and open the door
        for i in range(self.key_num):
            if not self.key_picked[i] and agent_loc == key_locs[i]:
                self.key_picked[i] = True
                self.grid.set(key_locs[i][0], key_locs[i][1], None)
                door_obj = env.grid.get(door_locs[i][0], door_locs[i][1])
                door_obj.is_locked = False
                door_obj.is_open = True

        return obs, reward, terminated, truncated, info


class RGBImgObsRevChannelWrapper(ObservationWrapper):
    """
    The wrapper to use fully observable RGB image as observation,
    with the channel order as [channel, width, height].
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                3,
                self.unwrapped.width * tile_size,
                self.unwrapped.height * tile_size,
            ),
            dtype="uint8",
        )

        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces, "image": new_image_space})

    def observation(self, obs):
        rgb_image = self.get_frame(highlight=self.unwrapped.highlight, tile_size=self.tile_size)

        # change the channel order to [channel, width, height]
        rgb_image = rgb_image.transpose(2, 0, 1)

        return {**obs, "image": rgb_image}


class NormalRevChannelWrapper(ObservationWrapper):
    """
    The wrapper to transpose the channel from [W,H,C] to [C,W,H], for normal observation.
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space.spaces["image"].shape

        new_image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, obs_shape[0], obs_shape[1]),
            dtype="uint8",
        )

        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces, "image": new_image_space})

    def observation(self, obs):
        reversed_grid = obs["image"].transpose(2, 0, 1)

        return {**obs, "image": reversed_grid}


class FloatObservationWrapper(ObservationWrapper):
    """
    The wrapper to change the date type of the observation space to float32.
    """

    def __init__(self, env):
        super(FloatObservationWrapper, self).__init__(env)

        obs_space = env.observation_space.spaces["image"]
        low = obs_space.low
        high = obs_space.high

        new_float_obs_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces, "image": new_float_obs_space})

    def observation(self, obs):
        return {**obs, "image": obs["image"].astype("float32")}
