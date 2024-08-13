import gymnasium as gym
from gymnasium import spaces, utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key, MeshEnt
from miniworld.miniworld import MiniWorldEnv

import numpy as np


class PickupObjects2(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +1 when agent picked up object

    ## Arguments

    ```python
    PickupObjects(size=12, num_objs=5)
    ```

    `size`: size of world

    `num_objs`: number of objects

    """

    def __init__(self, env_size=12, num_objs=1, obj_size=1.0, en_rand_obj_size=False, obj_pos=(10, 1), agent_pos=(1, 1),
                 threshold=1.5, obj_type="ball", obj_color="red", **kwargs):
        assert env_size >= 2
        self.env_size = env_size
        self.num_objs = num_objs
        if type(obj_size) == dict:
            self.obj_size = obj_size
        elif type(obj_size) == float:
            self.obj_size = {"ball": obj_size, "box": obj_size, "key": obj_size}
        self.en_rand_obj_size = en_rand_obj_size
        if self.en_rand_obj_size:
            print("Random object size enabled")

        self.obj_pos = obj_pos
        self.agent_pos = agent_pos

        self.threshold = threshold
        self.obj_type = obj_type
        self.obj_color = obj_color

        MiniWorldEnv.__init__(self, max_episode_steps=400, **kwargs)
        utils.EzPickle.__init__(self, env_size, num_objs, **kwargs)

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup + 1)

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.env_size,
            min_z=0,
            max_z=self.env_size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        # + only use one Ball object at a specific location

        if self.obj_type == "ball":
            self.place_entity(Ball(color=self.obj_color, size=1.0), pos=[self.obj_pos[0], 0, self.obj_pos[1]])
        elif self.obj_type == "cube":
            self.place_entity(Box(color=self.obj_color, size=1.0), pos=[self.obj_pos[0], 0, self.obj_pos[1]])
        elif self.obj_type == "key":
            self.place_entity(Key(color=self.obj_color), pos=[self.obj_pos[0], 0, self.obj_pos[1]])
        else:
            self.place_entity(MeshEnt(mesh_name="medkit", height=0.40, static=False))

        self.place_agent(dir=5.5, min_x=self.agent_pos[0], max_x=self.agent_pos[0], min_z=self.agent_pos[1],
                         max_z=self.agent_pos[1])

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # check if the agent is near enough to the object
        dist = np.linalg.norm(self.entities[1].pos - self.entities[0].pos)

        if dist < self.threshold:
            self.entities.remove(self.entities[0])
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_objs:
                termination = True

        return obs, reward, termination, truncation, info


gym.register(
    id="MiniWorld-3DPickup-ball-red",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 8, "obj_pos": (6, 6), "agent_pos": (1, 1), "obj_type": "ball", "obj_color": "red"},
)

gym.register(
    id="MiniWorld-3DPickup-cube-green",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 8, "obj_pos": (3, 6), "agent_pos": (1, 1), "obj_type": "cube", "obj_color": "green"},
)

gym.register(
    id="MiniWorld-3DPickup-key-blue",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 8, "obj_pos": (6, 4), "agent_pos": (1, 1), "obj_type": "key", "obj_color": "blue"},
)

gym.register(
    id="MiniWorld-3DPickup-healthkit",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 8, "obj_pos": (1, 5), "agent_pos": (1, 1), "obj_type": "healthkit"},
)

gym.register(
    id="MiniWorld-3DPickup-cube-yellow",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 8, "obj_pos": (5, 3), "agent_pos": (1, 1), "obj_type": "cube", "obj_color": "yellow"},
)
