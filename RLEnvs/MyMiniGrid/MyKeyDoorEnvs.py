from gymnasium.envs.registration import register

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Door, Key
from minigrid.core.mission import MissionSpace

AvailableColors = ["yellow", "blue", "green", "red", "purple"]


###### Type A: agent needs to pick up the key to open the door by himself ######

class MyDoorKeyEnvTypeA(MiniGridEnv):
    """
    Environment with a key (keys) and the corresponding door (doors).
    """

    def __init__(self, width, height, obj_lists, max_steps, agent_init_loc=None, goal_loc=None, **kwargs):
        self.obj_lists = obj_lists
        self._agent_default_pos = agent_init_loc if agent_init_loc is not None else (1, 1)
        self._goal_default_pos = goal_loc if goal_loc is not None else (width - 2, height - 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super(MyDoorKeyEnvTypeA, self).__init__(mission_space=mission_space, width=width, height=height,
                                                max_steps=max_steps,
                                                **kwargs)

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # to create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = 0

        self.put_obj(Goal(), *self._goal_default_pos)

        for wall in self.obj_lists['walls']:
            self.grid.set(wall[0], wall[1], Wall())

        for d in range(len(self.obj_lists['doors'])):
            self.put_obj(Door(AvailableColors[d], is_locked=True),
                         self.obj_lists['doors'][d][0], self.obj_lists['doors'][d][1])

        for k in range(len(self.obj_lists['keys'])):
            self.place_obj(obj=Key(AvailableColors[k]),
                           top=(self.obj_lists['keys'][k][0], self.obj_lists['keys'][k][1]),
                           size=(1, 1))


####### Type B: keys and doors are automatically picked up and opened #######

# + re-write the Key to be able to overlap
class MyKey(Key):
    def __init__(self, color="yellow"):
        super().__init__(color)

    def can_overlap(self):
        return True


class MyDoorKeyEnvTypeB(MiniGridEnv):
    """
    Environment with one key and one door, the key is auto picked and the door is auto open.
    """

    def __init__(self, width, height, obj_lists, max_steps, agent_init_loc=None, goal_loc=None, **kwargs):
        self.obj_lists = obj_lists
        self._agent_default_pos = agent_init_loc if agent_init_loc is not None else (1, 1)
        self._goal_default_pos = goal_loc if goal_loc is not None else (width - 2, height - 2)

        self.key_locs = obj_lists['keys'][0] if len(obj_lists['keys']) > 0 else []
        self.door_locs = obj_lists['doors'][0] if len(obj_lists['doors']) > 0 else []

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super(MyDoorKeyEnvTypeB, self).__init__(mission_space=mission_space, width=width, height=height,
                                                max_steps=max_steps,
                                                **kwargs)

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # to create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = 0

        self.put_obj(Goal(), *self._goal_default_pos)

        for wall in self.obj_lists['walls']:
            self.grid.set(wall[0], wall[1], Wall())

        for d in range(len(self.obj_lists['doors'])):
            self.put_obj(Door(AvailableColors[d], is_locked=True),
                         self.obj_lists['doors'][d][0], self.obj_lists['doors'][d][1])

        for k in range(len(self.obj_lists['keys'])):
            self.place_obj(obj=MyKey(AvailableColors[k]),
                           top=(self.obj_lists['keys'][k][0], self.obj_lists['keys'][k][1]),
                           size=(1, 1))


wall_type1 = {"walls": [(6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10)],
              "doors": [(6, 3)],
              "keys": [(2, 8)]}

wall_type2 = {"walls": [(1, 4), (2, 4), (3, 4), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6)],
              "doors": [(4, 4)],
              "keys": [(6, 2)]}

wall_type3 = {"walls": [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 7), (8, 7), (9, 7), (10, 7)],
              "doors": [(7, 7)],
              "keys": [(2, 9)]}

wall_type4 = {"walls": [(6, 5), (6, 5), (6, 6), (6, 8), (6, 9), (6, 10), (7, 4), (8, 4), (9, 4), (10, 4)],
              "doors": [(6, 7)],
              "keys": [(7, 2)]}

wall_type5 = {"walls": [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 5), (9, 5), (10, 5)],
              "doors": [(8, 5)],
              "keys": [(3, 8)]}

register(
    id='MiniGrid-Type1',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnvTypeB',
    kwargs={"width": 12, "height": 12, "obj_lists": wall_type1, "max_steps": 500}
)

register(
    id='MiniGrid-Type2',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnvTypeB',
    kwargs={"width": 12, "height": 12, "obj_lists": wall_type2, "max_steps": 500, "goal_loc": (5, 10)}
)

register(
    id='MiniGrid-Type3',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnvTypeB',
    kwargs={"width": 12, "height": 12, "obj_lists": wall_type3, "max_steps": 500, "goal_loc": (10, 4)}
)

register(
    id='MiniGrid-Type4',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnvTypeB',
    kwargs={"width": 12, "height": 12, "obj_lists": wall_type4, "max_steps": 500}
)

register(
    id='MiniGrid-Type5',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnvTypeB',
    kwargs={"width": 12, "height": 12, "obj_lists": wall_type5, "max_steps": 500, "goal_loc": (10, 1)}
)
