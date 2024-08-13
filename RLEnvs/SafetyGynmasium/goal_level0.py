# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Goal level 0."""

from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.common_utils import ResamplingError
import mujoco
import numpy as np


class GoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        self.goal_xy = config["goal_xy"]
        self.reward_type = config["reward_type"]
        self.agent_xy = config["agent_xy"]
        self.agent_rot = config["agent_rot"]

        config["agent_name"] = "Racecar"

        del (
            config["goal_xy"],
            config["reward_type"],
            config["agent_xy"],
            config["agent_rot"],
        )  # since extra keys are not allowed in the config, we have to delete them before calling super()
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(Goal(keepout=0.305))

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        if self.reward_type == "continuous":
            reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
            self.last_dist_goal = dist_goal
            if self.goal_achieved:
                reward += self.goal.reward_goal
        elif self.reward_type == "sparse":
            if self.goal_achieved:
                reward += self.goal.reward_goal
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal_position(self, goal_xy=None):
        """Build a new goal position, maybe with resampling due to hazards."""
        # Resample until goal is compatible with layout
        if 'goal' in self.world_info.layout:
            del self.world_info.layout['goal']
        if goal_xy == None:
            for _ in range(10000):  # Retries
                if self.random_generator.sample_goal_position():
                    break
            else:
                raise ResamplingError('Failed to generate goal')
        else:
            self.world_info.layout['goal'] = np.array(goal_xy)
        # print(f"Goal position: {self.world_info.layout['goal']}")
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout['goal']
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position(goal_xy=self.goal_xy)
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

    def _build_world_config(self, layout: dict) -> dict:
        """Create a world_config from our own config."""
        world_config = {
            'floor_type': self.floor_conf.type,
            'floor_size': self.floor_conf.size,
            'agent_base': self.agent.base,
            'agent_xy': layout['agent'] if self.agent_xy is None else self.agent_xy,
        }
        if self.agent.rot is None:
            world_config['agent_rot'] = (
                self.random_generator.random_rot() if self.agent_rot is None else float(self.agent_rot)
            )
        else:
            world_config['agent_rot'] = float(self.agent.rot)

        self.task_name = self.__class__.__name__.split('Level', maxsplit=1)[0]
        world_config['task_name'] = self.task_name

        # process world config via different objects.
        world_config.update(
            {
                'geoms': {},
                'free_geoms': {},
                'mocaps': {},
            },
        )
        for obstacle in self._obstacles:
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            obstacle.process_config(world_config, layout, self.random_generator.generate_rots(num))
        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])

        return world_config
