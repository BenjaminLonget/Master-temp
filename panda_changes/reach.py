from math import e
from shlex import join
from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        get_ee_orientation,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.get_ee_orientation = get_ee_orientation
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-1.0)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        joint_angles = []
        for i in range(7):
            joint_angles.append(self.sim.get_joint_angle("panda", joint=i))
        '''Final observation is then ee-position, ee-velocity, joint-angles'''
        return np.array(joint_angles)
        #return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        ee_orientation = np.array(self.get_ee_orientation())
        achieved_goal = np.concatenate([ee_position, ee_orientation])
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        goal_position = self.goal[:3]
        goal_orientation = self.goal[3:]
        self.sim.set_base_pose("target", goal_position, goal_orientation)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal_position = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        goal = np.concatenate([goal_position, goal_orientation])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
