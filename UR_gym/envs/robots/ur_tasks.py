import numpy as np

from UR_gym.envs.core import RobotTaskEnv
from UR_gym.envs.robots.UR5 import UR5Ori
from UR_gym.envs.tasks.reach import ReachDyn
from UR_gym.pyb_setup import PyBullet
#from panda_gym.pybullet import PyBullet

class UR5DynReachEnv(RobotTaskEnv):
    """Reach task wih UR5 robot. (Added obstacle reward)

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
    """

    def __init__(self, render: bool = False, render_mode="rgb_array", max_episode_steps=1500, renderer = "Tiny") -> None:
        
        # sim = PyBullet(render=render, render_mode=render_mode)
        steps = max_episode_steps
        sim = PyBullet(render_mode=render_mode, n_substeps=5, renderer=renderer)
        robot = UR5Ori(sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]))
        task = ReachDyn(sim, robot=robot)
        super().__init__(robot, task, steps)

        #self.max_episode_steps = max_episode_steps
