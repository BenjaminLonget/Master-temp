from math import inf
import gymnasium as gym
from gymnasium.core import Env

class CustomEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super(CustomEnvironmentWrapper, self).__init__(env)
        # Add other initial stuff here, maybe the autoencoders goes here?

    def step(self, action):
        # the original environment's step action:
        obs, reward, done, truncated, info = self.env.step(action)

        # Call the custom reward function:
        custom_reward = self.CustomRewardFunction(obs, reward, done, truncated, info)

        return obs, custom_reward, done, truncated, info

    def CustomRewardFunction(self, obs, reward, done, truncated, info):
        reward = reward
        return reward
