import gymnasium as gym
from gymnasium.core import Env
import os
import torch
import numpy as np
from stable_baselines3.common.logger import Logger

class LSTMEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super(LSTMEnvironmentWrapper, self).__init__(env)
        self.fitlist = []
        self.fit_mean_list = []


    def step(self, action):
        # the original environment's step action:
        obs, reward, done, truncated, info = self.env.step(action)

        if done or truncated:
            self.fit_mean_list.append(np.sum(self.fitlist))
            self.fitlist = []

        return obs, reward, done, truncated, info
        

    def get_fit(self):
        fit = self.fit_mean_list
        self.fit_mean_list = []
        self.fitlist = []
        if not fit:
            print("no fit")
            fit = np.sum(self.fitlist)
        return fit
