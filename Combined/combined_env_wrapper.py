from collections import deque
from math import exp, inf
import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces.utils import flatten_space
import os
import torch
from combined_Autoencoder import Autoencoder, preprocess_states
import numpy as np
from combined_dataPlotter import DataPlotter
from stable_baselines3.common.logger import Logger
from sklearn.metrics import mean_squared_error
import random

class CombinedEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: Env, autoencoder_folder, n_states, obs_space, autoencoders, mimic=False, open_maze_test=False):
        super(CombinedEnvironmentWrapper, self).__init__(env)
        # Add other initial stuff here
        self.novel_reward_plot = DataPlotter("Novelty reward (no weight)", "Mini-batch", "Novelty score")
        self.autoencoder_folder = autoencoder_folder
        self.input_dim = (n_states, obs_space)  #should be an array where [0] and [1] are the sizes
        #n_states and obs space^
        self.n_states = n_states
        self.state_sequence = []    # A rolling buffer of states

        #instantiate the potential autoencoders
        self.autoencoders = autoencoders

        self.fitlist = []
        self.fit_mean_list = []
        self.novlist = []
        self.nov_mean_list = []

        self.highest_mse = 0
        self.lowest_mse = float('inf')
        self.mse_list = []
        self.mse_list_mean = 0
        self.mse_list_std = 1

        self.reward_list = []
        self.reward_list_mean = 0
        self.reward_list_std = 1

        self.alpha = 0.5
        self.open_maze_test = open_maze_test

        self.mimic = mimic

        self.fit_max = -float('inf')
        self.fit_min = float('inf')
        self.window_size = 10000
        self.fit_buffer = deque(maxlen=self.window_size)

    # def reset(self, **kwargs):
    #     """Resets the environment and normalizes the observation."""
    #     obs, info = self.env.reset(**kwargs)
    #     self.state_sequence = []
    #     return obs, info
    
    def step(self, action):
        # the original environment's step action:
        obs, reward, done, truncated, info = self.env.step(action)

        if self.open_maze_test:
            if self.autoencoders:
                # reward = 0      # Test for pure novelty
                reward = obs[2] * 0.1 # x-velocity/10
            else:
                reward = obs[2] * 0.1

        # if not done and not truncated:
        #     self.reward_list.append(reward)
        #     dynamic_midpoint = self.reward_list_mean 
        #     dynamic_slope = 1.0/self.reward_list_std
        #     x = reward
        #     reward = 2.0 / (1 + np.exp(-dynamic_slope * (x - dynamic_midpoint))) - 1.0



        self.state_sequence.append(obs)
        

        custom_reward = self.CustomRewardFunction(obs, reward, done, truncated, info)

        if done or truncated:
            self.nov_mean_list.append(np.sum(self.novlist))
            self.fit_mean_list.append(np.sum(self.fitlist))
            self.novlist = []
            self.fitlist = []
            #self.state_sequence = []

        return obs, custom_reward, done, truncated, info

    def CustomRewardFunction(self, obs, fitness, done, truncated, info):
        n_reward = self.novelty_reward()
        
        if self.mimic:
            n_reward = n_reward * -1,
        self.novlist.append(n_reward)
        self.fitlist.append(fitness)
        if self.autoencoders:
            reward = (1 - self.alpha) * fitness + self.alpha * n_reward
            # reward = n_reward
            #reward = (fitness / 3) + (n_reward / 3)
            # if self.open_maze_test:
            #     reward = n_reward
            # else:
            #     reward = (1 - self.alpha) * (fitness / 3) + self.alpha * (n_reward / 3)

            # reward = 0.5 * fitness + 0.5 * n_reward
        else:
            reward = fitness

        return reward
    
    
    def novelty_reward(self):

        if len(self.state_sequence) == self.n_states and self.autoencoders:
            mini_batch = torch.tensor(np.stack(self.state_sequence, axis=0), dtype=torch.float32) # maybe device="cuda"
            mini_batch = mini_batch.view(-1, np.shape(self.state_sequence)[0] * np.shape(self.state_sequence)[1])
            self.state_sequence.pop(0)
            min_mse = float('inf')
            for ae in self.autoencoders:
                with torch.no_grad():
                    output = ae(mini_batch)
                mse = mean_squared_error(mini_batch.numpy(), output.numpy())
                if mse < min_mse:
                    min_mse = mse
               
                if mse > self.highest_mse:
                    self.highest_mse = mse
                if mse < self.lowest_mse:
                    self.lowest_mse = mse
            
                    
            self.mse_list.append(min_mse)       

            dynamic_midpoint = self.mse_list_mean #self.lowest_mse
            dynamic_slope = 1.0/self.mse_list_std
            x = min_mse
            sigmoid = 2.0 / (1 + np.exp(-dynamic_slope * (x - dynamic_midpoint))) - 1.0
            lowest_n = sigmoid
            self.novel_reward_plot.add_data(lowest_n)

        else:
            lowest_n = 0
        return lowest_n
    

    def get_fit(self, alpha):
        fit = self.fit_mean_list
        self.fit_mean_list = []
        self.mse_list_mean = np.mean(self.mse_list)
        self.mse_list_std = np.std(self.mse_list)
        # self.mse_list = []

        self.reward_list_mean = np.mean(self.reward_list)
        self.reward_list_std = np.std(self.reward_list)
        if not self.mse_list_mean:
            self.mse_list_mean = 0
            self.mse_list_std = 0
        print(f"highest fit: {self.fit_max}, lowest fit: {self.fit_min}, mean mse: {self.mse_list_mean}, std mse: {self.mse_list_std}")

        self.state_sequence = []
        self.novlist = []
        self.fitlist = []
        if not fit:
            print("no fit")
            fit = np.sum(self.fitlist)
        
        # Alpha regulation for two-objective compromise
        self.alpha = alpha

        return fit, self.mse_list_mean, self.mse_list_std, self.alpha

    def get_nov(self):
        nov = self.nov_mean_list
        self.nov_mean_list = []
        if not nov:
            print("no nov")
            nov = np.sum(self.novlist)
        return nov
        

