from collections import deque
from math import exp, inf
import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces.utils import flatten_space
import os
import torch
from Autoencoder import Autoencoder, preprocess_states
import numpy as np
from dataPlotter import DataPlotter
from stable_baselines3.common.logger import Logger
from sklearn.metrics import mean_squared_error
import random




class CustomEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: Env, autoencoder_folder, input_dim, n_states, obs_space, autoencoders, mimic=False):
        super(CustomEnvironmentWrapper, self).__init__(env)
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
        self.alpha_max = 0.9
        self.alpha_min = 0.2
        self.alpha_step = 0.05
        self.quality_buffer = []



        self.mimic = mimic

        self.fit_max = -float('inf')
        self.fit_min = float('inf')
        self.window_size = 10000
        self.fit_buffer = deque(maxlen=self.window_size)

    def step(self, action):
        # the original environment's step action:
        obs, reward, done, truncated, info = self.env.step(action)
        

        if not done and not truncated:
            self.reward_list.append(reward)
            # self.fit_buffer.append(reward)
            dynamic_midpoint = self.reward_list_mean 
            dynamic_slope = 1.0/self.reward_list_std
            x = reward
            # x = mean_current_mse
            reward = 2.0 / (1 + np.exp(-dynamic_slope * (x - dynamic_midpoint))) - 1.0
            # if reward > self.fit_max:
            #     self.fit_max = reward
            # if reward < self.fit_min:
            #     self.fit_min = reward
            # reward = 2 * (reward - self.fit_min) / (self.fit_max - self.fit_min + 1e-8) - 1


        self.state_sequence.append(obs)
        
        # Update reward normalization statistics
        # self.mean_reward = 0.99 * self.mean_reward + 0.01 * reward
        # self.std_reward = 0.99 * self.std_reward + 0.01 * (reward - self.mean_reward)**2

        # # Normalize the reward
        # normalized_reward = (reward - self.mean_reward) / max(1e-8, self.std_reward)


        custom_reward = self.CustomRewardFunction(obs, reward, done, truncated, info)

        if done or truncated:
        #     # self.logger.record("train/novelty_score", np.sum(self.novlist))
        #     # self.logger.record("train/fitness_score", np.sum(self.fitlist))
        #     #log(self.logger, np.sum(self.novlist), np.sum(self.fitlist))
        #     #np.sum(self.novlist)

            self.nov_mean_list.append(np.sum(self.novlist))
            self.fit_mean_list.append(np.sum(self.fitlist))
            self.novlist = []
            self.fitlist = []

        return obs, custom_reward, done, truncated, info

    def CustomRewardFunction(self, obs, fitness, done, truncated, info):
        n_reward = self.novelty_reward()
        
        if self.mimic:
            n_reward = n_reward * -1,
        self.novlist.append(n_reward)
        self.fitlist.append(fitness)
        if self.autoencoders:
            reward = (1 - self.alpha) * fitness + self.alpha * n_reward
            #reward = 0.5 * fitness + 0.5 * n_reward
        else:
            reward = fitness

        return reward
    
    def get_min_mse(self, state_sequence, input_dim):
        files = os.listdir(self.autoencoder_folder)
        autoencoder_model_extension = ".pth"  
        autoencoder_models = sorted([file for file in files if file.endswith(autoencoder_model_extension)])
        min_mse = float('inf')
        for autoencoder_file in autoencoder_models:
            autoencoder = Autoencoder(input_dim=input_dim)
            autoencoder.load_state_dict(torch.load(self.autoencoder_folder + autoencoder_file))
            autoencoder.eval()
            input_data = torch.tensor(np.stack(state_sequence, axis=0), dtype=torch.float32)
            input_data = input_data.view(-1, np.shape(state_sequence)[0] * np.shape(state_sequence)[1])
            with torch.no_grad():
                output_data = autoencoder(input_data)
            mse = mean_squared_error(input_data.numpy(), output_data.numpy())
            if mse < min_mse:
                min_mse = mse
        return min_mse
    
    def novelty_reward(self):
        '''r_novel according to TNB paper:
        r_novel = -exp(-w_novel * min ||D(s)-s||²)
        where w_novel changes the sensitivity of the reward, i think they used 100 to 500 but that might just be for the wheighted sum of rewards??
        min of D(s) - s is the prior trained autoencoder that has the lowest error
        '''
        # lowest_r = float('inf')
        if len(self.state_sequence) == self.n_states and self.autoencoders:
            #mse_list = []
            mini_batch = torch.tensor(np.stack(self.state_sequence, axis=0), dtype=torch.float32) # maybe device="cuda"
            mini_batch = mini_batch.view(-1, np.shape(self.state_sequence)[0] * np.shape(self.state_sequence)[1])
            # min_mse = self.get_min_mse(self.state_sequence, self.input_dim)
            self.state_sequence.pop(0)
            min_mse = float('inf')
            # mse_list_short = []
            for ae in self.autoencoders:
                with torch.no_grad():
                    output = ae(mini_batch)
                mse = mean_squared_error(mini_batch.numpy(), output.numpy())
                # mse_list_short.append(mse)
                if mse < min_mse:
                    min_mse = mse
               
                if mse > self.highest_mse:
                    self.highest_mse = mse
                if mse < self.lowest_mse:
                    self.lowest_mse = mse
            
                #self.mse_list.append(mse)
                    
            self.mse_list.append(min_mse)       
            # mean_current_mse = np.mean(mse_list_short)
            
            #range = self.highest_mse - self.lowest_mse + 0.00001
            #range = self.mse_list_std * 2 + 0.00001
            #dynamic_midpoint =  self.lowest_mse + range / 2
            dynamic_midpoint = self.mse_list_mean #self.lowest_mse
            dynamic_slope = 1.0/self.mse_list_std
            x = min_mse
            # x = mean_current_mse
            sigmoid = 2.0 / (1 + np.exp(-dynamic_slope * (x - dynamic_midpoint))) - 1.0
            # sigmoid = 2.0 / (1 + np.exp(-slope * (x - midpoint))) - 1.0
            lowest_n = sigmoid
            # lowest_n = 1 - 4 * np.exp(-0.5 * min_mse) 
            #-np.exp(-100 * lowest_r ** 2)   # According to paper
            self.novel_reward_plot.add_data(lowest_n)

        else:
            lowest_n = 0
        return lowest_n
    
    def set_lowest_mse(self, states, n_states, batch_size, autoencoder_number):
        print("oi")
        autoencoder_dir = "PPO/BipedalWalker-v3-dyn_sigmoid/autoencoders/"
        autoencoder = Autoencoder(self.input_dim)  # Instantiate the model
        autoencoder.load_state_dict(torch.load(autoencoder_dir + f"/autoencoder_{autoencoder_number}.pth"))
        # Put the model in evaluation mode
        autoencoder.eval()
        # autoencoder = autoencoder
        # autoencoder.eval()
        preprocessed_states = preprocess_states(states, n_states, batch_size)
        random_state_lists = random.sample(preprocessed_states, 1000)       # 1000 is arbitrary, we just need enough to find a low mse
        for batch in random_state_lists:
            mini_batch = torch.tensor(np.stack(self.state_sequence, axis=0), dtype=torch.float32)
            mini_batch = mini_batch.view(-1, np.shape(self.state_sequence)[0] * np.shape(self.state_sequence)[1])
            with torch.no_grad():
                output = autoencoder(mini_batch)
            mse = mean_squared_error(mini_batch.numpy(), output.numpy())
            if mse < self.lowest_mse:
                self.lowest_mse = mse
        # Find the mse for 100? random state-sequences

        return 0

    def get_fit(self, alpha):
        fit = self.fit_mean_list
        self.fit_mean_list = []
        self.mse_list_mean = np.mean(self.mse_list)
        self.mse_list_std = np.std(self.mse_list)
        self.reward_list_mean = np.mean(self.reward_list)
        self.reward_list_std = np.std(self.reward_list)
        if not self.mse_list_mean:
            self.mse_list_mean = 0
            self.mse_list_std = 0
        print(f"highest fit: {self.fit_max}, lowest fit: {self.fit_min}, mean mse: {self.mse_list_mean}, std mse: {self.mse_list_std}")
        #self.highest_mse = 0
        #self.lowest_mse = float('inf')
        self.state_sequence = []
        #self.mse_list = []
        self.novlist = []
        self.fitlist = []
        if not fit:
            print("no fit")
            fit = np.sum(self.fitlist)
        
        # Alpha regulation for two-objective compromise
        self.alpha = alpha
        # if len(self.quality_buffer) > 5:
        #     if np.mean(self.quality_buffer) > np.mean(fit):
        #         if self.alpha < self.alpha_max:
        #             self.alpha += self.alpha_step
        #         else:
        #             self.alpha = 0.5
            
        #     if np.mean(self.quality_buffer) < np.mean(fit):
        #         if self.alpha > self.alpha_min:
        #             self.alpha -= self.alpha_step
        #         else:
        #             self.alpha = 0.5

        #     self.quality_buffer.pop(0)
        
        # self.quality_buffer.append(np.mean(fit))
        return fit, self.mse_list_mean, self.mse_list_std, self.alpha

    def get_nov(self):
        nov = self.nov_mean_list
        self.nov_mean_list = []
        if not nov:
            print("no nov")
            nov = np.sum(self.novlist)
        return nov
        


