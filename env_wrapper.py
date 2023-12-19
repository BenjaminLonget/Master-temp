from math import exp, inf
import gymnasium as gym
from gymnasium.core import Env
import os
import torch
from Autoencoder import Autoencoder, preprocess_states
import numpy as np
from dataPlotter import DataPlotter
from stable_baselines3.common.logger import Logger
from sklearn.metrics import mean_squared_error
import random




class CustomEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env: Env, autoencoder_folder, input_dim, n_states, obs_space, autoencoders):
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

    def step(self, action):
        # the original environment's step action:
        obs, reward, done, truncated, info = self.env.step(action)
        self.done = done
        self.truncated = truncated
        self.state_sequence.append(obs)

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
        self.novlist.append(n_reward)
        self.fitlist.append(fitness)
        if self.autoencoders:
            reward = 0.5 * fitness + 0.5 * n_reward
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
            mse_list = []
            mini_batch = torch.tensor(np.stack(self.state_sequence, axis=0), dtype=torch.float32) # maybe device="cuda"
            mini_batch = mini_batch.view(-1, np.shape(self.state_sequence)[0] * np.shape(self.state_sequence)[1])
            # min_mse = self.get_min_mse(self.state_sequence, self.input_dim)
            self.state_sequence.pop(0)
            min_mse = float('inf')
            for ae in self.autoencoders:
                with torch.no_grad():
                    output = ae(mini_batch)
                mse = mean_squared_error(mini_batch.numpy(), output.numpy())
                mse_list.append(mse)
                if mse < min_mse:
                    min_mse = mse
               
                if mse > self.highest_mse:
                    self.highest_mse = mse
                if mse < self.lowest_mse:
                    self.lowest_mse = mse
            
            range = self.highest_mse - self.lowest_mse
            dynamic_midpoint =  range / 2
            # dynamic_midpoint = self.lowest_mse
            dynamic_slope = 10 / range      # 10 seems to result in values close to lowest/highest being about 0.98
                                            # dynamic sigmoid has not been tested yet.
            # its a tanh, not a sigmoid.
            
            slope = 18.2
            x = min_mse
            midpoint = 0.274
            midpoint = 0.2
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

    def get_fit(self):
        fit = self.fit_mean_list
        self.fit_mean_list = []
        return fit

    def get_nov(self):
        nov = self.nov_mean_list
        self.nov_mean_list = []
        return nov
        


