from math import exp, inf
import gymnasium as gym
from gymnasium.core import Env
import os
import torch
from Autoencoder import Autoencoder
import numpy as np
from dataPlotter import DataPlotter


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
        # files = os.listdir(self.autoencoder_folder)
        # autoencoder_model_extension = ".pth"  # Adjust the extension as needed
        # autoencoder_models = [file for file in files if file.endswith(autoencoder_model_extension)]

        # if autoencoder_models:
        #     print(f"{len(autoencoder_models)} trained autoencoder model(s) found in the folder:")
        #     print(f"ae names: {autoencoder_models}")
        #     for model_file in autoencoder_models:
        #         print(model_file)
        #         autoencoder = Autoencoder(self.input_dim)  # Instantiate the model
        #         # Put the model in evaluation mode
        #         autoencoder.eval()
        #         self.autoencoders.append(autoencoder)
        # else:
        #     print("No trained autoencoder models found in the folder.")
        

    def step(self, action):
        # the original environment's step action:
        obs, reward, done, truncated, info = self.env.step(action)

        self.state_sequence.append(obs)

        # Call the custom reward function:
        custom_reward = self.CustomRewardFunction(obs, reward, done, truncated, info)

        return obs, custom_reward, done, truncated, info

    def CustomRewardFunction(self, obs, reward, done, truncated, info):
        n_reward = self.novelty_reward()
        return 0.5 * reward + 0.5 * n_reward
    
    def novelty_reward(self):
        '''r_novel according to TNB paper:
        r_novel = -exp(-w_novel * min ||D(s)-s||²)
        where w_novel changes the sensitivity of the reward, i think they used 100 to 500 but that might just be for the wheighted sum of rewards??
        min of D(s) - s is the prior trained autoencoder that has the lowest error
        '''
        lowest_r = 10
        if len(self.state_sequence) == self.n_states:
            mini_batch = torch.tensor(np.stack(self.state_sequence, axis=0), dtype=torch.float32) # maybe device="cuda"
            mini_batch = mini_batch.view(-1, np.shape(self.state_sequence)[0] * np.shape(self.state_sequence)[1])
            self.state_sequence.pop(0)
            for ae in self.autoencoders:
                ae_out = ae(mini_batch)
                error = np.mean(abs(ae_out.detach().cpu().numpy() - mini_batch.cpu().numpy()))
                if error < lowest_r:
                    lowest_r = error
            lowest_n = 1 + 2 * -np.exp(-10 * lowest_r ** 4)
            #-np.exp(-100 * lowest_r ** 2)   # According to paper
            self.novel_reward_plot.add_data(lowest_n)
        else:
            lowest_n = 0
        return lowest_n
        


