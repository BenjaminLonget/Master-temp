from gc import freeze
from multiprocessing import freeze_support
from tracemalloc import start
from cv2 import exp, mean
import gymnasium as gym
from stable_baselines3 import PPO
import os
from Autoencoder import Autoencoder, train_autoencoder
from env_wrapper import CustomEnvironmentWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import time
from multiprocessing import Pool
import multiprocessing as mp

model_dir = "PPO/models/"
autoencoder_dir = "PPO/autoencoders/"

if __name__ == '__main__':
    n_states = 16
    env_name = 'BipedalWalker-v3'
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    input_dim = (n_states, obs_space)

    #instantiate the potential autoencoders
    autoencoders = []
    files = os.listdir(autoencoder_dir)
    autoencoder_model_extension = ".pth"  # Adjust the extension as needed
    autoencoder_models = [file for file in files if file.endswith(autoencoder_model_extension)]

    if autoencoder_models:
        print(f"{len(autoencoder_models)} trained autoencoder model(s) found in the folder:")
        print(f"ae names: {autoencoder_models}")
        for model_file in autoencoder_models:
            print(model_file)
            autoencoder = Autoencoder(input_dim)  # Instantiate the model
            # Put the model in evaluation mode
            autoencoder.eval()
            autoencoders.append(autoencoder)
    else:
        print("No trained autoencoder models found in the folder.")

    policies = os.listdir(model_dir)
    while True:
        for i in range(len(policies)):
            model = PPO.load(model_dir + policies[i] + "/model_final.zip")
            print(policies[i] + "/model_final.zip")
            env = gym.make('BipedalWalker-v3', render_mode='human')
            obs, _ = env.reset()
            rewards = 0
            novelty_rewards = 0
            obs_buffer = []
            while True:
                action = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action[0])
                obs_buffer.append(obs)
                rewards += reward
                if len(obs_buffer) == n_states:
                        mini_batch = torch.tensor(np.stack(obs_buffer, axis=0), dtype=torch.float32, device="cpu")
                        mini_batch = mini_batch.view(-1, np.shape(obs_buffer)[0] * np.shape(obs_buffer)[1])
                        #error = abs(np.mean(autoencoder(mini_batch).cpu().numpy() - mini_batch.cpu().numpy()))
                        lowest_novelty = 10
                        highest_novelty = -10
                        for ae in autoencoders:
                            ae_out = ae(mini_batch)
                            error = np.mean(abs(ae_out.detach().cpu().numpy() - mini_batch.cpu().numpy()))
                            if error < lowest_novelty:
                                lowest_novelty = error
                            if error > highest_novelty:
                                highest_novelty = error
                        paper_novelty = -np.exp(-100 * lowest_novelty ** 2)
                        lowest_novelty = 1 + 2 * -np.exp(-10 * lowest_novelty ** 4)
                        highest_novelty = 1 + 2 * -np.exp(-10 * highest_novelty ** 4)
                        novelty_rewards += lowest_novelty
                        
                        #print(f"lowest novelty: {lowest_novelty}, highest: {highest_novelty}, paper: {paper_novelty}")

                        obs_buffer.pop(0)   # Rolling buffer
                if(terminated or truncated):

                    print(f"reward: {rewards}, novelty: {novelty_rewards}")
                    rewards = 0
                    novelty_rewards = 0
                    obs, _ = env.reset()
                    break