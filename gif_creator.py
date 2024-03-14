import os
import imageio
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

def create_gif(model_path, env, gif_path):
    model = PPO.load(model_path)
    obs, _ = env.reset()
    images = []
    fitness = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info, truncated = env.step(action)
        frame = env.render()
        images.append(frame)
        fitness += reward
        
        if done or truncated:
            print(f"Fitness: {fitness}")
            fitness = 0
            obs, _ = env.reset()
            break
    imageio.mimsave(gif_path, images, duration=25)

if __name__ == '__main__':
    model_root = "tests/BipedalWalker/AE_test/BipedalWalker-autoencoder_2/"
    model_dir = model_root + "models/"
    gif_dir = model_root + "gifs/"

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    
    policies = os.listdir(model_dir)
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    # env = gym.make("UR5DynReach-v1", render_mode="rgb_array", max_episode_steps=1600)
    # env = gym.make("UR5DynReach-v1", render_mode="rgb_array", max_episode_steps=1600)
    # env = gym.make("UR5DynReach-v1", render_mode="rgb_array", max_episode_steps=1600)
    
    for i in range(len(policies)):
        print(f"Model {i}")
        create_gif(model_dir + f"policy_{i}" + "/model_final.zip", env, gif_dir + f"policy_{i}.gif")
        #input('Press enter to continue')
        
    #input('Press enter to continue')
    env.close()