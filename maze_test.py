from time import sleep
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

model_dir = "tests/Deceptive_maze/LSTM_AE/Deceptive_maze_LSTM_simple_2/models/policy_1/"

MEDIUM_DECEPTIVE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, "r", 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, "g", 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                
                
LARGE_DECEPTIVE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, "r", 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, "g", 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
LARGE_DECEPTIVE_MAZE_NUMERIC = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

def plot_map(map_data):
    
    #map_data = np.array(map_data)
    #map_data = np.where(np.char.isnumeric(map_data.astype(str)), map_data.astype(float), 1)
    #mask = np.vectorize(lambda x: isinstance(x, (int, float)))(map_data)
    #plt.imshow(np.where(mask, map_data.astype(float), np.nan), cmap='gray', origin='lower', interpolation='none')
    plt.imshow(map_data, cmap='gray', origin='upper')
    plt.title('Map')
    plt.show()

def plot_trajectories(map_data, trajectories):
    cmap = mcolors.ListedColormap(['white', 'gray'])
    #map_data = np.logical_xor(map_data, 1).astype(int)
    plt.imshow(map_data, cmap=cmap, origin='upper', extent=(-len(map_data[0])/2, len(map_data[0])/2, -len(map_data)/2, len(map_data)/2))

    for i, trajectory in enumerate(trajectories):
        x, y = zip(*trajectory)
        plt.plot(x, y, label=f'Policy {i + 1}')

    plt.legend()
    plt.title('Trajectories on double-deceptive Map')
    plt.show()

if __name__ == '__main__':
    
    env = gym.make('PointMaze_UMazeDense-v3', max_episode_steps=1024, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=False, render_mode="human")
    # env = gym.make('PointMaze_UMazeDense-v3', render_mode="human", max_episode_steps=1024, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=False)
    policies = os.listdir(model_dir)
    trajectories = []
    files = os.listdir(model_dir)
    model_extension = ".zip" 
    models = [file for file in files if file.endswith(model_extension)]
    print(models)

    #while True:
        #trajectory = []
    for mod in models:#range(len(policies)):
        traj = []
        # model_path = model_dir + f"/policy_{i}" + "/model_best.zip"
        print(f"model {mod}")
        model = PPO.load(model_dir + mod)
        observation, _= env.reset(seed=0)
        fitness = 0
        while True:
            action = model.predict(observation, deterministic=True)[0]
            #action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            traj.append(observation[:2])
            #trajectory.append(observation[:2])
            fitness += reward
            #print(f"reward: {reward}")
            #sleep(0.1)
            if done:
                print("done")
            if truncated:
                print("truncated")

            if done or truncated:
                #sleep(1)
                print(f"Fitness: {fitness}")
                fitness = 0
                observation, _ = env.reset(seed=0)
                break
        trajectories.append(traj)
    #plot_map(LARGE_DECEPTIVE_MAZE_NUMERIC)
    # print(trajectories[0][-1])
    # plot_trajectories(LARGE_DECEPTIVE_MAZE_NUMERIC, trajectories)