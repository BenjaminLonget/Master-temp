from math import e
from time import sleep
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import gymnasium_robotics

import sys
sys.setrecursionlimit(262145)   #state sequence squared

import matplotlib
matplotlib.use('TkAgg')

model_dir = "tests/Deceptive_maze/Combined_Final_test/LSTM/Open_maze_LSTM_0/models/"#intermediate_models/"
test_name = "LSTM-AE intermediate policies"

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

    # plt.plot(0, 2, color="green", marker='o', markersize=10, label="Start")
    # plt.plot(0, -4, color="orange", marker='o', markersize=15, label="Goal")
    colormaps = [plt.cm.viridis, plt.cm.plasma, plt.cm.inferno, plt.cm.magma, plt.cm.cividis, plt.cm.spring, plt.cm.summer, plt.cm.autumn, plt.cm.winter, plt.cm.cool, plt.cm.hot, plt.cm.gray, plt.cm.bone, plt.cm.pink, plt.cm.jet]  
    cmap_traj = mcolors.ListedColormap(['black', 'blue', 'green', 'red', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray'])
    max_abs_x = 0
    max_abs_y = 0
    for i, trajectory in enumerate(trajectories):
        x, y = zip(*trajectory)
        max_abs_x = max(max_abs_x, max(np.abs(x)))
        max_abs_y = max(max_abs_y, max(np.abs(y)))
        # Create an array with the colors for each point in the trajectory
        colors = colormaps[i % len(colormaps)](np.linspace(0, 1, len(x)))
        # Plot the trajectory with decreasing size and color
        plt.scatter(x, y, color=cmap_traj(i), s=np.linspace(10, 0, len(x)), label=f'Policy {i + 1}')
        #plt.plot(x, y, label=f'Policy {i + 1}', markersize=20)
    
    max_lim = max(max_abs_x, max_abs_y) + 0.5
    plt.xlim(-max_lim, max_lim)
    plt.ylim(-max_lim, max_lim)
    plt.legend()
    plt.title(f'Trajectories on open map, {test_name}')
    plt.show()

def plot_coordinates(coordinates_path):
    import pandas as pd
    df = pd.read_csv(coordinates_path, header=None)
    coordinates = df.values
    cmap = mcolors.ListedColormap(['white', 'gray'])
    #map_data = np.logical_xor(map_data, 1).astype(int)
    plt.imshow(LARGE_DECEPTIVE_MAZE_NUMERIC, cmap=cmap, origin='upper', extent=(-len(LARGE_DECEPTIVE_MAZE_NUMERIC[0])/2, len(LARGE_DECEPTIVE_MAZE_NUMERIC[0])/2, -len(LARGE_DECEPTIVE_MAZE_NUMERIC)/2, len(LARGE_DECEPTIVE_MAZE_NUMERIC)/2))

    # for i, trajectory in enumerate(trajectories):
    #     x, y = zip(*trajectory)
    plt.plot(0, 2, color="green", marker='o', markersize=10, label="Start")
    plt.plot(0, -4, color="orange", marker='o', markersize=15, label="Goal")
    # print(coordinates[0])
    max_abs_x = 0
    max_abs_y = 0
    for point in coordinates:
        x, y = point
        if np.abs(x) > max_abs_x:
            max_abs_x = np.abs(x)
        if np.abs(y) > max_abs_y:
            max_abs_y = np.abs(y)
        plt.plot(x, y, color='red', marker='o', markersize=2)

    # max_lim = max(max_abs_x, max_abs_y) + 0.5
    # plt.xlim(-max_lim, max_lim)
    # plt.ylim(-max_lim, max_lim)
    plt.legend()
    plt.title(f'Final coordinates during exploration of deceptive map, {test_name}')
    plt.show()

def plot_policy_scatter(model_dir):
    import pandas as pd
    cmap = mcolors.ListedColormap(['red', 'blue', 'green', 'black', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray'])
    max_abs_x = 0
    max_abs_y = 0
    for i in range(len(os.listdir(model_dir))):
        df = pd.read_csv(model_dir + f"/policy_{i}/final_coordinates_with_LSTM_{i}.csv")
        coordinates = df.values
        # for point in coordinates:
            # x, y = point
        x, y = zip(*coordinates)
        max_abs_x = max(max_abs_x, max(np.abs(x)))
        max_abs_y = max(max_abs_y, max(np.abs(y)))
        plt.scatter(x, y, color=cmap(i), s=5, marker='o', label=f"Policy {i + 1}")
    max_lim = max(max_abs_x, max_abs_y) + 0.5
    plt.xlim(-max_lim, max_lim)
    plt.ylim(-max_lim, max_lim)
    plt.legend(bbox_to_anchor=(1.08, 1), loc='upper right')
    plt.title(f'Exploration coordinates of {test_name}')
    plt.show()

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def _c(ca, i, j, p, q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]

def discrete_frechet_distance(p, q, p_n, q_n):
    '''Function based on the implementation (https://github.com/spiros/discrete_frechet/tree/master) 
    of the discrete Frechet distance function from Eiter, T. and Mannila, H., 1994 Computing discrete Fréchet distance.
    '''
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')
        
    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)

    # plt.figure(figsize=(32, 32))
    # plt.imshow(ca, cmap='viridis', origin='upper')#, interpolation='nearest')
    # plt.colorbar(label='Distance')
    # plt.xlabel('Index of Trajectory q')
    # plt.ylabel('Index of Trajectory p')
    # plt.title(f'Frechet Matrix Between Policy {p_n + 1} and {q_n + 1}')
    # plt.show()
    
    return dist#, frechet_matrix

def get_dissimilar_trajectories(trajectories, threshold):
    dissimilar_trajectories = []
    for i in range(len(trajectories)):
        print(f"Calculating frechet from policy {i}")
        for j in range(i + 1, len(trajectories)):
            frechetd = discrete_frechet_distance(trajectories[i], trajectories[j])
            if frechetd > threshold:
                dissimilar_trajectories.append((i, j))
    return dissimilar_trajectories

def get_similar_trajectories(trajectories, threshold):
    similar_trajectories = []
    for i in range(len(trajectories)):
        
        for j in range(i + 1, len(trajectories)):
            print(f"Calculating frechet between policy {i} and {j}...")
            frechetd = discrete_frechet_distance(trajectories[i], trajectories[j])
            if frechetd < threshold:
                similar_trajectories.append((i, j))
    return similar_trajectories

def get_unique_trajectories(trajectories, threshold):
    unique_trajectories = []
    # sequentially for intermediately trained policies
    for i in range(len(trajectories) - 1):
        print(f"Calculating Frechet from policy {i} to {i + 1}...")
        frechetd = discrete_frechet_distance(trajectories[i], trajectories[i + 1], i, i + 1)
        if frechetd > threshold:
            unique_trajectories.append(i)

    # for i in range(len(trajectories)):
    #     unique = True
    #     for j in range(i + 1, len(trajectories)):
    #         print(f"Calculating frechet between policy {i} and {j}...")
    #         frechetd = discrete_frechet_distance(trajectories[i], trajectories[j], i, j)
    #         if frechetd < threshold:
    #             unique = False
    #             break
    #     if unique:
    #         unique_trajectories.append(i)

    return unique_trajectories

if __name__ == '__main__':
    # plot_policy_scatter(model_dir)
    LARGE_DECEPTIVE_MAZE = [[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,"r",0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]]
    
    LARGE_DECEPTIVE_MAZE_NUMERIC = [[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]]
    
    env = gym.make('PointMaze_UMazeDense-v3', max_episode_steps=512, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=True)
    # env = gym.make('PointMaze_UMazeDense-v3', render_mode="human", max_episode_steps=1024, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=False)
    policies = os.listdir(model_dir)
    # model_dir = model_dir.replace("models/", "intermediate_models/policy_0/")
    # policies = os.listdir(model_dir)
    policies = os.listdir(model_dir.replace("models/", "intermediate_models/policy_0/"))
    trajectories = []
    # files = os.listdir(model_dir)
    # model_extension = ".zip" 
    # models = [file for file in files if file.endswith(model_extension)]
    # print(models)

    #while True:
        #trajectory = []

    for i in range(len(policies) - 1):
        traj = []
        # model_path = model_dir + f"/policy_{i}" + "/model_final.zip"
        # model_path = model_dir + f"policy_{i}" + "/model_best"
        # model_path = model_dir + f"/policy_{i}" + "/final_good_model.zip"
        # model_path = model_dir + f"policy_{i}" + "/fastets_model"
        '''For intermediate models: '''
        # tests/Deceptive_maze/Combined/Open_maze_pure_LSTM_NOWWITHTRAINEDAE_eps_decay_1/intermediate_models/policy_0
        model_path = model_dir.replace("models", "intermediate_models") + "/policy_0" + f"/model_it_{(i+1)*10}.zip"
        # csv_path = model_dir + f"policy_{i}/final_coordinates_with_LSTM_{i}.csv"
        # plot_coordinates(csv_path)
        print(model_path)
        print(f"model {i}")
        model = PPO.load(model_path)
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
    #trajectories = np.array(trajectories)
    plot_trajectories(LARGE_DECEPTIVE_MAZE_NUMERIC, trajectories)
    print(f"Number of unique trajectories: {len(get_unique_trajectories(trajectories, 2.5))}")
    