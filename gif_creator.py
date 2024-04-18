import os
import imageio
import gymnasium as gym
from stable_baselines3 import PPO
import cv2
import numpy as np
import UR_gym
import gymnasium_robotics

def create_gif(model_path, env, gif_path, policy_number):
    model = PPO.load(model_path)
    obs, _ = env.reset()
    env_name = env.spec.id

    images = []
    fitness = 0
    i = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        fitness += reward
        frame = np.array(env.render())
        cv2.putText(frame, f"Policy# {policy_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Fitness: {round(fitness, 1)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        images.append(frame)
        
        # input("Press Enter to continue...")
        if done or truncated:
            print(f"iterations: {i}")
            print(f"Fitness: {fitness}")
            if "UR5" in env_name:
                # save last frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(gif_path + f"_pol_{policy_number}_last_frame.png", frame_rgb)
            fitness = 0
            obs, _ = env.reset()
            break
        i += 1
    imageio.mimsave(gif_path, images, duration=50, loop=0)
    print(f"Saved gif to {gif_path}")
    '''Duration is the time between frames in milliseconds
    25 for walker, 50 for UR
    '''

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

if __name__ == '__main__':
    model_type = ""
    # best = True UR5_ee_lowerrot_direct_expo_reward_1
    model_root = "tests/UR5/Combined/UR5_ee_lowerrot_direct_expo_reward_1/" # Use the relative path to the root of the model dir
    model_dir = model_root + "best_models/"
    if model_type == "best":
        gif_dir = model_root + "gifs_best/"
    elif model_type == "fastest":
        gif_dir = model_root + "gifs_fastest/"
    elif model_type == "final_good":
        gif_dir = model_root + "gifs_final/"
    else:
        model_dir = model_root + "models/"
        gif_dir = model_root + "gifs/"
    
    

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    
    policies = os.listdir(model_dir)
    # env = gym.make('BipedalWalker-v3', render_mode='rgb_array', max_episode_steps=1200)
    # env = gym.make('Swimmer-v4', render_mode='rgb_array', max_episode_steps=750, camera_id=0, width=640, height=640)
    
    # env = gym.make('PointMaze_UMazeDense-v3', max_episode_steps=1024, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=False, render_mode="rgb_array")
    
    # i=1
    # create_gif(model_dir + f"policy_{i}" + "/model_final.zip", env, gif_dir + f"policy_{i}.gif")
    
    for i in range(len(policies)):
        env = gym.make("UR5DynReach-v1", render_mode="rgb_array", max_episode_steps=512, renderer = "OpenGL")
        print(f"Model {i}")
        if model_type == "best":
            create_gif(model_dir + f"policy_{i}" + "/model_best.zip", env, gif_dir + f"policy_{i}.gif", i)
        elif model_type == "fastest":
            create_gif(model_dir + f"policy_{i}" + "/fastest_model.zip", env, gif_dir + f"policy_{i}.gif", i)
        elif model_type == "final_good":
            create_gif(model_dir + f"policy_{i}" + "/final_good_model.zip", env, gif_dir + f"policy_{i}.gif", i)
        else:
            create_gif(model_dir + f"policy_{i}" + "/model_final.zip", env, gif_dir + f"policy_{i}.gif", i)
        # if best:
        #     create_gif(model_dir + f"policy_{i}" + "/model_best.zip", env, gif_dir + f"policy_{i}.gif", i)
        # else:
        #     create_gif(model_dir + f"policy_{i}" + "/model_final.zip", env, gif_dir + f"policy_{i}.gif", i)
        env.close()

    env.close()