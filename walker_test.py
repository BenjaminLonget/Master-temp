from gc import freeze
from gettext import find
from multiprocessing import freeze_support
from sre_parse import State
from tracemalloc import start
from unittest import result
from cv2 import exp, mean
from graphviz import render
import gymnasium as gym
from pytest import skip
from stable_baselines3 import PPO
import os

from sympy import det
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
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# env_name = 'HalfCheetah-v4'
env_name = 'BipedalWalker-v3'
# save_root = "BipedalWalker-v3-dyn_sigmoid"
model_dir = "tests/BipedalWalker/interesting_models/models/"
# log_dir = f"PPO/{save_root}/logs"
# autoencoder_dir = f"PPO/{save_root}/autoencoders/"
# save_root = "BipedalWalker-min_mse_alpha_2"
# model_dir = f"tests/BipedalWalker/AE_test/{save_root}/best_models/"
# log_dir = f"tests/BipedalWalker/AE_test/{save_root}/logs"
# autoencoder_dir = f"tests/BipedalWalker/AE_test/{save_root}/autoencoders/"

def capture_frames_with_vel(env, policy, frame_interval, num_frames):
    frames = []
    state, _ = env.reset()
    x_vel = 0
    y_vel = 0
    velocity_scale = 30

    for step in range(num_frames):
        action = policy.predict(state, deterministic=True)
        state, _, done, truncated, _ = env.step(action[0])
        x_vel += state[2]
        y_vel += state[3]

        if step % frame_interval == 0:
            vel_vector = (x_vel, y_vel)
            frame = env.render()
            start_point = (150, 230)
            end_point = (int(start_point[0] + velocity_scale * vel_vector[0]), int(start_point[1] + velocity_scale * vel_vector[1]))
            color = (255, 0, 0)
            thickness = 5
            print(f"Drawing arrow from {start_point} to {end_point}")
            frame = cv2.arrowedLine(frame, start_point, end_point, color, thickness)
            #frames.append((frame.copy(), state.copy()))
            frames.append(frame.copy())
            x_vel = 0
            y_vel = 0

        if done or truncated:
            state = env.reset()

    env.close()
    print(f"Captured {len(frames)} frames")
    return frames


def create_img(frames, output_path):
    #panorama = np.concatenate(frames, axis=1)
    num_frames = len(frames)
    frames_per_row = num_frames // 2  # Assuming an even number of frames

    # Split frames into two rows
    row1_frames = frames[:frames_per_row]
    row2_frames = frames[frames_per_row:]

    # Concatenate frames in each row
    row1 = np.concatenate(row1_frames, axis=1)
    row2 = np.concatenate(row2_frames, axis=1)

    # Concatenate the two rows vertically
    panorama = np.concatenate([row1, row2], axis=0)
    cv2.imwrite(output_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))

def create_panorama(model_number):
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')  
    num_frames = 130
    frame_interval = 13
    output_folder = f"{model_dir}/sequential_images/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = f'{output_folder}/model_{model_number}_image.png'

    policy = PPO.load(model_dir + f"/policy_{model_number}" + "/model_final.zip")
    #find_walker_limits(env, policy)
    # frames = capture_frames_with_vel(env, policy, frame_interval, num_frames)
    frames = capture_frames_with_distance(env, policy, frame_interval, num_frames)

    create_img(frames, output_path)

def capture_frames_with_distance(env, policy, frame_interval, num_frames):
    frames = []
    state, _ = env.reset()
    walker_center_x = int((180 - 60) / 2 + 60)
    pixel_distance_x = 0
    pixel_distance_y = 230
    pixel_distance_y_prev = pixel_distance_y
    pre_steps = 15
    for step in range(15):
        action = policy.predict(state, deterministic=True)
        state, _, done, truncated, _ = env.step(action[0])
    for step in range(num_frames):
        action = policy.predict(state, deterministic=True)
        state, _, done, truncated, _ = env.step(action[0])
        pixel_distance_x += state[2] / 0.12
        pixel_distance_y -= state[3] / 0.12        

        if step % frame_interval == 0:
            frame = env.render()
            print(f"X position: {pixel_distance_x}, Y position: {pixel_distance_y}")
            start_point = (int(walker_center_x - pixel_distance_x), int(pixel_distance_y_prev))
            end_point = (int(walker_center_x), int(pixel_distance_y))
            color = (255, 0, 0)
            thickness = 5
            print(f"Drawing arrow from {start_point} to {end_point}")
            frame = cv2.arrowedLine(frame, start_point, end_point, color, thickness)
            #frames.append((frame.copy(), state.copy()))
            frames.append(frame.copy())

            pixel_distance_x = 0
            pixel_distance_y_prev = pixel_distance_y

        if done or truncated:
            state = env.reset()

    return frames

def alpha_blend(background, overlay, alpha):
    # plt.imshow(background)
    # plt.show()
    # plt.imshow(overlay)
    # plt.show()
    return (background  * (1 - alpha)+ overlay * alpha).astype(np.uint8)

def capture_and_calculate_panorama(env, policy, frame_interval, num_frames, output_path):

    panorama_img = []
    ghost_frames = []
    state, _ = env.reset(seed=1)
    walker_lower_limit = 60
    walker_upper_limit = 190
    walker_range = (walker_upper_limit - walker_lower_limit) / 2
    x_position_prev = (walker_upper_limit - walker_lower_limit) / 2 + walker_lower_limit
    x_position = (walker_upper_limit - walker_lower_limit) / 2 + walker_lower_limit
    for step in range(num_frames):
        action = policy.predict(state, deterministic=True)
        state, _, done, truncated, _ = env.step(action[0])
        frame = env.render()
        if step == 0:
            #print(f"X position: {x_position}")
            panorama_img = frame[:, :walker_upper_limit]
        x_position += state[2] /0.12#* (5 - 14/30.0)#(50/(0.3 * (600 / 30)))
        #print(f"X position: {x_position}")
        #input("Press Enter to continue...")
        if x_position - walker_lower_limit > x_position_prev + walker_upper_limit:
            #print(f"X position: {x_position}")
            #print(f"step: {step}")
            #plt.imshow(frame)
            #plt.show()
            frame = frame[:, walker_lower_limit: walker_upper_limit]

            panorama_img = panorama_img[:, :int(x_position_prev) + (walker_upper_limit - walker_lower_limit) // 2]
            #plt.imshow(panorama_img)
            #plt.show()
            panorama_img = np.hstack([panorama_img, frame])

            #plt.imshow(panorama_img)
            #plt.show()
            x_position_prev = x_position
        elif step % 10 == 0:
            frame = frame[:, walker_lower_limit: walker_upper_limit]
            ghost_frames.append([frame, x_position])
            #panorama_img[x_position:] = ghost_frame
    
    panorama_img_ghost = panorama_img.copy()
    for ghost_frame, x_positions in ghost_frames:
        alpha = 0.2
        if x_positions + walker_upper_limit < panorama_img.shape[1]:
            panorama_img_ghost[:, int(x_positions - walker_range): int(x_positions + walker_range)] = alpha_blend(panorama_img[:, int(x_positions - walker_range): int(x_positions + walker_range)], ghost_frame, alpha)
    
    panorama_img = alpha_blend(panorama_img, panorama_img_ghost, 0.5)
    plt.imshow(panorama_img)
    plt.show()
    cv2.imwrite(output_path, cv2.cvtColor(np.array(panorama_img), cv2.COLOR_RGB2BGR))

def find_walker_limits(env, policy):
    state, _ = env.reset()
    lower_line_x = 60
    upper_line_x = 180
    x_position = int((upper_line_x - lower_line_x) / 2 + lower_line_x)
    y_position = 230
    for i in range(100):
        action = policy.predict(state, deterministic=True)
        state, _, done, truncated, _ = env.step(action[0])
        y_position -= state[3] / 0.12
        if i % 10 == 0:
            frame = env.render()
            
            # x_position += int(state[2] / (0.3 * (600 / 30)) * 50) # constants found in the walker environment
            print(f"X position: {x_position}, Y position: {y_position}")

            cv2.line(frame, (lower_line_x, 0), (lower_line_x, 640), (0, 255, 0), 2)
            cv2.line(frame, (upper_line_x, 0), (upper_line_x, 640), (0, 255, 0), 2)
            cv2.line(frame, (x_position, 0), (x_position, 640), (0, 0, 255), 2)
            cv2.line(frame, (0, int(y_position)), (640, int(y_position)), (0, 0, 255), 2)
            plt.imshow(frame)
            plt.show()
        #input("Press Enter to continue...")


def create_fluid_panorama(model_number):
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')  
    num_frames = 300  
    frame_interval = 10  
    output_path = f'{model_dir}/policy_{model_number}/panorama_image.png'

    policy = PPO.load(model_dir + f"/policy_{model_number}" + "/model_final.zip")
    capture_and_calculate_panorama(env, policy, frame_interval, 1000, output_path)

    #create_panorama_img(frames, output_path)

def get_min_mse(state_sequence, autoencoders, pol_number):

    min_mse = float('inf')
    # input_data = torch.tensor(state_sequence, dtype=torch.float32)
    mse_list = []
    input_data = torch.tensor(np.stack(state_sequence, axis=0), dtype=torch.float32)
    input_data = input_data.view(-1, np.shape(state_sequence)[0] * np.shape(state_sequence)[1])
    i = 0
    for autoencoder in autoencoders:
        if i == pol_number:
            skip
        with torch.no_grad():
            output_data = autoencoder(input_data)
        mse = mean_squared_error(input_data.numpy(), output_data.numpy())
        mse_list.append(mse)
        if mse < min_mse:
            min_mse = mse
    # print(mse_list)
    return min_mse

if __name__ == '__main__':
    #create_fluid_panorama(0)
    n_states = 32
    # env = gym.make(env_name)
    env = gym.make(env_name, render_mode='human')

    obs_space = env.observation_space.shape[0]
    input_dim = (n_states, obs_space)
    batch_size = 1
    batch = []
    
    # autoencoders = []
    # files = os.listdir(autoencoder_dir)
    # autoencoder_model_extension = ".pth" 
    # autoencoder_models = [file for file in files if file.endswith(autoencoder_model_extension)]
    # #autoencoder_models = []
    # if autoencoder_models:
    #     print(f"{len(autoencoder_models)} trained autoencoder model(s) found in the folder.")
    #     print(f"ae names: {autoencoder_models}")
    #     for model_file in autoencoder_models:
    #         #print(model_file)
    #         autoencoder = Autoencoder((n_states, obs_space))  # Instantiate the model
    #         autoencoder.load_state_dict(torch.load(autoencoder_dir + model_file))
    #         # Put the model in evaluation mode
    #         autoencoder.eval()
    #         autoencoders.append(autoencoder)

    policies = os.listdir(model_dir)
    #while True:
    for i in range(len(policies)):
        create_panorama(i)
        #create_fluid_panorama(i)   # can't get it to blend the frames properly
        model_path = model_dir + f"/policy_{i}" + "/model_final.zip"
        print(f"model {i}")
        # model_path = "tests/BipedalWalker/LSTM_AE/BipedalWalker-LSTM_test_1/best_models/policy_3/model_best.zip"
        model = PPO.load(model_path)
        # env = gym.make('BipedalWalker-v3', render_mode='human')
        obs, _ = env.reset()
        rewards = 0
        novelty_reward = 0
        highest_novel_reward = 0
        novelty_self = 0
        obs_buffer = []
        # while True:
        #     action = model.predict(obs, deterministic=True)
        #     obs, reward, terminated, truncated, info = env.step(action[0])
        #     obs_buffer.append(obs)
        #     rewards += reward 
        #     # if len(obs_buffer) == n_states:
            #         # batch.append(obs_buffer)
            #         #obs_buffer = []
            #         #if len(batch) == batch_size:
            #         # mini_batch = torch.tensor(np.stack(obs_buffer, axis=0), dtype=torch.float32, device="cpu")
            #         # mini_batch = mini_batch.view(-1, np.shape(obs_buffer)[0] * np.shape(obs_buffer)[1])
            #         #error = abs(np.mean(autoencoder(mini_batch).cpu().numpy() - mini_batch.cpu().numpy()))
            #         lowest_mse = get_min_mse(obs_buffer, autoencoders, i)
            #         obs_buffer.pop(0)   # Rolling buffer

            #             # batch = []
            #         # lowest_mse = get_min_mse(obs_buffer, input_dim=input_dim)
            #         # novelty = 1 - 4 * np.exp(-0.5 * lowest_mse)
            
            #         x = lowest_mse
            #         slope = 18.2
            #         midpoint = 0.274
            #         # sigmoid = 2.0 / (1 + np.exp(-dynamic_slope * (x - dynamic_midpoint))) - 1.0
            #         sigmoid = 2.0 / (1 + np.exp(-slope * (x - midpoint))) - 1.0

            #         novelty = sigmoid
            #         novelty_reward += novelty
                    # print(novelty)
                    # novelty_reward += 1 +
                    # lowest_novelty = 10
                    # highest_novelty = -10
                    # highest_mse = 0
                    # for ae in range(len(autoencoders)):
                    #     # if ae == i:
                    #     #     skip
                    #     current_ae = Autoencoder((n_states, obs_space))
                    #     ae_out = autoencoders[ae](mini_batch).detach().cpu().numpy()
                    #     #mse = mean((ae_out - mini_batch.cpu().numpy())**2)
                    #     mse = mean_squared_error(ae_out, mini_batch.cpu().numpy())
                    #     # print(mse)
                    #     # mse = mse[0]
                    #     if mse < lowest_mse:
                    #         lowest_novelty = 1 + 2 * -np.exp(-0.1 * mse)
                    #         lowest_mse = mse
                    #     if mse > highest_mse:
                    #         highest_novelty = 1 + 2 * -np.exp(-0.1 * mse)
                    #         highest_mse = mse
                    #     # error = abs(ae_out.detach().cpu().numpy() - mini_batch.cpu().numpy())
                    #     #error = 1 + 2 * -np.exp(-0.1 * error ** 2)

                    #     # if error < lowest_novelty:
                    #     #     lowest_novelty = error
                    #     # if error > highest_novelty:
                    #     #     highest_novelty = error
                        
                    #     if ae == i:
                    #         novelty_self += 1 + 2 * -np.exp(-0.1 * mse)
                    
                    # paper_novelty = -np.exp(-100 * lowest_novelty ** 2)
                    # lowest_novelty = 1 + 2 * -np.exp(-0.1 * lowest_novelty ** 2)
                    # highest_novelty = 1 + 2 * -np.exp(-0.1 * highest_novelty ** 2)
                    # novelty_rewards += lowest_novelty
                    # highest_novel_reward += highest_novelty
                    
                    # print(f"lowest novelty: {lowest_novelty}")
                    # print(f"lowest novelty: {lowest_novelty}, highest: {highest_novelty}, paper: {paper_novelty}")

            # if(terminated or truncated):

            #     print(f"reward: {rewards}")
            #     rewards = 0
            #     novelty_rewards = 0
            #     obs, _ = env.reset()
            #     break