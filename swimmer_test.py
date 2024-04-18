import gymnasium as gym
from stable_baselines3 import PPO
import os
import torch
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

# env_name = 'HalfCheetah-v4'
# env_name = 'BipedalWalker-v3'
env_name = 'Swimmer-v4'
save_root = "Swimmer_LSTM_alpha_AE_fit_1"
model_dir = f"tests/Swimmer/Combined/{save_root}/models/"
log_dir = f"tests/Swimmer/Combined/{save_root}/logs"
autoencoder_dir = f"tests/Swimmer/Combined/{save_root}/autoencoders/"

def capture_frames_with_vel(env, policy, frame_interval, num_frames, policy_num):
    frames = []
    state, _ = env.reset(seed=1)
    x_vel = 0
    y_vel = 0

    position_scale = 100

    for step in range(100): # Period to settle in a position for the behavior
    #     # action = policy.predict(state[:len(state)-2])
        action = policy.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action[0])
    initial_distance = info["distance_from_origin"]
    #x_pos_old = info["x_position"]  #state[3]
    #y_pos_old = info["y_position"]  #state[4]

    for step in range(num_frames):
        # action = policy.predict(state[:len(state)-2])
        action = policy.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action[0])
        # x_vel += state[8]
        # y_vel += state[9]
        x_pos = info["x_position"]  #state[3]
        y_pos = info["y_position"]

        if step % frame_interval == 0:
            #vel_vector = (x_vel, y_vel)
            #pos_vector = (x_pos - x_pos_old, y_pos - y_pos_old)
            #x_pos_old = x_pos
            #y_pos_old = y_pos
            frame = np.array(env.render())
            frame[:130, :, :] = [0, 0, 0]
            start_point = (10, 90)
            distance_from_origin = info["distance_from_origin"]
            print(f"Distance from origin: {distance_from_origin}")
            # end_point = (start_point[0] + int(x_pos * position_scale), start_point[0] + int(y_pos * position_scale))
            end_point = (start_point[0] + int((distance_from_origin - initial_distance) * position_scale), start_point[1])
            #end_point = (int(start_point[0] + velocity_scale * pos_vector[0]), int(start_point[1] + velocity_scale * pos_vector[1]))
            color = (255, 0, 0)
            thickness = 10
            print(f"Drawing arrow from {start_point} to {end_point}")
            frame = cv2.arrowedLine(frame, start_point, end_point, color, thickness)
            #frames.append((frame.copy(), state.copy()))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 0, 0)  
            line_type = 3
            if step == 0:
                cv2.putText(frame, f'Policy #: {policy_num}', (10, 30), font, font_scale, font_color, line_type)
            #cv2.putText(frame, f'Policy #: {policy_num}', (10, 30), font, font_scale, font_color, line_type)
            cv2.putText(frame, f'Total distance travelled: {round(distance_from_origin, 4)}', (10, 60), font, font_scale, font_color, line_type)
            frames.append(frame.copy())
            x_vel = 0
            y_vel = 0

        if done or truncated:
            state, _ = env.reset()
            print("Done or truncated")

    env.close()
    print(f"Captured {len(frames)} frames")
    return frames


def create_panorama_img(frames, output_path):
    #panorama = np.concatenate(frames, axis=1)
    frames = [cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT) for frame in frames]

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
    # exclude_current_positions_from_observation # remove these two obs from the observation when using the policy
    env = gym.make(env_name, render_mode='rgb_array', camera_id=0, width=640, height=640)  
    num_frames = 160  
    frame_interval = 10  
    output_path = f'{model_dir}/policy_{model_number}/panorama_image_with_overlay.png'

    policy = PPO.load(model_dir + f"/policy_{model_number}" + "/model_final.zip")
    frames = capture_frames_with_vel(env, policy, frame_interval, num_frames, model_number)

    create_panorama_img(frames, output_path)
    env.close()


if __name__ == '__main__':
    #create_panorama(0)
    # env = gym.make(env_name, render_mode='human', **{"camera_id": 2})
    # env = gym.make(env_name)

    policies = os.listdir(model_dir)
    #while True:
    #env = gym.make('Swimmer-v4', render_mode='human')

    for i in range(len(policies)):
        # create_panorama(i)
        env = gym.make(env_name, render_mode='human', **{"camera_id": 0})
        model_path = model_dir + f"/policy_{i}" + "/model_final.zip"
        print(f"model {i}")
        model = PPO.load(model_path)
        obs, _ = env.reset(seed=1)
        rewards = 0
        while True:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            rewards += reward 

            if(terminated or truncated):

                print(f"reward: {rewards}")
                rewards = 0
                env.close()
                #obs, _ = env.reset()
                break