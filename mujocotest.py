'''@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}'''
# export LD_LIBRARY_PATH=/home/benjamin/.mujoco/mujoco2.3.7/bin:/usr/lib/nvidia
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/benjamin/.mujoco/mujoco2.3.7/bin
from time import sleep
import gymnasium as gym
from multiprocessing import freeze_support
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import panda_gym
# from panda_gym.envs import PandaReachEnv  # To potentially change to pure inverse kinematics

# from gymnasium_robotics.mamujoco_v0 import get_parts_and_edges
model_dir = "testing/models/"
log_dir = "testing/mujoco_test/logs"
autoencoder_dir = "testing/autoencoders/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(autoencoder_dir):
    os.makedirs(autoencoder_dir)


# env = gym.make("FetchReach-v2", render_mode='human')

LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, "r", 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, "g", 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    freeze_support()
    iterations = 100
    n_env = 8   # Mujoco seems to require one core to work
    n_steps_per_core = 1024
    steps_per_update = n_env * n_steps_per_core
    # total_timesteps = 1500000    # Enough to actually learn a good policy, different for each environment
    total_timesteps = iterations * steps_per_update
    # n_steps_per_update = 9984  # 12000 according to TNB-paper 2048 according to stable baselines
    # To match with number of cores and batch size: n_steps_per_update = (n_steps_per_core // batchsize) * batchsize * n_env
    # n_steps_per_core = int(n_steps_per_update // n_env)
    batchsize=32
    print(f"steps should be: {(n_steps_per_core // batchsize) * batchsize * n_env}")


    #model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=n_steps_per_core, batch_size=batchsize, n_epochs=3, stats_window_size=1000)
    # MultiInputPolicy when obs is a dict
    #model.learn(total_timesteps=total_timesteps, progress_bar=True)


    #env = gym.make('PointMaze_UMazeDense-v3', render_mode="human", max_episode_steps=1000, maze_map=LARGE_MAZE)
    # env = gym.make('Humanoid-v4', render_mode='human')
    # env = gym.make('PandaReach-v3', reward_type="dense", control_type="joints")
    env_fns = [lambda: Monitor(gym.make('PandaReach-v3', reward_type="dense", control_type="joints")) for _ in range(n_env)]
    env = SubprocVecEnv(env_fns)
    # observation, _= env.reset()
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=n_steps_per_core, batch_size=batchsize, n_epochs=3)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    env = gym.make('PandaReach-v3', render_mode="human", reward_type="dense", control_type="joints")

    observation, _= env.reset()
    '''observation er et dictionary med '''

    fitness = 0
    while True:
        action = model.predict(observation)[0]
        # action = env.action_space.sample()
        #action = model.predict(observation)[0]
        observation, reward, done, truncated, info = env.step(action)
        fitness += reward
        sleep(0.01)
        #print(reward)
        if done or truncated:
            observation, _ = env.reset()
            print(f"Fitness: {fitness}")
            fitness = 0

      # action = env.action_space.sample()
      # observation, reward, done, truncated, info = env.step(action)

      # if done or truncated:
      #     # Reset the environment if the episode is done
      #     observation = env.reset()

    # Close the environment
    env.close()