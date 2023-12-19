'''@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}'''
# export LD_LIBRARY_PATH=/home/benjamin/.mujoco/mujoco2.3.7/bin:/usr/lib/nvidia
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/benjamin/.mujoco/mujoco2.3.7/bin
import gymnasium as gym
from multiprocessing import freeze_support
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
# from gymnasium_robotics.mamujoco_v0 import get_parts_and_edges
model_dir = "PPO/models/"
log_dir = "PPO/mujoco_test/logs"
autoencoder_dir = "PPO/autoencoders/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(autoencoder_dir):
    os.makedirs(autoencoder_dir)


# env = gym.make("FetchReach-v2", render_mode='human')

LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, "r", 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, "g", 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
if __name__ == '__main__':
    mp.set_start_method('forkserver')
    freeze_support()

    n_env = 7   # Mujoco seems to require one core to work
    total_timesteps = 1000    # Enough to actually learn a good policy, different for each environment
    n_steps_per_update = 11200  # 12000 according to TNB-paper
    # To match with number of cores and batch size: n_steps_per_update = (n_steps_per_core // batchsize) * batchsize * n_env
    n_steps_per_core = int(n_steps_per_update // n_env)
    batchsize=32
    print(f"steps should be: {(n_steps_per_core // batchsize) * batchsize * n_env}")

    # env_fns = [lambda: Monitor(gym.make('PointMaze_UMazeDense-v3', max_episode_steps=1000, maze_map=LARGE_MAZE)) for _ in range(n_env)]
    # env = SubprocVecEnv(env_fns)
    #model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=n_steps_per_core, batch_size=batchsize, n_epochs=3, stats_window_size=1000)
    # MultiInputPolicy when obs is a dict
    #model.learn(total_timesteps=total_timesteps, progress_bar=True)


    #env = gym.make('PointMaze_UMazeDense-v3', render_mode="human", max_episode_steps=1000, maze_map=LARGE_MAZE)
    env = gym.make('Humanoid-v4', render_mode='human')
    observation, _= env.reset()
    '''observation er et dictionary med '''

    fitness = 0
    while True:
        action = env.action_space.sample()
        #action = model.predict(observation)[0]
        observation, reward, done, truncated, info = env.step(action)
        fitness += reward
        print(reward)
        if done or truncated:
            observation, _ = env.reset()
            print(fitness)
            fitness = 0

      # action = env.action_space.sample()
      # observation, reward, done, truncated, info = env.step(action)

      # if done or truncated:
      #     # Reset the environment if the episode is done
      #     observation = env.reset()

    # Close the environment
    env.close()