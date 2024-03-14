import gymnasium as gym
from stable_baselines3 import PPO
import os
import multiprocessing as mp
from multiprocessing import freeze_support
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from LSTM import LSTMAutoencoder
from LSTM_env_wrapper import LSTMEnvironmentWrapper
from LSTM_callback import LSTMCallback

#maze = copy.deepcopy(LARGE_DECEPTIVE_MAZE)
save_root = "Deceptive_maze_LSTM_simple_2"
model_dir = f"tests/Deceptive_maze/LSTM_AE/{save_root}/models/"
log_dir = f"tests/Deceptive_maze/LSTM_AE/{save_root}/logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
    mp.set_start_method('forkserver')
    freeze_support()

    # env_name = 'BipedalWalker-v3'
    env_name = 'PointMaze_UMazeDense-v3'
    iterations = 100
    n_env = 8
    n_steps_per_core = 1024
    steps_per_update = n_env * n_steps_per_core
    total_timesteps = iterations * steps_per_update

    state_seq_len = 1

    i = 14
    # for i in range(1, 5):
    # max_episode_steps=1024, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=False
    env_fns = [lambda: Monitor(LSTMEnvironmentWrapper(gym.make(env_name, max_episode_steps=n_steps_per_core, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=False))) for _ in range(n_env)]  
    env = SubprocVecEnv(env_fns)
    lstm_callback = LSTMCallback(env=env_name, verbose=1, n_env=n_env, total_timesteps=total_timesteps, n_steps_per_update=steps_per_update, model_dir=model_dir, n_states=state_seq_len, policy_number=i)

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=n_steps_per_core, batch_size=32, n_epochs=3)
    model.learn(total_timesteps=total_timesteps, callback=lstm_callback, progress_bar=True)
    
    env.close()
    # input("Press Enter to continue...")
    # env = gym.make(env_name, max_episode_steps=n_steps_per_core)
    # state_size = env.observation_space.shape[0]
    # seq_size = 10
    # lstmautoencoder = LSTMAutoencoder(seq_len=seq_size, state_size=state_size, embedding_size=4, lr=0.001, epochs=100, max_grad_norm=1)
    # state_sequence_pre = []
    # model_path = "tests/BipedalWalker/AE_test/BipedalWalker-autoencoder_w_det_2/models/policy_0/model_final.zip"
    # model = PPO.load(model_path)
    # obs, _ = env.reset()
    # state_seq_collection = []
    # for i in range(n_steps_per_core*10):
    #     action = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action[0])
    #     state_sequence_pre.append(obs)
    #     if len(state_sequence_pre) == seq_size:
    #         #print(f"state_sequence_pre: {state_sequence_pre}")
    #         mse, est = lstmautoencoder.get_novelty(state_sequence_pre[-10:])
    #         #input("Press Enter to continue...")
    #         state_seq_collection.append(state_sequence_pre)
    #         state_sequence_pre = state_sequence_pre[-5:]
    #         #state_sequence_pre.pop(0)
    #         print(f"mse: {mse}")
    #     if terminated or truncated:
    #         obs, _ =env.reset()

    # lstmautoencoder.train_LSTMAutoencoder(state_seq_collection)
    # input("Press Enter to continue...")

    # state_sequence_pre = []
    # obs, _ = env.reset()
    # for i in range(n_steps_per_core):
    #     action = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action[0])
    #     state_sequence_pre.append(obs)
    #     if len(state_sequence_pre) == seq_size:
    #         mse, est = lstmautoencoder.get_novelty(state_sequence_pre[-10:])
    #         print(f"mse: {mse}")
    #         state_sequence_pre.pop(0)
    #     if terminated or truncated:
    #         obs, _ =env.reset()