import gymnasium as gym
from stable_baselines3 import PPO
import os
import multiprocessing as mp
from multiprocessing import freeze_support
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from combined_LSTM import LSTMAutoencoder
from combined_env_wrapper import CombinedEnvironmentWrapper
from combined_callback import LSTMCallback
from combined_Autoencoder import Autoencoder, train_autoencoder
import torch
import time
from multiprocessing import Pool
import multiprocessing as mp
#from . import UR_gym
import sys
sys.path.append('/home/benjamin/Desktop/Master/Workspace_NoveltySearch')
import UR_gym

#maze = copy.deepcopy(LARGE_DECEPTIVE_MAZE)
# save_root = "Deceptive_maze_LSTM_alpha_fitAE_all_wheighted_no_determ_2"

def collect_states_multi(models, model_dir, env_name, states_per_model, max_steps, maze, cont_task):
    try:
        print(model_dir + models)
        # env = gym.make(env_name, max_episode_steps=1024, maze_map=maze, continuing_task=False, reward_type="dense")
        # env = gym.make(env_name, max_episode_steps=1600)
        env = gym.make(env_name, max_episode_steps=max_steps)
        
        # env = gym.make(env_name, max_episode_steps=max_steps, maze_map=maze, continuing_task=cont_task, reward_type="dense")
        model = PPO.load(model_dir + models, device="cpu")

        obs, _ = env.reset()
        s_idx = 0
        state_list = []

        while True:
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            state_list.append(obs)
            #state_list.append(obs['observation'])
            s_idx += 1
            
            if terminated or truncated:
                obs, _ = env.reset()

            if len(state_list) >= states_per_model: #s_idx >= states_per_model:
                env.close()
                break
        print(f"{models} collected {len(state_list)} states.")
        env.close()
        del model
        return state_list
    except Exception as e:
        print(f"Error in collect_states_multi: {e}")
        return []
    
if __name__ == '__main__':
    mp.set_start_method('forkserver')
    #mp.Queue(1000)
    freeze_support()
    for test_number in range(0, 1):
        save_root = f"UR_AE_LSTM_fit_eval_alpha_{test_number}"
        model_dir = f"tests/UR5/Combined_Final_test/AE_LSTM_fit/{save_root}/models/"
        log_dir = f"tests/UR5/Combined_Final_test/AE_LSTM_fit/{save_root}/logs"
        autoencoder_dir = f"tests/UR5/Combined_Final_test/AE_LSTM_fit/{save_root}/autoencoders/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(autoencoder_dir):
            os.makedirs(autoencoder_dir)

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



            
            
    #if __name__ == '__main__':
        t_start = time.time()

        cont_task = False
        use_LSTM = True     # for the callback
        open_maze_test = False  # for the environment wrapper
        if open_maze_test:
            LARGE_DECEPTIVE_MAZE = [[0,0,0,0,0],        # If the environment is unable to run without a goal due to the assertion, 
                                    [0,0,0,0,0],        # The goal could be added somewhere that cannot be reached by the agent.
                                    [0,0,"r",0,0],      # Starting the env as a continuing task and setting the goal to a random location could also be an option
                                    [0,0,0,0,0],
                                    [0,0,0,0,0]]
            cont_task = True
            
        # env_name = 'BipedalWalker-v3'
        # env_name = 'PointMaze_UMazeDense-v3'
        # env_name = 'Swimmer-v4'
        env_name = 'UR5DynReach-v1'

        iterations = 250#200
        n_env = 8
        n_steps_per_core = 512#1024#768#1536# 512 * 2
        steps_per_update = n_env * n_steps_per_core
        total_timesteps = iterations * steps_per_update

        state_seq_len = n_steps_per_core

        n_models = 8
        n_states = 32
        n_policies_for_AE = 10
        batch_size_AE = 256 
        epochs_AE = 150
        n_states_for_AE = epochs_AE * batch_size_AE * n_states    
        #env = gym.make(env_name, max_episode_steps=1024, maze_map=maze, continuing_task=False, reward_type="dense")
        env = gym.make(env_name)
        obs_space = env.observation_space.shape[0]

        model_files = os.listdir(model_dir)
        for i in range(len(model_files), n_models):
            print(f"Training model #{i}.")
            
            autoencoders = []
            files = os.listdir(autoencoder_dir)
            autoencoder_model_extension = ".pth" 
            autoencoder_models = [file for file in files if file.endswith(autoencoder_model_extension)]

            if autoencoder_models:
                print(f"{len(autoencoder_models)} trained autoencoder model(s) found in the folder.")
                print(f"ae names: {autoencoder_models}")
                for model_file in autoencoder_models:
                    #print(model_file)
                    autoencoder = Autoencoder((n_states, obs_space))  # Instantiate the model
                    autoencoder.load_state_dict(torch.load(autoencoder_dir + model_file))
                    # Put the model in evaluation mode
                    autoencoder.eval()
                    autoencoders.append(autoencoder)
            else:
                print("No trained autoencoder models found in the folder.")

            env_fns = [lambda: Monitor(CombinedEnvironmentWrapper(gym.make(env_name, max_episode_steps=n_steps_per_core), autoencoder_dir, n_states=n_states, obs_space=obs_space, autoencoders=autoencoders, mimic=False, open_maze_test=open_maze_test)) for _ in range(n_env)]  
            # env_fns = [lambda: Monitor(CombinedEnvironmentWrapper(gym.make(env_name, max_episode_steps=n_steps_per_core, maze_map=LARGE_DECEPTIVE_MAZE, continuing_task=cont_task, reward_type="dense"), autoencoder_dir, n_states=n_states, obs_space=obs_space, autoencoders=autoencoders, mimic=False, open_maze_test=open_maze_test)) for _ in range(n_env)]  
            env = SubprocVecEnv(env_fns)
            lstm_callback = LSTMCallback(env=env_name, verbose=1, n_env=n_env, total_timesteps=total_timesteps, 
                                        n_steps_per_update=steps_per_update, n_last_policies=n_policies_for_AE,
                                        model_dir=model_dir, n_states=state_seq_len, policy_number=i,
                                        use_LSTM=use_LSTM, open_map=open_maze_test)

            model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=n_steps_per_core, batch_size=32, n_epochs=3)
            model.learn(total_timesteps=total_timesteps, callback=lstm_callback, progress_bar=True)
            
            if autoencoder_models:
                del autoencoders
                del autoencoder
            env.close()
            del model

            t1 = time.time()
            model_files = os.listdir(model_dir + f"policy_{i}/")
            model_files = [file for file in model_files if file.endswith(".zip")]
            # model_extension = ".zip" 
            # models_for_ae = [file for file in files if file.endswith(autoencoder_model_extension)]
            # model_files = models_for_ae
            print(f"Collecting states from {len(model_files)} models...")
            print(model_files)
            states_per_model = n_states_for_AE // (len(model_files)) # - 1)
            state_lists = []
                # Multiprocessing
            
            with Pool(processes=7) as pool:
                jobs = []
                for model in model_files:
                    jobs.append(pool.apply_async(collect_states_multi, (model, model_dir + f"policy_{i}/", env_name, states_per_model, n_steps_per_core, LARGE_DECEPTIVE_MAZE, cont_task)))
                    
                for job in jobs:
                    print("Waiting for job completion...")
                    state_list = job.get()

                    state_lists.extend(state_list)
                    print(f"Total amount of states: {len(state_lists)}")
            
            pool.close()
            pool.join()
            pool.terminate()

            #state_lists = np.load('states.npy')
            print(f"State acquisition took: {time.time() - t1} seconds.")
            print(f"n states acquired: {len(state_lists)}")
            #np.save('states.npy', state_lists)

            t1 = time.time()

            autoencoder = Autoencoder((n_states, obs_space))
            train_autoencoder(autoencoder=autoencoder, state_list=state_lists, n_states=n_states, batch_size=batch_size_AE, ae_number=i, autoencoder_dir=autoencoder_dir)
            print(f"Autoencoder trained with {len(state_lists)} for {time.time() - t1} seconds")
            torch.save(autoencoder.state_dict(), autoencoder_dir + f'autoencoder_{i}.pth')


            del state_lists
            del autoencoder
        t_end = time.time()
        print(f"Training {n_models} models with {total_timesteps} timesteps took {t_end - t_start} seconds, {(t_end - t_start)/60} minutes or {((t_end - t_start)/60)/60} hours.")

