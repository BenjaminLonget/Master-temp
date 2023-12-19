'''
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.MultiInputPolicy

deterministic=True for PPO agenten'''
from gc import freeze
from multiprocessing import freeze_support
from tracemalloc import start
# from venv import logger
from cv2 import exp, mean
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import os
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
import copy
from itertools import chain

save_root = "BipedalWalker-v3-SAC-dyn_sigmoid"
model_dir = f"PPO/{save_root}/models/"
log_dir = f"PPO/{save_root}/logs"
autoencoder_dir = f"PPO/{save_root}/autoencoders/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(autoencoder_dir):
    os.makedirs(autoencoder_dir)


# Callback example from stable-baselines3 docs:
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env=None, verbose=0, n_env=1, total_timesteps = 0, n_steps_per_update = 0, n_last_policies = 0, model_dir = "/models", n_states=0, policy_number=0):
        super().__init__(verbose)
        
        self.n_env = n_env
        self.total_timesteps = total_timesteps
        self.n_steps_per_rollout = n_steps_per_update
        self.n_last_policies = n_last_policies
        self.model_dir = model_dir + f"policy_{policy_number}/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.n_states = n_states
        self.state_lists = []
        self.model_idx = 0
        self.start_PPO_time = time.time()


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        '''for env_idx, info in enumerate(self.locals['infos']):
            if env_idx == ENV_INDEX:  
                current_state = info['current_state']
                self.collected_states.append(current_state)'''
        # dones = self.training_env.unwrapped.get_attr("done")
        # print(dones)
        # self.locals.ge
        # novelty_lists = self.training_env.get_attr("novlist")[0]
        # fitness_lists = self.training_env.get_attr("fitlist")[0]
        # dones = self.training_env.get_attr("done")[0]
        # truncateds = self.training_env.get_attr("truncated")[0]
        # # condition = np.logical_or(dones, truncateds)
        # # if np.any(condition):
        # if dones or truncateds:
        #     # indexes = list(map(int, np.where(condition)[0]))
        #     #print(f"idx: {indexes}")
        #     # indexes = np.where(condition)
        #     # print(indexes)
        #     # novelty = [novelty_lists[i-1] for i in indexes]
        #     # fitness = [fitness_lists[i-1] for i in indexes]


        #     # novelty = self.training_env.get_attr("novlist", indexes)
        #     # fitness = self.training_env.get_attr("fitlist", indexes)
        #     self.logger.record("train/novelty_score", np.mean(novelty_lists))
        #     self.logger.record("train/fitness_score", np.mean(fitness_lists))
            #print(f"nov: {np.mean(novelty)}, fit: {np.mean(fitness)}")

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Might be triggered once per environment??
        """
        fitnesses = self.training_env.env_method("get_fit")
        novelties = self.training_env.env_method("get_nov")

        fitnesses = list(chain.from_iterable(entry for entry in fitnesses if entry))
        novelties = list(chain.from_iterable(entry for entry in novelties if entry))
        
        self.logger.record("train/fitness_score", np.mean(fitnesses)) 
        self.logger.record("train/novelty_score", np.mean(novelties))  # Mean of sums (mean episodic reward)
        # print(fitnesses)
        # self.training_env.unwrapped.set_attr("fit_mean_list", [])
        # save models for the n_last_policies
        # print(f"num_timesteps: {self.num_timesteps}")
        # print(f"- stuff: {(self.n_last_policies - 1) * self.n_steps_per_rollout}")
        if (self.num_timesteps + (self.n_last_policies - 2) * self.n_steps_per_rollout) > self.total_timesteps:
            print(f"saving model {self.model_idx}")
            self.model.save(self.model_dir + "model_" + str(self.model_idx))
            self.model_idx += 1
        
        pass


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Might need to first save the final model as well?
        print(f"saving model_final")
        self.model.save(self.model_dir + "model_final")

        pass


def collect_states_multi(models, model_dir, env_name, states_per_model):
    try:
        print(model_dir + models)

        env = gym.make(env_name)
        model = PPO.load(model_dir + models, device="cpu")

        obs, _ = env.reset()
        s_idx = 0
        state_list = []

        while True:
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action[0])
            state_list.append(obs)
            s_idx += 1
            
            if terminated or truncated:
                obs, _ = env.reset()

            if s_idx >= states_per_model:
                env.close()
                break
        print(f"{models} collected {s_idx} states.")
        return state_list
    except Exception as e:
        print(f"Error in collect_states_multi: {e}")
        return []

if __name__ == '__main__':
    t_start = time.time()
    # mp.set_start_method('spawn')
    mp.set_start_method('forkserver')
    env_name = 'BipedalWalker-v3'
    n_models = 2
    n_states = 16   # amount of states the autoencoder should use as a sequence, stride in paper
                    # Theres a posibillity that they call this epochs or batch-size for the AE network training
    n_env = 8
    total_timesteps = 2500000    # Enough to actually learn a good policy, different for each environment
    n_steps_per_update = 9984  # 12000 according to TNB-paper 2048 according to stable baselines
    n_steps_per_core = int(n_steps_per_update // n_env)
    n_policies_for_AE = 10
    batch_size_AE = 512
    epochs_AE = 200
    n_states_for_AE = epochs_AE * batch_size_AE * n_states    # 200 "epochs" with a batchsize of 1024 as per TNB-paper
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    input_dim_flat = n_states * obs_space
    #print(f"input_dim: {input_dim_flat}")
    freeze_support()
    #for i in range(3, n_models + 1):
    for i in range(n_models):
        print(f"Training model #{i}.")
        
        autoencoders = []
        files = os.listdir(autoencoder_dir)
        autoencoder_model_extension = ".pth"  # Adjust the extension as needed
        autoencoder_models = [file for file in files if file.endswith(autoencoder_model_extension)]

        if autoencoder_models:
            print(f"{len(autoencoder_models)} trained autoencoder model(s) found in the folder.")
            print(f"ae names: {autoencoder_models}")
            for model_file in autoencoder_models:
                print(model_file)
                autoencoder = Autoencoder((n_states, obs_space))  # Instantiate the model
                autoencoder.load_state_dict(torch.load(autoencoder_dir + model_file))
                # Put the model in evaluation mode
                autoencoder.eval()
                autoencoders.append(autoencoder)
        else:
            print("No trained autoencoder models found in the folder.")
        
        env_fns = [lambda: Monitor(CustomEnvironmentWrapper(gym.make(env_name), autoencoder_dir, input_dim_flat, n_states=n_states, obs_space=obs_space, autoencoders=autoencoders)) for _ in range(n_env)]  # Adjust the number of environments as needed
        # Create the parallelized vectorized environment
        env = SubprocVecEnv(env_fns)
        
        #env = make_vec_env(CustomEnvironmentWrapper(gym.make(env_name), autoencoder_dir, input_dim_flat), n_envs=8, seed=0, vec_env_cls=SubprocVecEnv) # Helper class provided by stable baselines3

        '''You have specified a mini-batch size of 64, but because the `RolloutBuffer` is of size `n_steps * n_envs = 12000`, after every 187 untruncated mini-batches, there will be a truncated mini-batch of size 32
        We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
        Info: (n_steps=1500 and n_envs=8)'''
        custom_callback = CustomCallback(env=env_name, verbose=1, n_env=n_env, total_timesteps=total_timesteps, 
                                        n_steps_per_update=n_steps_per_update, n_last_policies=n_policies_for_AE, 
                                        model_dir=model_dir, n_states=n_states_for_AE, policy_number=i)

        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, buffer_size=n_steps_per_update, batch_size=32, stats_window_size=50) #n_epochs=3, n_steps=n_steps_per_core

        # Set the logger for each environment
        # for i in range(n_env):
        #     env.venv.envs[i].unwrapped.set_logger(model.logger)
        #env.env_method("set_logger", model.logger(), indices=2)

        model.learn(total_timesteps=total_timesteps, callback=custom_callback, progress_bar=True)
        # num_cpu affects the amount of data gathered before each update step of the PPO model
        # n_steps=2048 * 8 cpu cores = 16384 interactions with the environment before policy update
        # n_steps should be 256 for it to equal a single core process
        
        
        #model.env.env_method("display_novelty")

        # Find highest mse for this policy on other AE's -> done in env:
        if autoencoder_models:
            del autoencoders
            del autoencoder

        env.close()
        del model
        
        t1 = time.time()
        model_files = os.listdir(model_dir + f"policy_{i}/")
        print(f"Collecting states from {len(model_files)} models...")
        print(model_files)
        states_per_model = n_states_for_AE // len(model_files)
        state_lists = []
            # Multiprocessing
        
        with Pool(processes=7) as pool:
            jobs = []
            for model in model_files:
                jobs.append(pool.apply_async(collect_states_multi, (model, model_dir + f"policy_{i}/", env_name, states_per_model)))
                
            for job in jobs:
                print("Waiting for job completion...")
                state_list = job.get(timeout=6000)

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
        train_autoencoder(autoencoder=autoencoder, state_list=state_lists, n_states=n_states, batch_size=batch_size_AE, ae_number=i)
        print(f"Autoencoder trained with {len(state_lists)} for {time.time() - t1} seconds")
        torch.save(autoencoder.state_dict(), autoencoder_dir + f'autoencoder_{i}.pth')
        # Find lowest mse for the policy's own AE
        # env.env_method("set_lowest_mse", (state_lists, n_states, batch_size_AE, autoencoder))
        # autoencoder.eval()
        # env.unwrapped.env_method("set_lowest_mse", state_lists, n_states, batch_size_AE, i)

        del state_lists
        del autoencoder
    t_end = time.time()
    print(f"Training {n_models} models with {total_timesteps} timesteps took {t_end - t_start} seconds, {(t_end - t_start)/60} minutes or {((t_end - t_start)/60)/60} hours.")

        #test below:
        # model = PPO.load("PPO/close_to_300.zip")
        # env = gym.make('BipedalWalker-v3', render_mode='human')
        # obs, _ = env.reset()
        # rewards = 0
        # autoencoder = Autoencoder(input_dim_flat)  # Instantiate the model
        # autoencoder.load_state_dict(torch.load("PPO/autoencoders/autoencoder_0.pth"))
        # autoencoder.eval()
        # obs_buffer = []
        # while True:
        #     action = model.predict(obs)
        #     obs, reward, terminated, truncated, info = env.step(action[0])
        #     obs_buffer.append(obs)
        #     rewards += reward
        #     if len(obs_buffer) == n_states:
        #             mini_batch = torch.tensor(np.stack(obs_buffer, axis=0), dtype=torch.float32, device="cuda")
        #             mini_batch = mini_batch.view(-1, np.shape(obs_buffer)[0] * np.shape(obs_buffer)[1])
        #             #error = abs(np.mean(autoencoder(mini_batch).cpu().numpy() - mini_batch.cpu().numpy()))
        #             ae_out = autoencoder(mini_batch)
        #             error = np.mean((ae_out.detach().cpu().numpy() - mini_batch.cpu().numpy()) ** 2)
        #             print(f"err: {error}")
        #             obs_buffer.pop(0)   # Rolling buffer
        #     if(terminated or truncated):

        #         print(f"reward: {rewards}")
        #         rewards = 0
        #         obs, _ = env.reset()

    '''PPO policies: MlpPolicy, CnnPolicy, MultiInputPolicy'''
