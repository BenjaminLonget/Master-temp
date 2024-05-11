from calendar import c
import copy
import os
import re
from typing import final
import gymnasium as gym
import numpy as np
from itertools import chain
import copy
# Callback example from stable-baselines3 docs:
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from LSTM import LSTMAutoencoder
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class LSTMCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env=None, verbose=0, n_env=1, total_timesteps = 0, n_steps_per_update = 0, model_dir = "/models", n_states=0, policy_number=0):
        super().__init__(verbose)
        self.env_name = env
        self.n_env = n_env
        self.policy_number = policy_number
        self.total_timesteps = total_timesteps
        self.n_steps_per_rollout = n_steps_per_update
        self.model_dir = model_dir + f"policy_{policy_number}/"
        self.lstm_dir = self.model_dir.replace("models", "lstm_autoencoder")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.lstm_dir):
            os.makedirs(self.lstm_dir)
        self.best_model_dir = self.model_dir.replace("models", "best_models")
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)
        self.csv_path = self.model_dir + "final_coordinates_with_LSTM.csv"
        
        self.n_states = n_states
        self.state_lists = []
        env = gym.make(self.env_name)
        self.state_size = env.observation_space.shape[0]
        env.close()
        self.lstm_autoencoder = LSTMAutoencoder(seq_len=self.n_states, state_size=self.state_size, embedding_size=2, lr=0.001, epochs=10, max_grad_norm=1)

        #added for maze
        self.save_if_hit = False
        self.fastest_time = float('inf')
        self.best_eval_reward = float('-inf')
        self.it = 0
        self.coordinates = []
        self.successfull_trajectories = []

        self.mean_reward_buffer = []
        self.local_optima_state_buffer = []
        self.local_optima_thresh = 20
        self.optima_count = 0
        
        self.highest_mse_reward = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.initial_params = copy.deepcopy(self.model.get_parameters())
        # print(f"Initial params: {self.initial_params}")
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def detect_local_optima(self):
        '''
        Trying to determine whether a local optima is reached.
        '''
        rewards = self.model.rollout_buffer.rewards
        obs = self.model.rollout_buffer.observations
        # current_mean_reward = np.mean(rewards)
        episode_starts = self.model.rollout_buffer.episode_starts
        current_mean_reward = sum(sum(rewards)) / sum(sum(episode_starts))
        
        if len(self.mean_reward_buffer) > 10:
            mean_change = abs(current_mean_reward - np.mean(self.mean_reward_buffer))
            print(f"Mean change: {mean_change}")
            self.logger.record("train/mean_change", mean_change)
            if mean_change < self.local_optima_thresh:
                print("Local optima detected.")
                self.model.save(self.model_dir + f"model_optima_{self.optima_count}")
                self.optima_count += 1
                # self.model.set_parameters(self.initial_params)
                # print(f"Initial params: {self.initial_params}")
                # print(f"Current params: {self.model.get_parameters()}")
                self.mean_reward_buffer.append(current_mean_reward)
                self.local_optima_state_buffer.append(obs)
                return True
            
            # self.mean_reward_buffer.pop(0)
            # self.local_optima_state_buffer.pop(0)

        self.mean_reward_buffer.append(current_mean_reward)
        self.local_optima_state_buffer.append(obs)
        return False


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
        if self.model.num_timesteps % self.n_steps_per_rollout == 0:

            self.update_rewards()
            self.it += 1
        
        infos = self.locals['infos']
        success_list = [info['success'] for info in infos]
        obs = self.model.rollout_buffer.observations
        for env in range(obs.shape[1]):
            if success_list[env]:
                env_obs = obs[:, env, :]
                empty_obs_filter = np.any(env_obs != 0, axis=1)
                filtered_obs = env_obs[empty_obs_filter]
                final_position = filtered_obs[-1, :2]
                # print(f"Successfull run detected in env {env}.")
                # print(f"All obs: {env_obs[np.any(env_obs != 0, axis=1)]}")
                # input("Press Enter to continue...")
                # print(f"Should be goal: {env_obs[np.any(env_obs != 0, axis=1)][-1, :2]}")
                # input("Press Enter to continue...")
                self.coordinates.append(final_position)
                # self.successfull_trajectories.append(obs[-1, env, :2])
            



        # if self.save_if_hit:
        #     print("Successfull run detected, aborting training.")
        #     return False
        # if infos["success"]:
        #     self.model.save(self.model_dir + "model_final")
        #     self.save_if_hit = True
        # if self.save_if_hit:
        #     print("Successfull run detected, aborting training.")
        #     return False
        return True

    def evaluate(self, force_evaluate=False):
        self.training_env.reset()
        #if self.it % 10 == 0 or force_evaluate:
        # env = gym.make(self.env_name)
        # env.reset()
        eval_rewards, eval_steps = evaluate_policy(
            self.model,
            self.training_env,
            n_eval_episodes=8,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
            warn=True,
        )
        # env.close()
        self.training_env.reset()
        self.logger.record("train/mean_evaluation_reward", np.mean(eval_rewards))
        self.logger.record("train/mean_evaluation_steps", np.mean(eval_steps))
        #self.it += 1
        #self.training_env.reset()
        
        # if a model completes the map, save the best one. This does not stop the training as autoencoder training is still needed.
        # timesteps are currently used for the maze env

        # if np.mean(eval_steps) < 1023 and np.mean(eval_rewards) > self.best_eval_reward:# and np.mean(eval_steps) < self.fastest_time:
        if np.mean(eval_rewards) > self.best_eval_reward:# and np.mean(eval_steps) < self.fastest_time:
            self.best_eval_reward = np.mean(eval_rewards)
            self.model.save(self.best_model_dir + "model_best")
            self.save_if_hit = True
            print("Successfull run detected, saving current best model.")
                

    def calculate_mse(self, state_sequence, n_envs, reward_list, train_lstm):
        mse_list = np.zeros_like(reward_list)
        lstm_loss = 0
        # if self.optima_count > 0:
        for env in range(n_envs):
            mse, next_state_estimate = self.lstm_autoencoder.get_novelty(state_sequence[:, env])
            mse_list[:, env] = mse
            highest_mse = np.max(mse)
            if highest_mse > self.highest_mse_reward:
                self.highest_mse_reward = highest_mse
                # mse_list[1:, env] = mse
                # print(f"mse_shape: {mse.shape}")
                # print(f"reward_shape: {reward_list[1:, env].shape}")
        mse_list = 2 * mse_list / self.highest_mse_reward - 1
        self.highest_mse_reward = 0

        if train_lstm:
            # n_states_in_optima = len(self.local_optima_state_buffer)
            # for sequences in self.local_optima_state_buffer:
            for env in range(n_envs):
                lstm_loss += self.lstm_autoencoder.train_LSTMAutoencoder(state_sequence[:, env])
                # lstm_loss += self.lstm_autoencoder.train_LSTMAutoencoder(sequences[:, env])
            # self.local_optima_state_buffer = []
            # self.mean_reward_buffer = []
            # self.local_optima_state_buffer.pop(0)
            # self.mean_reward_buffer.pop(0)

        return mse_list, lstm_loss/n_envs #lstm_loss/n_envs/n_states_in_optima
    
    def update_rewards(self):
        print("Updating rewards")
        rollout_buffer = self.model.rollout_buffer
        states = rollout_buffer.observations
        # optima_detected = self.detect_local_optima()
        optima_detected = True
        mses, lstm_loss = self.calculate_mse(states, states.shape[1], rollout_buffer.rewards, optima_detected)
        # print(f"mse shape: {mses.shape}")
        episode_starts = rollout_buffer.episode_starts
        # print(f"Episode starts: {sum(sum(episode_starts))}")
        rewards = copy.deepcopy(rollout_buffer.rewards)
        mean_fit = sum(sum(rewards)) / sum(sum(episode_starts)) # seems correct
        mean_mse_reward = sum(sum(mses)) / sum(sum(episode_starts))

        
        
        # self.logger.record("train/fitness_score", np.mean(mean_fit)) 
        # self.logger.record("train/mse_reward", np.mean(mean_mse_reward))
        self.logger.record("train/fitness_score", mean_fit) 
        self.logger.record("train/mse_reward", mean_mse_reward)
        self.logger.record("train/LTSM_loss", lstm_loss)
        
        self.model.rollout_buffer.rewards += mses
        # gauss = np.random.normal(0, 10, size=rollout_buffer.rewards.shape)
        # self.model.rollout_buffer.rewards += gauss
        # self.logger.record("train/mean_ep_gauss_reward", sum(sum(gauss)) / sum(sum(episode_starts)))

        mean_reward_combined = sum(sum(self.model.rollout_buffer.rewards)) / sum(sum(episode_starts))
        self.logger.record("train/combined_reward", mean_reward_combined)
        print(f"mean_fit: {mean_fit}, mean_mse_reward: {mean_mse_reward}, lstm_loss: {lstm_loss}, mean_reward_combined: {mean_reward_combined}")


        mean_fit = 0
        mean_mse_reward = 0
        lstm_loss = 0
        mean_reward_combined = 0

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.

        rollout_buffer conatins:
        observations: np.ndarray
        actions: np.ndarray
        rewards: np.ndarray
        advantages: np.ndarray
        returns: np.ndarray
        episode_starts: np.ndarray
        log_probs: np.ndarray
        values: np.ndarray

        info["success"]
        """

        obs = self.model.rollout_buffer.observations
        # print(f"obs: {obs.shape}")
        for env in range(obs.shape[1]):     # This only gives the final position, meaning it will not add positions of the ones that hit.
            # print(f"env: {obs[env]}")
            # print(f"obs: {obs[-1, env, :2]}")
            # self.coordinates.append(obs[env][-1][:2].tolist())
            self.coordinates.append(obs[-1, env, :2])
        # print(f"coordinates: {self.coordinates}")

        # self.logger.record("train/novelty_score", np.mean(novelties))  # Mean of sums (mean episodic reward)

        # if (self.num_timesteps + (self.n_last_policies - 2) * self.n_steps_per_rollout) >= self.total_timesteps:
        #     print(f"saving model {self.model_idx}")
        #     self.model.save(self.model_dir + "model_" + str(self.model_idx))
        #     self.model_idx += 1
        
        # Evaluate policy performance, a single evaluation is fine if no stochasticity in the environment.
        self.evaluate(False)

        pass

    def plot_coordinates(self, coordinates):
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
        cmap = mcolors.ListedColormap(['white', 'gray'])
        #map_data = np.logical_xor(map_data, 1).astype(int)
        plt.imshow(LARGE_DECEPTIVE_MAZE_NUMERIC, cmap=cmap, origin='upper', extent=(-len(LARGE_DECEPTIVE_MAZE_NUMERIC[0])/2, len(LARGE_DECEPTIVE_MAZE_NUMERIC[0])/2, -len(LARGE_DECEPTIVE_MAZE_NUMERIC)/2, len(LARGE_DECEPTIVE_MAZE_NUMERIC)/2))

        # for i, trajectory in enumerate(trajectories):
        #     x, y = zip(*trajectory)
        plt.plot(0, 2, color="green", marker='o', markersize=10, label="Start")
        plt.plot(0, -4, color="orange", marker='o', markersize=15, label="Goal")
        for point in coordinates:
            x, y = point
            plt.plot(x, y, color='red', marker='o', markersize=2)
        for point in self.successfull_trajectories:
            x, y = point
            plt.plot(x, y, color='blue', marker='o', markersize=2)

        plt.legend()
        plt.title('Final coordinates during exploration of double-deceptive Map')
        plt.show()
        
    def save_to_csv(self, data, filename):
        import pandas as pd 
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, header=False) # mode='a' to append to existing file

        # with open(filename, 'w', newline=''):  # Creates the empty csv file
        #     pass
        # with open(filename, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data)


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        
        self.save_to_csv(self.coordinates, self.csv_path)
        self.plot_coordinates(self.coordinates)
        self.evaluate(True)
        self.model.save(self.model_dir + "model_final")
        torch.save(self.lstm_autoencoder.state_dict(), self.lstm_dir + f'lstm_{self.policy_number}.pth')
        pass
