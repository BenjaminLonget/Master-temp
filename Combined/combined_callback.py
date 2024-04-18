import copy
import os
import re
import gymnasium as gym
import numpy as np
from itertools import chain
import copy
# Callback example from stable-baselines3 docs:
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from combined_LSTM import LSTMAutoencoder
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('TkAgg')

class LSTMCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env=None, verbose=0, n_env=1, total_timesteps = 0, n_steps_per_update = 0, n_last_policies = 0, model_dir = "/models", n_states=0, policy_number=0, use_LSTM=True, open_map=False):
        super().__init__(verbose)
        self.env_name = env
        self.n_env = n_env
        self.policy_number = policy_number
        self.total_timesteps = total_timesteps
        self.n_steps_per_rollout = n_steps_per_update
        self.model_dir = model_dir + f"policy_{policy_number}/"
        self.lstm_dir = self.model_dir.replace("models", "lstm_autoencoder")
        self.intermediate_model_dir = self.model_dir.replace("models", "intermediate_models")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.lstm_dir):
            os.makedirs(self.lstm_dir)
        self.best_model_dir = self.model_dir.replace("models", "best_models")
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)
        # self.fastest_model_dir = self.model_dir.replace("models", "fastest_models")
        # if not os.path.exists(self.fastest_model_dir):
        #     os.makedirs(self.fastest_model_dir)
        # self.final_good_model_dir = self.model_dir.replace("models", "final_good_models")
        # if not os.path.exists(self.final_good_model_dir):
        #     os.makedirs(self.final_good_model_dir)
        if not os.path.exists(self.intermediate_model_dir):
            os.makedirs(self.intermediate_model_dir)
        self.csv_path = self.model_dir + f"final_coordinates_with_LSTM_{self.policy_number}.csv"
        self.use_LSTM = use_LSTM
        self.open_map = open_map
        self.n_states = n_states
        self.state_lists = []
        env = gym.make(self.env_name)
        self.state_size = env.observation_space.shape[0]
        env.close()
        self.lstm_autoencoder = LSTMAutoencoder(seq_len=self.n_states, state_size=self.state_size, embedding_size=8, lr=0.001, epochs=10, max_grad_norm=1)

        self.n_last_policies = n_last_policies
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

        self.model_idx = 0

        self.alpha = 0.5
        self.alpha_max = 0.9
        self.alpha_min = 0.1
        self.alpha_step = 0.05
        self.quality_buffer = []

        self.initial_epsilon = 1.0
        self.epsilon = self.initial_epsilon
        self.decay_rate = 0.01
        self.final_epsilon = 0.0

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

    def exponential_decay(self, initial_epsilon, final_epsilon, decay_rate, episode):
        epsilon = final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-decay_rate * episode)
        self.logger.record("train/LSTM_epsilon", epsilon)
        return epsilon

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
            self.epsilon = self.exponential_decay(self.initial_epsilon, self.final_epsilon, self.decay_rate, self.it)


        if not self.open_map:
            infos = self.locals['infos']
            success_list = [info['success'] for info in infos]
            obs = self.model.rollout_buffer.observations
            for env in range(obs.shape[1]):
                if success_list[env]:
                    env_obs = obs[:, env, :]
                    empty_obs_filter = np.any(env_obs != 0, axis=1)
                    filtered_obs = env_obs[empty_obs_filter]
                    final_position = filtered_obs[-1, :2]

                    self.coordinates.append(final_position)
            

        return True

    def evaluate(self, force_evaluate=False):
        self.training_env.reset()
        #if self.it % 10 == 0 or force_evaluate:
        # env = gym.make(self.env_name, max_episode_steps=self.n_steps_per_rollout/self.n_env)
        # env.reset()

        eval_rewards, eval_steps = evaluate_policy(
            self.model,
            self.training_env,#env,
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


        if len(self.quality_buffer) >= 5:
            if np.mean(self.quality_buffer) > np.mean(eval_rewards):
                if self.alpha < self.alpha_max:
                    self.alpha += self.alpha_step
                else:
                    self.alpha = 0.5
            
            if np.mean(self.quality_buffer) < np.mean(eval_rewards):
                if self.alpha > self.alpha_min:
                    self.alpha -= self.alpha_step
                else:
                    self.alpha = 0.5

            self.quality_buffer.pop(0)
        #self.alpha = 0.5    # Force alpha to be 0.5 for now, uncomment to use alpha regulation
        self.quality_buffer.append(np.mean(eval_rewards))

        # For environments that can terminate early due to completion (maze and UR)
        if np.mean(eval_steps) < self.n_steps_per_rollout/self.n_env:
            self.model.save(self.best_model_dir + "final_good_model")
            
        if np.mean(eval_steps) < self.fastest_time:
            self.fastest_time = np.mean(eval_steps)
            self.model.save(self.best_model_dir + "fastets_model")


        if np.mean(eval_rewards) > self.best_eval_reward:# and np.mean(eval_steps) < self.fastest_time:
            self.best_eval_reward = np.mean(eval_rewards)
            self.model.save(self.best_model_dir + "model_best")
            # self.model.save(self.model_dir + "model_best")
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
        # LSTM_mean = np.mean(mse_list)
        # LSTM_std = np.std(mse_list)
        # LSTM_slope = 1.0/LSTM_std
        # mse_list = 2.0 / (1 + np.exp(-LSTM_slope * (mse_list - LSTM_mean))) - 1.0
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
    
    def plot_pareto(self, mse, rewards):
        for env_number in range(mse.shape[1]):
            plt.scatter(rewards[env_number], mse[env_number])

        plt.xlabel('Fitness')
        plt.ylabel('Novelty')
        plt.title('Fitness vs Novelty')
        #plt.legend()
        plt.grid(True)
        plt.show()

        pareto_front = []
        for env_number in range(mse.shape[1]):
            current_fit = rewards[env_number]
            current_novelty = mse[env_number]
            pareto_front.append(np.where((rewards <= current_fit) & (mse <= current_novelty))[1])

        print(f"Pareto front: {pareto_front}")
        pareto_rewards = rewards[pareto_front]
        pareto_mse = mse[pareto_front]
        # Plot all solutions
        plt.scatter(rewards, mse, color='blue', label='All Solutions')
        # Plot Pareto solutions
        plt.scatter(pareto_rewards, pareto_mse, color='red', label='Pareto Solutions')
        # Add labels and legend
        plt.xlabel('Fitness')
        plt.ylabel('Novelty')
        plt.legend()

    def update_rewards(self):
        print("Updating rewards")
        rollout_buffer = self.model.rollout_buffer
        states = rollout_buffer.observations
        # optima_detected = self.detect_local_optima()
        optima_detected = True
        mses, lstm_loss = self.calculate_mse(states, states.shape[1], rollout_buffer.rewards, optima_detected)
        mses = mses #* self.epsilon
        # print(f"mse shape: {mses.shape}")
        episode_starts = rollout_buffer.episode_starts
        # print(f"Episode starts: {sum(sum(episode_starts))}")
        rewards = copy.deepcopy(rollout_buffer.rewards)
        mean_fit = sum(sum(rewards)) / sum(sum(episode_starts)) # almost correct, using starts might skew the results for environments that hit the goal
        mean_mse_reward = sum(sum(mses)) / sum(sum(episode_starts))

        
        
        # self.logger.record("train/fitness_score", np.mean(mean_fit)) 
        # self.logger.record("train/mse_reward", np.mean(mean_mse_reward))
        self.logger.record("train/fitness_score", mean_fit) 
        self.logger.record("train/mse_reward", mean_mse_reward)
        self.logger.record("train/LTSM_loss", lstm_loss)

        # if self.it % 10 == 0:
        #     self.plot_pareto(mses, rewards)
        
        # self.model.rollout_buffer.rewards += mses   # A third of the reward from LSTM novelty, similar for reward coming from env wrapper with fit and AE novelty
        if self.use_LSTM:
            if self.policy_number == 0:
                '''Pure LSTM novelty'''
                # self.model.rollout_buffer.rewards = mses
                # print(f"mse shape: {np.array(mses).shape}")
                # print(f"reward shape: {np.array(rollout_buffer.rewards).shape}")
                '''Combined test with LSTM + AE novelty or LSTM + fitness or all'''
                self.model.rollout_buffer.rewards = self.model.rollout_buffer.rewards / 2 + mses / 2
                # self.model.rollout_buffer.rewards /= 2
            else:
                # self.model.rollout_buffer.rewards = self.model.rollout_buffer.rewards / 2 + mses / 2
                self.model.rollout_buffer.rewards = self.model.rollout_buffer.rewards + mses / 2  # essentially 1/3 when using both fit/2 and AE novelty/2
        
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
        # # print(f"obs: {obs.shape}")
        for env in range(obs.shape[1]):     # Final coordinates for maze.
            self.coordinates.append(obs[-1, env, :2])
        # print(f"coordinates: {self.coordinates}")

        # self.logger.record("train/novelty_score", np.mean(novelties))  # Mean of sums (mean episodic reward)

        # if (self.num_timesteps + (self.n_last_policies - 2) * self.n_steps_per_rollout) >= self.total_timesteps:
        #     print(f"saving model {self.model_idx}")
        #     self.model.save(self.model_dir + "model_" + str(self.model_idx))
        #     self.model_idx += 1
        
        # Evaluate policy performance, a single evaluation is fine if no stochasticity in the environment.
        self.evaluate(False)

        fitnesses, means, stds, alphas = zip(*self.training_env.env_method("get_fit", alpha=self.alpha))
        novelties = self.training_env.env_method("get_nov")
        alpha = self.alpha
        fitnesses = list(chain.from_iterable(entry for entry in fitnesses if entry))
        novelties = list(chain.from_iterable(entry for entry in novelties if entry))
        self.logger.record("train/fitness_from_env_wrapper", np.mean(fitnesses)) 
        self.logger.record("train/AE_novelty_score", np.mean(novelties))  # Mean of sums (mean episodic reward)
        self.logger.record("train/alpha", alpha)
        self.logger.record("train/AE_mse_means", np.mean(means))
        self.logger.record("train/AE_mse_stds", np.mean(stds))

        if self.it % 10 == 0:
            self.model.save(self.intermediate_model_dir + "model_it_" + str(self.it))

        if (self.num_timesteps + (self.n_last_policies - 2) * self.n_steps_per_rollout) >= self.total_timesteps:
            print(f"saving model {self.model_idx}")
            self.model.save(self.model_dir + "model_" + str(self.model_idx))
            self.model_idx += 1

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

    def plot_policy_scatter(self, coordinates):
        import pandas as pd
        cmap = mcolors.ListedColormap(['black', 'blue', 'green', 'red', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray'])

        x, y = zip(*coordinates)
        plt.scatter(x, y, marker='o', label=f"Policy {self.policy_number}")
        plt.legend()
        plt.title('Final coordinates of novel policies')
        plt.show()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        
        self.save_to_csv(self.coordinates, self.csv_path)
        self.save_to_csv(self.coordinates, self.best_model_dir + f"final_coordinates_with_LSTM_{self.policy_number}.csv")
        self.save_to_csv(self.coordinates, self.intermediate_model_dir + f"final_coordinates_with_LSTM_{self.policy_number}.csv")
        #self.plot_policy_scatter(self.coordinates)
        #self.plot_coordinates(self.coordinates)
        self.evaluate(True)
        self.model.save(self.model_dir + "model_final")
        torch.save(self.lstm_autoencoder.state_dict(), self.lstm_dir + f'lstm_{self.policy_number}.pth')
        del self.lstm_autoencoder
        pass
