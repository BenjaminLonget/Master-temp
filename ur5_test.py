from cgi import test
from time import sleep
import gymnasium as gym
from pygments import highlight
from stable_baselines3 import PPO
import UR_gym
import os

def test_model(model_dir, env):
    # env = gym.make("UR5DynReach-v1", render_mode="human", max_episode_steps=100)
    # env = gym.make("UR5DynReach-v1",render_mode="human", max_episode_steps=500)
    #render=True, max_episode_steps=500)
    model = PPO.load(model_dir)
    obs, _ = env.reset()
    fitness = 0
    i = 0
    lowest_reward = 100
    higest_reward = -100
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info, truncated = env.step(action)
        #env.render()
        #sleep(0.5)
        fitness += reward
        # print(f"reward: {reward}")
        if reward < lowest_reward and not done:
            lowest_reward = reward

        if reward > higest_reward and not done:
            higest_reward = reward
        #sleep(0.01)
        if done:
            print(f"Episode {i}")
            #obs, _ = env.reset()
            print(f"Fitness: {fitness}")
            print(f"Final reward: {reward}")
            print(f"Lowest reward: {lowest_reward}")
            print(f"Higest reward: {higest_reward}")
            fitness = 0
            i = 0
            break
        i += 1

if __name__ == '__main__':
    model_dir = "tests/UR5/AE_test/UR5_manual_norm_reward_no_ee/models/"
    
    policies = os.listdir(model_dir)
    
    # test_model("tests/UR5/AE_test/UR5_manual_norm_reward/models/policy_0/model_8.zip", env)
    # input('Press enter to continue')
    for i in range(len(policies)):
        env = gym.make("UR5DynReach-v1",render_mode="human", max_episode_steps=1600)
        print(f"Model {i}")
        test_model(model_dir + f"policy_{i}" + "/model_final.zip", env)
        input('Press enter to continue')
        env.close()
        
    input('Press enter to continue')
    env.close()
    #test_model("tests/UR5/AE_test/UR5_deceptive_reach_2/best_models/policy_0/model_best.zip")
    #input('Press enter to continue')
    
    
    # env = gym.make("UR5DynReach-v1", render=True, max_episode_steps=500)
    # while True:
    #     observation, _= env.reset()
    #     fitness = 0
    #     i=0
    #     while True:
    #         action = env.action_space.sample()
    #         observation, reward, done, truncated, info = env.step(action)
    #         #print(observation)
    #         #input('Press enter to continue')
    #         fitness += reward
    #         sleep(0.1)
    #         if done or truncated:
    #             observation, _ = env.reset()
    #             print(f"Fitness: {fitness}")
    #             print(f"Final reward: {reward}")
    #             fitness = 0
    #             print(f"Episode {i}")
    #             i=0
    #             break
    #         i+=1