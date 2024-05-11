from time import sleep
import gymnasium as gym
import cv2
import UR_gym

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

if __name__=='__main__':
    # env_name = 'BipedalWalker-v3'
    # env_name = 'PointMaze_UMazeDense-v3'
    # env_name = 'Swimmer-v4'
    # env_name = 'UR5DynReach-v1'
    # env = gym.make('PointMaze_UMazeDense-v3', maze_map=LARGE_DECEPTIVE_MAZE, render_mode='human')
    # env = gym.make('BipedalWalker-v3', render_mode='human', hardcore=True)
    # env = gym.make('UR5DynReach-v1', render_mode='human')
    env = gym.make('Swimmer-v4', render_mode='rgb_array', max_episode_steps=750, camera_id=0, width=860, height=640)
    
    env.reset()
    while True:

        env.step(env.action_space.sample())
        frame = env.render()
        # Show the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Frame', frame)
        sleep(0.02)
        cv2.waitKey(1)
        # input("Press Enter to continue...")