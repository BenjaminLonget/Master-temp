from Dynamics_RND import dynamics_main, maze_test
import os

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-bipedal-walker.txt')
    # dynamics_main.run(config_path)
    
    config_path = os.path.join(local_dir, 'config-maze-neat.txt')
    maze_test.run(config_path)