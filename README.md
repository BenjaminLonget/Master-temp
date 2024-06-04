# Novelty Search Using Generalized Behavior Metrics - Master Thesis
> Thesis submitted at University of Souther Denmark for MSc in Advanced Robotics.
> This repository contains the code devised for the project, as well as gifs and images that better show the dynamical models behaving differently.
> The latest version of the code is found in the [Combined](Combined) folder.

All test result are found at [TestRepo](https://github.com/BenjaminLonget/Master_thesis_test_results)

## Author
*  Benjamin B. Longet

## Supervisor
*  Anders L. Christensen

## Abstract
In the domain of evolutionary computation and reinforcement learning, novelty search encompasses algorithms
aimed at enhancing exploration by emphasizing diversity of the phenotypes encountered by a model. By
prioritizing novel behavior, these algorithms enable the model to break free from the monotonous, preconceived
behaviors often enforced by the fitness function, thereby overcoming deception in complex domains. The
effectiveness of novelty search hinges entirely on the careful design of the behavior metric, which typically
must be tailored specially to a task. Expert knowledge is therefore required to reap the benefits of novelty
search.

In this project, we propose to generalize the behavior descriptor based on sequential state data. We investigate
the application of novelty search across a diverse set of environments, with this generalized metric. To accomplish
generalization, two types of novelty search are applied. An off-policy version, which attempts to train multiple
novel policies by adding a novelty score to the reward, based on the reconstruction error from a linear autoencoder.
An on-policy version is meant to overcome immediate deception by training a LSTM-autoencoder on all state
trajectories encountered during exploration.

We test the two algorithms, separately and in combination, across four distinct environments: Bipedal Walker,
Swimmer, Deceptive Maze, and UR5. To visualize and measure the efficacy of the proposed approach, an open
version of the maze without a target position is utilized. Using the Fréchet distance on resulting models in
an open version of the maze environment, we determine the diversity of intermediate policies found with the
on-policy algorithm. This measurement is also used to determine the diversity between the final models, found
with the novel policy-seeking algorithm.

Generalized behavior descriptors enables the off-policy version to find a wide array of distinct behaviors across
all domains, while maintaining the ability to complete the specific tasks. Numerous behaviors are found for
the Walker, one of the found behaviors for the Swimmer even outperforms the existing benchmarks, multiple
inverse kinematics solutions are found for the UR5, and multiple distinct solutions are found for the Deceptive
Maze.

The on-policy version allowed the Deceptive Maze to be solved consistently, usually within just a handful of
iterations. The combination of both algorithms find both of the two distinct paths through the maze using two
separate policies.

## Environments
Four different environments running on 3 different physics engines are used to show the level of generalization achieved in this project.
### Box2D BipedalWalker-v3
<img src="imgs/Env/my_walker_env.png" alt="image" style="width:300px;height:auto;">

### MuJoCo Swimmer-v4
<img src="imgs/Env/swimmer_env_2.png" alt="image" style="width:300px;height:auto;">

### PyBullet UR5
This environment is based on the implementation from https://github.com/WanqingXia/HiTS_Dynamic/tree/main.

<img src="imgs/Env/UR_env.png" alt="image" style="width:300px;height:auto;">

### MuJoCo PointMaze
Some slight changes are made to the original reward function, these are found in [maze](maze) and should replace the corresponding folders in the GymnasiumRobotics site-packages.

#### Deceptive
<img src="imgs/Env/Deceptive_maze_env.png" alt="image" style="width:300px;height:auto;">

Additionally, an open variant of the maze is used, where all walls are removed, and the fitness score is either changed to be a factor of the x-velocity or removed all together.

## Interesting Walker Behaviors
None of the encoutered novel walker behaviors had any major increase in sample effeciency or final reward compared to the fitness-based solution. A significant amount of vastly different behaviors were encountered, many of which were able to achieve a decent score, often as good as the fitness-based solution. A few selected behaviors are seen in the following gifs, the first of which being the behavior that usually occurs from following the fitness gradient alone.

<img src="gifs/Interesting_walker_gifs/generic_fit (copy).gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/back_leg_jumper.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/decisive_walking (copy).gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/fast_crawler.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/fast_skipper.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/front_leg_jumper_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/smooth_runner.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/two_legged_jumper.gif" alt="image" style="width:300px;height:auto;"> 

## Swimmer vs Benchmark
The generic fitness based Swimmer opts to swim sideways, as this obtains a constant positive reward.
One specific behavior for the Swimmer turned out to be quite decent. Here the Swimmer learns to swim more like an eel where the overall reward is much higher, but a lot of the stepwise rewards are in the negative. This behavior is quite rare, most of the other encountered behaviors seem to primarily follow the novelty gradient.

<img src="gifs/Swimmer_1/policy_0.gif" alt="image" style="width:300px;height:auto;">, 
<img src="gifs/Swimmer_1/policy_1.gif" alt="image" style="width:300px;height:auto;">,
<img src="gifs/Swimmer_1/policy_2.gif" alt="image" style="width:300px;height:auto;">,
<img src="gifs/Swimmer_1/policy_3.gif" alt="image" style="width:300px;height:auto;">, 
<img src="gifs/Swimmer_1/policy_4.gif" alt="image" style="width:300px;height:auto;">,
<img src="gifs/Swimmer_1/policy_5.gif" alt="image" style="width:300px;height:auto;">

Note that these gifs does not show the episode to completion. The evaluation reward after every policy update is used to generate the following graph, comparing policy 3 from above, a similar policy found with the combined algorithm, and the pure fitness based solution:

<img src="imgs/Swimmer/Good_swimmer_comparison.png" alt="image" style="width:800px;height:auto;">

This specific behavior outperforms the benchmark, found at https://spinningup.openai.com/en/latest/spinningup/bench.html, both in terms of final score, and i terms of timesteps to converge:

<img src="imgs/Swimmer/Swimmer_bench.png" alt="image" style="width:700px;height:auto;">

## UR5
5 tests were conducted with the linear AE, training 8 novel policies. The average amount of different inverse kinematics solutions found were 3.8 / 8.0

The following gifs show the result of a test that found 5 sepperate configurations:

<img src="gifs/UR/policy_0.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_1.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_3.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_4.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_5.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_6.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_7.gif" alt="image" style="width:300px;height:auto;"> 

## Maze
Fitness with high reward noise result:

<img src="imgs/Maze/fitness_w_reward_noise.png" alt="image" style="width:650px;height:auto;">

### Linear Autoencoder
#### Deceptive Maze Trajectories
<img src="imgs/Maze/DMaze_AE_4.svg" alt="image" style="width:650px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/pure_ae_trajectories.png" alt="image" style="width:400px;height:auto;"> ,
<img src="imgs/Maze/ae_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### LSTM-Autoencoder
#### Deceptive Maze Scatterplot
<img src="imgs/Maze/deceptive_LSTM_fit_scatter.png" alt="image" style="width:650px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/pure_lstm_open_trajectories.png" alt="image" style="width:400px;height:auto;">, 
<img src="imgs/Maze/lstm_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### Combined Novelty Search
#### Deceptive Maze Dual Trajectories
<img src="imgs/Maze/deceptive_LSTM_AE_fit_traj.png" alt="image" style="width:650px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/open_lstm_and_ae_traj.png" alt="image" style="width:400px;height:auto;"> ,
<img src="imgs/Maze/lstm_ae_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### Frechet Distance Results
5 tests on the open maze variant were conducted, where the Frechet distance with a threshold of 2.5 was used to determine whether trajectories were different. The tests using linear autoencoders had to be above the treshold for all other trajectories to be deemed unique. The tests where only LSTM-AE or fitness were used had to have trajectories being different for intermediate policies that were saved 10 iterations apart. 

The final average results are: 

| Test  | Average Score |
| ------------- | ------------- |
| Fitness  | 3.0 / 10.0  |
| Linear AE  | 9.6 / 10.0  |
| Linear AE + Fitness  | 9.6 / 10.0  |
| LSTM-AE  | 8.6 / 10.0  |
| LSTM-AE + Fitness  | 5.4 / 10.0  |
| Linear AE + LSTM-AE  | 9.8 / 10.0  |
| Linear AE + LSTM-AE + Fitness  | 8.8 / 10.0  |

## Combined Novelty Search on the Swimmer and Walker
### Swimmer
<img src="gifs/Swimmer_combined/policy_0.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Swimmer_combined/policy_1.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Swimmer_combined/policy_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Swimmer_combined/policy_3.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Swimmer_combined/policy_4.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Swimmer_combined/policy_5.gif" alt="image" style="width:300px;height:auto;"> 

### Walker
<img src="gifs/Walker_combined/policy_0.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Walker_combined/policy_1.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Walker_combined/policy_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Walker_combined/policy_3.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Walker_combined/policy_4.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Walker_combined/policy_5.gif" alt="image" style="width:300px;height:auto;"> ,
