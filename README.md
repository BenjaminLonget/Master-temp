# Novelty Search Using Generalized Behavior Metrics - Master Thesis
This repository contains the spaghetti code I've used in my masters thesis
## Abstract

## Environments

## Interesting Walker Behaviors
<img src="gifs/Interesting_walker_gifs/generic_fit.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/back_leg_jumper.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/decisive_walking.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/fast_crawler.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/fast_skipper.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/front_leg_jumper_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/smooth_runner.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/Interesting_walker_gifs/two_legged_jumper.gif" alt="image" style="width:300px;height:auto;"> 

## Swimmer vs Benchmark
The generic fitness based Swimmer opts to swim sideways, as this obtains a constant positive reward.
One specific behavior for the Swimmer turned out to be quite decent. Here the Swimmer learns to swim more like an eel where the overall reward is much higher, but a lot of the stepwise rewards are in the negative.

<img src="gifs/Swimmer_0/policy_0.gif" alt="image" style="width:300px;height:auto;">, 
<img src="gifs/Swimmer_1/policy_3.gif" alt="image" style="width:300px;height:auto;"> 

Note that these gifs does not show the episode to completion. The final evaluation reward is used to generate the following graph:

<img src="imgs/Swimmer/Good_swimmer_comparison.png" alt="image" style="width:500px;height:auto;">

Comparing with the benchmark from https://spinningup.openai.com/en/latest/spinningup/bench.html

<img src="imgs/Swimmer/Swimmer_bench.png" alt="image" style="width:400px;height:auto;">

## Maze
Fitness with high reward noise result:

<img src="imgs/Maze/fitness_w_reward_noise.png" alt="image" style="width:400px;height:auto;">

### Linear Autoencoder
#### Deceptive Maze Trajectories
<img src="imgs/Maze/DMaze_AE_4.svg" alt="image" style="width:400px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/pure_ae_trajectories.png" alt="image" style="width:400px;height:auto;"> ,
<img src="imgs/Maze/ae_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### LSTM-Autoencoder
#### Deceptive Maze Scatterplot
<img src="imgs/Maze/deceptive_LSTM_fit_scatter.png" alt="image" style="width:400px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/pure_lstm_open_trajectories.png" alt="image" style="width:400px;height:auto;">, 
<img src="imgs/Maze/lstm_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### Combined Novelty Search
#### Deceptive Maze Dual Trajectories
<img src="imgs/Maze/deceptive_LSTM_AE_fit_traj.png" alt="image" style="width:400px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/open_lstm_and_ae_traj.png" alt="image" style="width:400px;height:auto;"> ,
<img src="imgs/Maze/lstm_ae_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### Frechet Distance Results
5 tests on the open maze variant were conducted, where the Frechet distance with a threshold of 2.5 was used to determine whether trajectories were different. The tests using linear autoencoders had to be above the treshold for all other trajectories to be deemed unique. The tests where only LSTM-AE or fitness were used had to have trajectories being different for intermediate policies that were saved 10 iterations apart. 

The final average results are: 

| Test  | Average Score |
| ------------- | ------------- |
| Fitness  | 3.0 / 10.0  |
| Linear AE  | 9.6  |
| Linear AE + Fitness  | 9.6  |
| LSTM-AE  | 8.6  |
| LSTM-AE + Fitness  | 5.4  |
| Linear AE + LSTM-AE  | 9.8  |
| Linear AE + LSTM-AE + Fitness  | 8.8  |

## UR5
<img src="gifs/UR/policy_0.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_1.gif" alt="image" style="width:300px;height:auto;"> 
<img src="gifs/UR/policy_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_3.gif" alt="image" style="width:300px;height:auto;"> 
<img src="gifs/UR/policy_4.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_5.gif" alt="image" style="width:300px;height:auto;"> 
<img src="gifs/UR/policy_6.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_7.gif" alt="image" style="width:300px;height:auto;"> 
