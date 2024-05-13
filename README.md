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

<img src="imgs/Swimmer/Good_swimmer_comparison.png" alt="image" style="width:500px;height:auto;">,

Comparing with the benchmark from https://spinningup.openai.com/en/latest/spinningup/bench.html

<img src="imgs/Swimmer/Swimmer_bench.png" alt="image" style="width:400px;height:auto;">

## Maze
Fitness with high reward noise result:
<img src="imgs/Maze/fitness_w_reward_noise.png" alt="image" style="width:400px;height:auto;">

### Linear Autoencoder
#### Deceptive Maze Trajectories
<img src="imgs/Maze/DMaze_AE_4.svg" alt="image" style="width:400px;height:auto;">

#### Open Maze Variant
<img src="imgs/Maze/pure_ae_trajectories.png" alt="image" style="width:400px;height:auto;">
<img src="imgs/Maze/ae_fit_trajectories.png" alt="image" style="width:400px;height:auto;">

### LSTM-Autoencoder
#### Deceptive Maze Scatterplot
<img src="imgs/Maze/deceptive_LSTM_fit_scatter.png" alt="image" style="width:400px;height:auto;">



## UR5
<img src="gifs/UR/policy_0.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_1.gif" alt="image" style="width:300px;height:auto;"> 
<img src="gifs/UR/policy_2.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_3.gif" alt="image" style="width:300px;height:auto;"> 
<img src="gifs/UR/policy_4.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_5.gif" alt="image" style="width:300px;height:auto;"> 
<img src="gifs/UR/policy_6.gif" alt="image" style="width:300px;height:auto;"> ,
<img src="gifs/UR/policy_7.gif" alt="image" style="width:300px;height:auto;"> 
