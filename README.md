# Tabular Reinforcement Learning

# Description 
This repository contains the work done for the practical assignment for the Reinforcement Learning course where all the methods that represent the basic principles in a value-based tabular reinforcement learning are considered.  Specifically, it covers the topic of **Dynamic Programming** which is considered a method that stands in between planning and reinforcement learning. In this context, we are aware of the model and in particular of the transition probabilities and the rewards for each state s and action a. It also highlights the method's strengths and disadvantages and complexities such as the curse of dimensionality. We then move on to the Model-free topic where we do not have access to the model where specifically off-policy methods (**Q-learning**) and on-policy methods (**SARSA**) are compared. It then concludes with the implementation of n-step methods using a depth target function such as the **n-step Q-learning** and the **Monte Carlo** methods. For all the methods implemented, the results and their interpretations are reported based on the parameter tuning. All experiments were tested on the same environment called Stochastic Windy Gridworld.


# Environment
The **Stochastic Windy Gridworld** is a 10x7 grid, where each cell is numbered from 0 to 70 starting from the bottom left cell and moving upwards until reaching the seventieth cell located at the bottom right. The agent moves up, down, left and right and his initial position indicated as ’S’ in the figure below is (0,3). The goal is to move the agent to the final position (7,3), indicated as ’G’. In the environment and especially in columns 3,4,5,8 there is a wind that makes the agent move one step upwards while in columns 6 and 7 the wind pushes it upwards by two steps. The fact that the presence of the
wind is random, since it blows 80% of the time makes the environment stochastic. The agent’s reward is -1 in each step and +35 if the goal is reached. Achieving the final state leads to the termination of the episode. 


<p align="center">
<img src="environment.png" width="400" height="300">
</p>
