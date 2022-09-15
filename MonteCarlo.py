#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import random
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        n = random.uniform(0,1)

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            if n > epsilon:
                a = argmax(self.Q_sa[s])
            else: 
                a = random.choice(range(len(self.Q_sa[s])))
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            prob = softmax(self.Q_sa[s],temp)
            index = range(len(self.Q_sa[s]))
            a = np.random.choice(index, p=prob)
            
        return a
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        
        pass

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    
    
    
    while n_timesteps>0: 

        s = env.reset()
        states = []
        actions = []
        reward = []
        dones = []
        states.append(s) 
        for t in range(min(max_episode_length, n_timesteps)): 
                
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.01)   
            
            states.append(s_next)
            actions.append(a)
            dones.append(done)
            reward.append(r)
            rewards.append(r)
            
            s = s_next
            n_timesteps-=1
            if done: 
                
                break

        t=len(actions)
        pi.G = np.zeros(t+1)  
        
        for i in range(t-1,-1,-1): 
            pi.G[i]= rewards[i] + pi.gamma*pi.G[i+1]
            
            pi.Q_sa[states[i], actions[i]] =  pi.Q_sa[states[i], actions[i]] + (pi.learning_rate*(pi.G[i] - pi.Q_sa[states[i],actions[i]]))
        
        


        # if plot:
        #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution
    
    return rewards 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    # print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
