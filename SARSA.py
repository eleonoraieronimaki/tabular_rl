#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
import random 
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

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
            else: 
                prob = softmax(self.Q_sa[s],temp)
                index = range(len(self.Q_sa[s]))
                a = np.random.choice(index, p=prob)
        return a
        
    def update(self,s,a,r,s_next,a_next,done):

        G_t= r + self.gamma * self.Q_sa[s_next][a_next]
        self.Q_sa[s][a] =  self.Q_sa[s][a] + self.learning_rate*(G_t - self.Q_sa[s][a])

        pass
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 

    env = StochasticWindyGridworld(initialize_model=False)
    env.render()
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    
    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)
    
    while n_timesteps>0: 
        
        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next, policy, epsilon, temp)
        rewards.append(r)

        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.01) 
        
        pi.update(s,a,r,s_next,a_next,done)
        
        if done: 
        
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            s = s_next
            a = a_next
        
        n_timesteps = n_timesteps-1


    return rewards 


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))        
    
if __name__ == '__main__':
    test()
