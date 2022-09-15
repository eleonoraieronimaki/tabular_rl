#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

from statistics import mean
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s '''
        return argmax(self.Q_sa[s])
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        
        max_Q=[]

        for i in range(len(self.Q_sa)):
    
            max_Q.append(self.Q_sa[i][self.select_action(i)])
            
        self.Q_sa[s][a]= np.sum(np.multiply(p_sas[s][a],(r_sas[s][a]) + self.gamma * np.array(max_Q)))
    pass
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)  

    max_error = 10000000
    i = 0
    env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    env.fig.savefig(str(i) + '.png')
    while max_error > threshold:
        max_error = 0 
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions): 
                x = QIagent.Q_sa[s][a]
                QIagent.update(s,a,env.p_sas,env.r_sas)
                max_error = max(max_error,abs(x - QIagent.Q_sa[s][a]))
                
        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        i+=1
        env.fig.savefig(str(i) + '.png')
        print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
        
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

       
    
    # TO DO: Compute mean reward per timestep under the optimal policy
    # 
    V_initial = max(QIagent.Q_sa[env.reset()])
    mean_reward_per_timestep = V_initial / (35 - V_initial +1)
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    print('Optimal value at the start state', V_initial)

if __name__ == '__main__':
    experiment()
