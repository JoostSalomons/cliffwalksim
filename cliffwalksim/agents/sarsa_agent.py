import random
import numpy as np
from gymnasium.spaces.discrete import Discrete
from cliffwalksim.agents.tabularagent import TabularAgent


class SarsaAgent(TabularAgent):

    def __init__(self, state_space: Discrete, action_space: Discrete, learning_rate=0.1, discount_rate=0.9):
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        self.epsilon = 0.1
        self.action_space = action_space.n

    def update(self, trajectory: tuple) -> None:
        S = trajectory[0]
        A = trajectory[1]
        R = trajectory[2]
        S_pr = trajectory[3]
        A_pr = trajectory[4]
        alpha = self.learning_rate
        gamma = self.discount_rate
        self.q_table[S,A] = self.q_table[S,A] + alpha * (R + gamma * self.q_table[S_pr,A_pr] - self.q_table)[S,A]

    def policy(self, state):
        if self.epsilon < random.random():
            A = random.randint(0,self.action_space - 1)
        else:
            A = np.argmax(self.q_table[state])
        return A
        
