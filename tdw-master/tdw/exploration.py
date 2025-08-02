import numpy as np


class EpsilonGreedy:
    def __init__(self, num_actions, init_value, final_value, duration, seed):
        self.num_actions = num_actions
        self.base= init_value - final_value
        self.init_value = init_value
        self.final_value = final_value
        self.duration = duration
        self.rng = np.random.RandomState(seed)

    def get(self, t, greedy_action):
        decay = t / self.duration
        if decay > 1.0:
            decay = 1.0
        epsilon = (1.0 - decay) * self.base + self.final_value
        if self.rng.rand() < epsilon:
            return self.rng.randint(self.num_actions)
        return greedy_action
