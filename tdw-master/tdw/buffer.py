import random

from collections import deque


class Buffer:
    def __init__(self, maxlen=10 ** 5, batch_size=32, seed=0):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=maxlen)
        random.seed(seed)

    def add(self, obs_t, action_t, reward_tp1, obs_tp1, done_tp1):
        experience = dict(obs_t=obs_t, action_t=[action_t],
                          reward_tp1=[reward_tp1], obs_tp1=obs_tp1,
                          done_tp1=[done_tp1])
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)
