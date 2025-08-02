import numpy as np
import matplotlib.pyplot as plt


class QTable:
    def __init__(self, state_size, action_size, lr=1e-2, gamma=0.95):
        self.table = np.zeros((state_size, action_size), dtype=np.float32)
        self.lr = lr
        self.gamma = gamma

    def act(self, obs):
        return np.argmax(self.q_value(obs))

    def q_value(self, obs):
        return self.table[obs]

    def update(self, obs_t, action_t, reward_tp1, obs_tp1, done_tp1):
        q_t = self.table[obs_t][action_t]
        if done_tp1:
            q_tp1 = 0.0
        else:
            q_tp1 = np.max(self.table[obs_tp1])
        td = reward_tp1 + self.gamma * q_tp1 - q_t
        self.table[obs_t][action_t] += self.lr * td
        return td

    def visualize(self):
        max_table = np.max(self.table, axis=1)
        size = int(np.sqrt(max_table.shape[0]))
        plt.pcolor(np.reshape(max_table, (size, size)))


class AverageQTable(QTable):
    def __init__(self, tables):
        self.table = np.mean(np.transpose(tables, (1, 2, 0)), axis=2)

    def act(self, obs):
        return np.argmax(self.table[obs])


class MaxQTable:
    def __init__(self, tables):
        self.tables = tables

    def act(self, obs):
        actions = []
        values = []
        for i in range(len(self.tables)):
            actions.append(np.argmax(self.tables[i][obs]))
            values.append(np.max(self.tables[i][obs]))
        return actions[np.argmax(values)]

    def visualize(self):
        max_table = np.max(np.max(np.transpose(self.tables, (1, 2, 0)), axis=2), axis=1)
        size = int(np.sqrt(max_table.shape[0]))
        plt.pcolor(np.reshape(max_table, (size, size)))


