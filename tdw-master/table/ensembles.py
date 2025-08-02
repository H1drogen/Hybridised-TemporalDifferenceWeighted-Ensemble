import numpy as np


def softmax(q):
    return np.exp(q - np.max(q)) / np.sum(np.exp(q - np.max(q)))


class Single:
    def __init__(self, tables):
        self.table = tables[0]

    def act(self, obs):
        q = self.table.q_value(obs)
        return np.argmax(q), None

    def observe(self, action, obs, reward, done):
        pass


class AverageEnsemble:
    def __init__(self, tables):
        self.tables = tables

    def act(self, obs):
        qs = []
        for table in self.tables:
            qs.append(table.q_value(obs))
        return np.argmax(np.mean(qs, axis=0)), None

    def observe(self, action, obs, reward, done):
        pass


class VoteEnsemble:
    def __init__(self, tables):
        self.tables = tables

    def act(self, obs):
        qs = []
        for table in self.tables:
            qs.append(table.q_value(obs))
        vote = np.zeros(4)
        for q in qs:
            vote[np.argmax(q)] += 1
        return np.argmax(vote), None

    def observe(self, action, obs, reward, done):
        pass


class TDWAverageEnsemble:
    def __init__(self, tables, gamma=0.95, decay=1.0):
        self.tables = tables
        self.gamma = gamma
        self.decay = decay
        self.cumulative_errors = np.zeros(len(tables))
        self.prev_qs = None

    def _update_prev_qs(self, obs):
        self.prev_qs = []
        for table in self.tables:
            self.prev_qs.append(table.q_value(obs))

    def act(self, obs):
        self._update_prev_qs(obs)

        weights = softmax(-self.cumulative_errors)

        weighted_q = np.reshape(weights, [-1, 1]) * np.array(self.prev_qs)
        q = np.sum(weighted_q, axis=0)
        action = np.argmax(q)
        self.prev_action = action
        return action, weights

    def observe(self, action, obs, reward, done):
        if done:
            self.cumulative_errors *= 0.0
            return
        self.cumulative_errors *= self.decay
        for i, table in enumerate(self.tables):
            next_q = np.max(table.q_value(obs))
            q = self.prev_qs[i][action]
            error = reward + self.gamma * next_q - q
            self.cumulative_errors[i] += error ** 2


class TDWVoteEnsemble(TDWAverageEnsemble):
    def act(self, obs):
        self._update_prev_qs(obs)

        weights = softmax(-self.cumulative_errors)

        votes = np.zeros(4, dtype=np.float32)
        for w, q in zip(weights, self.prev_qs):
            max_index = np.argmax(q)
            votes[max_index] += w

        action = np.argmax(votes)
        self.prev_action = action
        return action, weights
