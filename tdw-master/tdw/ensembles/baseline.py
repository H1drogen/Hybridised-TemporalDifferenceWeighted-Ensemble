import numpy as np


def softmax(q, temp=1.0):
    q /= temp
    return np.exp(q - np.max(q)) / np.sum(np.exp(q - np.max(q)))


class Single:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        q = self.model.infer(obs)[0]
        return np.argmax(q)

    def observe(self, action, obs, reward, done):
        pass


class AverageEnsemble:
    def __init__(self, models):
        self.models = models

    def act(self, obs):
        qs = []
        for model in self.models:
            qs.append(model.infer(obs)[0])
        return np.argmax(np.mean(qs, axis=0))

    def observe(self, action, obs, reward, done):
        pass


class WeightedAverageEnsemble:
    def __init__(self, models, scores):
        self.models = models
        self.weights = np.reshape(softmax(np.array(scores)), [-1, 1])

    def act(self, obs):
        qs = []
        for model in self.models:
            qs.append(model.infer(obs)[0])
        return np.argmax(np.sum(self.weights * np.array(qs), axis=0))

    def observe(self, action, obs, reward, done):
        pass


class VoteEnsemble:
    def __init__(self, models):
        self.models = models

    def act(self, obs):
        qs = []
        for model in self.models:
            qs.append(model.infer(obs)[0])
        vote = np.zeros(qs[0].shape[0])
        for q in qs:
            vote[np.argmax(q)] += 1
        return np.argmax(vote)

    def observe(self, action, obs, reward, done):
        pass


class WeightedVoteEnsemble:
    def __init__(self, models, scores):
        self.models = models
        self.weights = softmax(np.array(scores))

    def act(self, obs):
        qs = []
        for model in self.models:
            qs.append(model.infer(obs)[0])
        vote = np.zeros(qs[0].shape[0])
        for w, q in zip(self.weights, qs):
            vote[np.argmax(q)] += w
        return np.argmax(vote)

    def observe(self, action, obs, reward, done):
        pass
