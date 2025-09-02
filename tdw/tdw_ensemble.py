import numpy as np

def softmax(q, temp=1.0):
    q /= temp
    return np.exp(q - np.max(q)) / np.sum(np.exp(q - np.max(q)))

class AccumulateErrorEnsemble:
    def __init__(self, models, gamma=0.99, decay=1.0, temp=1.0):
        self.models = models
        self.gamma = gamma
        self.decay = decay
        self.temp = temp
        self.cumulative_errors = np.zeros(len(models))
        self.prev_qs = None

    def _update_prev_qs(self, obs):
        if self.prev_qs is None:
            self.prev_qs = []
            for model in self.models:
                self.prev_qs.append(model.infer(obs)[0])

    def _get_weights(self):
        return softmax(-self.cumulative_errors, self.temp)

    def act(self, obs):
        raise NotImplementedError

    def observe(self, action, obs, reward, done):
        if done:
            self.cumulative_errors *= 0.0
            self.prev_qs = None
            return
        self.cumulative_errors *= self.decay
        next_qs = []
        for i, model in enumerate(self.models):
            next_q = model.infer(obs)[0]
            next_qs.append(next_q)
            q = self.prev_qs[i][action]
            error = reward + self.gamma * np.max(next_q) - q
            self.cumulative_errors[i] += error ** 2
        self.prev_qs = next_qs

class TDWVoteEnsemble(AccumulateErrorEnsemble):
    def __init__(self, models, gamma=0.99, decay=1.0, temp=1.0, visualizer=None):
        self.visualizer = visualizer
        super().__init__(models, gamma, decay, temp)

    def act(self, obs):
        self._update_prev_qs(obs)
        # check self.prev_qs

        weights = self._get_weights()
        print("weights:", weights)

        if self.visualizer is not None:
            self.visualizer.update(weights)

        votes = np.zeros(len(self.prev_qs[0]), dtype=np.float32)
        # for w, q in zip(weights, self.prev_qs):
        #     votes[np.argmax(q)] += w

        agent_contributions = np.zeros(len(self.prev_qs), dtype=np.float32)
        for i, (w, q) in enumerate(zip(weights, self.prev_qs)):
            action_index = np.argmax(q)
            votes[action_index] += w
            agent_contributions[i] += w if action_index == np.argmax(votes) else 0

        action = np.argmax(votes)
        most_contributing_agent = np.argmax(agent_contributions)

        self.prev_action = action
        return action, most_contributing_agent
