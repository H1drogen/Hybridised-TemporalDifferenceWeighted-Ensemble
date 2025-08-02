import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S


def cnn_network(obs, num_actions, rng, scope):
    with nn.parameter_scope(scope):
        out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), rng=rng,
                             name='conv1')
        out = F.relu(out)
        out = PF.convolution(out, 64, (4, 4), stride=(2, 2), rng=rng,
                             name='conv2')
        out = F.relu(out)
        out = PF.convolution(out, 64, (3, 3), stride=(1, 1), rng=rng,
                             name='conv3')
        out = F.relu(out)
        out = PF.affine(out, 512, rng=rng, name='fc1')
        out = F.relu(out)
        return PF.affine(out, num_actions, rng=rng, name='output')


class DQN:
    def __init__(self, num_actions, batch_size, gamma, lr, seed, name='dqn'):
        self.name = name

        rng = np.random.RandomState(seed)

        # infer variable
        self.infer_obs_t = infer_obs_t = nn.Variable((1, 4, 84, 84))
        with nn.parameter_scope(name):
            # inference output
            self.infer_q_t = cnn_network(infer_obs_t, num_actions, rng,
                                         scope='q_func')

        # train variables
        self.obs_t = obs_t = nn.Variable((batch_size, 4, 84, 84))
        self.actions_t = actions_t = nn.Variable((batch_size, 1))
        self.rewards_tp1 = rewards_tp1 = nn.Variable((batch_size, 1))
        self.obs_tp1 = obs_tp1 = nn.Variable((batch_size, 4, 84, 84))
        self.dones_tp1 = dones_tp1 = nn.Variable((batch_size, 1))

        with nn.parameter_scope(name):
            # training output
            q_t = cnn_network(obs_t, num_actions, rng, scope='q_func')
            q_tp1 = cnn_network(obs_tp1, num_actions, rng,
                                scope='target_q_func')

        # select one dimension
        a_one_hot = F.one_hot(actions_t, (num_actions,))
        q_t_selected = F.sum(q_t * a_one_hot, axis=1, keepdims=True)
        q_tp1_best = F.max(q_tp1, axis=1, keepdims=True)

        # loss calculation
        y = self.rewards_tp1 + gamma * q_tp1_best * (1.0 - self.dones_tp1)
        self.loss = F.mean(F.huber_loss(q_t_selected, y))

        # optimizer
        self.solver = S.RMSprop(lr, 0.95, 1e-2)

        with nn.parameter_scope(name):
            # weights and biases
            with nn.parameter_scope('q_func'):
                self.params = nn.get_parameters()
            with nn.parameter_scope('target_q_func'):
                self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array(obs_t)
        self.infer_q_t.forward(clear_buffer=True)
        return self.infer_q_t.d

    def train(self, obs_t, actions_t, rewards_tp1, obs_tp1, dones_tp1):
        self.obs_t.d = np.array(obs_t)
        self.actions_t.d = np.array(actions_t)
        self.rewards_tp1.d = np.array(rewards_tp1)
        self.obs_tp1.d = np.array(obs_tp1)
        self.dones_tp1.d = np.array(dones_tp1)
        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient clipping by norm
        for name, variable in self.params.items():
            g = 10.0 * variable.g / max(np.sqrt(np.sum(variable.g ** 2)), 10.0)
            variable.g = g
        self.solver.update()
        return self.loss.d

    def update_target(self):
        for key in self.target_params.keys():
            self.target_params[key].data.copy_from(self.params[key].data)

    def save(self, path):
        with nn.parameter_scope(self.name):
            nn.save_parameters(path)

    def load(self, path):
        with nn.parameter_scope(self.name):
            nn.load_parameters(path)
