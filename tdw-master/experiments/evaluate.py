import numpy as np
import nnabla as nn
import argparse
import gym
import os

from datetime import datetime
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries

from tdw.score_table import SCORE_TABLE
from tdw.dqn import DQN
from tdw.env import AtariWrapper
from tdw.experiment import evaluate
from tdw.ensembles.baseline import Single
from tdw.ensembles.baseline import AverageEnsemble, WeightedAverageEnsemble
from tdw.ensembles.baseline import VoteEnsemble, WeightedVoteEnsemble
from tdw.ensembles.tdw import TDWAverageEnsemble, TDWVoteEnsemble
from tdw.visualizer import WeightVisualizer


class Random:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        return np.random.randint(self.num_actions)

    def observe(self, action, obs, reward, done):
        pass


def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)

    env = AtariWrapper(gym.make(args.env), args.seed, render=args.render,
                       episodic=False, random_start=True)
    num_actions = env.action_space.n

    models = []
    if args.load:
        for i, path in enumerate(args.load[0]):
            model = DQN(num_actions, 1, 0.99, 0.0, args.seed, 'dqn{}'.format(i))
            model.load(path)
            models.append(model)

    if args.visualize:
        visualizer = WeightVisualizer(len(models))
    else:
        visualizer = None

    if args.ensemble == 'single':
        method = Single(models[0])
    elif args.ensemble == 'average':
        method = AverageEnsemble(models)
    elif args.ensemble == 'weighted_average':
        method = WeightedAverageEnsemble(models, SCORE_TABLE[args.env])
    elif args.ensemble == 'vote':
        method = VoteEnsemble(models)
    elif args.ensemble == 'weighted_vote':
        method = WeightedVoteEnsemble(models, SCORE_TABLE[args.env])
    elif args.ensemble == 'tdw_average':
        method = TDWAverageEnsemble(models, decay=args.decay,
                                    visualizer=visualizer)
    elif args.ensemble == 'tdw_vote':
        method = TDWVoteEnsemble(models, decay=args.decay,
                                 visualizer=visualizer)
    elif args.ensemble == 'random':
        method = Random(num_actions)
    else:
        raise ValueError('invalid ensemble method')

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = os.path.join('eval_logs', args.logdir + '_' + date)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)

    rng = np.random.RandomState(args.seed)
    for i in range(args.num_eval):
        reward = evaluate(env, method, args.epsilon, rng)
        reward_monitor.add(i, reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--logdir', type=str, default='dqn')
    parser.add_argument('--num-eval', type=int, default=1000)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--ensemble', type=str)
    parser.add_argument('--load', nargs='*', action='append')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    main(args)
