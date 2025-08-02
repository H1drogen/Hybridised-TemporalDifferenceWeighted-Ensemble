import nnabla as nn
import argparse
import gym
import os

from datetime import datetime
from nnabla.ext_utils import get_extension_context
from tdw.dqn import DQN
from tdw.env import AtariWrapper
from tdw.experiment import train
from tdw.exploration import EpsilonGreedy
from tdw.buffer import Buffer


def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)

    env = AtariWrapper(gym.make(args.env), args.seed, random_start=True)
    num_actions = env.action_space.n

    model = DQN(num_actions, args.batch_size, args.gamma, args.lr, args.seed)

    buffer = Buffer(args.buffer_size, args.batch_size, args.seed)

    exploration = EpsilonGreedy(num_actions, args.epsilon, 0.1,
                                args.schedule_duration, args.seed)

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = os.path.join('train_logs', args.logdir + '_' + date)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    train(env, model, buffer, exploration, logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--schedule-duration', type=int, default=10 ** 6)
    parser.add_argument('--logdir', type=str, default='dqn')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    main(args)
