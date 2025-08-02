import numpy as np
import pickle
import argparse

from visualization import Visualizer
from env import SlitEnv, DoubleSlitEnv
from ensembles import AverageEnsemble, VoteEnsemble
from ensembles import TDWAverageEnsemble, TDWVoteEnsemble


def main(args):
    rng = np.random.RandomState(args.seed)

    if args.env == 'slit':
        env = SlitEnv(13, goal_reward=100.0, step_penalty=-0.1)
    elif args.env == 'doubleslit':
        env = DoubleSlitEnv(13, goal_reward=100.0, step_penalty=-0.1)
    else:
        raise ValueError

    with open(args.load, 'rb') as f:
        q_tables = pickle.load(f)

    if args.ensemble == 'tdw_average':
        ensemble = TDWAverageEnsemble(q_tables, decay=args.decay)
    elif args.ensemble == 'tdw_vote':
        ensemble = TDWVoteEnsemble(q_tables, decay=args.decay)
    elif args.ensemble == 'average':
        ensemble = AverageEnsemble(q_tables)
    elif args.ensemble == 'vote':
        ensemble = VoteEnsemble(q_tables)
    else:
        raise ValueError

    visualizer = Visualizer(13, len(q_tables))

    while True:
        obs_t = env.reset()
        while True:
            action_t, weights_t = ensemble.act(obs_t)
            if weights_t is None:
                weights_t = np.zeros(len(q_tables))
            if rng.rand() < args.epsilon:
                action_t = rng.randint(4)
            obs_t, reward_t, done_t, _ = env.step(action_t)
            ensemble.observe(action_t, obs_t, reward_t, done_t)
            visualizer.update(env.map, env.pos, weights_t)
            if done_t:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='slit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--load', type=str)
    parser.add_argument('--ensemble', type=str, default='tdw_average')
    args = parser.parse_args()
    main(args)
