import numpy as np
import os

from itertools import combinations


def train(env, q_table, seed, steps=1000, epsilon=0.1):
    rng = np.random.RandomState(seed)
    rewards = {}
    step = 0
    while step < steps:
        sum_of_reward = 0.0
        done_tp1 = False
        obs_t = env.reset()
        while not done_tp1:
            action_t = q_table.act(obs_t)
            if rng.rand() < epsilon:
                action_t = np.random.randint(4)
            obs_tp1, reward_tp1, done_tp1, _ = env.step(action_t)

            q_table.update(obs_t, action_t, reward_tp1, obs_tp1, done_tp1)

            sum_of_reward += reward_tp1
            obs_t = obs_tp1
            step += 1
            if step == steps:
                break
        rewards[step] = sum_of_reward
    return list(rewards.keys()), list(rewards.values())


def evaluate(env, q_table, seed=0, num_episodes=100, epsilon=0.05):
    rng = np.random.RandomState(seed)
    episode = 0
    rewards = []
    while episode < num_episodes:
        sum_of_reward = 0.0
        obs_t = env.reset()
        while True:
            action_t = q_table.act(obs_t)
            if rng.rand() < epsilon:
                action_t = np.random.randint(4)
            obs_t, reward_t, done_t, _ = env.step(action_t)
            sum_of_reward += reward_t
            if done_t:
                break
        rewards.append(sum_of_reward)
        episode += 1
    return rewards


def evaluate_ensemble(env,
                      ensemble,
                      seed,
                      num_episodes=100,
                      epsilon=0.05):
    rng = np.random.RandomState(seed)
    episode = 0
    rewards = []
    while episode < num_episodes:
        sum_of_reward = 0.0
        obs_t = env.reset()
        while True:
            action_t, _ = ensemble.act(obs_t)
            if rng.rand() < epsilon:
                action_t = rng.randint(4)
            obs_t, reward_t, done_t, _ = env.step(action_t)
            ensemble.observe(action_t, obs_t, reward_t, done_t)
            sum_of_reward += reward_t
            if done_t:
                break
        rewards.append(sum_of_reward)
        episode += 1
    return rewards


def evaluate_combinations(n,
                          env,
                          tables,
                          ensemble_fn,
                          seed,
                          num_episodes=500,
                          epsilon=0.05):
    rewards = []
    for comb in combinations(tables, n):
        ensemble = ensemble_fn(comb)
        rewards.append(evaluate_ensemble(env, ensemble, seed,
                                         num_episodes, epsilon))
    return np.reshape(rewards, [-1])


def evaluate_or_load(n,
                     dir_name,
                     name,
                     env,
                     tables,
                     ensemble_fn,
                     seed,
                     num_episodes=500,
                     epsilon=0.05):
    path = os.path.join(dir_name, name + '.npy')
    if not os.path.exists(path):
        rewards = evaluate_combinations(n, env, tables, ensemble_fn, seed,
                                        num_episodes=num_episodes,
                                        epsilon=epsilon)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        np.save(path, rewards)
        print(path + ' saved')
    else:
        rewards = np.load(path)
        print(path + ' loaded')
    return rewards
