import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np


def run_FL():
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), map_name="4x4", is_slippery=False, render_mode='human')

    q = np.zeros((env.observation_space.n, env.action_space.n)) # Q-table for the agent
    learning_rate = 0.9
    discount_factor = 0.9

    observation, info = env.reset()

    env.render()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'observation tile: {observation}')

        episode_over = terminated or truncated

    print('final reward:', reward)

    env.close()
