import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# Initialise the environment
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), map_name="4x4", is_slippery=True)

observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
