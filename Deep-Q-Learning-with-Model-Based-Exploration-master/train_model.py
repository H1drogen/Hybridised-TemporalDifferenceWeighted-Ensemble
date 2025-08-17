import gymnasium as gym
import ale_py
import time
from DQN_Agent import DQN_Agent
from DQN_Guided_Exploration import DQN_Guided_Exploration
from Evaluation.Evaluation_Plots import plot_rewards_and_length, plot_state_scatter, save_actor_distribution, save_rewards_and_length
import numpy as np
from tdw.Atari import AtariWrapper

from tdw.tdw_ensemble import TDWVoteEnsemble

env_name = "LunarLander-v3"
seed = 0
# "ALE/Breakout-v5"#"ALE/Asterix-v5"
#"Acrobot-v1"#"LunarLander-v3"#"BipedalWalker-v2"#"CartPole-v0"#"HalfCheetah-v2"#MountainCar-v0
max_episodes = 0
max_evaluations = 100

def main():
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    env.reset(seed=seed)
    # env = wrappers.Monitor(env, 'replay', video_callable=lambda e: e%record_video_every == 0,force=True)


    state_shape = (1,env.observation_space.shape[0])
    agents = []
    agents.append(DQN_Guided_Exploration(env=env))
    agents.append(DQN_Agent(env=env, name='DQN Agent'))

    for agent in agents:
        start_time = time.time()
        total_reward_list = []
        episode_length_list = []
        for episode in range(max_episodes):
            agent.on_episode_start()
            state, _ = env.reset()
            cur_state = state.reshape(state_shape)
            steps = 0
            total_reward = 0
            done = False
            while not done:
                steps += 1
                action = agent.act(cur_state)
                new_state, reward, done, truncated, _ = env.step(action)
                new_state = new_state.reshape(state_shape)
                agent.update_model(cur_state, action, reward, new_state, done)
                cur_state = new_state
                total_reward += reward
                if done:
                    break

            agent.on_episode_end()
            total_reward_list.append(total_reward)
            episode_length_list.append(steps)
            print('episode {} steps: {}, total reward: {},  elapsed time: {}s'.format(episode, steps, total_reward, int(time.time()-start_time)))

        # plot_state_scatter(agent,title1=f'{env_name} {agent.name}',title2='',xlabel1='position',ylabel1='velocity',xlabel2='x-velocity',ylabel2='y-velocity',color= '#6666ff')
        # plot_rewards_and_length(total_reward_list, -200.,0., episode_length_list, agent.name)

    method = TDWVoteEnsemble(agents)

    def pixel_to_float(obs):
        return np.array(obs, dtype=np.float32) / 255.0

    def atari_evaluation(env, method, epsilon, rng):
        # env = AtariWrapper(gym.make(env_name), seed, render=False, episodic=False, random_start=True)
        cur_state, _ = env.reset()
        cumulative_reward = 0.0
        agent_distribution = {agent.name: 0 for agent in agents}
        terminated = False
        while not terminated:
            action, actor_index = method.act(pixel_to_float([cur_state]))
            if rng.rand() < epsilon:
                action = rng.randint(env.action_space.n)
            obs, reward, terminated, truncated = env.step(action)
            agent_distribution[agents[actor_index].name] += 1
            clipped_reward = np.clip(reward, -1.0, 1.0)
            agents[actor_index].update_model(cur_state, action, reward, obs, truncated)

            # Update both agents
            # agents[0].update_model(cur_state, action, reward, obs, done)
            # agents[1].update_model(cur_state, action, reward, obs, done)

            cur_state = obs
            method.observe(action, pixel_to_float([obs]), clipped_reward, truncated)
            cumulative_reward += reward

        save_actor_distribution(agent_distribution, 'Data/actor_distribution.csv')
        return cumulative_reward

    def ensemble_training(env, method, epsilon, rng):
        obs, _ = env.reset()
        cumulative_reward = 0.0
        agent_distribution = {agent.name: 0 for agent in agents}
        cur_state = obs.reshape(state_shape)
        terminated = False
        while not terminated:
            action, actor_index = method.act(obs)
            if rng.rand() < epsilon:
                action = rng.randint(env.action_space.n)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent_distribution[agents[actor_index].name] += 1
            clipped_reward = np.clip(reward, -1.0, 1.0)
            new_state = obs.reshape(state_shape)
            # agents[actor_index].update_model(cur_state, action, reward, new_state, truncated)

            # Update both agents
            agents[0].update_model(cur_state, action, reward, new_state, terminated)
            agents[1].update_model(cur_state, action, reward, new_state, terminated)

            cur_state = new_state
            method.observe(action, obs, clipped_reward, truncated)
            cumulative_reward += reward

        save_actor_distribution(agent_distribution, 'Data/actor_distribution.csv')
        return cumulative_reward


    for i in range(max_evaluations):
        if env_name.startswith("ALE/"):
            reward = atari_evaluation(env, method, epsilon=0.05, rng=np.random.RandomState(0))
        else:
            reward = ensemble_training(env, method, epsilon=0.05, rng=np.random.RandomState(0))
        save_rewards_and_length([reward], 'Data/tdw_rewards.csv')


if __name__ == "__main__":
    main()