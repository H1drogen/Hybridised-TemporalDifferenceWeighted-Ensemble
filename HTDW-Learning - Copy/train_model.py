import gymnasium as gym
import ale_py
import time
from DQN_Agent import DQN_Agent
from DQN_Guided_Exploration import DQN_Guided_Exploration
from Evaluation.Evaluation_Plots import plot_rewards_and_length, plot_state_scatter, save_actor_distribution, save_rewards_and_length
import numpy as np
import pickle

from tdw.tdw_ensemble import TDWVoteEnsemble

env_name = "CartPole-v0"
# env_name = "MountainCar-v0"
# env_name = "LunarLander-v3"
seed = 0

initial_training = 10
max_evaluations = 100

dqn_path = None
model_dqn_path = None

def main():
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    env.reset(seed=seed)

    state_shape = (1,env.observation_space.shape[0])
    agents = []

    def train_agent(agent, env, max_episodes, state_shape):
        """ Train a single agent on the given environment"""
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
                if truncated:
                    done = True
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

        plot_state_scatter(agent,title1=f'{env_name} {agent.name}',title2='',xlabel1='position',ylabel1='velocity',xlabel2='x-velocity',ylabel2='y-velocity',color= '#6666ff')
        plot_rewards_and_length(total_reward_list, -200.,0., episode_length_list, agent.name)
        with open(f'Data/agents/{agent.name}_{env_name}model_{initial_training}episode.pkl', 'wb') as f:
            pickle.dump(agent, f)

    def ensemble_training(env, method, epsilon, rng):
        obs, _ = env.reset(seed=seed)
        cumulative_reward = 0.0
        agent_distribution = {agent.name: 0 for agent in agents}
        cur_state = obs.reshape(state_shape)
        terminated = False
        while not terminated:
            action, actor_index = method.act(obs)
            if rng.rand() < epsilon:
                action = rng.randint(env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                terminated = True
            agent_distribution[agents[actor_index].name] += 1
            clipped_reward = np.clip(reward, -1.0, 1.0)
            new_state = obs.reshape(state_shape)
            agents[actor_index].update_model(cur_state, action, reward, new_state, terminated)

            # Update both agents
            # agents[0].update_model(cur_state, action, reward, new_state, terminated)
            # agents[1].update_model(cur_state, action, reward, new_state, terminated)

            cur_state = new_state
            method.observe(action, obs, clipped_reward, terminated)
            cumulative_reward += reward

        save_actor_distribution(agent_distribution, f'Data/actor_distribution_{seed}.csv')
        return cumulative_reward


    def pixel_to_float(obs):
        return np.array(obs, dtype=np.float32) / 255.0

    def atari_evaluation(env, method, epsilon, rng):
        pass


    if dqn_path is not None:
        with open(dqn_path, 'rb') as f:
            agents.append(pickle.load(f))
    else:
        agents.append(DQN_Agent(env=env, name='DQN_Agent'))
        train_agent(agents[0], env, initial_training, state_shape)

    if model_dqn_path is not None:
        with open(model_dqn_path, 'rb') as f:
            agents.append(pickle.load(f))
    else:
        agents.append(DQN_Guided_Exploration(env=env, name='DQN_Guided_Exploration'))
        train_agent(agents[1], env, initial_training, state_shape)

    method = TDWVoteEnsemble(agents)

    # train with ensemble
    for i in range(max_evaluations):
        if env_name.startswith("ALE/"):
            reward = atari_evaluation(env, method, epsilon=0.05, rng=np.random.RandomState(0))
        else:
            reward = ensemble_training(env, method, epsilon=0.05, rng=np.random.RandomState(0))
        save_rewards_and_length([reward], f'Data/tdw_rewards_{seed}.csv')


if __name__ == "__main__":
    main()