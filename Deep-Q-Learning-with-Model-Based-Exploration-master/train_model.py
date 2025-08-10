import gym
import time
from gym import wrappers
from DQN_Agent import DQN_Agent
from DQN_Guided_Exploration import DQN_Guided_Exploration
from Helpers import plot_rewards_and_length,plot_state_scatter
import numpy as np

from tdw.tdw_ensemble import TDWVoteEnsemble

env_name = "CartPole-v0"
#"Acrobot-v1"#"LunarLander-v2"#"BipedalWalker-v2"#"CartPole-v0"#"HalfCheetah-v2"#MountainCar-v0
max_episodes = 2
record_video_every = 100

def main():
    env = gym.make(env_name)
    env.reset(seed=0)
    env = wrappers.Monitor(env, 'replay', video_callable=lambda e: e%record_video_every == 0,force=True)


    state_shape = (1,env.observation_space.shape[0])
    agents = [DQN_Agent(env=env), DQN_Guided_Exploration(env=env)]

    for agent in agents:
        start_time = time.time()
        total_reward_list = []
        episode_length_list = []
        for episode in range(max_episodes):
            agent.on_episode_start()
            cur_state = env.reset().reshape(state_shape)
            steps = 0
            total_reward = 0
            done = False
            while not done:
                steps += 1
                action = agent.act(cur_state)
                new_state, reward, done, _ = env.step(action)
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
        plot_rewards_and_length(total_reward_list, -200.,0., episode_length_list)

    method = TDWVoteEnsemble(agents)

    def pixel_to_float(obs):
        return np.array(obs, dtype=np.float32) / 255.0

    def evaluate(env, method, epsilon, rng):
        obs = env.reset()
        cumulative_reward = 0.0
        done = False
        while not done:
            action = method.act(pixel_to_float([obs]))
            if rng.rand() < epsilon:
                action = rng.randint(env.action_space.n)
            obs, reward, done, _ = env.step(action)
            clipped_reward = np.clip(reward, -1.0, 1.0)
            method.observe(action, pixel_to_float([obs]), clipped_reward, done)
            cumulative_reward += reward
        return cumulative_reward

    for i in range(3):
        reward = evaluate(env, method, epsilon=0.05, rng=np.random.RandomState(0))
        print('reward: {}'.format(reward))

if __name__ == "__main__":
    main()