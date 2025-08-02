import numpy as np
import os

from itertools import combinations
from nnabla.monitor import Monitor, MonitorSeries


def pixel_to_float(obs):
    return np.array(obs, dtype=np.float32) / 255.0


def update(model, buffer):
    experiences = buffer.sample()
    obs_t = []
    actions_t = []
    rewards_tp1 = []
    obs_tp1 = []
    dones_tp1 = []
    for experience in experiences:
        obs_t.append(experience['obs_t'])
        actions_t.append(experience['action_t'])
        rewards_tp1.append(experience['reward_tp1'])
        obs_tp1.append(experience['obs_tp1'])
        dones_tp1.append(experience['done_tp1'])
    return model.train(pixel_to_float(obs_t), actions_t, rewards_tp1,
                       pixel_to_float(obs_tp1), dones_tp1)


def train(env, model, buffer, exploration, logdir):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=10000)
    loss_monitor = MonitorSeries('loss', monitor, interval=10000)
    # copy parameters to target network
    model.update_target()

    step = 0
    while step < 10 ** 7:
        obs_t = env.reset()
        reward_t = 0.0
        done_tp1 = False
        cumulative_reward = 0.0

        while step < 10 ** 7 and not done_tp1:
            # infer q values
            q_t = model.infer(pixel_to_float([obs_t]))[0]
            # epsilon-greedy exploration
            action_t = exploration.get(step, np.argmax(q_t))
            # move environment
            obs_tp1, reward_tp1, done_tp1, _ = env.step(action_t)
            # clip reward between [-1.0, 1.0]
            clipped_reward_tp1 = np.clip(reward_tp1, -1.0, 1.0)
            # store transition
            buffer.add(obs_t, action_t, clipped_reward_tp1, obs_tp1, done_tp1)

            # update parameters
            if step > 10000 and step % 4 == 0:
                loss = update(model, buffer)
                loss_monitor.add(step, loss)

            # synchronize target parameters with the latest parameters
            if step % 10000 == 0:
                model.update_target()

            step += 1
            cumulative_reward += reward_tp1
            obs_t = obs_tp1

        # record metrics
        reward_monitor.add(step, cumulative_reward)

    # save parameters
    model.save(os.path.join(logdir, 'model.h5'))


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
