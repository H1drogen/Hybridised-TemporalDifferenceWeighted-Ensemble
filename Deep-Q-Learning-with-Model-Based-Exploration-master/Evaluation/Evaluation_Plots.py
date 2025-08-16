import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import seaborn as sns


def plot_state_scatter(agent,title1,title2,xlabel1,ylabel1,xlabel2,ylabel2,color, lim1 = [-0.1,0.1,-1.4,0.6],lim2=[-2.0,1.0,-2.0,2.0]):
    fig = plt.figure()

    a = []
    b = []
    sample_size = min(2000,len(agent.replay_memory))
    for sample in random.sample(agent.replay_memory, sample_size):
        a.append(sample[0][0][0])
        b.append(sample[0][0][1])

    sub1 = fig.add_subplot(2,2,1)
    sub1.grid(True,linewidth='0.4',color='white')
    sub1.set_xlabel(xlabel1)
    sub1.set_ylabel(ylabel1)
    sub1.set_ylim(bottom=lim1[0],top = lim1[1])
    sub1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    sub1.set_xlim(left=lim1[2],right=lim1[3])
    sub1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    sub1.set_facecolor('#e6f3ff')
    sub1.scatter(a,b,s=3,color = color)

    if len(sample[0][0]) <= 2:
        return
    c = []
    d = []
    for sample in random.sample(agent.replay_memory, sample_size):
        c.append(sample[0][0][2])
        d.append(sample[0][0][3])

    sub2 = fig.add_subplot(2,2,2)
    sub2.grid(True,linewidth='0.4',color='white')
    sub2.set_xlabel(xlabel2)
    sub2.set_ylabel(ylabel2)
    sub2.set_ylim(bottom=lim2[0],top = lim2[1])
    sub2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    sub2.set_xlim(left=lim2[2],right=lim2[3])
    sub2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    sub2.set_facecolor('#e6f3ff')
    sub2.scatter(c,d,s=3,color = color)


def plot_rewards_and_length(rewards, min_reward,max_reward, lengths, agent_name):

    rewards_df = pd.DataFrame(rewards)
    rewards_df.to_csv(f'Data/rewards_{agent_name}.csv')

    fig = plt.figure()
    sub1 = fig.add_subplot(2,2,1)
    sub1.set_title('Reward')
    sub1.set_ylim(bottom=min_reward,top=max_reward)
    sub1.set_xlabel('episodes')
    sub1.set_ylabel('reward')
    sub1.plot(rewards)

    '''
    sub2 = fig.add_subplot(2,2,2)
    sub2.set_title('episode length')
    sub2.set_xlabel('episodes')
    sub2.plot(lengths)
    '''


    avg_reward = [0.] * len(rewards)
    cumulative_rewards = [0.] * len(rewards)
    cumulated_r = 0.
    for i in range(len(rewards)):
        cumulated_r += rewards[i]
        cumulative_rewards[i] = cumulated_r
    #interval = 10

    for i in range(len(rewards)):
        if i <= 0:
            avg_reward[i] = rewards[i]
        else:
            avg_reward[i] = (cumulative_rewards[i] - cumulative_rewards[0])/i
    sub3 = fig.add_subplot(2,2,2)
    sub3.set_ylim(bottom=min_reward,top=max_reward)
    sub3.set_title('average rewards')
    sub3.set_xlabel('episodes')
    sub3.plot(avg_reward)
    plt.show()




def get_merged_df(prefix, name):
    sns.set(rc={'figure.figsize': (11.7, 8.27), 'legend.fontsize': 18})

    mountain = ['Data/rewards_400_ours_mountain', 'Data/rewards_400_dqn_mountain', 'Data/rewards_500_pg_mountain']
    lunar = ['Data/rewards_500_our_lunar', 'Data/rewards_500_dqn_lunar', 'Data/rewards_500_pg_lunar']

    df1 = pd.read_csv(prefix + '_1.csv').iloc[:, 1]
    df2 = pd.read_csv(prefix + '_2.csv').iloc[:, 1]
    df3 = pd.read_csv(prefix + '_3.csv').iloc[:, 1]
    merged = pd.concat([df1, df2, df3], axis=1)
    merged.columns = [name] * 3
    return merged

def plot_rewards():
    a_df = pd.read_csv('../Data/rewards_DQN_Guided_Exploration.csv').iloc[:, 1]
    b_df = pd.read_csv('../Data/rewards_DQN_Agent.csv').iloc[:, 1]
    df = pd.concat([a_df, b_df], axis=1)
    df = df.expanding().mean()
    df = df.iloc[:500, :]
    df.tail()

    ax = sns.lineplot(data=df, color="red", dashes=False)
    # ax.lines[0].set_linestyle("-")
    # ax.lines[1].set_linestyle("-")
    # ax.lines[2].set_linestyle("-")

    ax.set_xlabel('Number of training episodes')
    ax.set_ylabel('Average rewards')

    fig = ax.get_figure()  # Get the figure object from the Axes
    fig.savefig('plot.png', dpi=300, bbox_inches='tight')

def save_actor_distribution(agent_distribution, path=None):
    df = pd.DataFrame.from_dict(agent_distribution, orient='index', columns=['count'])
    df.index.name = 'actor'
    if path is not None:
        df.to_csv(path, mode='a', header=False)

def save_rewards_and_length(rewards, lengths, path=None):
    df = pd.DataFrame({'rewards': rewards, 'lengths': lengths})
    if path is not None:
        df.to_csv(path, mode='a', header=False)