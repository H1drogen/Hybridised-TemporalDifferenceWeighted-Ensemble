import os
import pandas as pd
from scipy.stats import ttest_1samp

directory = '/Users/simonh/Documents/GitHub/RL-Experiments/Deep-Q-Learning-with-Model-Based-Exploration-master/Data/Combined_MountainCar_Rewards'


for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        values = df.iloc[:30, 1]  # First 30 values from the first column
        mean_value = values.mean()
        std_value = values.std()
        t_stat, p_value = ttest_1samp(values, 0)
        print(f'{filename}: mean = {mean_value}, std = {std_value}, p-value = {p_value}')