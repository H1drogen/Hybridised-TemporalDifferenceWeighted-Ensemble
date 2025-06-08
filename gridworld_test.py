import gymnasium

# Initialise the environment
env = gymnasium.make('GridWorldEnv', render_mode='human')
print(env)