import numpy as np

def theta_W(w):
    # Projection operator for weights (e.g., non-negative, sum to 1)
    w = np.maximum(w, 0)
    return w / np.sum(w)

def theta_pi(pi):
    # Projection operator for policy (e.g., non-negative, sum to 1)
    pi = np.maximum(pi, 0)
    return pi / np.sum(pi)

def MCAC(S, actions, V_list, r_list, T, gamma=0.99, delta_W=0.01, delta_pi=0.01):
    m = len(V_list)  # number of value vectors
    N = len(r_list)  # number of actors
    W = np.ones(m) / m  # Initial weights
    pi = np.ones((len(S), len(actions))) / len(actions)  # Uniform initial policy

    s = np.random.choice(S)  # Initial state
    for t in range(T):
        # Draw action from current policy
        a = np.random.choice(actions, p=pi[s])

        # Simulate environment transition (replace with your environment)
        s_prime = np.random.choice(S)  # Placeholder for next state
        r_hat = np.random.rand()  # Placeholder for observed reward

        # Step 8: Compute E_hat for each actor
        E_hat_i = np.array([
            V_list[i][s] - gamma * V_list[i][s_prime] + r_list[i][s, a, s_prime]
            for i in range(m)
        ])

        # Step 9: Update weights
        W = theta_W(W + delta_W * np.sqrt(W) * E_hat_i)

        # Step 10: Compute E_hat (ensemble)
        E_hat = np.sum(W * (np.array([V_list[i][s] - gamma * V_list[i][s_prime] for i in range(m)]))) + r_hat

        # Step 11: Update policy
        pi[s, a] = theta_pi(pi[s, a] - delta_pi * np.sqrt(pi[s, a]) * E_hat)

        # Step 12: Update state
        s = s_prime

    return pi

# Example usage (with dummy data)
S = list(range(16))  # 16 states for FrozenLake 4x4
actions = list(range(4))  # 4 actions
V_list = [np.random.rand(16) for _ in range(3)]  # 3 pre-trained value vectors
r_list = [np.random.rand(16, 4, 16) for _ in range(3)]  # 3 reward tensors
T = 1000

policy = MCAC(S, actions, V_list, r_list, T)
print("Learned policy:", policy)