import numpy as np
import gym
import random
from gym import spaces

class MultiArmedBanditEnv(gym.Env):
    def __init__(self, k):
        super(MultiArmedBanditEnv, self).__init__()
        self.k = k
        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Discrete(1)
        self.reward_means = np.random.normal(0, 1, k)

    def step(self, action):
        reward = np.random.normal(self.reward_means[action], 1)
        return 0, reward, True, {}

    def reset(self):
        return 0

    def render(self, mode='human'):
        pass

class MultiArmedBanditAgent:
    def __init__(self, k, alpha=0.1, epsilon=0.1, c=2):
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.c = c
        self.q_values = np.zeros(k)
        self.action_counts = np.zeros(k)
        self.total_steps = 0

    def epsilon_greedy_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.k - 1)
        else:
            return np.argmax(self.q_values)

    def ucb_action(self):
        if self.total_steps < self.k:
            return self.total_steps
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.total_steps + 1) / (self.action_counts + 1))
        return np.argmax(ucb_values)

    def update_q_values(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def run(self, env, num_steps, strategy='epsilon_greedy'):
        rewards = []
        for step in range(num_steps):
            self.total_steps += 1
            if strategy == 'epsilon_greedy':
                action = self.epsilon_greedy_action()
            elif strategy == 'ucb':
                action = self.ucb_action()
            else:
                raise ValueError("Strategy not recognized. Use 'epsilon_greedy' or 'ucb'.")
            _, reward, _, _ = env.step(action)
            self.update_q_values(action, reward)
            rewards.append(reward)
            env.reset()
        return rewards

# Example usage
k = 10
num_steps = 1000
env = MultiArmedBanditEnv(k)
agent = MultiArmedBanditAgent(k)

rewards_epsilon_greedy = agent.run(env, num_steps, strategy='epsilon_greedy')
agent = MultiArmedBanditAgent(k)  # Reset agent
rewards_ucb = agent.run(env, num_steps, strategy='ucb')

print("Epsilon-Greedy Rewards:", np.sum(rewards_epsilon_greedy))
print("UCB Rewards:", np.sum(rewards_ucb))
