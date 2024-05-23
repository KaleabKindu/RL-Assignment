import numpy as np
import gym
import random
from collections import defaultdict

class GridWorld:
    def __init__(self, env, discount_factor=1.0, theta=0.0001, alpha=0.1, epsilon=0.1, c=2) -> None:
        self.env = env
        self.actions = list(range(env.action_space.n))
        self.discount_factor = discount_factor
        self.theta = theta
        self.alpha = alpha
        self.epsilon = epsilon
        self.c = c
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.action_counts = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.total_steps = 0

    def value_iteration(self):
        V = np.zeros(self.env.observation_space.n)
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = V[s]
                action_values = []
                for a in self.actions:
                    q_value = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        next_state = int(next_state)
                        q_value += prob * (reward + self.discount_factor * V[next_state])
                    action_values.append(q_value)
                V[s] = max(action_values)
                delta = max(delta, abs(v - V[s]))
            if delta < self.theta:
                break
        return V

    def policy_iteration(self):
        policy = np.random.choice(self.actions, self.env.observation_space.n)
        V = np.zeros(self.env.observation_space.n)

        while True:
            while True:
                delta = 0
                for s in range(self.env.observation_space.n):
                    v = V[s]
                    a = policy[s]
                    v_new = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        next_state = int(next_state)
                        v_new += prob * (reward + self.discount_factor * V[next_state])
                    V[s] = v_new
                    delta = max(delta, abs(v - V[s]))
                if delta < self.theta:
                    break

            policy_stable = True
            for s in range(self.env.observation_space.n):
                old_action = policy[s]
                action_values = []
                for a in self.actions:
                    q_value = 0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        next_state = int(next_state)
                        q_value += prob * (reward + self.discount_factor * V[next_state])
                    action_values.append(q_value)
                policy[s] = np.argmax(action_values)
                if old_action != policy[s]:
                    policy_stable = False

            if policy_stable:
                break

        return policy, V

    def epsilon_greedy_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def q_learning(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            while not done:
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_delta
                state = next_state
                if done and reward == 1.0:
                    reward = 10
                if next_state in [5, 7, 11, 12]:
                    reward = -10

    def ucb_action(self, state):
        ucb_values = self.q_table[state] + self.c * np.sqrt(np.log(self.total_steps + 1) / (self.action_counts[state] + 1))
        return np.argmax(ucb_values)

    def ucb_learning(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            while not done:
                self.total_steps += 1
                action = self.ucb_action(state)
                self.action_counts[state][action] += 1
                next_state, reward, done, truncated, _ = self.env.step(action)
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_delta
                state = next_state
                if done and reward == 1.0:
                    reward = 10
                if next_state in [5, 7, 11, 12]:
                    reward = -10

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()

grid_world = GridWorld(env)

# Run value iteration and print results
value_function = grid_world.value_iteration()
print("Value Function from Value Iteration:")
print(value_function.reshape((4, 4)))

# Run policy iteration and print results
policy, value_function = grid_world.policy_iteration()
print("Policy from Policy Iteration:")
print(policy.reshape((4, 4)))
print("Value Function from Policy Iteration:")
print(value_function.reshape((4, 4)))

# Run Q-Learning and print Q-table
grid_world.q_learning(num_episodes=1000)
q_table = np.array([grid_world.q_table[i] for i in range(env.observation_space.n)])
print("Q-Table from Q-Learning:")
print(q_table.reshape((4, 4, env.action_space.n)))

# Run UCB Learning and print Q-table
grid_world.ucb_learning(num_episodes=1000)
q_table = np.array([grid_world.q_table[i] for i in range(env.observation_space.n)])
print("Q-Table from UCB Learning:")
print(q_table.reshape((4, 4, env.action_space.n)))
