import numpy as np

class QLearningAgent:
    def __init__(self, observation_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros((observation_space_size, action_space_size))

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_space_size)  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        """Updates the Q-table based on the Q-learning formula."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state, :])

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def save_q_table(self, filename):
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.load(filename)

if __name__ == \"__main__\":
    # Example usage (simplified, typically integrated with an environment)
    obs_size = 10  # Example observation space size
    act_size = 4   # Example action space size

    agent = QLearningAgent(obs_size, act_size)
    print("Q-table initialized:\n", agent.q_table)

    # Simulate a learning step
    state = 0
    action = agent.choose_action(state)
    reward = 1
    next_state = 1
    done = False
    agent.learn(state, action, reward, next_state, done)
    print("\nQ-table after one learning step:\n", agent.q_table)

    # Simulate another learning step leading to episode end
    state = 1
    action = agent.choose_action(state)
    reward = 10
    next_state = 9 # Terminal state
    done = True
    agent.learn(state, action, reward, next_state, done)
    print("\nQ-table after episode end:\n", agent.q_table)
    print(f"Epsilon after episode end: {agent.epsilon}")

    # Save and load example
    agent.save_q_table(\"q_table_test.npy\")
    new_agent = QLearningAgent(obs_size, act_size)
    new_agent.load_q_table(\"q_table_test.npy\")
    print("\nLoaded Q-table:\n", new_agent.q_table)
