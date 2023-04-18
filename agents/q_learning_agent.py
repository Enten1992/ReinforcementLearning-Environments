
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.001):
        """
        Initializes a Q-Learning agent.

        Args:
            state_space_size (tuple): Dimensions of the state space.
            action_space_size (int): Number of possible actions.
            learning_rate (float): The learning rate (alpha).
            discount_factor (float): The discount factor (gamma).
            exploration_rate (float): Initial exploration rate (epsilon).
            min_exploration_rate (float): Minimum exploration rate.
            exploration_decay_rate (float): Rate at which exploration rate decays.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros(state_space_size + (action_space_size,))

    def choose_action(self, state):
        """
        Chooses an action based on an epsilon-greedy policy.

        Args:
            state (tuple): The current state.

        Returns:
            int: The chosen action.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(self.action_space_size)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-value for a given state-action pair using the Q-Learning formula.

        Args:
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The next state.
            done (bool): Whether the episode has ended.
        """
        current_q = self.q_table[state + (action,)]
        max_future_q = np.max(self.q_table[next_state]) if not done else 0

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state + (action,)] = new_q

    def decay_exploration_rate(self, episode):
        """
        Decays the exploration rate.

        Args:
            episode (int): The current episode number.
        """
        self.exploration_rate = self.min_exploration_rate + \
                                (1.0 - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

    def save_q_table(self, filename="q_table.npy"):
        """
        Saves the Q-table to a file.
        """
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        """
        Loads the Q-table from a file.
        """
        if os.path.exists(filename):
            self.q_table = np.load(filename)
            print(f"Q-table loaded from {filename}")
        else:
            print(f"Q-table file not found at {filename}")

if __name__ == "__main__":
    # Example usage with a simple environment (e.g., FrozenLake from Gymnasium)
    print("Running example Q-Learning agent with a dummy environment.")
    try:
        import gymnasium as gym
    except ImportError:
        print("Gymnasium not found. Please install it: pip install gymnasium")
        exit()

    env = gym.make("FrozenLake-v1", is_slippery=False) # For simplicity, no slippery
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    # Convert discrete state space to a tuple for Q-table indexing
    agent = QLearningAgent(state_space_size=(state_space_size,), action_space_size=action_space_size)

    num_episodes = 1000
    max_steps_per_episode = 100

    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        state = (state,) # Convert to tuple for Q-table indexing
        done = False
        truncated = False
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            new_state = (new_state,) # Convert to tuple

            agent.learn(state, action, reward, new_state, done or truncated)
            state = new_state
            rewards_current_episode += reward

            if done or truncated:
                break
        
        agent.decay_exploration_rate(episode)
        rewards_per_episode.append(rewards_current_episode)

    print("\nTraining complete.")
    print(f"Average reward over last 100 episodes: {np.mean(rewards_per_episode[-100:])}")
    agent.save_q_table()
    env.close()
