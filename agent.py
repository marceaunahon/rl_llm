from env import Env, Ladder
import numpy as np

class RLAgent():
    """
    RLAgent class represents a reinforcement learning agent.

    Args:
        env: The environment in which the agent interacts.

    Attributes:
        env: The environment in which the agent interacts.
        state: The current state of the agent.
        actions: The available actions in the environment.
        states: The possible states in the environment.
        nb_states: The number of possible states.
        nb_actions: The number of available actions.
        rewards: An array to store the rewards for each state.
        policy: An array to store the policy for each state.
        rewards_table: The rewards table of the environment.

    Methods:
        choose_action: Chooses an action based on the current policy.
        train: Trains the agent using the environment.
        run: Runs the agent in the environment.
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes the RLAgent with the given environment.

        Args:
            env: The environment in which the agent interacts.
        """
        self.env = env
        self.state = env.reset()
        self.actions = env.actions
        self.states = env.states
        self.nb_states = len(self.states)
        self.nb_actions = len(self.actions)
        self.rewards = np.zeros(self.nb_states)
        self.policy = np.zeros(self.nb_states)

        # self.rewards_table = env.rewards_table

    def e_greedy(self) -> int:
        """
        Chooses an action based on an epsilon-greedy policy.

        Returns:
            The chosen action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.policy[self.state]

    def train(self) -> None:
        """
        Trains the agent using the environment.
        """
        pass

    def run(self) -> None:
        """
        Runs the agent in the environment.
        """
        self.train()
        state = self.env.reset()
        done = False
        while not done:
            action = self.policy[state]
            next_state, reward, done = self.env.step(action)
            state = next_state
            self.env.render()
            print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")

class MBRLAgent(RLAgent):
    """
    MBRLAgent class represents a model-based reinforcement learning agent.

    Args:
        env: The environment in which the agent interacts.

    Attributes:
        gamma: Discount factor for future rewards.
        transition_matrix: Transition probabilities between states.

    Methods:
        train: Trains the agent using the model-based reinforcement learning algorithm.
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes the MBRLAgent with the given environment.

        Args:
            env: The environment in which the agent interacts.
        """
        super().__init__(env)
        self.transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        self.rewards = np.zeros(self.nb_states)
        self.nb_visits = np.zeros(self.nb_states)
        self.actions_history = [],[],[]
        self.actions_history_strange_value = [],[],[]
        self.likelihood_threshold = 0.01
        self.models = []
        self.change_idx = [0]

    def likelihood(self, state, history) -> int:
        """
        Computes the likelihood of a state given the history of actions.

        Args:
            state: The current state.
            history: The history of actions.

        Returns:
            The likelihood of the state.
        """
        return np.sum([s == state for s in history]) / len(history)

    def update_history(self, action) -> None: 
        
        #TODO : ça va pas aller 
        #si une action "étrange" se repète beaucoup, elle ne sera plus considérée comme étrange
        """
        Keeps track of the history of the agent.

        Args:
            action: The action taken.
        """
        self.actions_history[action].append(self.state)

        if len(self.actions_history[action][self.change_idx[-1]:]) > 100: #on part du principe que le modèle ne change pas avant 100 actions
            if self.likelihood(self.state, self.actions_history[action][self.change_idx[-1]:]) < self.likelihood_threshold :
                self.actions_history_strange_value[action].append(True)
        else :
            self.actions_history_strange_value[action].append(False)

        # if 5 of the 10 last actions are strange, we create a new model
        if sum(self.actions_history_strange_value[action][-100:]) >= 10:
            if len(self.models) < 2: #first model change or second model change : we create a new model (we do not change to the first model)
                self.models.append(self.transitions)
                self.transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
            else: #there has already been a model change
                likelihoods = np.zeros(len(self.change_idx)-1)
                for i in range(len(self.models)): #compute likelihoods for each model, does one fit the history ?
                    likelihoods[i] = self.likelihood(self.state, self.actions_history[action][self.change_idx[i]:self.change_idx[i+1]])
                if np.max(likelihoods) > (1-self.likelihood_threshold): #there is a model that fits the history
                    self.models.append(self.transitions)
                    self.transitions = self.models[np.argmax(likelihoods)]
                else: #no model fits the history, we create a new one
                    self.models.append(self.transitions)
                    self.transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
            self.change_idx.append(np.sum([len(self.actions_history[action]) for action in self.actions])-1)

        #TODO : regarder en arrière à quel moment on a eu un changement de modèle


    def update_model(self, state, action, reward, next_state) -> None:
        """
        Updates the model of the environment.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
        """
        self.nb_visits[next_state] += 1 
        self.transitions[state][int(action)][next_state] += 1
        self.rewards[next_state] += reward 

    def finalise_model(self) -> None:
        """
        Finalises the model of the environment.
        """
        actions_history = [np.sum(self.transitions[0][action]) for action in self.actions]
        for s in self.states:
            self.rewards[s] /= self.nb_visits[s] #RuntimeWarning: invalid value encountered in true_divide car rewards[0] = nan
            for a in self.actions:
                self.transitions[s][int(a)] /= actions_history[int(a)]

        # TODO : ne garder qu'un nb_visits (pas visits et nb_visits)


class QLearningAgent(RLAgent):
    """
    QLearningAgent class represents a Q-learning agent.

    Args:
        env: The environment in which the agent interacts.

    Attributes:
        gamma: Discount factor for future rewards.
        epsilon: Probability of choosing a random action.
        Q: Q-value table.
        nb_episodes: Number of training episodes.

    Methods:
        choose_action: Chooses an action based on an epsilon-greedy policy.
        train: Trains the agent using the Q-learning algorithm.
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes the QLearningAgent with the given environment.

        Args:
            env: The environment in which the agent interacts.
        """
        super().__init__(env)
        self.gamma = 0.9
        self.epsilon = 0.01
        self.Q = np.zeros((self.nb_states, len(self.actions)))
        self.nb_episodes = 1000

    def train(self) -> None:
        """
        Trains the agent using the Q-learning algorithm.
        """
        for _ in range(self.nb_episodes):
            self.state = self.env.reset()
            done = False
            while not done:
                action = self.e_greedy()
                next_state, reward, done = self.env.step(action)
                self.env.render()
                
                # Update Q-value using the Q-learning update rule
                self.Q[self.state, action] = reward + self.gamma * np.max(self.Q[next_state])
                self.state = next_state 

        # Derive policy from the Q-value table
        self.policy = np.argmax(self.Q, axis=1)


class ValueIterationAgent(MBRLAgent):
    """
    ValueIterationAgent class represents an agent that uses value iteration.

    Args:
        env: The environment in which the agent interacts.

    Attributes:
        gamma: Discount factor for future rewards.
        epsilon: Convergence threshold.
        V: Value function table.

    Methods:
        init_rewards: Initializes the rewards for the agent.
        train: Trains the agent using the value iteration algorithm.
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes the ValueIterationAgent with the given environment.

        Args:
            env: The environment in which the agent interacts.
        """
        super().__init__(env)
        self.gamma = 0.9
        self.epsilon = 0.01
        self.V = np.zeros(self.nb_states)

    def init_rewards(self) -> None:
        """
        Initializes the rewards from a random profile.
        """
        profile = np.random.randint(4)
        self.rewards = np.zeros(4)
        self.rewards[1:] = self.rewards_table[profile]

    def value_iteration(self) -> None:
        """
        Trains the agent using the value iteration algorithm.
        """
        while True:
            delta = 0
            for s in self.states:
                # self.init_rewards()
                v = self.V[s]
                # Update value function using the Bellman equation
                for a in self.actions:
                    for s_prime in self.states:
                        # V[s] = max(V[s], self.transition_matrix[a][s][s_prime] * (self.rewards[s_prime] + self.gamma * V[s_prime]))
                        self.V[s] = max(self.V[s], self.transitions[s][int(a)][s_prime] * (self.rewards[s_prime] + self.gamma * self.V[s_prime]))
                delta = max(delta, abs(v - self.V[s]))
                

            if delta < self.epsilon:
                break

        # Derive policy from the value function table
        # TODO plus propre
        interests = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        for s in self.states:
            for a in self.actions:
                for s_prime in self.states:
                    if s_prime == 0: #pour éviter le nan et pas avoir de problème de somme
                        interests[s][int(a)][s_prime] = 0 
                    else:
                        interests[s][int(a)][s_prime] = self.transitions[s][int(a)][s_prime] * (self.rewards[s_prime] + self.gamma * self.V[s_prime])

            self.policy[s] = np.argmax(np.sum(interests[s], axis=1))

    def explore(self) -> None:
        """
        Explores the environment.
        """
        for _ in range(1000):
            self.state = self.env.reset()
            action = np.random.choice(self.actions)
            next_state, reward, done = self.env.step(action)
            #self.env.render()
            self.update_model(self.state, action, reward, next_state)
            self.state = next_state
        self.finalise_model()
        
    def train(self) -> None:
        """
        Trains the agent using the value iteration algorithm.
        """
        self.explore()
        self.value_iteration()

if __name__ == '__main__':
    # Create an instance of the environment
    env = Ladder()
    # Create an instance of the agent
    agent = ValueIterationAgent(env)
    # Run the agent in the environment
    agent.run()
    print(agent.transitions)