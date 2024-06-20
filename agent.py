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
        self.last_20 = [], [], []
        self.likelihood_threshold = 20
        self.models = []
        self.change_idx = [0]
        self.converged = False
        self.likelihoods = [], [], []

    def proba(self, state, history) -> float:
        """
        Computes the probability of a state given the history of actions.

        Args:
            state: The current state.
            history: The history of actions.    

        Returns:
            The probability of the state.
        """
        return np.sum([s == state for s in history]) / len(history)
    
    def log_likelihood(self, state, history) -> float:
        """
        Computes the log-likelihood of a state given the history of actions.

        Args:
            state: The current state.
            history: The history of actions.

        Returns:
            The log-likelihood of the state.
        """
        return - np.log(self.proba(state, history))


    def update_history(self, action) -> None: 
        
        #TODO : ça va pas aller 
        #si une action "étrange" se repète beaucoup, elle ne sera plus considérée comme étrange
        """
        Keeps track of the history of the agent.

        Args:
            action: The action taken.
        """
        self.actions_history[action].append(self.state) #add the state to the history of the corresponding action
        if len(self.last_20[action]) == 20:
            self.last_20[action].pop(0) #remove the first state of the last 20 states of the corresponding action
        self.last_20[action].append(self.state) #add the state to the last 20 states of the corresponding action
        last_20_likeliood = np.sum([self.log_likelihood(state, self.actions_history[action]) for state in self.last_20[action]]) #compute the log-likelihood of the last 20 states of the corresponding action
        self.likelihoods[action].append(last_20_likeliood) #add the log-likelihood to the likelihoods of the corresponding action
        if last_20_likeliood > self.likelihood_threshold: #if the log-likelihood is greater than 20, the action is considered as strange
            self.actions_history_strange_value[action].append(True)
        else:
            self.actions_history_strange_value[action].append(False)

        if sum(self.actions_history_strange_value[action][-20:]) >= 5: #if 5 of the 10 last actions are strange, we create a new model
            if len(self.models) == 0 : #first model change : we create a new model 
                self.finalise_model()
                self.models.append(self.transitions)
                self.transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
                self.actions_history = [],[],[]
                self.actions_history_strange_value = [],[],[]
                self.last_20 = [], [], []
            # if len(self.models) > 0 : #there has already been a model change
            #     self.finalise_model()
            #     self.models.append(self.transitions)
            #     likelihoods = np.zeros(len(self.change_idx)-1)
            #     for i in range(len(self.models)): #compute likelihoods for each model, does one fit the history ?
                    #TODO calculer directement en fonction des modèles et pas des historiques

        # self.likelihoods[action].append(self.likelihood(self.state, self.actions_history[action][self.change_idx[-1]:]))

        

        # if len(self.actions_history[action][self.change_idx[-1]:]) > 100: #on part du principe que le modèle ne change pas avant 100 actions
            
        #     if self.likelihood(self.state, self.actions_history[action][self.change_idx[-1]:]) < self.likelihood_threshold :
        #         self.actions_history_strange_value[action].append(True)
        #     else :
        #         self.actions_history_strange_value[action].append(False)
        # else :
        #     self.actions_history_strange_value[action].append(False)
        # if len(self.actions_history[action][self.change_idx[-1]:]) > 200:
        #     if sum(self.actions_history_strange_value[action][-100:]) <= 20:
        #         print("Rien de bizarre")
        #         self.converged = True

        # if 5 of the 10 last actions are strange, we create a new model
        # if sum(self.actions_history_strange_value[action][-100:]) >= 25:
        #     if len(self.models) < 2: #first model change or second model change : we create a new model (we do not change to the first model)
        #         self.models.append(self.transitions)
        #         self.transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        #     else: #there has already been a model change
        #         likelihoods = np.zeros(len(self.change_idx)-1)
        #         for i in range(len(self.models)): #compute likelihoods for each model, does one fit the history ?
        #             likelihoods[i] = self.likelihood(self.state, self.actions_history[action][self.change_idx[i]:self.change_idx[i+1]])
        #         if np.max(likelihoods) > (1-self.likelihood_threshold): #there is a model that fits the history
        #             self.models.append(self.transitions)
        #             self.transitions = self.models[np.argmax(likelihoods)]
        #         else: #no model fits the history, we create a new one
        #             self.models.append(self.transitions)
        #             self.transitions = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        #     self.change_idx.append(np.sum([len(self.actions_history[action]) for action in self.actions])-1)

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
            if _ == 500:
                print("\n ############################### \n")
                print("CHANGEMENT DE MODELE")
                print("\n ############################### \n")
                self.env.transition_matrix = np.array([
                [ # action 0
                    [0.0, 0.775, 0.225, 0.0], #state 0
                    [0.0, 0.0, 0.0, 0.0], #state 1
                    [0.0, 0.0, 0.0, 0.0], #state 2
                    [0.0, 0.0, 0.0, 0.0]  #state 3
                ]
                ,
                [ # action 1
                    [0.0, 0.135, 0.73, 0.135], #state 0
                    [0.0, 0.0, 0.0, 0.0], #state 1
                    [0.0, 0.0, 0.0, 0.0], #state 2
                    [0.0, 0.0, 0.0, 0.0]  #state 3
                ]
                ,
                [ # action 2
                    [0.0, 0.0, 0.075, 0.925], #state 0
                    [0.0, 0.0, 0.0, 0.0], #state 1
                    [0.0, 0.0, 0.0, 0.0], #state 2
                    [0.0, 0.0, 0.0, 0.0]  #state 3
                ]
                    ])
            self.state = self.env.reset()
            action = np.random.choice(self.actions)
            # action = 0
            next_state, reward, done = self.env.step(action)
            #self.env.render()
            self.update_model(self.state, action, reward, next_state)
            self.state = next_state
            self.update_history(action)
        self.finalise_model()
        self.models.append(self.transitions)
        
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
    # print(agent.transitions)
    print(len(agent.models))
    print("\n ############################### \n")
    print(agent.models[0][0])
    print("\n ############################### \n")
    print(agent.models[1][0])

    # for i in range(len(agent.actions)):
    #     print(f"Action {i}")
    #     print(agent.actions_history[i])
    #     print("len : ", len(agent.actions_history[i]))
    #     print("\n ############################### \n")
    #     print("Before 1000")
    #     print(agent.actions_history_strange_value[i][:333])
    #     print("After 1000")
    #     print(agent.actions_history_strange_value[i][333:])
    #     print("len : ", len(agent.actions_history_strange_value[i]))

    # print(agent.actions_history_strange_value[0])
    # print(agent.actions_history[0])
    # print(agent.likelihoods[0])
