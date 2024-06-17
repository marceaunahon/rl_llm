import numpy as np

class Env():
    def __init__(self):
        self.actions = np.array([])
        self.states = np.array([])
        self.rewards = np.array([])
        self.P = np.array([])

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

class Ladder(Env):
    def __init__(self):
        super().__init__()
        self.states = np.array([0, 1, 2, 3])
        self.actions = np.array([0, 1, 2])
        self.transition_matrix = np.array([
                [ # action 0
                    [0.0, 0.8, 0.2, 0.0], #state 0
                    [0.0, 0.0, 0.0, 0.0], #state 1
                    [0.0, 0.0, 0.0, 0.0], #state 2
                    [0.0, 0.0, 0.0, 0.0]  #state 3
                ]
                ,
                [ # action 1
                    [0.0, 0.1, 0.8, 0.1], #state 0
                    [0.0, 0.0, 0.0, 0.0], #state 1
                    [0.0, 0.0, 0.0, 0.0], #state 2
                    [0.0, 0.0, 0.0, 0.0]  #state 3
                ]
                ,
                [ # action 2
                    [0.0, 0.0, 0.2, 0.8], #state 0
                    [0.0, 0.0, 0.0, 0.0], #state 1
                    [0.0, 0.0, 0.0, 0.0], #state 2
                    [0.0, 0.0, 0.0, 0.0]  #state 3
                ]
                    ])
        self.rewards = np.array([0, -1, 0, 1])

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        done = False
        # if action == 0 : self.state = 1
        # if action == 1 : self.state = 2
        # if action == 2 : self.state = 3


        #print(f"state : {self.state}, action : {action}, p : {self.transition_matrix[action][self.state]}")
        try :
            self.state = np.random.choice(self.states, p=self.transition_matrix[int(action)][self.state])
        except ValueError:
            print("PROBLEM")
            print(f"state : {self.state}, action : {action}, p : {self.transition_matrix[int(action)][self.state]}")
        

        if self.state in [1,2,3]:
            done = True

        # profile = np.random.randint(4)
        # self.rewards = np.zeros(4)
        # self.rewards[1:] = self.rewards_table[profile]



        # reward = self.rewards[self.state]
        # print(reward)

        reward = self.rewards[self.state]
        
        return self.state, reward, done

    def render(self):
        # TODO: remplacer par le scénario
        if self.state == 0 : print("Début")
        if self.state == 1 : print("Mauvais")
        if self.state == 2 : print("Neutre")
        if self.state == 3 : print("Bon")