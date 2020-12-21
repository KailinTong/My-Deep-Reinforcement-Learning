import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.counter = 0
        self.gamma = 0.8
        self.alpha = 0.2

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        eps = min(1 / (self.counter + 1), 0.1)
        return epsilon_greedy(self.Q, state, self.nA, eps)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        """
        Q-learning 
        """
        if done:
            self.counter += 1
            max_Q_next = 0
        else:
            max_Q_next = max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_Q_next - self.Q[state][action])



def epsilon_greedy(Q, state, nA, eps=0.1):
    """Selects epsilon-greedy action for supplied state.

    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if np.random.random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return np.random.choice(np.arange(nA))