class TabularValueFunction:
    def __init__(self):
        self.values = {}

    def get_value(self, state):
        return self.values.get(state, 0.0)

    def add(self, state, value):
        self.values[state] = value

    def get_q_value(self, mdp, state, action, gamma=0.9):
        next_states = mdp.get_transition_states_and_probs(state, action)
        q_value = 0.0
        for next_state, prob in next_states:
            reward = mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + gamma * self.get_value(next_state))
        return q_value
