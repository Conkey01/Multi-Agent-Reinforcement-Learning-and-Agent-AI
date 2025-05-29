class QTable:
    def __init__(self, alpha=1.0):
        self.q = {}
        self.alpha = alpha

    def update(self, state, action, value):
        self.q[(state, action)] = value

    def get_q_value(self, state, action):
        return self.q.get((state, action), 0.0)

    def get_argmax_q(self, state, actions):
        max_value = float('-inf')
        best_action = actions[0]
        for action in actions:
            q = self.get_q_value(state, action)
            if q > max_value:
                max_value = q
                best_action = action
        return best_action
