class TabularPolicy:
    def __init__(self, default_action=None):
        self.policy = {}
        self.default_action = default_action

    def select_action(self, state, actions):
        # Return the current policy's action for this state, or the default
        return self.policy.get(state, self.default_action or actions[0])

    def update(self, state, action):
        # Update the policy for a given state
        self.policy[state] = action

