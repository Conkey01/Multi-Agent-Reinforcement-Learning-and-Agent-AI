from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction
from qtable import QTable

class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001):
        while True:
            delta = 0.0
            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                old_value = values.get_value(state)
                new_value = values.get_q_value(
                    self.mdp, state, policy.select_action(state, actions)
                )
                values.add(state, new_value)
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:
                break
        return values

    def policy_iteration(self, max_iterations=100, theta=0.001):
        values = TabularValueFunction()
        for i in range(1, max_iterations + 1):
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta)
            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                if not actions:
                    continue
                old_action = self.policy.select_action(state, actions)
                q_values = QTable(alpha=1.0)
                for action in actions:
                    q_values.update(state, action, values.get_q_value(self.mdp, state, action))
                new_action = q_values.get_argmax_q(state, actions)
                self.policy.update(state, new_action)
                policy_changed = True if new_action != old_action else policy_changed
            if not policy_changed:
                return i
        return max_iterations
