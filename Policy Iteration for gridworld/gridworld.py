import matplotlib.pyplot as plt
import numpy as np

class GridWorld:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, width=4, height=3, terminals={(3, 2): 1, (3, 1): -1}, obstacles={(1, 1)}):
        self.width = width
        self.height = height
        self.terminals = terminals
        self.obstacles = obstacles

    def get_states(self):
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.obstacles
        ]

    def get_actions(self, state):
        if state in self.terminals:
            return []
        return self.ACTIONS

    def get_transition_states_and_probs(self, state, action):
        if state in self.terminals:
            return [(state, 1.0)]

        x, y = state
        next_state = {
            self.UP: (x, y + 1),
            self.DOWN: (x, y - 1),
            self.LEFT: (x - 1, y),
            self.RIGHT: (x + 1, y),
        }[action]

        # Stay in place if hitting wall or obstacle
        if (
            next_state[0] < 0
            or next_state[0] >= self.width
            or next_state[1] < 0
            or next_state[1] >= self.height
            or next_state in self.obstacles
        ):
            next_state = state

        return [(next_state, 1.0)]

    def get_reward(self, state, action, next_state):
        return self.terminals.get(next_state, -0.04)

    def visualise_policy(self, policy):
        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        grid = [["" for _ in range(self.width)] for _ in range(self.height)]
        for state in self.get_states():
            x, y = state
            if state in self.terminals:
                grid[self.height - y - 1][x] = str(self.terminals[state])
            elif state in self.obstacles:
                grid[self.height - y - 1][x] = "X"
            else:
                action = policy.select_action(state, self.get_actions(state))
                grid[self.height - y - 1][x] = action_symbols[action]

        for row in grid:
            print(" | ".join(row))
        print()
