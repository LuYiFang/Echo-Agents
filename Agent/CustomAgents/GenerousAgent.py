import random

from Agent.AgentBase import Agent


class GenerousAgent(Agent):
    def decide_action(self, others):
        weights = [0.2, 0.5, 0.2, 0.1]  # prefers give
        return random.choices(self.actions, weights)[0]
