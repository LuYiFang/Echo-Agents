import random

from Agent.AgentBase import Agent


class GenerousAgent(Agent):
    def decide_action(self, others):
        actions = ["combine", "give", "use", "none"]
        weights = [0.2, 0.5, 0.2, 0.1]  # prefers give
        return random.choices(actions, weights)[0]
