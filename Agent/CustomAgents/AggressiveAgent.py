import random

from Agent.AgentBase import Agent


class AggressiveAgent(Agent):
    def decide_action(self, others):
        actions = ["combine", "give", "use", "none"]
        weights = [0.3, 0.1, 0.5, 0.1]  # prefers use
        return random.choices(actions, weights)[0]
