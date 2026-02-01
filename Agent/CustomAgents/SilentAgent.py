import random

from Agent.AgentBase import Agent


class SilentAgent(Agent):
    def decide_speech(self, others):
        weights = [0.7, 0.1, 0.1, 0.1]  # mostly silent
        target = random.choice(others) if others else None
        speech = random.choices(self.speeches, weights)[0]
        return target, speech
