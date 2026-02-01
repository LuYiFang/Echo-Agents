import random

from Agent.AgentBase import Agent


class RandomAgent(Agent):
    def decide_receive(self, item, giver):
        return random.choice(self.choices)

    def decide_speech(self, others):
        speech = random.choice(self.speeches)
        target = random.choice(others) if speech and others else None
        return target, speech

    def decide_action(self, others):
        return random.choice(self.actions)
