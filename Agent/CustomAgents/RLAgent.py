import random

from Agent.AgentBase import Agent


class RLAgent(Agent):
    def __init__(self, name, base_item_name, alpha=0.1, gamma=0.9, epsilon=0.2):
        super().__init__(name, base_item_name)
        self.q_table = {}  # {state: {decision_type: {choice: value}}}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_decision = None

    def get_state(self):
        return (self.hp, len(self.items), len(self.incoming))

    def _choose(self, decision_type, choices):
        state = self.get_state()
        if state not in self.q_table:
            self.q_table[state] = {decision_type: {c: 0.0 for c in choices}}

        if random.random() < self.epsilon:
            choice = random.choice(choices)
        else:
            choice = max(self.q_table[state][decision_type], key=self.q_table[state][decision_type].get)

        self.last_state = state
        self.last_decision = (decision_type, choice)
        return choice

    def decide_receive(self, item, giver):
        return self._choose("receive", self.choices)

    def decide_speech(self, others):
        speech = self._choose("speech", self.speeches)
        target = random.choice(others) if speech and others else None
        return target, speech

    def decide_action(self, others):
        return self._choose("action", self.actions)

    def update_q(self, reward, next_state):
        if self.last_state is None or self.last_decision is None:
            return
        decision_type, choice = self.last_decision
        if next_state not in self.q_table:
            self.q_table[next_state] = {decision_type: {c: 0.0 for c in self.choices if decision_type=="receive"}}
            self.q_table[next_state]["speech"] = {c: 0.0 for c in self.speeches}
            self.q_table[next_state]["action"] = {c: 0.0 for c in self.actions}

        old_value = self.q_table[self.last_state][decision_type][choice]
        next_max = max(self.q_table[next_state][decision_type].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[self.last_state][decision_type][choice] = new_value