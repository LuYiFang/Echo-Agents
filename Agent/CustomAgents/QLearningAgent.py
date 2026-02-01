import random

from Agent.AgentBase import Agent


class QLearningAgent(Agent):
    def __init__(self, name, base_item_name, alpha=0.1, gamma=0.9,
                 epsilon=0.2):
        super().__init__(name, base_item_name)
        self.q_table = {}  # {state: {decision_type: {choice: value}}}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_decision = None

    def get_state(self):
        return (self.hp, len(self.items), len(self.incoming))

    def _choose_and_learn(self, decision_type, choices, reward=0):
        state = self.get_state()
        # 初始化 Q-table
        if state not in self.q_table:
            self.q_table[state] = {
                "receive": {c: 0.0 for c in self.choices},
                "speech": {c: 0.0 for c in self.speeches},
                "action": {c: 0.0 for c in self.actions},
            }

        # epsilon-greedy
        if random.random() < self.epsilon:
            choice = random.choice(choices)
        else:
            choice = max(self.q_table[state][decision_type],
                         key=self.q_table[state][decision_type].get)

        # 更新 Q-table (邊玩邊學)
        if self.last_state is not None and self.last_decision is not None:
            self.update_q(reward, state)

        self.last_state = state
        self.last_decision = (decision_type, choice)
        return choice

    def decide_receive(self, item, giver):
        # 即時 reward：存活 +1
        reward = 1 if self.is_alive() else -10
        return self._choose_and_learn("receive", self.choices, reward)

    def decide_speech(self, others):
        reward = 1 if self.is_alive() else -10
        speech = self._choose_and_learn("speech", self.speeches, reward)
        target = random.choice(others) if speech and others else None
        return target, speech

    def decide_action(self, others):
        reward = 1 if self.is_alive() else -10
        return self._choose_and_learn("action", self.actions, reward)

    def update_q(self, reward, next_state):
        decision_type, choice = self.last_decision
        if next_state not in self.q_table:
            self.q_table[next_state] = {
                "receive": {c: 0.0 for c in self.choices},
                "speech": {c: 0.0 for c in self.speeches},
                "action": {c: 0.0 for c in self.actions},
            }

        old_value = self.q_table[self.last_state][decision_type][choice]
        next_max = max(self.q_table[next_state][decision_type].values())
        new_value = old_value + self.alpha * (
                    reward + self.gamma * next_max - old_value)
        self.q_table[self.last_state][decision_type][choice] = new_value
