import random
import torch
import torch.nn as nn

from Agent.AgentBase import Agent


class PolicyNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class NNBaseAgent(Agent):
    def __init__(self, name, base_item_name, epsilon=0.1):
        super().__init__(name, base_item_name)

        self.model_receive = PolicyNN(5, len(self.choices))
        self.model_speech = PolicyNN(5, len(self.speeches))
        self.model_action = PolicyNN(5, len(self.actions))

        self.optimizer = torch.optim.Adam(
            list(self.model_receive.parameters()) +
            list(self.model_speech.parameters()) +
            list(self.model_action.parameters()), lr=0.01
        )

        # exploration rate (epsilon)
        self.epsilon = epsilon

    def get_state(self):
        base = [self.hp, len(self.items), len(self.incoming)]

        if self.heard_log:
            speaker, speech = self.heard_log[-1]
            # encode speech
            a_count = speech.count("A")
            b_count = speech.count("B")
            speech_vec = [a_count, b_count]
        else:
            speech_vec = [0, 0]

        return torch.tensor(base + speech_vec, dtype=torch.float32)

    def _choose_and_learn(self, model, choices, reward=0):
        state = self.get_state()
        probs = model(state).clone()
        probs = probs / probs.sum()

        # epsilon-greedy: personality decides exploration rate
        if random.random() < self.epsilon:
            action = random.choice(range(len(choices)))  # exploration
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

        # learning step
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action))
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return choices[action]

    def decide_receive(self, item, giver):
        reward = 1 if self.is_alive() else -10
        return self._choose_and_learn(self.model_receive, self.choices, reward)

    def decide_speech(self, others):
        reward = 1 if self.is_alive() else -10
        speech = self._choose_and_learn(self.model_speech, self.speeches, reward)
        target = random.choice(others) if speech and others else None
        return target, speech

    def decide_action(self, others):
        reward = 1 if self.is_alive() else -10
        return self._choose_and_learn(self.model_action, self.actions, reward)


# -----------------------------
# Aggressive personality (high exploration)
# -----------------------------
class AggressiveNNAgent(NNBaseAgent):
    def __init__(self, name, base_item_name):
        super().__init__(name, base_item_name, epsilon=0.3)  # high exploration


# -----------------------------
# Generous personality (moderate exploration, cooperative bias)
# -----------------------------
class GenerousNNAgent(NNBaseAgent):
    def __init__(self, name, base_item_name):
        super().__init__(name, base_item_name, epsilon=0.15)  # moderate exploration


# -----------------------------
# Conservative personality (low exploration)
# -----------------------------
class ConservativeNNAgent(NNBaseAgent):
    def __init__(self, name, base_item_name):
        super().__init__(name, base_item_name, epsilon=0.05)  # low exploration
