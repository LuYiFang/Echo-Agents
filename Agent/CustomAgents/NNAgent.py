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


class NNAgent(Agent):
    def __init__(self, name, base_item_name):
        super().__init__(name, base_item_name)

        self.model_receive = PolicyNN(3, len(self.choices))
        self.model_speech  = PolicyNN(3, len(self.speeches))
        self.model_action  = PolicyNN(3, len(self.actions))

        self.optimizer = torch.optim.Adam(
            list(self.model_receive.parameters()) +
            list(self.model_speech.parameters()) +
            list(self.model_action.parameters()), lr=0.01
        )

    def get_state(self):
        return torch.tensor([self.hp, len(self.items), len(self.incoming)], dtype=torch.float32)

    def _choose_and_learn(self, model, choices, reward=0):
        state = self.get_state()
        probs = model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return choices[action.item()]

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