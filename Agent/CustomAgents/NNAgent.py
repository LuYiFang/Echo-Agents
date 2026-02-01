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

        # 三個獨立的 policy network
        self.model_receive = PolicyNN(input_dim=3, output_dim=len(
            self.choices))  # accept/reject
        self.model_speech = PolicyNN(input_dim=3, output_dim=len(
            self.speeches))  # "", A, B, AB
        self.model_action = PolicyNN(input_dim=3, output_dim=len(
            self.actions))  # combine, give, use, none

    def get_state(self):
        # 狀態向量：HP, inventory 數量, incoming 數量
        return torch.tensor([self.hp, len(self.items), len(self.incoming)],
                            dtype=torch.float32)

    def decide_receive(self, item, giver):
        state = self.get_state()
        probs = self.model_receive(state)
        idx = torch.multinomial(probs, 1).item()
        return self.choices[idx]

    def decide_speech(self, others):
        state = self.get_state()
        probs = self.model_speech(state)
        idx = torch.multinomial(probs, 1).item()
        speech = self.speeches[idx]
        target = random.choice(others) if speech and others else None
        return target, speech

    def decide_action(self, others):
        state = self.get_state()
        probs = self.model_action(state)
        idx = torch.multinomial(probs, 1).item()
        return self.actions[idx]
