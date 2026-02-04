import random
import torch
import torch.nn as nn

from Agent.AgentBase import Agent


class SpeechEncoder(nn.Module):
    def __init__(self, vocab_size=3, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # vocab: "", A, B
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, seq_idx):
        emb = self.embedding(seq_idx)
        _, h = self.rnn(emb)
        return h.squeeze(0).squeeze(0)  # (hidden_dim,)


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

        self.speech_encoder = SpeechEncoder()
        state_dim = 3 + 32  # hp, items, incoming + speech embedding

        self.model_receive = PolicyNN(state_dim, len(self.choices))
        self.model_speech = PolicyNN(state_dim, len(self.speeches))
        self.model_action = PolicyNN(state_dim, len(self.actions))

        self.optimizer = torch.optim.Adam(
            list(self.model_receive.parameters()) +
            list(self.model_speech.parameters()) +
            list(self.model_action.parameters()) +
            list(self.speech_encoder.parameters()), lr=0.01
        )

        self.epsilon = epsilon

    def encode_speech(self, speech: str):
        mapping = {"": 0, "A": 1, "B": 2}
        seq_idx = [mapping[ch] for ch in speech if ch in mapping]
        if not seq_idx:
            seq_idx = [0]
        seq_tensor = torch.tensor(seq_idx, dtype=torch.long).unsqueeze(0)
        return self.speech_encoder(seq_tensor)

    def get_state(self):
        base = torch.tensor([self.hp, len(self.items), len(self.incoming)],
                            dtype=torch.float32)
        if self.heard_log:
            _, speech = self.heard_log[-1]
            speech_vec = self.encode_speech(speech).view(-1)
        else:
            speech_vec = torch.zeros(32)
        return torch.cat([base, speech_vec], dim=0)

    def _choose_and_learn(self, model, choices, reward=0):
        state = self.get_state()
        probs = model(state).clone()
        probs = probs / probs.sum()

        if random.random() < self.epsilon:
            action = random.choice(range(len(choices)))
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action))
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return choices[action]

    # --- Decision methods with HP-based reward ---
    def decide_receive(self, item, giver):
        prev_hp = self.hp

        # 選擇動作
        state = self.get_state()
        probs = self.model_receive(state).clone()
        probs = probs / probs.sum()

        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(self.choices)))
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        choice = self.choices[action_idx]

        # 執行後 HP 可能變化（例如接受物品之後）
        reward = self.hp - prev_hp

        # 更新模型
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action_idx))
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return choice

    def decide_speech(self, others):
        prev_hp = self.hp

        # 選擇動作
        state = self.get_state()
        probs = self.model_speech(state).clone()
        probs = probs / probs.sum()

        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(self.speeches)))
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        speech = self.speeches[action_idx]
        target = random.choice(others) if speech and others else None

        # reward = HP 變化量
        reward = self.hp - prev_hp

        # 更新模型
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action_idx))
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return target, speech

    def decide_action(self, others):
        prev_hp = self.hp

        # 選擇動作
        state = self.get_state()
        probs = self.model_action(state).clone()
        probs = probs / probs.sum()

        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(self.actions)))
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        action = self.actions[action_idx]

        # reward = HP 變化量
        reward = self.hp - prev_hp

        # 更新模型
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action_idx))
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return action

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
        super().__init__(name, base_item_name,
                         epsilon=0.15)  # moderate exploration


# -----------------------------
# Conservative personality (low exploration)
# -----------------------------
class ConservativeNNAgent(NNBaseAgent):
    def __init__(self, name, base_item_name):
        super().__init__(name, base_item_name, epsilon=0.05)  # low exploration
