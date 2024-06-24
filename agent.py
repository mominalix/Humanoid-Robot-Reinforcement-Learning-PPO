# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_values = self.critic(x)
        return action_probs, state_values


class PPO:
    def __init__(
        self, state_dim, action_dim, lr=0.002, gamma=0.99, clip_eps=0.2, K_epochs=4
    ):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs

        self.memory = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        print(f"State tensor shape: {state.shape}")  # Debug print
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        action = np.random.choice(
            len(action_probs.numpy()[0]), p=action_probs.numpy()[0]
        )
        return action

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        # Training code, possibly including batching, PPO update steps, etc.
        pass
