"""
src/agents/StaticRiskModel.py

Deterministic MLP actor for static-risk hedging.
"""

import numpy as np
import torch
import torch.nn as nn


class MLPActorStatic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 action_high=2.0, action_low=-2.0, init_action_mean=0.0):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias  = (action_high + action_low) / 2.0

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.05),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.q = nn.Parameter(torch.tensor(10.0))

        for i in [0, 2]:
            nn.init.orthogonal_(self.shared[i].weight, gain=np.sqrt(2))
            nn.init.constant_(self.shared[i].bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, np.arctanh(
            np.clip((init_action_mean - self.action_bias) / self.action_scale, -1 + 1e-6, 1 - 1e-6)
        ))

    def forward(self, s):
        h = self.shared(s)
        mu = self.mu_head(h)
        return self.action_scale * torch.tanh(mu) + self.action_bias

    def sample(self, s, deterministic=True):
        action = self.forward(s)
        log_prob = torch.zeros(s.shape[0], device=s.device)
        return action, log_prob
