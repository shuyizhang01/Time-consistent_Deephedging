"""
NestedCriticsModel.py
=====================
Model definition for the nested CVaR critic.

Place at: /content/DeepHedging/src/agents/NestedCriticsModel.py

Usage:
    from src.agents.NestedCriticsModel import CriticCVaR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticCVaR(nn.Module):
    """
    Vectorised CVaR critic for nested risk estimation.

    Each timestep in a group has its own two-layer MLP head stored as a
    batched parameter tensor: state_dim → head_dim → 1.

    Parameters
    ----------
    state_dim  : int
    group_size : int  — number of timesteps per critic group
    hidden_dim : int  — unused (kept for API compatibility)
    head_dim   : int  — hidden width for each head
    device     : str
    """

    def __init__(self, state_dim, group_size, hidden_dim=128, head_dim=64, device='cpu'):
        super().__init__()
        self.group_size = group_size
        self.device     = device

        self.W1 = nn.Parameter(torch.zeros(group_size, state_dim, head_dim, device=device))
        self.b1 = nn.Parameter(torch.zeros(group_size, head_dim,            device=device))
        self.W2 = nn.Parameter(torch.zeros(group_size, head_dim, 1,         device=device))
        self.b2 = nn.Parameter(torch.zeros(group_size, 1,                   device=device))

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            gain = torch.tensor(2.0).sqrt().item()
            for h in range(self.group_size):
                nn.init.orthogonal_(self.W1.data[h], gain=gain)
                nn.init.zeros_(self.b1.data[h])
                nn.init.orthogonal_(self.W2.data[h], gain=0.01)
                nn.init.zeros_(self.b2.data[h])

    def forward(self, states_batch):
        """states_batch: [group_size, B, state_dim] → [group_size, B]"""
        return (
            torch.einsum(
                'tbd,tdo->tbo',
                F.silu(
                    torch.einsum('tbs,tsd->tbd', states_batch, self.W1)
                    + self.b1.unsqueeze(1)
                ),
                self.W2,
            )
            + self.b2.unsqueeze(1)
        ).squeeze(-1)

    def forward_single_head(self, states, local_t):
        """states: [B, state_dim] → [B]"""
        return (
            F.silu(states @ self.W1[local_t] + self.b1[local_t])
            @ self.W2[local_t]
            + self.b2[local_t]
        ).squeeze(-1)

    def copy_to(self, other):
        with torch.no_grad():
            for p_src, p_dst in zip(self.parameters(), other.parameters()):
                p_dst.data.copy_(p_src.data)
