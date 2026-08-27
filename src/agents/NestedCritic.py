import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class CriticVaR(nn.Module):
    def __init__(self, state_dim, group_size, hidden_dim=64, n_layers=None, device='cpu'):
        super().__init__()
        self.group_size = group_size
        self.device = device
 
        self.W1 = nn.Parameter(torch.zeros(group_size, state_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(group_size, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(group_size, hidden_dim, 1))
        self.b2 = nn.Parameter(torch.zeros(group_size, 1))
 
        self._init_weights()
        self.to(device)
 
    def _init_weights(self):
        with torch.no_grad():
            for h in range(self.group_size):
                nn.init.orthogonal_(self.W1.data[h], gain=np.sqrt(2))
                nn.init.zeros_(self.b1.data[h])
                nn.init.orthogonal_(self.W2.data[h], gain=0.01)
                nn.init.zeros_(self.b2.data[h])
 
    def forward(self, states_batch):
        return (
            torch.einsum(
                'tbd,tdo->tbo',
                F.silu(torch.einsum('tbs,tsd->tbd', states_batch, self.W1) + self.b1.unsqueeze(1)),
                self.W2,
            ) + self.b2.unsqueeze(1)
        ).squeeze(-1)
 
    def forward_chunked(self, states_batch, chunk=16384):
        Bdim = states_batch.shape[1]
        if Bdim <= chunk:
            return self.forward(states_batch)
        outs = []
        for start in range(0, Bdim, chunk):
            end = min(start + chunk, Bdim)
            outs.append(self.forward(states_batch[:, start:end, :]))
        return torch.cat(outs, dim=1)
 
    def forward_single_head(self, states, local_t):
        return (F.silu(states @ self.W1[local_t] + self.b1[local_t]) @ self.W2[local_t] + self.b2[local_t]).squeeze(-1)
 
    def copy_to(self, other):
        with torch.no_grad():
            for p_src, p_dst in zip(self.parameters(), other.parameters()):
                p_dst.data.copy_(p_src.data)
