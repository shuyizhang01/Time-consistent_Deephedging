import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Stochastic actor with tanh-squashed Gaussian policy.

    Actions are squashed to [action_low, action_high] via tanh.

    Parameters
    ----------
    state_dim        : int
    action_dim       : int
    hidden_dim       : int
    action_high      : float
    action_low       : float
    init_action_mean : float — target mean action before squashing
    fixed_std        : float — fixed exploration std (not learned)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        action_high=2.0,
        action_low=-2.0,
        init_action_mean=0.0,
        fixed_std=0.5,
    ):
        super().__init__()
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias  = (action_high + action_low) / 2.0
        self.fixed_std    = fixed_std

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.05),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)

        for i in [0, 2]:
            nn.init.orthogonal_(self.shared[i].weight, gain=np.sqrt(2))
            nn.init.constant_(self.shared[i].bias, 0.0)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(
            self.mu_head.bias,
            np.arctanh(np.clip(
                (init_action_mean - self.action_bias) / self.action_scale,
                -1 + 1e-6, 1 - 1e-6
            ))
        )

    def forward(self, s):
        h   = self.shared(s)
        mu  = self.mu_head(h)
        std = torch.full_like(mu, self.fixed_std)
        return mu, std

    def _log_prob_from_raw(self, dist, raw_action):
        log_prob  = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1)
        return log_prob

    def sample(self, s, deterministic=False):
        mu, std    = self.forward(s)
        dist       = torch.distributions.Normal(mu, std)
        raw_action = mu if deterministic else dist.rsample()
        action     = self.action_scale * torch.tanh(raw_action) + self.action_bias
        log_prob   = self._log_prob_from_raw(dist, raw_action)
        return action, log_prob

    def log_prob(self, s, a):
        mu, std    = self.forward(s)
        dist       = torch.distributions.Normal(mu, std)
        a_norm     = torch.clamp(
            (a - self.action_bias) / self.action_scale,
            -1 + 1e-6, 1 - 1e-6
        )
        raw_action = torch.atanh(a_norm)
        return self._log_prob_from_raw(dist, raw_action)


class Critic_VaR_Excess(nn.Module):
    """
    Vectorised critic that jointly predicts VaR (a1) and excess-over-VaR (a2)
    for every timestep in a group via a single batched forward pass.

    Each timestep has its own two-layer MLP head stored as a batched
    parameter tensor: state_dim → head_dim → 1.

    Optimised externally with Nesterov SGD (see DynamicRiskTraining.py).

    Parameters
    ----------
    state_dim       : int
    group_size      : int  — number of timesteps per critic group
    head_dim_var    : int  — hidden width for VaR heads
    head_dim_excess : int  — hidden width for excess heads
    """

    def __init__(self, state_dim, group_size, head_dim_var=32, head_dim_excess=16):
        super().__init__()
        self.group_size = group_size

        # Batched parameter tensors: shape [group_size, in, out]
        self.heads = nn.ParameterList([
            nn.Parameter(torch.zeros(group_size, state_dim,       head_dim_var)),    # var_W1
            nn.Parameter(torch.zeros(group_size, head_dim_var)),                     # var_b1
            nn.Parameter(torch.zeros(group_size, head_dim_var,    1)),               # var_W2
            nn.Parameter(torch.zeros(group_size, 1)),                                # var_b2
            nn.Parameter(torch.zeros(group_size, state_dim,       head_dim_excess)), # exc_W1
            nn.Parameter(torch.zeros(group_size, head_dim_excess)),                  # exc_b1
            nn.Parameter(torch.zeros(group_size, head_dim_excess, 1)),               # exc_W2
            nn.Parameter(torch.zeros(group_size, 1)),                                # exc_b2
        ])
        (self.var_W1, self.var_b1, self.var_W2, self.var_b2,
         self.exc_W1, self.exc_b1, self.exc_W2, self.exc_b2) = self.heads

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for h in range(self.group_size):
                nn.init.orthogonal_(self.var_W1.data[h], gain=np.sqrt(2))
                nn.init.zeros_(self.var_b1.data[h])
                nn.init.orthogonal_(self.var_W2.data[h], gain=0.01)
                nn.init.zeros_(self.var_b2.data[h])
                nn.init.orthogonal_(self.exc_W1.data[h], gain=np.sqrt(2))
                nn.init.zeros_(self.exc_b1.data[h])
                nn.init.orthogonal_(self.exc_W2.data[h], gain=0.01)
                nn.init.constant_(self.exc_b2.data[h], 0.0)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, states_batch):
        """
        Parameters
        ----------
        states_batch : Tensor [T, B, state_dim]

        Returns
        -------
        var    : Tensor [T, B]
        excess : Tensor [T, B]
        """
        var = (
            torch.einsum(
                'tbd,tdo->tbo',
                F.silu(torch.einsum('tbs,tsd->tbd', states_batch, self.var_W1)
                       + self.var_b1.unsqueeze(1)),
                self.var_W2
            ) + self.var_b2.unsqueeze(1)
        ).squeeze(-1)

        excess = F.leaky_relu(
            torch.einsum(
                'tbd,tdo->tbo',
                F.silu(torch.einsum('tbs,tsd->tbd', states_batch, self.exc_W1)
                       + self.exc_b1.unsqueeze(1)),
                self.exc_W2
            ) + self.exc_b2.unsqueeze(1),
            negative_slope=0.05
        ).squeeze(-1)

        return var, excess

    def forward_single_head(self, states, local_t):
        """
        Forward for a single timestep head (used at group boundaries).

        Parameters
        ----------
        states  : Tensor [B, state_dim]
        local_t : int

        Returns
        -------
        var    : Tensor [B]
        excess : Tensor [B]
        """
        v = (
            F.silu(states @ self.var_W1[local_t] + self.var_b1[local_t])
            @ self.var_W2[local_t] + self.var_b2[local_t]
        ).squeeze(-1)

        e = F.leaky_relu(
            F.silu(states @ self.exc_W1[local_t] + self.exc_b1[local_t])
            @ self.exc_W2[local_t] + self.exc_b2[local_t],
            negative_slope=0.05
        ).squeeze(-1)

        return v, e

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy_to(self, other):
        """Hard-copy all weights into another Critic_VaR_Excess."""
        with torch.no_grad():
            for p_src, p_dst in zip(self.parameters(), other.parameters()):
                p_dst.data.copy_(p_src.data)
            for b_src, b_dst in zip(self.buffers(), other.buffers()):
                b_dst.data.copy_(b_src.data)

    def state_dict_heads(self):
        """Plain dict of head tensors for checkpointing."""
        return {name: getattr(self, name).detach().clone()
                for name in ('var_W1', 'var_b1', 'var_W2', 'var_b2',
                             'exc_W1', 'exc_b1', 'exc_W2', 'exc_b2')}
