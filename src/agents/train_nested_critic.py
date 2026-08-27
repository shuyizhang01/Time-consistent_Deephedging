import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from src.agents.NestedCritic import CriticVaR


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

AMP_DTYPE = torch.bfloat16


def empirical_cvar(x, alpha):
    B, M = x.shape
    var = torch.quantile(x, alpha, dim=1, keepdim=True)
    tail = x >= var
    return (x * tail).sum(dim=1) / tail.sum(dim=1).clamp_min(1)


def _expand_t(t, B, M, device):
    if torch.is_tensor(t):
        return t.to(device=device, dtype=torch.float32).unsqueeze(1).expand(B, M).reshape(B * M)
    return torch.full((B * M,), float(t), device=device, dtype=torch.float32)


def _one_step_nested_core(env, S_t, h_t, Q_t, positions_t, actions_t, B_account_pre, deriv_t, t, M, generator=None):
    B = S_t.shape[0]
    sim = env.simulator

    trade = actions_t - positions_t
    tc_cost = env.transaction_cost * torch.sum(torch.abs(trade) * S_t, dim=1)
    B_post = B_account_pre - torch.sum(trade * S_t, dim=1) - tc_cost
    B_account = B_post * torch.exp(env.r_daily)
    y_t = B_account_pre + torch.sum(positions_t * S_t, dim=1) - deriv_t

    S_e = S_t.unsqueeze(1).expand(B, M, -1).reshape(B * M, env.n_assets)
    h_e = h_t.unsqueeze(1).expand(B, M, -1).reshape(B * M, env.n_assets)
    Q_e = Q_t.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, env.n_assets, env.n_assets)
    a_e = actions_t.unsqueeze(1).expand(B, M, -1).reshape(B * M, env.n_assets)
    B_acc_e = B_account.unsqueeze(1).expand(B, M).reshape(B * M)
    y_t_e = y_t.unsqueeze(1).expand(B, M).reshape(B * M)

    Q_bar = sim.Q_bar

    bad_Qe = ~torch.isfinite(Q_e).all(dim=(-1, -2))
    if bad_Qe.any():
        Q_e[bad_Qe] = Q_bar.unsqueeze(0).expand(bad_Qe.sum(), -1, -1)

    d_t = torch.sqrt(torch.diagonal(Q_e, dim1=-2, dim2=-1))
    R_t = Q_e / (d_t.unsqueeze(-1) * d_t.unsqueeze(-2))

    bad_Rt = ~torch.isfinite(R_t).all(dim=(-1, -2))
    if bad_Rt.any():
        R_t[bad_Rt] = Q_bar.unsqueeze(0).expand(bad_Rt.sum(), -1, -1)
        Q_e[bad_Rt] = Q_bar.unsqueeze(0).expand(bad_Rt.sum(), -1, -1)

    L = torch.linalg.cholesky(R_t)

    Z_corr, X_dcc = sim._sample_t_copula_gaussian_margins(B * M, env.n_assets, L, env.device, generator=generator)

    sqrt_h = torch.sqrt(h_e)
    h_tp1 = torch.clamp(sim.omega + sim.beta_garch * h_e + sim.alpha_garch * (Z_corr - sim.gamma * sqrt_h) ** 2, min=1e-12)
    S_tp1 = S_e * torch.exp(sim.r_daily + sim.lambda_ * h_e + sqrt_h * Z_corr)
    scale = torch.sqrt(sim.nu_Q / (sim.nu_Q - 2))
    X_dcc_std = X_dcc / scale

    outer_products = torch.einsum("bi,bj->bij", X_dcc_std, X_dcc_std)

    Q_tp1 = (1 - sim.dcc_alpha - sim.dcc_beta) * sim.Q_bar + sim.dcc_alpha * outer_products + sim.dcc_beta * Q_e
    Q_tp1 = (Q_tp1 + Q_tp1.transpose(-1, -2)) / 2.0

    bad_Qtp1 = ~torch.isfinite(Q_tp1).all(dim=(-1, -2))
    if bad_Qtp1.any():
        Q_tp1[bad_Qtp1] = Q_bar.unsqueeze(0).expand(bad_Qtp1.sum(), -1, -1)

    diag_min = torch.diagonal(Q_tp1, dim1=-2, dim2=-1).min(dim=-1).values
    needs_fix = diag_min < 1e-6
    if needs_fix.any():
        sub = Q_tp1[needs_fix]
        min_eigvals_sub = torch.linalg.eigvalsh(sub).min(dim=-1).values
        bad = min_eigvals_sub < 1e-8
        if bad.any():
            idx = needs_fix.nonzero(as_tuple=True)[0][bad]
            nudge = torch.clamp(-min_eigvals_sub[bad] + 1e-8, min=0.0)
            eye = torch.eye(env.n_assets, device=env.device).unsqueeze(0)
            Q_tp1[idx] += nudge[:, None, None] * eye

    d_tp1 = torch.sqrt(torch.diagonal(Q_tp1, dim1=-2, dim2=-1))
    R_tp1 = Q_tp1 / (d_tp1.unsqueeze(-1) * d_tp1.unsqueeze(-2))
    t_idx = _expand_t(t, B, M, env.device) + 1.0
    deriv_tp1 = env._price_derivative_batch(S_tp1, h_tp1, R_tp1, t_idx)

    return B, S_tp1, h_tp1, Q_tp1, R_tp1, deriv_tp1, a_e, B_acc_e, y_t_e


def one_step_nested(env, S_t, h_t, Q_t, positions_t, actions_t, B_account_pre, deriv_t, t, M, generator=None):
    B, S_tp1, h_tp1, Q_tp1, R_tp1, deriv_tp1, a_e, B_acc_e, y_t_e = _one_step_nested_core(
        env, S_t, h_t, Q_t, positions_t, actions_t, B_account_pre, deriv_t, t, M, generator=generator
    )
    y_next = B_acc_e + torch.sum(a_e * S_tp1, dim=1) - deriv_tp1
    cost = (y_t_e - y_next).reshape(B, M)

    return (
        cost,
        S_tp1.reshape(B, M, env.n_assets),
        h_tp1.reshape(B, M, env.n_assets),
        Q_tp1.reshape(B, M, env.n_assets, env.n_assets),
        B_acc_e.reshape(B, M)[:, 0],
    )


def _forward_chunked_flat(critic, states_flat, forward_chunk=16384, n_timesteps=None, local_t_offset=0):
    if n_timesteps is None:
        raise ValueError("n_timesteps must be provided for _forward_chunked_flat")
    total_rows = states_flat.shape[0]
    if total_rows % n_timesteps != 0:
        raise ValueError(f"states_flat has {total_rows} rows, not divisible by n_timesteps={n_timesteps}")
    N = total_rows // n_timesteps
    state_dim = states_flat.shape[-1]
    states = states_flat.reshape(n_timesteps, N, state_dim)
    outputs = []
    for i in range(n_timesteps):
        local_t = local_t_offset + i
        states_t = states[i]
        if N <= forward_chunk:
            V_t = critic.forward_single_head(states_t, local_t=local_t)
        else:
            chunks = []
            for start in range(0, N, forward_chunk):
                end = min(start + forward_chunk, N)
                chunks.append(critic.forward_single_head(states_t[start:end], local_t=local_t))
            V_t = torch.cat(chunks, dim=0)
        outputs.append(V_t)
    return torch.cat(outputs, dim=0)


def one_step_nested_terminal(env, S_t, h_t, Q_t, positions_t, actions_t, B_account_pre, deriv_t, t, M, generator=None):
    B, S_tp1, h_tp1, Q_tp1, R_tp1, deriv_tp1, a_e, B_acc_e, y_t_e = _one_step_nested_core(
        env, S_t, h_t, Q_t, positions_t, actions_t, B_account_pre, deriv_t, t, M, generator=generator
    )
    tc_liquidate = env.transaction_cost * torch.sum(torch.abs(a_e) * S_tp1, dim=1)
    y_next = B_acc_e + torch.sum(a_e * S_tp1, dim=1) - tc_liquidate - deriv_tp1
    return (y_t_e - y_next).reshape(B, M)


def build_state_tp1(env, S_tp1, h_tp1, Q_tp1, t, prev_actions, B_account):
    BM = S_tp1.shape[0]
    if torch.is_tensor(t):
        tau = ((env.T_days - (t.to(env.device).float() + 1.0)) / env.T_days).reshape(BM, 1)
    else:
        tau = torch.full((BM, 1), (env.T_days - (t + 1)) / env.T_days, device=env.device)
    moneyness = S_tp1 / env.K
    vol_feat = torch.sqrt(252 * h_tp1) / env.vol_scale
    triu = env.triu_indices
    d = torch.sqrt(torch.diagonal(Q_tp1, dim1=-2, dim2=-1))
    R = Q_tp1 / (d.unsqueeze(-1) * d.unsqueeze(-2))
    corr = R[:, triu[0], triu[1]]
    return torch.cat([moneyness, tau, vol_feat, prev_actions, corr], dim=-1)


def precompute_inner_samples(env, actions, portfolio_values, derivative_values, S_paths, h_paths, Q_paths,
                              start_t, end_t, M, B, offload_to_cpu=True, time_chunk=32, seed=None, generator=None):
    if generator is None and seed is not None:
        generator = torch.Generator(device=env.device)
        generator.manual_seed(seed)

    inner_cache = {}
    n_assets = env.n_assets
    device = env.device
    Tl = end_t - start_t
    is_terminal_group_end = (end_t - 1) == env.T_days - 1
    n_nonterm = (Tl - 1) if is_terminal_group_end else Tl

    with torch.no_grad():
        if n_nonterm > 0:
            prev_actions_full = torch.cat([torch.zeros(1, B, n_assets, device=device), actions[:-1]], dim=0)
            y_outer_full = torch.cat([torch.zeros(B, 1, device=device), portfolio_values[:, :-1]], dim=1)

            for chunk_start in range(0, n_nonterm, time_chunk):
                chunk_len = min(time_chunk, n_nonterm - chunk_start)
                cs = start_t + chunk_start
                ce = cs + chunk_len

                t_abs = torch.arange(cs, ce, device=device)

                positions_blk = prev_actions_full[cs:ce]
                y_outer_blk = y_outer_full[:, cs:ce].permute(1, 0)

                S_blk = S_paths[:, cs:ce].permute(1, 0, 2)
                h_blk = h_paths[:, cs:ce].permute(1, 0, 2)
                Q_blk = Q_paths[:, cs:ce].permute(1, 0, 2, 3)
                deriv_blk = derivative_values[:, cs:ce].permute(1, 0)
                act_blk = actions[cs:ce]

                B_account_pre_blk = y_outer_blk - torch.sum(positions_blk * S_blk, dim=-1) + deriv_blk

                BT = chunk_len * B
                S_f = S_blk.reshape(BT, n_assets)
                h_f = h_blk.reshape(BT, n_assets)
                Q_f = Q_blk.reshape(BT, n_assets, n_assets)
                pos_f = positions_blk.reshape(BT, n_assets)
                act_f = act_blk.reshape(BT, n_assets)
                Bacc_f = B_account_pre_blk.reshape(BT)
                deriv_f = deriv_blk.reshape(BT)
                t_row = t_abs.unsqueeze(1).expand(chunk_len, B).reshape(BT)

                cost_inner, S_tp1, h_tp1, Q_tp1, B_account_post = one_step_nested(
                    env, S_f, h_f, Q_f, pos_f, act_f, Bacc_f, deriv_f, t_row, M, generator=generator
                )

                BM = BT * M
                a_exp = act_f.unsqueeze(1).expand(BT, M, -1).reshape(BM, n_assets)
                B_acc_exp = B_account_post.unsqueeze(1).expand(BT, M).reshape(BM, 1)
                t_exp = t_row.unsqueeze(1).expand(BT, M).reshape(BM)

                s_tp1_flat = build_state_tp1(
                    env, S_tp1.reshape(BM, n_assets), h_tp1.reshape(BM, n_assets),
                    Q_tp1.reshape(BM, n_assets, n_assets), t_exp, a_exp, B_acc_exp
                )

                cost_inner = cost_inner.reshape(chunk_len, B, M)
                s_tp1_flat = s_tp1_flat.reshape(chunk_len, B, M, -1)

                if offload_to_cpu:
                    cost_inner = cost_inner.to('cpu', non_blocking=True).pin_memory()
                    s_tp1_flat = s_tp1_flat.to('cpu', non_blocking=True).pin_memory()

                for i in range(chunk_len):
                    inner_cache[chunk_start + i] = (cost_inner[i], s_tp1_flat[i])

                del S_tp1, h_tp1, Q_tp1, B_account_post, a_exp, B_acc_exp, s_tp1_flat, cost_inner

        if is_terminal_group_end:
            t = env.T_days - 1
            local_t = Tl - 1
            positions_t = torch.zeros(B, n_assets, device=device) if t == 0 else actions[t - 1]
            y_t_outer = torch.zeros(B, device=device) if t == 0 else portfolio_values[:, t - 1]
            B_account_pre_t = y_t_outer - torch.sum(positions_t * S_paths[:, t], dim=1) + derivative_values[:, t]
            cost_inner = one_step_nested_terminal(
                env, S_paths[:, t], h_paths[:, t], Q_paths[:, t], positions_t, actions[t],
                B_account_pre_t, derivative_values[:, t], t, M, generator=generator
            )
            if offload_to_cpu:
                cost_inner = cost_inner.to('cpu', non_blocking=True).pin_memory()
            inner_cache[local_t] = (cost_inner, None)

    return inner_cache


def compute_target_cvars_from_cache(inner_cache, start_t, end_t, costs, targets, current_group, group_size,
                                     n_groups, alpha_f, B, M, env, b, forward_chunk=16384, time_chunk=32):
    k = max(1, math.ceil((1 - alpha_f) * M))
    target_cvars = torch.zeros(group_size, B, device=env.device)
    t_last = end_t - 1
    n_inner = group_size - 1
    dev_type = torch.device(env.device).type

    def _to_gpu(x):
        return x.to(env.device, non_blocking=True) if x.device.type == "cpu" else x

    if n_inner > 0:
        for cs in range(0, n_inner, time_chunk):
            ce = min(cs + time_chunk, n_inner)
            clen = ce - cs

            costs_stack = torch.stack([_to_gpu(inner_cache[lt][0]) for lt in range(cs, ce)], dim=0)
            states_stack = torch.stack([_to_gpu(inner_cache[lt][1]) for lt in range(cs, ce)], dim=0)
            D = states_stack.shape[-1]

            with torch.autocast(device_type=dev_type, dtype=AMP_DTYPE, enabled=(dev_type == "cuda")):
                V_tp1 = _forward_chunked_flat(
                    targets[current_group], states_stack.reshape(clen * B * M, D),
                    forward_chunk=forward_chunk, n_timesteps=clen, local_t_offset=cs + 1,
                ).reshape(clen, B, M)

            V_tp1 = V_tp1.float() + b[current_group]
            target_cvars[cs:ce] = torch.topk(costs_stack + V_tp1, k, dim=-1).values.mean(dim=-1)

            del costs_stack, states_stack, V_tp1

    cost_last = _to_gpu(inner_cache[group_size - 1][0])

    if t_last == env.T_days - 1:
        target_cvars[-1] = torch.topk(cost_last.reshape(B, M), k, dim=1).values.mean(dim=1)
    else:
        next_group = current_group + 1
        s_tp1_last = _to_gpu(inner_cache[group_size - 1][1])

        with torch.autocast(device_type=dev_type, dtype=AMP_DTYPE, enabled=(dev_type == "cuda")):
            V_tp1_last = targets[next_group].forward_single_head(s_tp1_last.reshape(B * M, -1), local_t=0).reshape(B, M)

        V_tp1_last = V_tp1_last.float() + b[next_group]
        target_cvars[-1] = torch.topk(cost_last.reshape(B, M) + V_tp1_last, k, dim=1).values.mean(dim=1)

    return target_cvars


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


def build_critics_targets_opts_scheds(env, n_groups, hidden_dim, n_layers, critic_reset_lr, Nepochs):
    group_size = env.T_days // n_groups
    assert env.T_days % n_groups == 0

    critics = nn.ModuleList([
        CriticVaR(env.state_dim, group_size, hidden_dim, n_layers, device=env.device) for _ in range(n_groups)
    ])
    targets = nn.ModuleList([
        CriticVaR(env.state_dim, group_size, hidden_dim, n_layers, device=env.device) for _ in range(n_groups)
    ])

    optimizers = [
        optim.SGD(critics[g].parameters(), lr=critic_reset_lr, momentum=0.9, nesterov=True)
        for g in range(n_groups)
    ]
    schedulers = [
        optim.lr_scheduler.ExponentialLR(optimizers[g], gamma=0.999999) for g in range(n_groups)
    ]
    return critics, targets, optimizers, schedulers, group_size


def load_critic_checkpoint(env, path, n_groups=1, hidden_dim=512, n_layers=3, critic_reset_lr=1e-4, Nepochs=20):
    critics, targets, optimizers, schedulers, _ = build_critics_targets_opts_scheds(
        env, n_groups, hidden_dim, n_layers, critic_reset_lr, Nepochs
    )

    ckpt = torch.load(path, map_location=env.device, weights_only=False)

    for g in range(n_groups):
        critics[g].load_state_dict(ckpt['critics'][g])
        targets[g].load_state_dict(ckpt['targets'][g])
        optimizers[g].load_state_dict(ckpt['optimizers'][g])
        schedulers[g].load_state_dict(ckpt['schedulers'][g])
        for p in targets[g].parameters():
            p.requires_grad_(False)

    start_epoch = ckpt['epoch'] + 1
    print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}), resuming at epoch {start_epoch}")
    return critics, targets, optimizers, schedulers, start_epoch


def load_outer_batch_from_disk(env, data_dir, alpha_label, scoring_key, B, path_set="fixed_stochastic_policy"):
    if path_set == "fixed_stochastic_policy":
        paths_dir = os.path.join(data_dir, alpha_label, "_shared_paths")
        roll_dir = os.path.join(paths_dir, scoring_key, "stochastic_policy")
    elif path_set == "fixed":
        paths_dir = os.path.join(data_dir, alpha_label, "_shared_paths")
        roll_dir = os.path.join(paths_dir, scoring_key)
    elif path_set == "stochastic":
        paths_dir = os.path.join(data_dir, alpha_label)
        roll_dir = os.path.join(paths_dir, scoring_key)
    else:
        raise ValueError(f"unknown path_set: {path_set!r}")

    def _load(path):
        arr = np.load(path)
        return torch.from_numpy(arr[:B]).to(env.device).float()

    S_paths = _load(os.path.join(paths_dir, "S_paths.npy"))
    h_paths = _load(os.path.join(paths_dir, "h_paths.npy"))
    Q_paths = _load(os.path.join(paths_dir, "Q_paths.npy"))
    derivative_values = _load(os.path.join(paths_dir, "deriv_prices.npy"))

    states = torch.from_numpy(np.load(os.path.join(roll_dir, "states.npy"))).to(env.device).float()
    actions = torch.from_numpy(np.load(os.path.join(roll_dir, "actions.npy"))).to(env.device).float()
    states = states[:, :B].contiguous()
    actions = actions[:, :B].contiguous()

    portfolio_values = _load(os.path.join(roll_dir, "portfolio_values.npy"))

    costs = torch.cat([
        -portfolio_values[:, 0:1],
        portfolio_values[:, :-1] - portfolio_values[:, 1:],
    ], dim=1).contiguous()

    assert states.shape[0] == env.T_days, f"expected states dim0=T_days({env.T_days}), got {states.shape}"
    assert actions.shape[0] == env.T_days, f"expected actions dim0=T_days({env.T_days}), got {actions.shape}"
    assert states.shape[1] == B and actions.shape[1] == B, \
        f"expected batch dim={B}, got states {states.shape}, actions {actions.shape}"

    return states, actions, costs, portfolio_values, derivative_values, S_paths, h_paths, Q_paths


def train_critics_nested(env, actor, n_groups=1, alpha=95, B=500, M=200, K_star=500, K_initial=1000, Nepochs=50,
                          critic_reset_lr=2.5e-3, target_update_freq=300, B1=256, hidden_dim=1024, n_layers=4,
                          offload_inner_cache=False, forward_chunk=16384, time_chunk=32, resume_path=None,
                          resume_state=None, data_dir=None, alpha_label=None, scoring_key=None,
                          path_set="fixed_stochastic_policy", nested_seed=None):
    alpha_f = alpha / 100.0

    if resume_state is not None:
        critics, targets, optimizers, schedulers, start_epoch = resume_state
        group_size = env.T_days // n_groups
        assert env.T_days % n_groups == 0
    elif resume_path is not None:
        critics, targets, optimizers, schedulers, start_epoch = load_critic_checkpoint(
            env, resume_path, n_groups=n_groups, hidden_dim=hidden_dim, n_layers=n_layers,
            critic_reset_lr=critic_reset_lr, Nepochs=Nepochs
        )
        group_size = env.T_days // n_groups
        assert env.T_days % n_groups == 0
    else:
        critics, targets, optimizers, schedulers, group_size = build_critics_targets_opts_scheds(
            env, n_groups, hidden_dim, n_layers, critic_reset_lr, Nepochs
        )
        for g in range(n_groups):
            critics[g].copy_to(targets[g])
            for p in targets[g].parameters():
                p.requires_grad_(False)
        start_epoch = 0

    b = torch.zeros(n_groups, device=env.device)

    dev_type = torch.device(env.device).type
    amp_enabled = dev_type == "cuda"
    scalers = [torch.amp.GradScaler(dev_type, enabled=False) for _ in range(n_groups)]

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=env.device, weights_only=False)
        if "scalers" in ckpt:
            for g in range(n_groups):
                scalers[g].load_state_dict(ckpt["scalers"][g])
        if "b" in ckpt:
            b = ckpt["b"].to(env.device)

    use_disk_outer_batch = data_dir is not None and alpha_label is not None and scoring_key is not None
    cached_outer_batch = None
    if use_disk_outer_batch:
        print(f"  Loading outer batch from disk: {data_dir}/{alpha_label}/{scoring_key} (path_set={path_set}, B={B})")
        cached_outer_batch = load_outer_batch_from_disk(env, data_dir, alpha_label, scoring_key, B, path_set=path_set)

    for epoch in range(start_epoch, Nepochs):
        print(f"\n{'='*50}\nEpoch {epoch}/{Nepochs}\n{'='*50}")

        nested_gen = None
        if nested_seed is not None:
            nested_gen = torch.Generator(device=env.device)
            nested_gen.manual_seed(nested_seed + epoch)

        if use_disk_outer_batch:
            states, actions, costs, portfolio_values, derivative_values, S_paths, h_paths, Q_paths = cached_outer_batch
        else:
            with torch.no_grad():
                (states, actions, log_probs, costs, next_states, dones, portfolio_values, derivative_values,
                 PnL, terminal_pnl, S_paths, h_paths, R_paths, Q_paths) = env.simulate_batch(actor, B, nested=True)

        states, actions, costs = states.detach(), actions.detach(), costs.detach()
        derivative_values = derivative_values.detach()
        portfolio_values = portfolio_values.detach()
        S_paths, h_paths, Q_paths = S_paths.detach(), h_paths.detach(), Q_paths.detach()

        for group_iter in range(n_groups):
            current_group = n_groups - 1 - group_iter
            start_t = current_group * group_size
            end_t = (current_group + 1) * group_size
            n_iters = K_initial if group_iter == 0 else K_star

            print(f"\n  Group {current_group} | t=[{start_t},{end_t}) | iters={n_iters}")

            inner_cache = precompute_inner_samples(
                env, actions, portfolio_values, derivative_values, S_paths, h_paths, Q_paths,
                start_t, end_t, M, B, offload_to_cpu=offload_inner_cache, time_chunk=time_chunk, generator=nested_gen
            )

            with torch.no_grad():
                target_cvars = compute_target_cvars_from_cache(
                    inner_cache, start_t, end_t, costs, targets, current_group, group_size, n_groups, alpha_f,
                    B, M, env, b, forward_chunk=forward_chunk, time_chunk=time_chunk
                )
                target_cvars_shifted = target_cvars

            total_updates = 0
            while total_updates < n_iters:
                iters_this_round = min(target_update_freq, n_iters - total_updates)

                for it in range(iters_this_round):
                    idx = torch.randint(0, B, (B1,), device=env.device)
                    states_batch = states[start_t:end_t, idx]

                    optimizers[current_group].zero_grad(set_to_none=True)

                    with torch.autocast(device_type=dev_type, dtype=AMP_DTYPE, enabled=amp_enabled):
                        V_pred = critics[current_group](states_batch)
                        loss = F.mse_loss(V_pred, target_cvars_shifted[:, idx].detach())

                    scalers[current_group].scale(loss).backward()
                    scalers[current_group].step(optimizers[current_group])
                    scalers[current_group].update()
                    schedulers[current_group].step()

                    if (total_updates + it) % 2500 == 0:
                        with torch.no_grad():
                            diag_mean = critics[current_group](states[start_t:end_t, :]).mean(dim=1)
                            indices = torch.linspace(0, len(diag_mean) - 1, steps=10).long()
                            diag_sample = diag_mean[indices]

                        print(f"    group={current_group} it={total_updates+it} loss={loss.item():.4f} "
                              f"val_mean={diag_sample.tolist()}")

                total_updates += iters_this_round
                critics[current_group].copy_to(targets[current_group])

                if total_updates < n_iters:
                    with torch.no_grad():
                        target_cvars = compute_target_cvars_from_cache(
                            inner_cache, start_t, end_t, costs, targets, current_group, group_size, n_groups,
                            alpha_f, B, M, env, b, forward_chunk=forward_chunk, time_chunk=time_chunk
                        )
                        target_cvars_shifted = target_cvars

            critics[current_group].copy_to(targets[current_group])
            del inner_cache
            if dev_type == "cuda":
                torch.cuda.empty_cache()

        if (epoch + 1) == Nepochs:
            torch.save({
                'epoch': epoch,
                'critics': [c.state_dict() for c in critics],
                'targets': [t.state_dict() for t in targets],
                'optimizers': [o.state_dict() for o in optimizers],
                'schedulers': [s.state_dict() for s in schedulers],
                'scalers': [s.state_dict() for s in scalers],
                'b': b.detach().cpu(),
            }, f'critic_checkpoint_shared_epoch{epoch+1}.pt')
            print(f"  \u2713 Checkpoint saved at epoch {epoch+1}")

    return critics, targets, schedulers
