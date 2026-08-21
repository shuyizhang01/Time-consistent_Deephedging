"""
src/agents/StaticRiskTraining.py

Training loop for the static-risk hedging agent.

Minimises CVaR of terminal loss via the Rockafellar-Uryasev dual:
    CVaR_α(L) = min_q { q + 1/(1-α) * E[max(L - q, 0)] }

Gradients flow through the full episode rollout → terminal_pnl by calling
env.simulate_batch(..., differentiable=True).
"""

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.agents.StaticRiskModel import MLPActorStatic


def train_static_hedging(
    env,
    alpha:             float        = 0.95,
    n_iter:            int          = 500,
    batch_size:        int          = 4096,
    lr:                float        = 1e-3,
    hidden_dim:        int          = 256,
    action_high:       float        = 2.0,
    action_low:        float        = -2.0,
    init_action_mean:  float        = 0.0,
    log_interval:      int          = 10,
    log_dir:           str          = "logs/static",
    resume_checkpoint: str | None   = None,
) -> tuple[MLPActorStatic, list[dict]]:
    """
    Train the static-risk hedging actor.

    Parameters
    ----------
    env                : HedgingEnv
    alpha              : CVaR confidence level
    n_iter             : total number of gradient steps
    batch_size         : trajectories used for evaluation; training uses
                         int(batch_size / (1-alpha)) to get enough tail samples
    lr                 : Adam learning rate for policy (shared + mu_head) parameters
    hidden_dim         : hidden units in the MLP
    action_high        : action bound (positions clipped to this upper bound)
    action_low         : action bound (positions clipped to this lower bound)
    init_action_mean   : initial mean action the policy is biased toward at init
    log_interval       : evaluate and print every this many iterations
    log_dir            : directory for checkpoints and results JSON
    resume_checkpoint  : path to a .pth checkpoint to resume from (or None)

    Returns
    -------
    actor   : trained MLPActorStatic
    results : list of dicts with per-eval metrics
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    actor = MLPActorStatic(
        state_dim        = env.state_dim,
        action_dim        = env.action_dim,
        hidden_dim        = hidden_dim,
        action_high       = action_high,
        action_low        = action_low,
        init_action_mean  = init_action_mean,
    ).to(env.device)

    policy_params = list(actor.shared.parameters()) + list(actor.mu_head.parameters())

    opt_actor = optim.Adam(policy_params, lr=lr)
    opt_q     = optim.Adam([actor.q], lr=lr * 1000)
    scheduler = optim.lr_scheduler.ExponentialLR(opt_actor, gamma=0.998)

    start_iter = 0
    results: list[dict] = []

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=env.device)
        actor.load_state_dict(ckpt['actor_state_dict'])
        opt_actor.load_state_dict(ckpt['opt_actor_state_dict'])
        opt_q.load_state_dict(ckpt['opt_q_state_dict'])
        start_iter = ckpt.get('iter', 0)
        results    = ckpt.get('results', [])
        print(f"✓ Resumed from {resume_checkpoint} at iteration {start_iter}")

    B_inflated = int(batch_size / (1 - alpha))

    for it in range(start_iter, n_iter):
        print(f"Iteration: {it}")

        opt_actor.zero_grad(set_to_none=True)
        opt_q.zero_grad(set_to_none=True)

        (
            states, actions, log_probs, costs,
            next_states, dones,
            portfolio_values, derivative_values, PnL, terminal_pnl,
            S_paths_return, h_paths_return, R_paths_return, Q_paths_return,
        ) = env.simulate_batch(
            actor, B_inflated,
            deterministic=True,
            seed=it * 1000,
            differentiable=True,
            nested=False
        )

        del (
            states, actions, log_probs, costs,
            next_states, dones,
            portfolio_values, derivative_values, PnL,
            S_paths_return, h_paths_return, R_paths_return, Q_paths_return,
        )

        loss_val = -terminal_pnl   # terminal loss = negative terminal P&L

        print(
            f"  terminal_pnl  "
            f"nan={torch.isnan(terminal_pnl).sum().item()} "
            f"inf={torch.isinf(terminal_pnl).sum().item()} "
            f"min={terminal_pnl.min().item():.3f} "
            f"max={terminal_pnl.max().item():.3f}"
        )

        if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
            print("  !! NaN/Inf in loss_val — skipping iteration")
            del terminal_pnl, loss_val
            torch.cuda.empty_cache()
            continue

        # CVaR primal: CVaR_α(L) = q + 1/(1-α) * E[max(L - q, 0)]
        q    = actor.q
        diff = loss_val - q
        cvar = q + (1.0 / (1 - alpha)) * torch.relu(diff).mean()

        cvar.backward()
        torch.nn.utils.clip_grad_norm_(policy_params, max_norm=10.0)

        nan_grads = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in policy_params
        )
        if nan_grads:
            print("  !! NaN gradient — skipping step")
            opt_actor.zero_grad(set_to_none=True)
            opt_q.zero_grad(set_to_none=True)
        else:
            opt_actor.step()
            opt_q.step()

        scheduler.step()

        del loss_val, diff, cvar

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if (it + 1) % log_interval == 0:
            with torch.no_grad():
                (
                    states, actions, log_probs, costs,
                    next_states, dones,
                    portfolio_values, derivative_values, PnL, te_det,
                    S_paths_return, h_paths_return, R_paths_return, Q_paths_return,
                ) = env.simulate_batch(
                    actor, batch_size,
                    deterministic=True,
                    seed=it*3000 + 100,
                    differentiable=True,
                    nested=False
                )

                del (
                    states, actions, log_probs, costs,
                    next_states, dones,
                    portfolio_values, derivative_values, PnL,
                    S_paths_return, h_paths_return, R_paths_return, Q_paths_return,
                )

                loss_det   = -te_det
                var_det    = torch.quantile(loss_det, alpha)
                cvar_det   = loss_det[loss_det >= var_det].mean()
                var_lower  = torch.quantile(te_det, 0.05)
                cvar_lower = te_det[te_det <= var_lower].mean()

                print(
                    f"Iter {it+1:4d} | "
                    f"CVaR_{alpha}={cvar_det:.4f} | "
                    f"VaR={var_det:.4f} | "
                    f"mean={te_det.mean():.4f} | "
                    f"std={te_det.std():.4f} | "
                    f"CVaR_lower={cvar_lower:.4f} | "
                    f"q={actor.q.item():.4f}"
                )

                results.append({
                    'iter':       it + 1,
                    'cvar':       cvar_det.item(),
                    'var':        var_det.item(),
                    'mean':       te_det.mean().item(),
                    'std':        te_det.std().item(),
                    'cvar_lower': cvar_lower.item(),
                })
                del te_det, loss_det, var_det, cvar_det, var_lower, cvar_lower

            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if (it + 1) % (log_interval * 10) == 0:
            ckpt_path = log_path / f'checkpoint_alpha{alpha}_iter{it+1}.pth'
            torch.save({
                'iter':                 it + 1,
                'actor_state_dict':     actor.state_dict(),
                'opt_actor_state_dict': opt_actor.state_dict(),
                'opt_q_state_dict':     opt_q.state_dict(),
                'results':              results,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved to {ckpt_path}")

        del terminal_pnl
        gc.collect()
        torch.cuda.empty_cache()

    # Save final results
    results_path = log_path / f'results_alpha{alpha}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_path}")

    return actor, results
