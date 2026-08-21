import gc
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.agents.DynamicRiskModel import Actor, Critic_VaR_Excess
from src.risk_measures.loss_functions import critic_loss, actor_loss


# ===========================================================================
# Logging
# ===========================================================================

class TrainingLogger:
    """Logs metrics to console and a timestamped file."""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        self.metrics = {
            'iteration':                    [],
            'mean_cost':                    [],
            'critic_loss':                  [],
            'actor_loss':                   [],
            'V_early':                      [],
            'mean_terminal_hedging_error':  [],
            'std_terminal_hedging_error':   [],
            'mean_action':                  [],
            **{f'cvar_{p}': [] for p in range(91, 100)},
        }
        self.actor_loss_files = []
        self.log_file = open(self.run_dir / "training.log", "w")
        self._write(f"Training started at {timestamp}\n{'='*80}")

    def _write(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def log(self, iteration, critic_loss, actor_loss, V_early, mean_cost,
            mean_terminal_hedging_error, std_terminal_hedging_error,
            mean_action, cvar_dict=None):

        self.metrics['iteration'].append(iteration)
        self.metrics['critic_loss'].append(critic_loss)
        self.metrics['actor_loss'].append(actor_loss)
        self.metrics['V_early'].append(V_early)
        self.metrics['mean_cost'].append(mean_cost)
        self.metrics['mean_terminal_hedging_error'].append(mean_terminal_hedging_error)
        self.metrics['std_terminal_hedging_error'].append(std_terminal_hedging_error)
        self.metrics['mean_action'].append(mean_action)

        for p in range(91, 100):
            self.metrics[f'cvar_{p}'].append(
                cvar_dict.get(p, float('nan')) if cvar_dict else float('nan')
            )

        self._write(
            f"Iter {iteration}: critic_loss={critic_loss:.6f}, "
            f"actor_loss={actor_loss:.6f}, V_early={V_early:.2f}, "
            f"mean_cost={mean_cost:.6f}, "
            f"mean_terminal_hedging_error={mean_terminal_hedging_error:.6f}, "
            f"std_terminal_hedging_error={std_terminal_hedging_error:.6f}, "
            f"mean_action={mean_action:.6f}"
        )
        if cvar_dict:
            self._write("  CVaR: " + ", ".join(
                f"CVaR_{p}%={cvar_dict[p]:.4f}" for p in range(91, 100)
            ))

    def log_actor_losses(self, outer_iter, actor_losses):
        filename = f"actor_losses_iter_{outer_iter}.log"
        filepath = self.run_dir / filename
        with open(filepath, 'w') as f:
            f.write(f"Actor Losses - Outer Iteration {outer_iter}\n{'='*40}\n")
            f.write(f"Total: {len(actor_losses)}, Mean: {np.mean(actor_losses):.6f}, "
                    f"Std: {np.std(actor_losses):.6f}\n{'='*40}\niter, loss\n")
            for i, loss in enumerate(actor_losses):
                f.write(f"{i}, {loss:.6f}\n")
        self.actor_loss_files.append(filename)

    def save_metrics(self):
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({**self.metrics, 'actor_loss_files': self.actor_loss_files},
                      f, indent=2)
        self._write(f"Metrics saved to {metrics_file}")

    def close(self):
        self.save_metrics()
        self.log_file.close()


# ===========================================================================
# Main training function
# ===========================================================================

def train_rl_hedging(
    env,
    # Architecture
    actor_dim=256,
    head_dim_var=128,
    head_dim_excess=128,
    # Outer loop
    K=400,
    # Critic inner loop
    K_initial=2000,
    K_star=2000,
    K_target=300,
    K_final=None,          
    n_groups=18,
    n_increment=0,
    B0=16384,
    B1=2048,
    # Actor inner loop
    K_2=3,
    B2=2048,
    # Risk level
    alpha=0.95,
    scoring_fn='arcsin',
    C_shift=10.0,
    # Optimiser
    eta_theta=1e-3,
    min_lr_actor=1e-4,
    critic_reset_lr=1e-3,
    critic_grad_clip=2.5,  
    b_shift=True,          # True: shift each group's targets by the empirical min
    log_interval=10,
    log_dir="logs",
    resume_checkpoint=None,
):
    """
    Train the deep hedging agent via a two-phase (critic / actor) loop.

    Parameters
    ----------
    env               : HedgingEnv
    actor_dim         : hidden width for Actor MLP
    head_dim_var      : hidden width for VaR heads in Critic
    head_dim_excess   : hidden width for excess heads in Critic
    K                 : total outer iterations
    K_initial         : critic inner iterations for the first group
    K_star            : critic inner iterations for subsequent groups
    K_target          : target-network update frequency (steps)
    K_final           : unused; accepted only for call-signature compatibility
    n_groups          : number of backward-induction critic groups
    n_increment       : extra steps added per group index
    B0                : batch size for critic trajectory generation
    B1                : batch size for diagnostics
    K_2               : actor inner iterations per outer step
    B2                : batch size for actor trajectory generation
    alpha             : CVaR confidence level (e.g. 0.95)
    scoring_fn        : scoring function key (see loss_functions.py)
    C_shift           : shift constant for scoring functions
    eta_theta         : actor Adam learning rate
    min_lr_actor      : minimum actor learning rate (floor)
    critic_reset_lr   : SGD learning rate for critics
    critic_grad_clip  : max-norm for critic gradient clipping; None disables clipping
    b_shift           : if True, shift each group's bootstrap target by its empirical
                         min (b_list[g] = (costs + V_next).min()) as each group is
                         frozen. If False, b_list stays at 0 for every group (no shift).
    log_interval      : log + checkpoint every N outer iterations
    log_dir           : root directory for logs and checkpoints
    resume_checkpoint : path to a .pt checkpoint to resume from (optional)
    """
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    CHUNK  = 5000
    T_days = env.T_days
    group_size = T_days // n_groups
    assert T_days % n_groups == 0, \
        f"T_days ({T_days}) must be divisible by n_groups ({n_groups})"

    logger = TrainingLogger(log_dir=log_dir)

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    actor = Actor(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=actor_dim,
        fixed_std=1e-3,
    ).to(env.device)

    def _make_critic():
        return Critic_VaR_Excess(
            env.state_dim, group_size,
            head_dim_var=head_dim_var,
            head_dim_excess=head_dim_excess,
        ).to(env.device)

    critic_backwards = nn.ModuleList([_make_critic() for _ in range(n_groups)])
    grouped_targets  = nn.ModuleList([_make_critic() for _ in range(n_groups)])

    for gt in grouped_targets:
        for p in gt.parameters():
            p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Optimisers
    # -----------------------------------------------------------------------
    opt_actor       = optim.Adam(actor.parameters(), lr=eta_theta)
    scheduler_actor = optim.lr_scheduler.ExponentialLR(opt_actor, gamma=0.997)

    opt_critics = [
        optim.SGD(critic_backwards[g].heads, lr=critic_reset_lr,
                  momentum=0.9, nesterov=True)
        for g in range(n_groups)
    ]
    scheduler_critics = [
        optim.lr_scheduler.ExponentialLR(opt_critics[g], gamma=0.99999)
        for g in range(n_groups)
    ]

    # -----------------------------------------------------------------------
    # Optional checkpoint resume
    # -----------------------------------------------------------------------
    k_start        = 0
    b_list_history = []

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=env.device)
        actor.load_state_dict(ckpt['actor'])
        for g in range(n_groups):
            critic_backwards[g].load_state_dict(ckpt['critics'][g])
            critic_backwards[g].copy_to(grouped_targets[g])

        if 'opt_actor' in ckpt:
            opt_actor.load_state_dict(ckpt['opt_actor'])
        if 'scheduler_actor' in ckpt:
            scheduler_actor.load_state_dict(ckpt['scheduler_actor'])
        for g in range(n_groups):
            if 'opt_critics' in ckpt:
                try:
                    opt_critics[g].load_state_dict(ckpt['opt_critics'][g])
                except Exception:
                    pass
            if 'scheduler_critics' in ckpt:
                try:
                    scheduler_critics[g].load_state_dict(ckpt['scheduler_critics'][g])
                except Exception:
                    pass

        k_start        = ckpt.get('k', 0) + 1
        b_list_history = ckpt.get('b_list_history', [])
        print(f"Resumed from {resume_checkpoint} at k={k_start}, "
              f"alpha={ckpt.get('alpha')}, scoring_fn={ckpt.get('scoring_fn')}")

    # -----------------------------------------------------------------------
    # Diagnostic state
    # -----------------------------------------------------------------------
    V_early                     = float('nan')
    mean_cost                   = float('nan')
    mean_terminal_hedging_error = float('nan')
    std_terminal_hedging_error  = float('nan')
    mean_action                 = float('nan')
    cvar_dict                   = None

    # -----------------------------------------------------------------------
    # Outer loop
    # -----------------------------------------------------------------------
    for k in range(k_start, K):
        print(f"\n{'='*60}\nOUTER ITERATION {k}/{K}\n{'='*60}")

        flag = 1.0 if k == 0 else 0.5
        if n_increment == 0:
            iters_per_group = [int(K_star * flag)] * n_groups
        else:
            iters_per_group = [int(K_star * flag + i * n_increment)
                               for i in range(n_groups)]
        iters_per_group[0] = int(K_initial * flag)

        group_freeze_points = [sum(iters_per_group[:i+1]) - 1 for i in range(n_groups)]
        K_1 = sum(iters_per_group)

        # ===================================================================
        # PHASE 1 — Generate critic trajectories
        # ===================================================================
        print(f"\n[GENERATING {B0} TRAJECTORIES FOR CRITIC]")
        with torch.no_grad():
            (states_critic, actions_critic, log_probs_critic, costs_critic,
             _, _, portfolio_values_critic, derivative_values_critic,
             PnL_critic, term_pnl, _, _, _, _) = env.simulate_batch(actor, B0, alpha, seed=k * 1000)

        states_critic    = states_critic.detach()
        actions_critic   = actions_critic.detach()
        log_probs_critic = log_probs_critic.detach()
        costs_critic     = costs_critic.detach()

        with torch.no_grad():
            per_t_cvar  = torch.stack([
                costs_critic[t][
                    costs_critic[t] >= torch.quantile(costs_critic[t], alpha)
                ].mean()
                for t in range(T_days)
            ])
            suffix_cvar = per_t_cvar.flip(0).cumsum(0).flip(0)

        var_95 = torch.quantile(costs_critic, alpha, dim=1)
        tail   = costs_critic >= var_95.unsqueeze(1)
        cvar_95 = (costs_critic * tail).sum(dim=1) / tail.sum(dim=1).clamp(min=1)

        # ===================================================================
        # PHASE 1 — Critic update (backward induction)
        # ===================================================================
        print(f"\n[PHASE 1: CRITIC UPDATE]")
        critic_losses_list = []

        current_group = n_groups - 1
        group_iter    = 0
        count         = 0

        start_t = T_days - (group_iter + 1) * group_size
        end_t   = T_days - group_iter * group_size
        T_slice = end_t - start_t

        b_list   = [torch.tensor(0.0, device=env.device) for _ in range(n_groups)]
        std_list = [torch.tensor(1.0, device=env.device) for _ in range(n_groups)]

        with torch.no_grad():
            V_next_full = torch.zeros(T_slice, B0, device=env.device)
            v_next, e_next = grouped_targets[current_group](
                states_critic[start_t:end_t]
            )
            V_next_full[:-1] = v_next[1:] + e_next[1:]

        if b_shift:
            b_list[group_iter] = (costs_critic[-1] + V_next_full[-1]).min()
        V_next_full[-1] -= b_list[group_iter]

        states_group = states_critic[start_t:end_t].contiguous()
        costs_group  = costs_critic[start_t:end_t].contiguous()

        for critic_iter in range(K_1):
            col_idx      = torch.randint(0, B0, (B1,), device=env.device)
            states_batch = states_group[:, col_idx]
            costs_batch  = costs_group[:, col_idx]

            opt_critics[current_group].zero_grad(set_to_none=True)
            a1_total, a2_total = critic_backwards[current_group](states_batch)

            loss = critic_loss(
                a1_total, a2_total,
                costs_batch / std_list[group_iter],
                V_next_full[:, col_idx],
                alpha, scoring_fn=scoring_fn, C=C_shift
            )
            loss.backward()
            if critic_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    critic_backwards[current_group].parameters(),
                    max_norm=critic_grad_clip, norm_type=2
                )
            opt_critics[current_group].step()
            scheduler_critics[current_group].step()
            critic_losses_list.append(loss.detach().item())

            if critic_iter % 10000 == 0:
                print(f"  [Critic iter {critic_iter}] loss={loss.item():.6f}  "
                      f"lr={opt_critics[current_group].param_groups[0]['lr']:.2e}")

            # Target network update
            if count > 0 and count % K_target == 0:
                critic_backwards[current_group].copy_to(grouped_targets[current_group])
                with torch.no_grad():
                    V_next_full = torch.zeros(T_slice, B0, device=env.device)
                    v, e = grouped_targets[current_group](states_critic[start_t:end_t])
                    V_next_full[:-1] = v[1:] + e[1:]
                    if current_group < n_groups - 1:
                        v_last, e_last = grouped_targets[current_group + 1].forward_single_head(
                            states_critic[end_t], local_t=0
                        )
                        V_next_full[-1] = (
                            (v_last + e_last) * std_list[group_iter - 1]
                            + b_list[group_iter - 1]
                            - b_list[group_iter]
                        ) / std_list[group_iter]
                    else:
                        V_next_full[-1] -= b_list[group_iter]
                        V_next_full[-1] /= std_list[group_iter]
                count = 0

            count += 1

            # Group freeze / transition
            if group_iter < n_groups and critic_iter == group_freeze_points[group_iter]:
                critic_backwards[current_group].copy_to(grouped_targets[current_group])

                current_group -= 1
                group_iter    += 1
                count          = 0

                if current_group == -1:
                    print(f"All groups complete at critic iter {critic_iter}.")
                    break

                start_t = T_days - (group_iter + 1) * group_size
                end_t   = T_days - group_iter * group_size
                T_slice = end_t - start_t

                with torch.no_grad():
                    V_next_full = torch.zeros(T_slice, B0, device=env.device)
                    v, e = grouped_targets[current_group](states_critic[start_t:end_t])
                    V_next_full[:-1] = v[1:] + e[1:]
                    if current_group < n_groups - 1:
                        v_last, e_last = grouped_targets[current_group + 1].forward_single_head(
                            states_critic[end_t], local_t=0
                        )
                        V_next_full[-1] = (
                            (v_last + e_last) * std_list[group_iter - 1]
                            + b_list[group_iter - 1]
                        )
                    if b_shift:
                        y_full = costs_critic[start_t:end_t] + V_next_full
                        b_list[group_iter] = y_full[-1].min()
                    V_next_full[-1] -= b_list[group_iter]
                    V_next_full     /= std_list[group_iter]

                states_group = states_critic[start_t:end_t].contiguous()
                costs_group  = costs_critic[start_t:end_t].contiguous()

        b_list_history.append([b.item() for b in b_list])
        avg_critic_loss = float(np.mean(critic_losses_list)) if critic_losses_list else float('nan')

        del (states_critic, actions_critic, log_probs_critic, costs_critic,
             portfolio_values_critic, derivative_values_critic, PnL_critic,
             V_next_full, suffix_cvar, per_t_cvar, var_95, cvar_95,
             states_group, costs_group)
        gc.collect()
        torch.cuda.empty_cache()

        # ===================================================================
        # PHASE 2 — Actor update
        # ===================================================================
        print(f"\n[PHASE 2: ACTOR UPDATE]")
        actor_losses = []

        for actor_iter in range(K_2):
            B2_adj = int(B2 / (1 - alpha))
            print(f"  Actor iter {actor_iter+1}/{K_2} — generating {B2_adj} trajectories")

            (states_actor, actions_actor, log_probs_actor, costs_actor,
             _, _, portfolio_values_actor, derivative_values_actor,
             PnL_actor, term_pnl, _, _, _, _) = env.simulate_batch(
                actor, B2_adj, alpha,
                seed=k * 1000 + 500 + actor_iter
            )

            states_actor  = states_actor.detach()
            costs_actor   = costs_actor.detach()
            actions_actor = actions_actor.detach()

            T_actor, B_actor = states_actor.shape[:2]

            with torch.no_grad():
                V_next = torch.zeros(T_actor, B_actor, device=env.device)
                a1     = torch.zeros(T_actor, B_actor, device=env.device)

                for g in range(n_groups):
                    actual  = n_groups - 1 - g
                    g_start = g * group_size
                    g_end   = (g + 1) * group_size

                    v_parts, e_parts = [], []
                    for b0 in range(0, B_actor, CHUNK):
                        b1_ = min(b0 + CHUNK, B_actor)
                        v_c, e_c = critic_backwards[g](
                            states_actor[g_start:g_end, b0:b1_]
                        )
                        v_parts.append(v_c)
                        e_parts.append(e_c)
                    v = torch.cat(v_parts, dim=1)
                    e = torch.cat(e_parts, dim=1)

                    a1[g_start:g_end]       = v * std_list[actual] + b_list[actual]
                    V_next[g_start:g_end-1] = (v[1:] + e[1:]) * std_list[actual] + b_list[actual]

                    if g < n_groups - 1:
                        v_b, e_b = critic_backwards[g + 1].forward_single_head(
                            states_actor[g_end], local_t=0
                        )
                        V_next[g_end-1] = (v_b + e_b) * std_list[actual - 1] + b_list[actual - 1]

            opt_actor.zero_grad(set_to_none=True)
            total_loss     = 0.0
            LOG_PROB_CHUNK = 131_072

            for t in range(T_actor):
                for b0 in range(0, B_actor, LOG_PROB_CHUNK):
                    b1_ = min(b0 + LOG_PROB_CHUNK, B_actor)
                    lp  = actor.log_prob(
                        states_actor[t, b0:b1_],
                        actions_actor[t, b0:b1_]
                    )
                    loss_chunk = actor_loss(
                        lp.unsqueeze(0),
                        a1[t, b0:b1_].unsqueeze(0),
                        costs_actor[t, b0:b1_].unsqueeze(0),
                        V_next[t, b0:b1_].unsqueeze(0),
                        alpha
                    ) * (b1_ - b0) / (T_actor * B_actor)
                    loss_chunk.backward()
                    total_loss += loss_chunk.item()

            opt_actor.step()
            scheduler_actor.step()
            opt_actor.param_groups[0]['lr'] = max(
                opt_actor.param_groups[0]['lr'], min_lr_actor
            )

            actor_losses.append(total_loss)
            print(f"  [Actor iter {actor_iter}] loss={total_loss:.6f}  "
                  f"lr={opt_actor.param_groups[0]['lr']:.2e}")

            # Diagnostics on final actor iter of a log interval
            if actor_iter == K_2 - 1 and k % log_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()
                with torch.no_grad():
                    (states_diag, _, _, costs_diag, _, _,
                     portfolio_values_diag, derivative_values_diag, _, term_pnl, _, _, _, _) = \
                        env.simulate_batch(actor, B1, alpha,
                                           deterministic=True,
                                           seed=k * 1000 + 900)

                    T_diag, B_diag = states_diag.shape[:2]
                    V_diag = torch.zeros(T_diag, B_diag, device=env.device)
                    for g in range(n_groups):
                        actual  = n_groups - 1 - g
                        g_start = g * group_size
                        g_end   = (g + 1) * group_size
                        v_parts, e_parts = [], []
                        for b0 in range(0, B_diag, CHUNK):
                            b1_ = min(b0 + CHUNK, B_diag)
                            v_c, e_c = grouped_targets[g](
                                states_diag[g_start:g_end, b0:b1_]
                            )
                            v_parts.append(v_c)
                            e_parts.append(e_c)
                        v = torch.cat(v_parts, dim=1)
                        e = torch.cat(e_parts, dim=1)
                        V_diag[g_start:g_end] = (v + e) * std_list[actual] + b_list[actual]

                    V_early     = V_diag[0].mean().item()
                    mean_cost   = costs_diag.mean().item()
                    mean_action = actions_actor.mean().item()

                    mean_terminal_hedging_error = term_pnl.mean().item()
                    std_terminal_hedging_error  = term_pnl.std().item()

                    neg_port  = -term_pnl
                    cvar_dict = {}
                    for p in range(91, 100):
                        thresh       = torch.quantile(neg_port, p / 100.0)
                        tail         = neg_port[neg_port >= thresh]
                        cvar_dict[p] = tail.mean().item()

                    print(f"  mean_cost={mean_cost:.4f}  V_early={V_early:.4f}")
                    print(f"  Terminal: mean={mean_terminal_hedging_error:.4f}  "
                          f"std={std_terminal_hedging_error:.4f}")
                    print("  CVaR: " + "  ".join(
                        f"CVaR_{p}%={cvar_dict[p]:.4f}" for p in range(91, 100)
                    ))

                    del (states_diag, costs_diag, portfolio_values_diag,
                         derivative_values_diag, V_diag, term_pnl)
                    torch.cuda.empty_cache()

            del (states_actor, actions_actor, log_probs_actor, costs_actor,
                 portfolio_values_actor, derivative_values_actor,
                 PnL_actor, V_next, a1)
            torch.cuda.empty_cache()

        avg_actor_loss = float(np.mean(actor_losses))
        print(f"\n  Avg actor loss: {avg_actor_loss:.6f}")

        # ===================================================================
        # Logging & checkpointing
        # ===================================================================
        if k % log_interval == 0:
            logger.log_actor_losses(k, actor_losses)
            logger.log(k, avg_critic_loss, avg_actor_loss, V_early, mean_cost,
                       mean_terminal_hedging_error, std_terminal_hedging_error,
                       mean_action, cvar_dict=cvar_dict)

            ckpt_path = logger.run_dir / f'checkpoint_k{k}.pt'
            torch.save({
                'k':                 k,
                'actor':             actor.state_dict(),
                'critics':           [cb.state_dict() for cb in critic_backwards],
                'targets':           [gt.state_dict() for gt in grouped_targets],
                'opt_actor':         opt_actor.state_dict(),
                'opt_critics':       [opt.state_dict() for opt in opt_critics],
                'scheduler_actor':   scheduler_actor.state_dict(),
                'scheduler_critics': [sc.state_dict() for sc in scheduler_critics],
                'b_list':            [b.item() for b in b_list],
                'b_list_history':    b_list_history,
                'std_list':          [s.item() if hasattr(s, 'item') else s
                                      for s in std_list],
                'alpha':             alpha,
                'scoring_fn':        scoring_fn,
                'b_shift':           b_shift,
                'config': {
                    'K':                env.K,
                    'T_days':           env.T_days,
                    'n_assets':         env.n_assets,
                    'transaction_cost': env.transaction_cost,
                },
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Finalise
    # -----------------------------------------------------------------------
    logger.close()
    return actor, critic_backwards, logger.run_dir
