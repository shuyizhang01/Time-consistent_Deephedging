import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

class HedgingEnv:
    def __init__(self, simulator, system, K, T_days, S0, r=0.04, transaction_cost=0.0001):
        """
        Args:
            simulator: DCCGARCHSimulator instance
            system: BasketOptionValuationSystem
            K: Strike price
            T_days: Number of trading days
            S0: Initial stock prices (will be converted to tensor if not already)
            r: Risk-free rate
            transaction_cost: Transaction cost rate
        """
        self.simulator = simulator
        self.K = K
        self.T_days = T_days
        self.r_daily = r


        self.transaction_cost = transaction_cost
        self.device = simulator.device  # Get device from simulator

        if isinstance(S0, torch.Tensor):
            self.S0 = S0.to(self.device)
        else:
            self.S0 = torch.tensor(S0, dtype=torch.float32, device=self.device)

        self.n_assets = len(self.S0)

        self.basket_weights = torch.tensor(
            system.weights, dtype=torch.float32, device=self.device
        )
        self.action_high = 1.0
        self.system = system
        self.system.call_model = self.system.call_model.to(self.device)
        if hasattr(self.system, 'put_model'):
            self.system.put_model = self.system.put_model.to(self.device)

        self.triu_indices = torch.triu_indices(
            self.n_assets, self.n_assets, offset=1, device=self.device
        )
        self._prepare_torch_scalers()
        self.h_unconditional = torch.tensor(
            [self.simulator.params[i]['h_unconditional'] for i in range(self.n_assets)],
            dtype=torch.float32,
            device=self.device
        )
        self.vol_scale = torch.tensor(
            np.sqrt(252 * np.array([self.simulator.params[i]['h_unconditional']
                                    for i in range(self.n_assets)])),
            dtype=torch.float32,
            device=self.device
        )
        self.n_corr_features = self.n_assets * (self.n_assets - 1) // 2
        self.state_dim = (
            self.n_assets +          # moneyness
            1 +                       # tau
            self.n_assets +           # volatilities
            self.n_assets +           # previous action
            self.n_corr_features
        )  # = 19
        self.action_dim = self.n_assets
        self.r_daily = torch.tensor(self.r_daily, dtype=torch.float32, device=self.device)


    def _correlation_to_features(self, R):
        """Extract upper triangle of correlation matrix"""
        triu_indices = np.triu_indices(self.n_assets, k=1)
        return R[triu_indices]

    def _prepare_torch_scalers(self):
        """Convert sklearn scalers to pure PyTorch for GPU (X only, no y)"""
        scaler_X = self.system.scaler_call['X']

        self.X_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=self.device)
        self.X_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=self.device)
    def _price_derivative_batch(self, S_batch, h_batch, R_batch, t_batch):
        """
        Feature layout MUST match training exactly:
            [moneyness(n_assets), vol_ratio(n_assets), T, log_moneyness, corr]
        Total price = discounted intrinsic (exact) + time value (NN, log1p space).
        """
        if not isinstance(S_batch, torch.Tensor):
            S_batch = torch.as_tensor(S_batch, dtype=torch.float32, device=self.device)
        elif S_batch.device != self.device:
            S_batch = S_batch.to(self.device)
        if not isinstance(h_batch, torch.Tensor):
            h_batch = torch.as_tensor(h_batch, dtype=torch.float32, device=self.device)
        elif h_batch.device != self.device:
            h_batch = h_batch.to(self.device)
        if not isinstance(R_batch, torch.Tensor):
            R_batch = torch.as_tensor(R_batch, dtype=torch.float32, device=self.device)
        elif R_batch.device != self.device:
            R_batch = R_batch.to(self.device)
        if not isinstance(t_batch, torch.Tensor):
            t_batch = torch.as_tensor(t_batch, dtype=torch.float32, device=self.device)
        elif t_batch.device != self.device:
            t_batch = t_batch.to(self.device)

        days_remaining = torch.clamp(self.T_days - t_batch, min=1.0)   # matches price()'s max(1, T*252)
        T_years = (self.T_days - t_batch) / 252.0

        basket_values = torch.sum(S_batch * self.basket_weights, dim=1)
        disc = torch.exp(-self.r_daily * days_remaining)
        intrinsic = torch.clamp(basket_values - self.K * disc, min=0.0)

        moneyness_feat = S_batch / 100.0
        vol_feat = torch.sqrt(h_batch / self.h_unconditional.unsqueeze(0))
        log_moneyness_feat = torch.log(basket_values / self.K).unsqueeze(1)
        corr_features = R_batch[:, self.triu_indices[0], self.triu_indices[1]]
        T_expanded = T_years.unsqueeze(1) if T_years.dim() == 1 else T_years

        X = torch.cat([moneyness_feat, vol_feat, T_expanded, log_moneyness_feat, corr_features], dim=1)
        X_scaled = (X - self.X_mean) / self.X_scale

        model = self.system.call_model
        model.eval()
        time_value = torch.clamp(torch.expm1(model(X_scaled).squeeze(-1)), min=0.0)
        return intrinsic + time_value

    def simulate_batch(self, actor, batch_size, alpha=0.05,
                      deterministic=False, seed=None, nested=False,
                      differentiable=False):
        """
        differentiable=True restores the old gradient-checkpointed actor forward
        (needed to backprop through the whole env, e.g. static model training).
        differentiable=False (default) wraps the actor call in no_grad -- pure
        inference, matches rollout_from_paths.
    
        Always returns the same 14-item tuple regardless of `nested`; fields
        that don't apply are None instead of changing the tuple's arity.
        """
        from torch.utils.checkpoint import checkpoint
    
        ACTOR_FORWARD_CHUNK = 131072
        CHUNK = 131072
        n_chunks = (batch_size + CHUNK - 1) // CHUNK
    
        all_S_paths_list, all_h_paths_list, all_R_paths_list, all_Q_paths_list, all_deriv_list = [], [], [], [], []
    
        for ci in range(n_chunks):
            cs = min(CHUNK, batch_size - ci * CHUNK)
            chunk_seed = None if seed is None else seed + ci
    
            S, h, R, Q = self._generate_all_paths_gpu(
                cs, seed=chunk_seed, randomize_init=True, nested=nested
            )
    
            with torch.no_grad():
                d = self._price_all_episodes_batched(S, h, R, chunk_size=16384)
    
            all_S_paths_list.append(S)
            all_h_paths_list.append(h)
            all_R_paths_list.append(R)
            if nested:
                all_Q_paths_list.append(Q)
            all_deriv_list.append(d)
            torch.cuda.empty_cache()
    
        all_S_paths = torch.cat(all_S_paths_list, dim=0)
        all_h_paths = torch.cat(all_h_paths_list, dim=0)
        all_R_paths = torch.cat(all_R_paths_list, dim=0)
        all_Q_paths = torch.cat(all_Q_paths_list, dim=0) if nested else None
        all_derivative_prices = torch.cat(all_deriv_list, dim=0)
    
        del all_S_paths_list, all_h_paths_list, all_R_paths_list, all_deriv_list
        if nested:
            del all_Q_paths_list
        torch.cuda.empty_cache()
    
        B = batch_size
        T = self.T_days
        self.MONEYNESS_MAX = 2.25
        self.VOL_MAX = 2.70
    
        # keep raw path tensors to return at the end (always, not just when nested)
        S_paths_return = all_S_paths
        h_paths_return = all_h_paths
        Q_paths_return = all_Q_paths
    
        all_corr_features = all_R_paths[:, :T, self.triu_indices[0], self.triu_indices[1]].clone()
        R_paths_return = all_R_paths  # keep a handle before any deletion
        torch.cuda.empty_cache()
    
        all_vol_feat = (torch.sqrt(252 * all_h_paths[:, :T]) / self.vol_scale).clone()
        torch.cuda.empty_cache()
    
        all_moneyness = (all_S_paths[:, :T] / self.K).clone()
    
        states_storage           = torch.zeros(B, T, self.state_dim, device=self.device)
        actions_storage          = torch.zeros(B, T, self.n_assets,  device=self.device)
        log_probs_storage        = torch.zeros(B, T,                 device=self.device)
        costs_storage            = torch.zeros(B, T,                 device=self.device)
        portfolio_values_storage = torch.zeros(B, T,                 device=self.device)
        PnL_storage               = torch.zeros(B, T,                device=self.device)
    
        prev_actions = torch.zeros((B, self.n_assets), device=self.device)
        positions    = torch.zeros(B, self.n_assets,   device=self.device)
        B_account    = all_derivative_prices[:, 0].clone()
    
        terminal_pnl = None
        if differentiable:
            dummy = next(actor.parameters()).sum() * 0
    
        for t in range(T):
            tau             = torch.full((B, 1), (T - t) / T, device=self.device)
            moneyness       = all_moneyness[:, t]
            vol_feat        = all_vol_feat[:, t]
            corr_features_t = all_corr_features[:, t]
            S_t             = all_S_paths[:, t]
            S_next          = all_S_paths[:, t + 1]
    
            states_t = torch.cat([
                moneyness, tau, vol_feat,
                prev_actions.detach() / self.action_high,
                corr_features_t
            ], dim=-1)
            states_storage[:, t] = states_t
    
            actions_chunks, logp_chunks = [], []
            for b0 in range(0, B, ACTOR_FORWARD_CHUNK):
                b1 = min(b0 + ACTOR_FORWARD_CHUNK, B)
                if differentiable:
                    def run_actor(s, d):
                        return actor.sample(s, deterministic=deterministic)
                    a_c, lp_c = checkpoint(run_actor, states_t[b0:b1], dummy, use_reentrant=False)
                else:
                    with torch.no_grad():
                        a_c, lp_c = actor.sample(states_t[b0:b1], deterministic=deterministic)
                actions_chunks.append(a_c)
                logp_chunks.append(lp_c)
    
            actions_t   = torch.cat(actions_chunks, dim=0)
            log_probs_t = torch.cat(logp_chunks,    dim=0)
            actions_storage[:, t]   = actions_t
            log_probs_storage[:, t] = log_probs_t
    
            trade         = actions_t - positions
            tc_cost       = self.transaction_cost * torch.sum(torch.abs(trade) * S_t, dim=1)
            B_account_pre = B_account
            B_post        = B_account - torch.sum(trade * S_t, dim=1) - tc_cost
            deriv_t       = all_derivative_prices[:, t]
            deriv_next    = all_derivative_prices[:, t + 1]
    
            y_t       = B_account_pre + torch.sum(positions * S_t, dim=1) - deriv_t
            B_account = B_post * torch.exp(self.r_daily)
    
            if t == T - 1:
                y_next = (
                    B_account + torch.sum(actions_t * S_next, dim=1)
                    - self.transaction_cost * torch.sum(torch.abs(actions_t) * S_next, dim=1)
                    - deriv_next
                )
                terminal_pnl = y_next          # explicitly captured, not just implicit in storage
            else:
                y_next = B_account + torch.sum(actions_t * S_next, dim=1) - deriv_next
    
            cost_t = y_t - y_next
            costs_storage[:, t]            = cost_t
            portfolio_values_storage[:, t] = y_next
            PnL_storage[:, t]              = torch.sum(actions_t * (S_next - S_t), dim=1)
    
            prev_actions = actions_t
            positions    = actions_t
    
        states    = states_storage.transpose(0, 1)
        actions   = actions_storage.transpose(0, 1)
        log_probs = log_probs_storage.transpose(0, 1)
        costs     = costs_storage.transpose(0, 1)
        PnL       = PnL_storage.transpose(0, 1)
    
        dones = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        dones[:, -1] = True
        dones = dones.transpose(0, 1)
    
        portfolio_values  = portfolio_values_storage
        derivative_values = all_derivative_prices
        next_states       = None   # never meaningfully populated in either old branch; kept for shape parity with rollout_from_paths
    
        # single return shape always -- 14 items, unused ones are None
        return (
            states, actions, log_probs, costs,
            next_states, dones,
            portfolio_values, derivative_values, PnL, terminal_pnl,
            S_paths_return, h_paths_return, R_paths_return, Q_paths_return,
        )


    def _price_all_episodes_batched(self, S_tensor, h_tensor, R_tensor, chunk_size=8192):
        B = S_tensor.shape[0]
        T_plus_1 = S_tensor.shape[1]

        S_flat = S_tensor.reshape(B * T_plus_1, self.n_assets)
        h_flat = h_tensor.reshape(B * T_plus_1, self.n_assets)
        R_flat = R_tensor.reshape(B * T_plus_1, self.n_assets, self.n_assets)
        t_indices = torch.arange(T_plus_1, device=self.device).repeat(B)

        total_samples = B * T_plus_1
        prices_list = []

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_prices = self._price_derivative_batch(
                S_flat[start_idx:end_idx],
                h_flat[start_idx:end_idx],
                R_flat[start_idx:end_idx],
                t_indices[start_idx:end_idx]
            )
            prices_list.append(chunk_prices)

        prices_flat = torch.cat(prices_list, dim=0)
        return prices_flat.reshape(B, T_plus_1)
    
    def _generate_all_paths_gpu(self, batch_size, seed=None, randomize_init=True, nested=False):
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)
    
        CHUNK = 20000
        device = self.device
    
        all_S_paths = torch.zeros(
            batch_size, self.T_days + 1, self.n_assets,
            device=device, dtype=torch.float32
        )
        all_h_paths = torch.zeros(
            batch_size, self.T_days + 1, self.n_assets,
            device=device, dtype=torch.float32
        )
        all_R_paths = torch.zeros(
            batch_size, self.T_days + 1, self.n_assets, self.n_assets,
            device=device, dtype=torch.float32
        )
        # always allocated now -- None-filled by caller logic below if not nested,
        # so the function has one return shape regardless of `nested`
        all_Q_paths = torch.zeros(
            batch_size, self.T_days + 1, self.n_assets, self.n_assets,
            device=device, dtype=torch.float32
        ) if nested else None
    
        omega = self.simulator.omega
        alpha_garch = self.simulator.alpha_garch
        beta_garch = self.simulator.beta_garch
        gamma = self.simulator.gamma
        lambda_ = self.simulator.lambda_
    
        # ==========================================================
        # INITIAL CONDITIONS
        # ==========================================================
        if randomize_init:
            log_shock = (
                torch.randn(batch_size, self.n_assets, device=device, generator=rng) * 0.15
            )
            all_S_paths[:, 0] = self.S0 * torch.exp(log_shock)
    
            h_noise = torch.exp(
                torch.randn(batch_size, self.n_assets, device=device, generator=rng) * 0.3
            )
            h_t_batch = self.simulator.h_unconditional.unsqueeze(0) * h_noise
        else:
            all_S_paths[:, 0] = self.S0.unsqueeze(0).expand(batch_size, -1)
            h_t_batch = self.simulator.h_unconditional.unsqueeze(0).expand(batch_size, -1).clone()
    
        h_t_batch = torch.clamp(h_t_batch, min=1e-12)
        all_h_paths[:, 0] = h_t_batch
    
        Q_t_batch = self.simulator.Q_bar.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    
        d_batch = torch.sqrt(torch.diagonal(Q_t_batch, dim1=1, dim2=2))
        all_R_paths[:, 0] = Q_t_batch / (d_batch.unsqueeze(-1) * d_batch.unsqueeze(-2))
    
        if nested:
            all_Q_paths[:, 0] = Q_t_batch
    
        r_daily = self.simulator.r_daily
        dcc_alpha = self.simulator.dcc_alpha
        dcc_beta = self.simulator.dcc_beta
        Q_bar = self.simulator.Q_bar

        # ==========================================================
        # SIMULATION LOOP
        # ==========================================================
        for day in range(self.T_days):
            d_batch = torch.sqrt(torch.diagonal(Q_t_batch, dim1=1, dim2=2))
            R_t_batch = Q_t_batch / (d_batch.unsqueeze(-1) * d_batch.unsqueeze(-2))
    
            if day % 10 == 0:
                bad_R = ~torch.isfinite(R_t_batch).all(dim=(-1, -2))
                if bad_R.any():
                    R_t_batch[bad_R] = Q_bar.unsqueeze(0).expand(bad_R.sum(), -1, -1)
                    Q_t_batch[bad_R] = Q_bar.unsqueeze(0).expand(bad_R.sum(), -1, -1)
    
            all_R_paths[:, day] = R_t_batch
    
            L_t_batch = torch.zeros_like(R_t_batch)
            for i in range(0, batch_size, CHUNK):
                j = min(i + CHUNK, batch_size)
                L_t_batch[i:j] = torch.linalg.cholesky(R_t_batch[i:j])
    
            Z_garch_batch, X_dcc_batch = self.simulator._sample_t_copula_gaussian_margins(
                batch_size, self.n_assets, L_t_batch, device, generator=rng
            )
    
            sqrt_h = torch.sqrt(h_t_batch)
            r_t_batch = r_daily + lambda_ * h_t_batch + sqrt_h * Z_garch_batch
            all_S_paths[:, day + 1] = all_S_paths[:, day] * torch.exp(r_t_batch)
    
            h_t_batch = omega + beta_garch * h_t_batch + alpha_garch * (Z_garch_batch - gamma * sqrt_h) ** 2
            h_t_batch = torch.clamp(h_t_batch, min=1e-12)
            all_h_paths[:, day + 1] = h_t_batch
    
            scale = torch.sqrt(self.simulator.nu_Q / (self.simulator.nu_Q - 2))
            X_dcc_std = X_dcc_batch / scale
            
            outer_products = torch.einsum("bi,bj->bij", X_dcc_std, X_dcc_std)
            
            Q_t_batch = (
                (1 - dcc_alpha - dcc_beta) * Q_bar
                + dcc_alpha * outer_products
                + dcc_beta * Q_t_batch
            )            
    
            if day % 10 == 0:
                bad_Q = ~torch.isfinite(Q_t_batch).all(dim=(-1, -2))
                if bad_Q.any():
                    Q_t_batch[bad_Q] = Q_bar.unsqueeze(0).expand(bad_Q.sum(), -1, -1)
    
            Q_t_batch = (Q_t_batch + Q_t_batch.transpose(-1, -2)) / 2.0
    
            diag_min = torch.diagonal(Q_t_batch, dim1=-2, dim2=-1).min(dim=-1).values
            needs_fix = diag_min < 1e-6
            if needs_fix.any():
                sub = Q_t_batch[needs_fix]
                min_eigvals_sub = torch.linalg.eigvalsh(sub).min(dim=-1).values
                bad = min_eigvals_sub < 1e-8
                if bad.any():
                    idx = needs_fix.nonzero(as_tuple=True)[0][bad]
                    nudge = torch.clamp(-min_eigvals_sub[bad] + 1e-8, min=0.0)
                    eye = torch.eye(self.n_assets, device=device).unsqueeze(0)
                    Q_t_batch[idx] += nudge[:, None, None] * eye
    
            if nested:
                all_Q_paths[:, day + 1] = Q_t_batch
    
        # ==========================================================
        # FINAL CORRELATION
        # ==========================================================
        d_batch = torch.sqrt(torch.diagonal(Q_t_batch, dim1=1, dim2=2))
        all_R_paths[:, self.T_days] = Q_t_batch / (d_batch.unsqueeze(-1) * d_batch.unsqueeze(-2))
    
        # single return shape always: (S, h, R, Q) -- Q is None if not nested
        return all_S_paths, all_h_paths, all_R_paths, all_Q_paths
    def rollout_from_paths(self, actor, S_paths, h_paths, R_paths, deriv_prices,
                          deterministic=False, seed=None):
        """
        Identical to simulate_batch (DR=True) but uses pre-generated paths
        instead of calling _generate_all_paths_gpu.

        Parameters
        ----------
        S_paths      : [B, T+1, n_assets]             torch.Tensor on self.device
        h_paths      : [B, T+1, n_assets]
        R_paths      : [B, T+1, n_assets, n_assets]   normalised correlation matrix
        deriv_prices : [B, T+1]
        seed         : int or None — unused, kept for API compatibility

        Returns
        -------
        states, actions, log_probs, costs, next_states, dones,
        portfolio_values, derivative_values, PnL, terminal_pnl,
        S_paths, h_paths, R_paths
        """
        ACTOR_CHUNK = 10_000
        actor = actor.to(S_paths.device)
        actor.eval()
        B = S_paths.shape[0]
        T = self.T_days

        S_paths = S_paths.clone()
        h_paths = h_paths.clone()

        # re-price derivatives with the passed-in paths directly
        with torch.no_grad():
            deriv_prices = self._price_all_episodes_batched(
                S_paths, h_paths, R_paths, chunk_size=16_384
            )

        # ── precompute normalised features ────────────────────────────────────
        all_corr = R_paths[:, :T, self.triu_indices[0], self.triu_indices[1]].clone()
        all_vol  = (torch.sqrt(252 * h_paths[:, :T]) / self.vol_scale).clone()
        all_mon  = (S_paths[:, :T] / self.K).clone()
        torch.cuda.empty_cache()

        # ── storage ───────────────────────────────────────────────────────────
        states_s  = torch.zeros(B, T, self.state_dim, device=self.device)
        actions_s = torch.zeros(B, T, self.n_assets,  device=self.device)
        logp_s    = torch.zeros(B, T,                 device=self.device)
        costs_s   = torch.zeros(B, T,                 device=self.device)
        portval_s = torch.zeros(B, T,                 device=self.device)
        pnl_s     = torch.zeros(B, T,                 device=self.device)

        prev_actions = torch.full((B, self.n_assets), 0.0, device=self.device)
        positions    = torch.zeros(B, self.n_assets,        device=self.device)
        B_account    = deriv_prices[:, 0].clone()

        terminal_pnl = None

        # ── timestep loop ─────────────────────────────────────────────────────
        for t in range(T):
            tau         = torch.full((B, 1), (T - t) / T, device=self.device)
            moneyness_t = all_mon[:, t]
            vol_t       = all_vol[:, t]
            corr_t      = all_corr[:, t]
            S_t         = S_paths[:, t]
            S_next      = S_paths[:, t + 1]


            state_t = torch.cat([
                moneyness_t, tau, vol_t,
                prev_actions.detach() / self.action_high,
                corr_t
            ], dim=-1)

            states_s[:, t] = state_t

            # ---- actor forward ----
            a_chunks, lp_chunks = [], []
            for b0 in range(0, B, ACTOR_CHUNK):
                b1 = min(b0 + ACTOR_CHUNK, B)
                with torch.no_grad():
                    a_c, lp_c = actor.sample(state_t[b0:b1], deterministic=deterministic)
                a_chunks.append(a_c)
                lp_chunks.append(lp_c)

            actions_t  = torch.cat(a_chunks,  dim=0)
            logprobs_t = torch.cat(lp_chunks, dim=0)

            actions_s[:, t] = actions_t
            logp_s[:, t]    = logprobs_t

            # ---- portfolio accounting ----
            trade   = actions_t - positions
            tc_cost = self.transaction_cost * torch.sum(torch.abs(trade) * S_t, dim=1)
            B_pre   = B_account
            B_post  = B_account - torch.sum(trade * S_t, dim=1) - tc_cost

            deriv_t    = deriv_prices[:, t]
            deriv_next = deriv_prices[:, t + 1]

            y_t       = B_pre + torch.sum(positions * S_t, dim=1) - deriv_t
            B_account = B_post * torch.exp(self.r_daily)

            if t == T - 1:
                y_next = (B_account
                          + torch.sum(actions_t * S_next, dim=1)
                          - self.transaction_cost * torch.sum(
                              torch.abs(actions_t) * S_next, dim=1)
                          - deriv_next)
                terminal_pnl = y_next
            else:
                y_next = B_account + torch.sum(actions_t * S_next, dim=1) - deriv_next

            costs_s[:, t]   = y_t - y_next
            portval_s[:, t] = y_next
            pnl_s[:, t]     = torch.sum(actions_t * (S_next - S_t), dim=1)

            prev_actions = actions_t
            positions    = actions_t

        # ── transpose ─────────────────────────────────────────────────────────
        states    = states_s.transpose(0, 1)
        actions   = actions_s.transpose(0, 1)
        log_probs = logp_s.transpose(0, 1)
        costs     = costs_s.transpose(0, 1)
        PnL       = pnl_s.transpose(0, 1)

        next_states = torch.zeros(B, T, dtype=torch.bool, device=self.device).transpose(0, 1)
        dones       = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        dones[:, -1] = True
        dones       = dones.transpose(0, 1)

        return (states, actions, log_probs, costs,
                next_states, dones,
                portval_s, deriv_prices, PnL, terminal_pnl,
                S_paths, h_paths, R_paths)
    def _build_all_states_batched(self, S_tensor, h_tensor, R_tensor):
        """
        Build states for all episodes in parallel

        Args:
            S_tensor: [B, T+1, n_assets]
            h_tensor: [B, T+1, n_assets]
            R_tensor: [B, T+1, n_assets, n_assets]

        Returns:
            states: [B, T, state_dim]
        """
        B = S_tensor.shape[0]

        # Moneyness for all timesteps (exclude last timestep)
        moneyness = S_tensor[:, :-1] / self.K  # [B, T, n_assets]

        t_indices = torch.arange(self.T_days, device=self.device)  # [T]
        tau = (self.T_days - t_indices) / self.T_days  # [T]
        tau = tau.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)  # [B, T, 1]

        # Volatilities (exclude last timestep)
        h = h_tensor[:, :-1]  # [B, T, n_assets]

        # Correlation features (exclude last timestep)
        R = R_tensor[:, :-1]  # [B, T, n_assets, n_assets]
        corr_features = R[:, :, self.triu_indices[0], self.triu_indices[1]]  # [B, T, n_corr]

        # Concatenate
        states = torch.cat([moneyness, tau, h, corr_features], dim=-1)  # [B, T, state_dim]

        return states
