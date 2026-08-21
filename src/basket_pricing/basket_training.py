"""
src/basket_pricing/basket_training.py
======================================
Generate + train logic for the basket-option DCC-GARCH surrogate.

Per-asset Heston-Nandi GARCH params (P-measure) and the DCC(1,1) +
Student-t copula params (Q_bar, dcc_alpha, dcc_beta, nu) are loaded
directly from cfgs/configDynamicRisk.yaml (as produced by
src/data/calibration.py's fit_all_garch / fit_dcc_parameters pipeline).
Risk-Neutral transformation is applied to Heston-Nandi GARCH params.

This module is imported by the thin CLI entry points in scripts/
(generate_basket_data.py, train_basket_surrogate.py)
"""
import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import yfinance as yf
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.basket_pricing.basket import (
    BasketOptionNet,
    HestonNandiGARCH_Q,
    correlation_matrix_to_features,
)

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG LOADING
# ============================================================================
def _load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# FEATURE VECTOR -> CORRELATION MATRIX
# ============================================================================
def features_to_correlation_matrix(corr_vec, n_assets):
    R = np.eye(n_assets)
    triu = np.triu_indices(n_assets, k=1)
    R[triu] = corr_vec
    R = R + R.T - np.diag(np.diag(R))
    return R


# ============================================================================
# GPU MONTE-CARLO SIMULATOR
# ============================================================================
class FastDCCGARCHSimulator:
    """GPU-accelerated DCC-GARCH path simulator used for training-data generation."""

    def __init__(self, garch_models, Q_bar, dcc_alpha, dcc_beta, nu, device="cuda", asset_names=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_assets = len(garch_models)
        assert Q_bar.shape == (self.n_assets, self.n_assets), \
            f"Q_bar shape {Q_bar.shape} does not match n_assets={self.n_assets}"
        if asset_names is not None:
            assert len(asset_names) == self.n_assets
        self.asset_names = asset_names
        self.dcc_alpha = dcc_alpha
        self.dcc_beta = dcc_beta
        self.nu = nu
        self._build_t_cdf_lut(nu)

        self.omega = torch.tensor([m.omega for m in garch_models], dtype=torch.float32, device=self.device)
        self.alpha = torch.tensor([m.alpha for m in garch_models], dtype=torch.float32, device=self.device)
        self.beta = torch.tensor([m.beta for m in garch_models], dtype=torch.float32, device=self.device)
        self.gamma = torch.tensor([m.gamma for m in garch_models], dtype=torch.float32, device=self.device)
        self.lambda_ = torch.tensor([m.lambda_ for m in garch_models], dtype=torch.float32, device=self.device)
        self.Q_bar = torch.tensor(Q_bar, dtype=torch.float32, device=self.device)

    def _build_t_cdf_lut(self, nu, grid_size=50_000, clip=1e-6, margin=1.0):
        """One-time CPU table build so _t_cdf_gpu never round-trips to scipy."""
        x_lo = float(t_dist.ppf(clip, df=nu))
        x_hi = float(t_dist.ppf(1 - clip, df=nu))
        x_grid = np.linspace(x_lo - margin, x_hi + margin, grid_size)
        cdf_grid = t_dist.cdf(x_grid, df=nu)
        self._t_lut_x = torch.tensor(x_grid, dtype=torch.float32, device=self.device)
        self._t_lut_cdf = torch.tensor(cdf_grid, dtype=torch.float32, device=self.device)
        self._t_lut_xlo, self._t_lut_xhi = x_lo, x_hi
        self._t_clip_lo, self._t_clip_hi = clip, 1 - clip

    def _t_cdf_gpu(self, X):
        flat = X.reshape(-1)
        xc = flat.clamp(self._t_lut_xlo, self._t_lut_xhi)
        idx = torch.searchsorted(self._t_lut_x, xc).clamp(1, self._t_lut_x.numel() - 1)
        x0, x1 = self._t_lut_x[idx - 1], self._t_lut_x[idx]
        y0, y1 = self._t_lut_cdf[idx - 1], self._t_lut_cdf[idx]
        U = y0 + (xc - x0) / (x1 - x0) * (y1 - y0)
        U = torch.where(flat <= self._t_lut_xlo, torch.full_like(U, self._t_clip_lo), U)
        U = torch.where(flat >= self._t_lut_xhi, torch.full_like(U, self._t_clip_hi), U)
        return U.reshape(X.shape)

    def _sample_t_copula_gaussian_margins(self, B, n, L_chol, generator=None):
        device = self.device
        nu = float(self.nu)

        g = torch.randn(B, n, device=device, generator=generator)
        g = torch.bmm(L_chol, g.unsqueeze(-1)).squeeze(-1)

        half_nu = torch.tensor(nu / 2.0, device=device)
        chi2 = 2.0 * torch._standard_gamma(half_nu.expand(B))
        X = g / torch.sqrt(chi2 / nu).unsqueeze(-1)

        U = self._t_cdf_gpu(X).clamp(1e-6, 1 - 1e-6)
        Z = torch.distributions.Normal(0.0, 1.0).icdf(U)
        return Z, X

    def simulate_batch_gpu(self, S0, h0, T_days, r_daily, R_t_initial, batch_size, generator=None):
        B, n = batch_size, self.n_assets

        S_t = torch.tensor(S0, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1).clone()
        h_t = torch.tensor(h0, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1).clone()
        log_return = torch.zeros((B, n), dtype=torch.float32, device=self.device)

        R_t_initial_t = torch.tensor(R_t_initial, dtype=torch.float32, device=self.device)
        d_initial = torch.sqrt(torch.diag(R_t_initial_t))
        Q_t = (torch.outer(d_initial, d_initial) * R_t_initial_t).unsqueeze(0).expand(B, -1, -1).clone()
        Q_bar_b = self.Q_bar.unsqueeze(0)
        eye = torch.eye(n, device=self.device)
        CHOL_CHUNK = 20000

        for day in range(T_days):
            d = torch.sqrt(torch.diagonal(Q_t, dim1=1, dim2=2))
            R_t = Q_t / (d.unsqueeze(2) * d.unsqueeze(1))

            if B > CHOL_CHUNK:
                L_t = torch.zeros_like(R_t)
                for i0 in range(0, B, CHOL_CHUNK):
                    i1 = min(i0 + CHOL_CHUNK, B)
                    try:
                        L_t[i0:i1] = torch.linalg.cholesky(R_t[i0:i1])
                    except Exception:
                        R_t[i0:i1] = R_t[i0:i1] + eye.unsqueeze(0) * 1e-6
                        L_t[i0:i1] = torch.linalg.cholesky(R_t[i0:i1])
            else:
                try:
                    L_t = torch.linalg.cholesky(R_t)
                except Exception:
                    R_t = R_t + eye.unsqueeze(0) * 1e-6
                    L_t = torch.linalg.cholesky(R_t)

            Z_corr, X_corr = self._sample_t_copula_gaussian_margins(B, n, L_t, generator=generator)

            sqrt_h = torch.sqrt(h_t)
            log_return += (r_daily - 0.5 * h_t) + sqrt_h * Z_corr

            h_t = (self.omega + self.beta * h_t +
                   self.alpha * (Z_corr - (self.gamma + self.lambda_ + 0.5) * sqrt_h) ** 2)
            h_t = torch.clamp(h_t, min=1e-12)

            Q_t = ((1 - self.dcc_alpha - self.dcc_beta) * Q_bar_b +
                   self.dcc_alpha * torch.bmm(X_corr.unsqueeze(2), X_corr.unsqueeze(1)) +
                   self.dcc_beta * Q_t)
            Q_t = (Q_t + Q_t.transpose(-1, -2)) / 2.0

            if day % 10 == 0:
                bad = ~torch.isfinite(Q_t).all(dim=(-1, -2))
                if bad.any():
                    Q_t[bad] = Q_bar_b.expand(bad.sum(), -1, -1)

                diag_min = torch.diagonal(Q_t, dim1=-2, dim2=-1).min(dim=-1).values
                needs_check = diag_min < 1e-6
                if needs_check.any():
                    sub = Q_t[needs_check]
                    min_eigvals_sub = torch.linalg.eigvalsh(sub).min(dim=-1).values
                    bad_eig = min_eigvals_sub < 1e-8
                    if bad_eig.any():
                        idx = needs_check.nonzero(as_tuple=True)[0][bad_eig]
                        nudge = torch.clamp(-min_eigvals_sub[bad_eig] + 1e-8, min=0.0)
                        Q_t[idx] += nudge[:, None, None] * eye.unsqueeze(0)

        S_T = S_t * torch.exp(log_return)
        return S_T.cpu().numpy()


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================
def generate_training_data_dcc(
    S0_base, raw_weights, garch_models, Q_bar, dcc_alpha, dcc_beta, nu_base, r, asset_names,
    n_samples=25000, n_paths=50000, T_min=0.0, T_max=1.0, weight_type="dollar",
    device="cuda:0", checkpoint_interval=1000, checkpoint_dir="checkpoints",
    near_expiry_T_threshold=0.05, near_expiry_path_multiplier=5, low_T_frac=0.30,
):
    """
    Builds (X, call_price) training data via GPU Monte-Carlo, with checkpointing
    so a run can resume after interruption.

    Feature layout: [moneyness (n_assets), vol (n_assets), T, log_moneyness,
                      corr_features (upper triangle of R_t)]
    """
    n_assets = len(S0_base)
    np.random.seed(2987465)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_files = sorted(f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_"))
    if checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
        checkpoint_data = pd.read_csv(checkpoint_path)
        start_idx = len(checkpoint_data)
        call_prices_existing = checkpoint_data["call_price"].values
        print(f"Resuming from checkpoint at sample {start_idx}")
    else:
        start_idx = 0
        call_prices_existing = None

    weights_normalized = raw_weights
    assert np.isclose(weights_normalized.sum(), 1.0)
    h_unconditional = np.array([m.h_unconditional for m in garch_models])

    # ------------------------------------------------------------------
    # Sample configuration (deterministic given the fixed seed above)
    # ------------------------------------------------------------------
    n_intrinsic = int(0.20 * n_samples)
    n_boundary = int(0.15 * n_samples)

    # T=0 intrinsic-value samples
    S0_intrinsic = np.random.uniform(10.0, 400.0, (n_intrinsic, n_assets))
    h_intrinsic = np.tile(h_unconditional, (n_intrinsic, 1))
    r_intrinsic = np.full(n_intrinsic, r)
    T_intrinsic = np.zeros(n_intrinsic)
    K_intrinsic = np.full(n_intrinsic, 100.0)
    R_intrinsic = np.tile(Q_bar, (n_intrinsic, 1, 1))

    # Boundary cases: deep ITM / OTM / near-expiry ATM / long-expiry ATM / exact ATM / high-vol
    n_bound_each = n_boundary // 6
    n_normal = n_samples - n_intrinsic - (n_bound_each * 6)

    def _uniform_block(n, lo=50.0, hi=200.0):
        return np.random.uniform(lo, hi, (n, n_assets))

    S0_b1, T_b1 = _uniform_block(n_bound_each), np.random.uniform(0.1, 1.0, n_bound_each)
    S0_b2, T_b2 = _uniform_block(n_bound_each), np.random.uniform(0.1, 1.0, n_bound_each)
    S0_b3, T_b3 = _uniform_block(n_bound_each), np.random.uniform(0.001, 0.05, n_bound_each)
    S0_b4, T_b4 = _uniform_block(n_bound_each), np.random.uniform(0.90, 1.0, n_bound_each)
    S0_b5, T_b5 = _uniform_block(n_bound_each), np.random.uniform(0.1, 1.0, n_bound_each)
    S0_b6, T_b6 = _uniform_block(n_bound_each), np.random.uniform(0.1, 1.0, n_bound_each)

    S0_boundary = np.vstack([S0_b1, S0_b2, S0_b3, S0_b4, S0_b5, S0_b6])
    K_boundary = np.full(len(S0_boundary), 100.0)
    T_boundary = np.concatenate([T_b1, T_b2, T_b3, T_b4, T_b5, T_b6])

    h_boundary = np.tile(h_unconditional, (len(S0_boundary), 1))
    h_boundary[-n_bound_each:] = h_unconditional * np.random.uniform(3.0, 5.0, (n_bound_each, n_assets))
    h_boundary = np.maximum(h_boundary, 1e-12)
    r_boundary = np.full(len(S0_boundary), r)
    R_boundary = np.tile(Q_bar, (len(S0_boundary), 1, 1))

    # "Normal" bucket
    S0_normal = np.clip(np.random.normal(100.0, 30.0, (n_normal, n_assets)), 10.0, 400.0)
    h_normal = np.zeros((n_normal, n_assets))
    for i in range(n_assets):
        h_normal[:, i] = np.maximum(np.random.lognormal(np.log(h_unconditional[i]), 0.5, n_normal), 1e-12)
    r_normal = np.full(n_normal, r)
    K_normal = np.full(n_normal, 100.0)

    # T-skewed mixture: mostly uniform, low_T_frac concentrated near expiry
    n_normal_low_T = int(low_T_frac * n_normal)
    n_normal_uniform = n_normal - n_normal_low_T
    T_normal = np.concatenate([
        np.random.uniform(T_min, T_max, n_normal_uniform),
        np.random.beta(0.5, 3.0, n_normal_low_T) * T_max,
    ])

    # Perturbed correlation matrices around Q_bar
    R_normal = []
    for _ in range(n_normal):
        noise_scale = np.random.uniform(0.0, 0.2)
        noise = np.random.randn(n_assets, n_assets) * noise_scale
        noise = (noise + noise.T) / 2
        Q_perturbed = Q_bar + noise
        eigvals, eigvecs = np.linalg.eigh(Q_perturbed)
        eigvals = np.maximum(eigvals, 1e-6)
        Q_perturbed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(Q_perturbed))
        R_normal.append(Q_perturbed / np.outer(d, d))
    R_normal = np.array(R_normal)

    # Combine + shuffle
    S0_samples = np.vstack([S0_intrinsic, S0_boundary, S0_normal])
    h_samples = np.vstack([h_intrinsic, h_boundary, h_normal])
    r_samples = np.concatenate([r_intrinsic, r_boundary, r_normal])
    T_samples = np.concatenate([T_intrinsic, T_boundary, T_normal])
    K_samples = np.concatenate([K_intrinsic, K_boundary, K_normal])
    R_t_samples = np.vstack([R_intrinsic, R_boundary, R_normal])

    shuffle_idx = np.random.permutation(n_samples)
    S0_samples, h_samples = S0_samples[shuffle_idx], h_samples[shuffle_idx]
    r_samples, T_samples = r_samples[shuffle_idx], T_samples[shuffle_idx]
    K_samples, R_t_samples = K_samples[shuffle_idx], R_t_samples[shuffle_idx]

    # ------------------------------------------------------------------
    # Price every sample with the GPU simulator
    # ------------------------------------------------------------------
    call_prices = np.zeros(n_samples)
    if start_idx > 0:
        call_prices[:start_idx] = call_prices_existing

    simulator = FastDCCGARCHSimulator(garch_models, Q_bar, dcc_alpha, dcc_beta, nu_base,
                                       device=str(device), asset_names=asset_names)

    pbar = tqdm(range(start_idx, n_samples), desc="GPU pricing")
    for idx in pbar:
        if T_samples[idx] < 1e-6:
            basket_now = S0_samples[idx] @ weights_normalized
            call_prices[idx] = max(basket_now - K_samples[idx], 0.0)
        else:
            T_days = max(1, int(T_samples[idx] * 252))
            r_daily = r
            basket_now = S0_samples[idx] @ weights_normalized
            this_n_paths = (n_paths * near_expiry_path_multiplier
                             if T_samples[idx] < near_expiry_T_threshold else n_paths)

            S_T = simulator.simulate_batch_gpu(
                S0=S0_samples[idx], h0=h_samples[idx], T_days=T_days, r_daily=r_daily,
                R_t_initial=R_t_samples[idx], batch_size=this_n_paths,
            )

            valid_mask = np.all(np.isfinite(S_T) & (S_T > 0), axis=1)
            S_T_valid = S_T[valid_mask]

            if len(S_T_valid) < 100:
                disc = np.exp(-r * T_days)
                call_prices[idx] = max(basket_now - K_samples[idx] * disc, 0.0)
            else:
                basket_T = S_T_valid @ weights_normalized
                call_payoff = np.maximum(basket_T - K_samples[idx], 0.0)
                disc = np.exp(-r * T_days)

                forward_basket = basket_now * np.exp(r * T_days)
                var_basket = np.var(basket_T)
                if var_basket > 1e-10:
                    beta_cv = np.cov(call_payoff, basket_T)[0, 1] / var_basket
                    adjusted_payoff = call_payoff - beta_cv * (basket_T - forward_basket)
                else:
                    adjusted_payoff = call_payoff

                lower_bound = max(basket_now - K_samples[idx] * disc, 0.0)
                call_prices[idx] = max(disc * adjusted_payoff.mean(), lower_bound)

        if ((idx + 1) % checkpoint_interval == 0) or (idx + 1 == n_samples):
            checkpoint_idx = idx + 1
            corr_feats = np.array([correlation_matrix_to_features(R) for R in R_t_samples[:checkpoint_idx]])
            basket_now_ck = S0_samples[:checkpoint_idx] @ weights_normalized
            log_moneyness_ck = np.log(basket_now_ck / K_samples[:checkpoint_idx]).reshape(-1, 1)

            X_checkpoint = np.column_stack([
                S0_samples[:checkpoint_idx], h_samples[:checkpoint_idx],
                T_samples[:checkpoint_idx], log_moneyness_ck, corr_feats,
            ])
            n_corr = n_assets * (n_assets - 1) // 2
            column_names = (
                [f"S{i}" for i in range(n_assets)] + [f"h{i}" for i in range(n_assets)] +
                ["T", "log_moneyness"] + [f"R_t_{i}" for i in range(n_corr)]
            )
            df_checkpoint = pd.DataFrame(X_checkpoint, columns=column_names)
            df_checkpoint["call_price"] = call_prices[:checkpoint_idx]
            df_checkpoint.to_csv(os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_idx:05d}.csv"), index=False)
            pbar.set_postfix({"checkpoint": checkpoint_idx})

    # ------------------------------------------------------------------
    # Final feature matrix (normalization matches the live RL/pricing state)
    # ------------------------------------------------------------------
    corr_feats = np.array([correlation_matrix_to_features(R) for R in R_t_samples])
    moneyness_feat = S0_samples / 100.0
    tau_feat = T_samples.reshape(-1, 1)
    vol_feat = np.sqrt(252 * h_samples) / np.sqrt(252 * h_unconditional)
    basket_now_all = S0_samples @ weights_normalized
    log_moneyness_feat = np.log(basket_now_all / K_samples).reshape(-1, 1)

    X = np.column_stack([moneyness_feat, vol_feat, tau_feat, log_moneyness_feat, corr_feats])

    print(f"Generated {n_samples} samples, {X.shape[1]} features. "
          f"call_price: min={call_prices.min():.4f} max={call_prices.max():.2f} mean={call_prices.mean():.4f}")

    return X, call_prices, None


# ============================================================================
# LOSS FUNCTION
# ============================================================================
def relative_price_loss(pred_log, true_log, true_orig, eps=1e-4, log_weight=0.3):
    """
    Symmetric (SMAPE-style) relative-price term on top of log1p-space MSE.
    Denominator is 0.5*(|pred|+true)+eps, so it stays well-behaved as
    true -> 0 without needing to clamp predictions or exclude rows.
    """
    log_mse = (pred_log - true_log) ** 2
    pred_price = torch.expm1(pred_log)
    denom = 0.5 * (pred_price.abs() + true_orig) + eps
    rel_sq = ((pred_price - true_orig) / denom) ** 2
    return log_weight * log_mse.mean() + (1 - log_weight) * rel_sq.mean()


# ============================================================================
# VALUATION SYSTEM
# ============================================================================
class BasketOptionValuationSystem:
    def __init__(self, n_assets, raw_weights=None, weight_type="dollar"):
        self.n_assets = n_assets
        self.raw_weights = np.array(raw_weights) if raw_weights is not None else None
        self.weight_type = weight_type
        self.weights = None
        self.asset_names = None
        self.S0_initial = None
        self.price_normalization_factors = None
        self.garch_models = [HestonNandiGARCH_Q() for _ in range(n_assets)]

        self.Q_bar = None
        self.dcc_alpha = None
        self.dcc_beta = None
        self.nu = None

        self.call_model = None
        self.scaler_call = None
        self.training_data = None

    # ------------------------------------------------------------------
    def calibrate_from_config(self, config_path, spot_prices):
        """
        Loads pre-calibrated Heston-Nandi GARCH params (one block per asset,
        under `garch_params:`) and the DCC(1,1) + Student-t copula params
        (`dcc:` block) straight from configDynamicRisk.yaml, instead of a
        hardcoded per-ticker dict.

        DCC/copula correlation dynamics are treated as measure-invariant:
        Q_bar / dcc_alpha / dcc_beta / nu are taken as-is from the config
        (exactly as calibration.fit_dcc_parameters produced them under P)
        and are NOT refit against risk-neutral residuals.

        Each asset's h_unconditional is recomputed locally under the
        risk-neutral gamma* = gamma + lambda + 1/2 shift, since that's the
        measure the GPU path simulator consumes -- the config's own
        h_unconditional (P-measure) is intentionally not used for that.
        """
        cfg = _load_config(config_path)
        garch_cfg = cfg["garch_params"]

        asset_order = sorted(garch_cfg, key=lambda k: int(k))
        asset_names = [garch_cfg[k]["asset"] for k in asset_order]
        assert len(asset_names) == self.n_assets, (
            f"config has {len(asset_names)} assets, system expects {self.n_assets}"
        )
        self.asset_names = asset_names

        spot_prices = np.asarray(spot_prices, dtype=float)
        self.price_normalization_factors = spot_prices / 100.0
        self.S0_initial = np.full(len(spot_prices), 100.0)

        for i, key in enumerate(asset_order):
            self._set_single_garch_from_params(self.garch_models[i], garch_cfg[key])

        if self.weight_type == "dollar":
            shares = 100.0 / spot_prices
            self.weights = shares / shares.sum()
        else:
            self.weights = self.raw_weights

        dcc_cfg = cfg["dcc"]
        self.Q_bar = np.array(dcc_cfg["Q_bar"], dtype=float)
        self.dcc_alpha = float(dcc_cfg["dcc_alpha"])
        self.dcc_beta = float(dcc_cfg["dcc_beta"])
        self.nu = float(dcc_cfg["nu_Q"])

        for i, name in enumerate(asset_names):
            assert self.garch_models[i].omega is not None, f"garch_models[{i}] ({name}) failed to calibrate"

        print(f"System calibrated from {config_path}: assets={asset_names}, weights={self.weights}")
        print(f"DCC (loaded from config, not refit): alpha={self.dcc_alpha:.6f} "
              f"beta={self.dcc_beta:.6f} nu={self.nu:.4f}")

    @staticmethod
    def _set_single_garch_from_params(model, p):
        """
        p holds the physical-measure (P) Heston-Nandi params for one asset,
        exactly as calibration.fit_hn_garch_pmeasure produced them: omega,
        alpha, beta, gamma, lambda (h_unconditional is also present in p,
        but it's the P-measure value and is intentionally not used here).
        """
        omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
        gamma, lambda_ = p["gamma"], p["lambda"]

        gamma_r = gamma + lambda_ + 0.5  # risk-neutral asymmetry
        persistence_r = beta + alpha * gamma_r ** 2
        h0 = (omega + alpha) / (1 - persistence_r) if persistence_r < 1 else np.nan

        model.omega, model.alpha, model.beta = omega, alpha, beta
        model.gamma, model.lambda_ = gamma, lambda_
        model.h_unconditional = h0

    # ------------------------------------------------------------------
    def train_surrogate(self, X_train, y_train, epochs=1000, batch_size=128, oversample_low_T=True):
        """
        Trains the call time-value surrogate using the composite
        (log-MSE + relative-error) loss, with optional T-bucket oversampling
        so near-expiry rows aren't drowned out by the much larger long-T bucket.
        """
        bad_mask = ~np.isfinite(X_train).all(axis=1)
        if bad_mask.sum() > 0:
            X_train, y_train = X_train[~bad_mask], y_train[~bad_mask]

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_train)
        y_log = np.log1p(y_train)

        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X_train))
        n_val = int(len(X_train) * 0.15)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = torch.FloatTensor(X_scaled[train_idx])
        y_train_t = torch.FloatTensor(y_log[train_idx]).reshape(-1, 1)
        y_train_orig_t = torch.FloatTensor(y_train[train_idx]).reshape(-1, 1)
        X_val_t = torch.FloatTensor(X_scaled[val_idx])
        y_val_t = torch.FloatTensor(y_log[val_idx]).reshape(-1, 1)
        y_val_orig = torch.FloatTensor(y_train[val_idx])
        X_val_raw = X_train[val_idx]

        # dropout=0.05 to match this project's original surrogate-training default
        # (basket.py's BasketOptionNet defaults to dropout=0.1, used elsewhere for inference).
        model = BasketOptionNet(input_dim=X_train.shape[1], dropout=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t, y_train_orig_t)

        T_col_idx = 2 * self.n_assets  # moneyness(n_assets) + vol(n_assets) + T
        if oversample_low_T:
            T_train_raw = X_train[train_idx, T_col_idx]
            bin_edges = np.array([0.0, 0.05, 0.25, 0.75, 1.01])
            bin_idx = np.clip(np.digitize(T_train_raw, bin_edges) - 1, 0, len(bin_edges) - 2)
            bin_counts = np.maximum(np.bincount(bin_idx, minlength=len(bin_edges) - 1), 1)
            sample_weights = torch.DoubleTensor(1.0 / bin_counts[bin_idx])
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_mae, best_mape, best_state = float("inf"), float("nan"), None
        patience_counter, patience_limit = 0, 250
        history = []

        pbar = tqdm(range(epochs), desc="training surrogate")
        for epoch in pbar:
            model.train()
            train_loss, n_batches = 0.0, 0
            for batch_X, batch_y, batch_y_orig in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = relative_price_loss(outputs.squeeze(), batch_y.squeeze(), batch_y_orig.squeeze())
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                    print(f"NaN/Inf weights at epoch {epoch + 1} — halting.")
                    n_batches = 0
                    break

                train_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                print(f"Epoch {epoch + 1}: all batches NaN — stopping.")
                break
            train_loss /= n_batches

            model.eval()
            with torch.no_grad():
                val_pred_log = model(X_val_t).squeeze()
                val_loss = relative_price_loss(val_pred_log, y_val_t.squeeze(), y_val_orig).item()
                val_pred_tv = torch.clamp(torch.expm1(val_pred_log), min=0.0)

                abs_err = torch.abs(val_pred_tv - y_val_orig)
                rmse_dollar = torch.sqrt(torch.mean((val_pred_tv - y_val_orig) ** 2)).item()
                mae_dollar = abs_err.mean().item()
                ss_res = torch.sum((y_val_orig - val_pred_tv) ** 2)
                ss_tot = torch.sum((y_val_orig - y_val_orig.mean()) ** 2)
                r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")

                nonzero_mask = y_val_orig > 0
                if nonzero_mask.sum() > 0:
                    pct_error = torch.abs((val_pred_tv[nonzero_mask] - y_val_orig[nonzero_mask]) / y_val_orig[nonzero_mask])
                    mape = (pct_error.mean() * 100).item()
                else:
                    mape = float("nan")

            scheduler.step(val_loss)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                             "mape_pct": mape, "rmse_dollar": rmse_dollar, "mae_dollar": mae_dollar, "r2": r2,
                             "lr": optimizer.param_groups[0]["lr"]})
            pbar.set_postfix({"MAE$": f"{mae_dollar:.4f}", "RMSE$": f"{rmse_dollar:.4f}",
                               "R2": f"{r2:.3f}", "MAPE": f"{mape:.2f}%"})

            # Model selection driven by MAE (robust to thin near-expiry outliers)
            if mae_dollar < best_mae:
                best_mae, best_mape = mae_dollar, mape
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stop at epoch {epoch + 1}")
                    break

        print(f"Best MAE: ${best_mae:.4f} (MAPE @ that epoch: {best_mape:.3f}%)")
        if best_state is not None:
            model.load_state_dict(best_state)

        pd.DataFrame(history).to_csv("training_history_call.csv", index=False)

        # Stratified validation report
        model.eval()
        with torch.no_grad():
            val_pred_tv = torch.clamp(torch.expm1(model(X_val_t).squeeze()), min=0.0).numpy()
        y_val_np = y_val_orig.numpy()
        T_raw = X_val_raw[:, T_col_idx]

        def bucket_mape(mask, label):
            m = mask & (y_val_np > 0.10)
            if m.sum() == 0:
                return
            err = np.abs((val_pred_tv[m] - y_val_np[m]) / y_val_np[m]) * 100
            print(f"  {label:<20s}: n={m.sum():5d}  MAPE={err.mean():6.2f}%  median={np.median(err):6.2f}%")

        print("Stratified validation (call):")
        bucket_mape(T_raw < 0.05, "near-expiry (<0.05)")
        bucket_mape((T_raw >= 0.05) & (T_raw < 0.25), "short T")
        bucket_mape((T_raw >= 0.25) & (T_raw < 0.75), "mid T")
        bucket_mape(T_raw >= 0.75, "long T")

        return model, {"X": scaler_X, "y": None}

    # ------------------------------------------------------------------
    def price(self, S0, h0, T, K, r, R_t, asset_names=None):
        if asset_names is not None and self.asset_names is not None:
            assert list(asset_names) == list(self.asset_names), \
                f"price() called with asset order {asset_names}, system trained on {self.asset_names}"

        basket = S0 @ self.weights
        T_days = max(1, int(T * 252))
        disc = np.exp(-r * T_days)
        intrinsic = max(basket - K * disc, 0.0)

        if T < 1e-6:
            return intrinsic

        moneyness_feat = S0 / 100.0
        vol_feat = np.sqrt(h0 / np.array([m.h_unconditional for m in self.garch_models]))
        log_moneyness_feat = np.array([np.log(basket / K)])
        corr_features = correlation_matrix_to_features(R_t)
        features = np.concatenate([moneyness_feat, vol_feat, [T], log_moneyness_feat, corr_features]).reshape(1, -1)

        features_scaled = self.scaler_call["X"].transform(features)
        features_tensor = torch.FloatTensor(features_scaled)

        self.call_model.eval()
        with torch.no_grad():
            pred_log = self.call_model(features_tensor).squeeze().item()

        time_value = max(np.expm1(pred_log), 0.0)
        return intrinsic + time_value

    # ------------------------------------------------------------------
    def save(self, filepath="basket_system_dcc.pkl"):
        state = {
            "n_assets": self.n_assets,
            "weights": self.weights,
            "asset_names": self.asset_names,
            "weight_type": self.weight_type,
            "S0_initial": self.S0_initial,
            "price_normalization_factors": self.price_normalization_factors,
            "garch_params": [m.get_params() for m in self.garch_models],
            "Q_bar": self.Q_bar,
            "dcc_alpha": self.dcc_alpha,
            "dcc_beta": self.dcc_beta,
            "nu": self.nu,
            "call_model_state": self.call_model.state_dict(),
            "scaler_call": self.scaler_call,
            "training_data": self.training_data,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Saved system to {filepath} ({os.path.getsize(filepath) / (1024 * 1024):.2f} MB)")

    @staticmethod
    def load(filepath="basket_system_dcc.pkl"):
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        system = BasketOptionValuationSystem(n_assets=state["n_assets"], weight_type=state["weight_type"])
        system.weights = state["weights"]
        system.asset_names = state.get("asset_names")
        system.S0_initial = state["S0_initial"]
        system.price_normalization_factors = state.get("price_normalization_factors")
        system.Q_bar = state["Q_bar"]
        system.dcc_alpha = state["dcc_alpha"]
        system.dcc_beta = state["dcc_beta"]
        system.nu = state["nu"]
        system.scaler_call = state["scaler_call"]
        system.training_data = state.get("training_data")

        for i, params in enumerate(state["garch_params"]):
            m = system.garch_models[i]
            m.omega, m.alpha, m.beta = params["omega"], params["alpha"], params["beta"]
            m.gamma, m.lambda_ = params["gamma"], params["lambda"]
            m.h_unconditional = params["h_unconditional"]

        input_dim = state["training_data"]["input_dim"]
        system.call_model = BasketOptionNet(input_dim)
        system.call_model.load_state_dict(state["call_model_state"])
        system.call_model.eval()

        print(f"Loaded system from {filepath} (alpha={system.dcc_alpha:.6f}, beta={system.dcc_beta:.6f}, nu={system.nu:.4f})")
        return system
