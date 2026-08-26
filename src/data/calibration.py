"""
src/data/calibration.py

Downloads historical price + risk-free data and fits the two-stage model:
  1. Heston-Nandi GARCH(1,1) per asset, P-measure MLE (omega, alpha, beta,
     gamma, lambda) via differential evolution -> standardised residuals
     z_t = (r_t - rf_t - lambda*h_t)/sqrt(h_t)
  2. Joint DCC(1,1) + dynamic Student-t copula MLE -> Q_bar, dcc_alpha,
     dcc_beta, nu_Q

"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t as t_dist
from scipy.special import gammaln


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_returns(
    tickers: dict[str, str],
    years_back: int = 10,
    end_date: datetime | None = None,
    rf_ticker: str = "^IRX",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Expected files:
        historical_prices.csv
        historical_rf.csv

    The price CSV contains adjusted prices for the assets,
    while the risk-free CSV contains the daily continuously-compounded
    risk-free rate derived from the annual quoted ^IRX yield.

    Returns:
        returns_matrix: (T, N) log returns, one column per asset.
        rf:             (T,) daily continuously-compounded risk-free rate,ht[0
                        aligned to returns_matrix.
        prices:         Adjusted closing prices.
    """
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent

    prices_path = data_dir / "historical_prices.csv"
    rf_path = data_dir / "historical_rf.csv"

    if not prices_path.exists():
        raise FileNotFoundError(
            f"Price data not found at {prices_path}"
        )

    if not rf_path.exists():
        raise FileNotFoundError(
            f"Risk-free data not found at {rf_path}"
        )

    prices = pd.read_csv(
        prices_path,
        index_col="Date",
        parse_dates=True,
    )

    asset_names = list(tickers.keys())

    missing_assets = [
        name for name in asset_names
        if name not in prices.columns
    ]

    if missing_assets:
        raise ValueError(
            f"Missing assets in historical_prices.csv: {missing_assets}"
        )

    prices = prices[asset_names]

    prices = (
        prices
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    log_returns = np.log(prices / prices.shift(1))

    log_returns = (
        log_returns
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    rf = pd.read_csv(
        rf_path,
        index_col="Date",
        parse_dates=True,
    ).squeeze("columns")

    rf = rf.reindex(log_returns.index).ffill().bfill()

    if rf.isna().any():
        raise ValueError(
            "Risk-free rate contains missing observations after alignment."
        )

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    returns_matrix = log_returns.to_numpy(dtype=float)
    rf_array = rf.to_numpy(dtype=float)

    assert len(rf_array) == returns_matrix.shape[0], (
        "rf/returns length mismatch"
    )

    assert returns_matrix.shape[1] == len(asset_names), (
        "Unexpected number of assets in returns matrix"
    )

    print(
        f"Loaded returns: shape={returns_matrix.shape}, "
        f"from {log_returns.index[0].date()} "
        f"to {log_returns.index[-1].date()}"
    )

    print(
        f"Loaded risk-free data: {len(rf_array)} observations"
    )

    return returns_matrix, rf_array, prices


# ============================================================================
# PHASE 1 — HESTON-NANDI GARCH(1,1), P-MEASURE MLE (per asset)
# ============================================================================

def fit_hn_garch_pmeasure(
    returns: np.ndarray,
    rf: np.ndarray,
    maxiter: int = 1000,
    popsize: int = 20,
    seed: int = 42,
    print_every: int = 50,
) -> dict:
    """
    MLE fit of Heston-Nandi (2000) GARCH(1,1) under the P-measure:

        r_t = rf_t + lambda*h_t + sqrt(h_t)*z_t
        h_t = omega + beta*h_{t-1} + alpha*(z_{t-1} - gamma*sqrt(h_{t-1}))^2

    h(0) is fixed at the sample variance (initialisation only, per footnote
    12 of the paper); omega, alpha, beta, gamma, lambda are all free MLE
    parameters, estimated with scipy.optimize.differential_evolution on the
    natural parameter scale.

    Returns a dict of fitted parameters plus persistence (h_unconditional
    is filled in by the caller, fit_all_garch, once the final params are
    known).
    """
    returns = np.asarray(returns, dtype=float)
    rf = np.asarray(rf, dtype=float)

    if len(returns) != len(rf):
        raise ValueError(
            f"returns and rf must have the same length. "
            f"Got {len(returns)} and {len(rf)}."
        )

    valid = np.isfinite(returns) & np.isfinite(rf)
    returns = returns[valid]
    rf = rf[valid]
    T = len(returns)

    if T == 0:
        raise ValueError("Returns array must contain valid numeric data.")

    h0 = np.var(returns, ddof=1)
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError(f"Invalid initial variance h0={h0}")

    print(f"Number of observations: {T}")
    print(f"Initial variance h0: {h0:.6e}")

    bounds = [
        (1e-9, 1e-3),      # omega
        (1e-9, 1e-3),      # alpha
        (1e-3, 0.995),     # beta
        (-1500.0, 1500.0), # gamma
        (-10.0, 10.0),     # lambda
    ]

    def negloglik(x):
        omega, alpha, beta, gamma, lam = x

        # Heston-Nandi persistence condition: beta + alpha*gamma^2 < 1
        persistence = beta + alpha * gamma ** 2
        if persistence >= 0.999 or persistence < 0 or omega <= 0 or alpha <= 0:
            return 1e12

        h = h0
        ll = 0.0
        for t in range(T):
            if not np.isfinite(h) or h <= 0:
                return 1e12

            sqrt_h = np.sqrt(h)
            z = (returns[t] - rf[t] - lam * h) / sqrt_h
            if not np.isfinite(z):
                return 1e12
	# conditional Gaussian log-likelihood
            ll += -0.5 * np.log(2.0 * np.pi) - 0.5 * np.log(h) - 0.5 * z ** 2
            h = omega + beta * h + alpha * (z - gamma * sqrt_h) ** 2
            if not np.isfinite(h):
                return 1e12

        neg_ll = -ll
        return neg_ll if np.isfinite(neg_ll) else 1e12

    generation = [0]

    def callback(xk, convergence):
        generation[0] += 1
        if generation[0] % print_every == 0 or generation[0] == 1:
            omega, alpha, beta, gamma, lam = xk
            persistence = beta + alpha * gamma ** 2
            loss = negloglik(xk)
            print(
                f"[DE {generation[0]:04d}] NLL={loss:.4f} | "
                f"omega={omega:.6e} | alpha={alpha:.6e} | beta={beta:.6f} | "
                f"gamma={gamma:.4f} | lambda={lam:.6f} | "
                f"persistence={persistence:.6f}"
            )
        return False

    result = differential_evolution(
        negloglik,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-8,
        atol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        seed=seed,
        callback=callback,
        disp=False,
        updating="immediate",
        workers=1,
    )

    omega, alpha, beta, gamma, lam = result.x
    persistence = beta + alpha * gamma ** 2

    print(f"✓ DE fit: success={result.success}, nit={result.nit}, "
          f"nfev={result.nfev}, NLL={result.fun:.6f}, "
          f"persistence={persistence:.8f}")

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lambda": lam,
        "persistence": persistence,
        "log_likelihood": -result.fun,
        "negative_log_likelihood": result.fun,
        "success": bool(result.success),
        "message": result.message,
        "n_iterations": result.nit,
        "n_function_evaluations": result.nfev,
    }


def fit_all_garch(
    returns_matrix: np.ndarray,
    rf: np.ndarray,
    asset_names: list[str],
    maxiter: int = 1000,
    popsize: int = 20,
    seed: int = 42,
) -> dict[int, dict]:
    """Fit fit_hn_garch_pmeasure independently for every asset column."""
    params = {}
    for i, name in enumerate(asset_names):
        p = fit_hn_garch_pmeasure(
            returns_matrix[:, i], rf=rf, maxiter=maxiter, popsize=popsize, seed=seed
        )
        p["asset"] = name

        #Compute h_unconditional
        denom = 1 - p["beta"] - p["alpha"] * (p["gamma"] ** 2)
        p["h_unconditional"] = (p["omega"] + p["alpha"]) / denom if denom > 0 else np.nan

        params[i] = p
        print(f"✓ GARCH fitted for {name}: h_unc={p['h_unconditional']:.10f}, "
              f"persistence={p['persistence']:.6f}")
    return params


# ============================================================================
# Standardize the residuals
# ============================================================================

def compute_garch_residuals(returns_matrix: np.ndarray, rf: np.ndarray, params: dict[int, dict]) -> np.ndarray:
    """
    z_t = (r_t - rf_t - lambda*h_t) / sqrt(h_t), with h_t from the fitted
    P-measure recursion.
    """
    T, N = returns_matrix.shape
    garch_residuals = np.zeros((T, N))

    for i in range(N):
        r = returns_matrix[:, i]
        p = params[i]
        omega, alpha, beta, gamma, lam = p["omega"], p["alpha"], p["beta"], p["gamma"], p["lambda"]

        h_t = np.zeros(T)
        h_t[0] = np.var(r, ddof=1)
        for step in range(1, T):
            sqrt_h_prev = np.sqrt(h_t[step - 1])
            z_prev = (r[step - 1] - rf[step - 1] - lam * h_t[step - 1]) / sqrt_h_prev if sqrt_h_prev > 0 else 0.0
            h_t[step] = omega + beta * h_t[step - 1] + alpha * (z_prev - gamma * sqrt_h_prev) ** 2
            h_t[step] = max(h_t[step], 1e-12)

        garch_residuals[:, i] = (r - rf - lam * h_t) / np.sqrt(h_t)

    print(f"✓ GARCH residuals computed: shape={garch_residuals.shape}")
    return garch_residuals


# ============================================================================
# STEP 2 — JOINT DCC(1,1) + DYNAMIC STUDENT-t COPULA MLE
# ============================================================================

def fit_dcc_parameters(
    garch_residuals: np.ndarray, x0: list | None = None, nu_bounds: tuple = (2.1, 200.0)
) -> tuple[np.ndarray, float, float, float]:
    """
    Jointly estimate DCC (alpha, beta) and copula nu by maximising the
    dynamic Student-t copula log-likelihood (Eq. B.9/B.10).

    A Student-t(nu) draw
    has variance nu/(nu-2) != 1, so we standardize to unit variance through
      Uvec_dcc = Uvec / sqrt(nu/(nu-2))     # unit-variance, used in
                                             # the Q_t recursion
    Returns:
        Q_bar:     (N, N) unconditional correlation target, computed at
                   nu_hat from the raw (non-standardised) copula draws.
        dcc_alpha, dcc_beta, nu_Q: fitted scalars.
    """
    T, N = garch_residuals.shape
    U_gauss = np.clip(norm.cdf(garch_residuals), 1e-10, 1 - 1e-10)

    def to_copula_space(nu):
        return t_dist.ppf(U_gauss, df=nu)

    def neg_loglik(theta):
        alpha, beta, log_nu = theta
        nu = np.exp(log_nu)
        if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10

        Uvec = to_copula_space(nu)                
        scale = np.sqrt(nu / (nu - 2))             
        Uvec_dcc = Uvec / scale                    # unit-variance version

        Q_bar = np.corrcoef(Uvec, rowvar=False)    #corrcoef is scale-free
        C_const = gammaln((nu + N) / 2.0) + (N - 1) * gammaln(nu / 2.0) - N * gammaln((nu + 1) / 2.0)
        Q_t = Q_bar.copy()
        ll = 0.0
        for tau in range(1, T):
            u_prev = Uvec_dcc[tau - 1]              # << standardized Uvecs
            Q_t = (1 - alpha - beta) * Q_bar + alpha * np.outer(u_prev, u_prev) + beta * Q_t
            if np.any(np.linalg.eigvalsh(Q_t) <= 1e-8):
                return 1e10
            d = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(d, d)
            try:
                invR = np.linalg.inv(R_t)
                _, logdet = np.linalg.slogdet(R_t)
            except np.linalg.LinAlgError:
                return 1e10
            u_t = Uvec[tau]                         # << raw, for the density
            quad = u_t @ invR @ u_t
            ll += (C_const - 0.5 * logdet - 0.5 * (nu + N) * np.log(1.0 + quad / nu)
                   + 0.5 * (nu + 1) * np.sum(np.log(1.0 + u_t ** 2 / nu)))
        return -ll
    if x0 is None:
        x0 = [0.02, 0.95, np.log(8.0)]
    bounds = [(1e-6, 0.2), (0.7, 0.999), (np.log(nu_bounds[0]), np.log(nu_bounds[1]))]

    res = minimize(neg_loglik, x0=x0, bounds=bounds, method="L-BFGS-B")
    dcc_alpha, dcc_beta, log_nu_hat = res.x
    nu_Q = float(np.exp(log_nu_hat))
    Q_bar = np.corrcoef(to_copula_space(nu_Q), rowvar=False)

    print(f"✓ DCC + copula jointly fitted: alpha={dcc_alpha:.6f}, beta={dcc_beta:.6f}, "
          f"nu={nu_Q:.4f}, alpha+beta={dcc_alpha + dcc_beta:.6f}")
    return Q_bar, dcc_alpha, dcc_beta, nu_Q


# ============================================================================
# Pipelin
# ============================================================================

def calibrate(
    tickers: dict[str, str], years_back: int = 10, end_date: datetime | None = None,
) -> dict:
    """
    Run the full calibration and return all fitted quantities:
        returns_matrix, rf, prices, garch_params, garch_residuals,
        Q_bar, dcc_alpha, dcc_beta, nu_Q
    """
    returns_matrix, rf, prices = download_returns(tickers, years_back, end_date)
    garch_params = fit_all_garch(returns_matrix, rf, list(tickers.keys()))
    garch_residuals = compute_garch_residuals(returns_matrix, rf, garch_params)
    Q_bar, dcc_alpha, dcc_beta, nu_Q = fit_dcc_parameters(garch_residuals)

    return {
        "returns_matrix": returns_matrix,
        "rf": rf,
        "prices": prices,
        "garch_params": garch_params,
        "garch_residuals": garch_residuals,
        "Q_bar": Q_bar,
        "dcc_alpha": dcc_alpha,
        "dcc_beta": dcc_beta,
        "nu_Q": nu_Q,
    }
