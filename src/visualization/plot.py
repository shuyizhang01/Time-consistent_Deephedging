"""
src/visualization/plot.py

All plotting utilities for:
  "Actor-critic Deep-hedging under Time-consistent Dynamic Risk"
Data layout expected at data_dir (default "content/data"):
    <data_dir>/<alpha_label>/
        S_paths.npy, h_paths.npy, R_paths.npy, deriv_prices.npy   (shared across scoring keys)
        <scoring_key>/
            states.npy, actions.npy, portfolio_values.npy, terminal_pnl.npy
        static/
            (same per-policy files)
    <data_dir>/<alpha_label>/_shared_paths/
        S_paths.npy, h_paths.npy, R_paths.npy, deriv_prices.npy   (fixed S0/h0 paths)
        <scoring_key>/, static/
            (same per-policy files as above, evaluated on the fixed paths)
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
import torch
import math
import gc
from src.risk_measures.loss_functions import SCORE_FUNCTIONS



# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SCORING_DISPLAY = {
    "arcsin":   r"$G(x) = \arcsin\!\left(\frac{1}{x+1}\right)$",
    "arcsinh":  r"$G(x) = -\mathrm{arcsinh}(x)$",
    "arctan":   r"$G(x) = -\arctan(x)$",
    "log":      r"$G(x) = -\log(x)$",
    "power03":  r"$G(x) = -x^{0.3}$",
    "rational": r"$G(x) = -\dfrac{x}{1+x}$",
}

SCORING_KEYS = ["arcsin", "arcsinh", "arctan", "log", "power03", "rational"]

_SCORING_COLORS = [
    "#4C8FE8", "#D4537E", "#1D9E75",
    "#7F77DD", "#BA7517", "#F5A623",
]
SCORING_COLOR = dict(zip(SCORING_KEYS, _SCORING_COLORS))
STATIC_COLOR  = "#E05CB8"
ASSET_NAMES   = ["JPM", "BAC", "WFC", "C"]

ALPHA_LABELS  = ["alpha925", "alpha95", "alpha975", "alpha99"]
ALPHA_DISPLAY = {
    "alpha925": r"$\alpha=0.925$", "alpha95":  r"$\alpha=0.95$",
    "alpha975": r"$\alpha=0.975$", "alpha99":  r"$\alpha=0.99$",
}
_ALPHA_COLORS = ["#4C8FE8", "#1D9E75", "#BA7517", "#D4537E"]
ALPHA_COLOR   = dict(zip(ALPHA_LABELS, _ALPHA_COLORS))


# ---------------------------------------------------------------------------
# Shared font sizes 
# ---------------------------------------------------------------------------
FONT_TITLE        = 13   # subplot / figure titles (single or few-panel plots)
FONT_LABEL         = 12   # axis labels (xlabel/ylabel)
FONT_TICK          = 10   # tick labels
FONT_LEGEND        = 10   # legend entries
FONT_ANNOT         = 9    # small in-panel annotations
FONT_SUPTITLE      = FONT_TITLE + 2

FONT_TITLE_SMALL   = 11   # panel titles in dense multi-panel grids
FONT_LABEL_SMALL   = 9    # axis labels in dense multi-panel grids
FONT_LEGEND_SMALL  = 8    # legend entries in dense multi-panel grids


def _scoring_keys_for_alpha(alpha_label, base_keys=None):
    """Scoring keys available for a given alpha."""
    return list(base_keys) if base_keys is not None else list(SCORING_KEYS)


def _load(data_dir, alpha_label, key, filename):
    """Load a single .npy file for a given actor key."""
    return np.load(os.path.join(data_dir, alpha_label, key, filename))


def _load_shared(data_dir, alpha_label, filename):
    """
    Load a market-path file (S_paths/h_paths/R_paths/deriv_prices) that is
    shared across all scoring keys for a given alpha -- these paths don't
    depend on the hedging policy, so generate_data.py saves them once at
    <data_dir>/<alpha_label>/<filename> instead of duplicating them under
    every <data_dir>/<alpha_label>/<scoring_key>/.
    """
    return np.load(os.path.join(data_dir, alpha_label, filename))


def _style_ax(ax, ticksize=FONT_TICK):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.set_facecolor("#f7f7f7")
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(colors="#444444", labelsize=ticksize)


def _save(fig, save_dir, stem, dpi_pdf=500, dpi_png=150):
    os.makedirs(save_dir, exist_ok=True)
    for ext, dpi in [("pdf", dpi_pdf), ("png", dpi_png)]:
        path = os.path.join(save_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# Checkpoint / training-log path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_srm_actor_path(project_root, alpha):
    """<project_root>/staticriskmodels/static_actor_alpha<alpha>.pth"""
    path = os.path.join(project_root, "staticriskmodels", f"static_actor_alpha{alpha}.pth")
    return path if os.path.isfile(path) else None


def _find_training_logs(logs_root, alpha_label):
    """
    Yields (label, path) for every training.log under
    <logs_root>/<alpha_label>/<scoring_key>/training.log.

    Falls back to <scoring_key>/<variant>/training.log (e.g. the log/modified,
    log/unmodified split that still exists for some alpha/scoring combos),
    yielding one entry per variant labeled '<scoring_key> (<variant>)'.
    """
    alpha_dir = Path(logs_root) / alpha_label
    if not alpha_dir.exists():
        raise FileNotFoundError(f"Alpha dir not found: {alpha_dir}")

    for scoring_dir in sorted(alpha_dir.iterdir()):
        if not scoring_dir.is_dir():
            continue
        flat = scoring_dir / "training.log"
        if flat.exists():
            yield scoring_dir.name, flat
            continue
        found_variant = False
        for variant_dir in sorted(scoring_dir.iterdir()):
            vlog = variant_dir / "training.log"
            if variant_dir.is_dir() and vlog.exists():
                yield f"{scoring_dir.name} ({variant_dir.name})", vlog
                found_variant = True
        if not found_variant:
            print(f"  ! no training.log found under {scoring_dir} -- skipping")


# ---------------------------------------------------------------------------
# Basket-gamma helpers (used by plot_extreme_trajectory's path selection)
# ---------------------------------------------------------------------------

def _basket_gamma_chunk(env, S_chunk, h_chunk, R_chunk, t_scalar):
    """
    d^2(price)/d(scale)^2 where S -> S + eps*basket_weights, via autograd
    through env._price_derivative_batch. This is basket Gamma: the real
    convexity driving gamma P&L ~= 0.5 * Gamma * (delta_basket)^2.
    """
    S_chunk = S_chunk.clone().detach().requires_grad_(True)
    t_batch = torch.full((S_chunk.shape[0],), float(t_scalar), device=env.device)
    price = env._price_derivative_batch(S_chunk, h_chunk, R_chunk, t_batch)
    grad1 = torch.autograd.grad(price.sum(), S_chunk, create_graph=True)[0]
    basket_delta = (grad1 * env.basket_weights).sum(dim=1)
    grad2 = torch.autograd.grad(basket_delta.sum(), S_chunk)[0]
    basket_gamma = (grad2 * env.basket_weights).sum(dim=1)
    return basket_gamma.detach()


def _compute_basket_gamma_window(env, S_paths_t, h_paths_t, R_paths_t, t_indices, chunk_size=256):
    """
    S_paths_t/h_paths_t/R_paths_t: torch tensors on env.device, [n, T+1, ...].
    t_indices: timesteps to score. Returns gamma as np.ndarray [n, len(t_indices)].
    """
    n = S_paths_t.shape[0]
    gamma_out = np.zeros((n, len(t_indices)), dtype=np.float32)
    for ti, t in enumerate(t_indices):
        S_t, h_t, R_t = S_paths_t[:, t], h_paths_t[:, t], R_paths_t[:, t]
        for b0 in range(0, n, chunk_size):
            b1 = min(b0 + chunk_size, n)
            gamma_out[b0:b1, ti] = _basket_gamma_chunk(
                env, S_t[b0:b1], h_t[b0:b1], R_t[b0:b1], t
            ).cpu().numpy()
    return gamma_out


# ---------------------------------------------------------------------------
# 1. G(x) functions  — no data needed
# ---------------------------------------------------------------------------

def plot_g2_functions(save_dir: str = ".") -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    x = np.linspace(0.01, 5, 500)
    funcs = {
        r"Logarithmic:  $-\log(x)$":                               -np.log(x),
        r"Fractional Power:  $-x^{0.3}$":                          -x ** 0.3,
        r"Arctangent:  $-\arctan(x)$":                             -np.arctan(x),
        r"Arcsinh:  $-\mathrm{arcsinh}(x)$":                       -np.arcsinh(x),
        r"Rational:  $-\frac{x}{1+x}$":                            -x / (1 + x),
        r"Arcsin:  $\arcsin\!\left(\frac{1}{x+1}\right)$":         np.arcsin(np.clip(1 / (x + 1), -1, 1)),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, y), color in zip(funcs.items(), _SCORING_COLORS):
        ax.plot(x, y, lw=1.8, color=color, label=name, alpha=0.9)

    ax.set_xlabel(r"$x$",    fontsize=FONT_LABEL, color="#444444")
    ax.set_ylabel(r"$G(x)$", fontsize=FONT_LABEL, color="#444444")
    ax.set_xlim(0, 5)
    ax.set_ylim(-4, 5)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper right", labelcolor="#444444")

    plt.tight_layout()
    _save(fig, save_dir, "g2_functions")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Portfolio delta scatter
# ---------------------------------------------------------------------------

def plot_portfolio_delta(
    alpha_label: str = "alpha95",
    data_dir:    str = "content/data",
    env                = None,
    save_dir:    str = ".",
    n_paths:     int = 1000,
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    T       = env.T_days
    weights = env.basket_weights.cpu().numpy()

    norm      = Normalize(vmin=0, vmax=T - 1)
    t_indices = np.repeat(np.arange(T), n_paths)

    # This reads the FIXED (S0/h0-fixed) shared paths, saved once under
    # _shared_paths/ regardless of scoring key -- unaffected by the
    # stochastic-path storage change.
    shared_S = np.load(os.path.join(data_dir, alpha_label, "_shared_paths", "S_paths.npy"))
    basket_moneyness = (shared_S[:n_paths, :T, :] * weights).sum(axis=-1) / env.K
    del shared_S

    keys   = _scoring_keys_for_alpha(alpha_label)
    ncols  = 3
    nrows  = int(np.ceil(len(keys) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, sk in enumerate(keys):
        ax = axes_flat[idx]

        actions_np = np.load(
            os.path.join(data_dir, alpha_label, "_shared_paths", sk, "actions.npy")
        )[:, :n_paths, :]
        port_delta = (actions_np * weights).sum(axis=-1)

        x = basket_moneyness.T.ravel()
        y = port_delta.ravel()

        ax.scatter(x, y, s=0.5, alpha=0.3,
                   c=t_indices, cmap="viridis", norm=norm,
                   rasterized=True, linewidths=0)

        ax.set_title(sk, fontsize=FONT_TITLE_SMALL, color="#444444")
        _style_ax(ax, ticksize=FONT_TICK)
        ax.set_axisbelow(True)
        if idx % ncols == 0:
            ax.set_ylabel("Weighted Aggregate Position", fontsize=FONT_LABEL_SMALL, color="#444444")
        if idx >= len(keys) - ncols:
            ax.set_xlabel("Moneyness", fontsize=FONT_LABEL_SMALL, color="#444444")

    for ax in axes_flat[len(keys):]:
        ax.axis("off")

    _save(fig, save_dir, "portfolio_delta_plot")
    plt.show()

# ---------------------------------------------------------------------------
# 3. PnL distributions
# ---------------------------------------------------------------------------

def plot_pnl_distributions(
    alpha_label:    str   = "alpha95",
    data_dir:       str   = "content/data",
    save_dir:       str   = ".",
    tail_prob:      float = 0.05,
    scoring_labels: list  = None,
) -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "axes.titlesize":   FONT_TITLE,
        "axes.labelsize":   FONT_LABEL,
        "xtick.labelsize":  FONT_TICK,
        "ytick.labelsize":  FONT_TICK,
        "legend.fontsize":  FONT_LEGEND,
        "axes.facecolor":   "#f7f7f7",
    })

    if scoring_labels is None:
        scoring_labels = _scoring_keys_for_alpha(
            alpha_label, base_keys=["rational", "arcsinh", "arctan", "arcsin", "power03", "log"]
        )

    drm_colors = [SCORING_COLOR.get(sl, "#888888") for sl in scoring_labels]

    # deriv_prices[:, 0] is identical across every scoring key / static for a
    # given alpha (it's priced from the underlying paths, not the hedging
    # policy), so load it once instead of per key.
    d0_shared = _load_shared(data_dir, alpha_label, "deriv_prices.npy")[:, 0]

    def _load_pnl(key):
        pnl = _load(data_dir, alpha_label, key, "terminal_pnl.npy")
        return -pnl / d0_shared

    def _cvar(losses):
        sorted_losses = np.sort(losses)[::-1]
        n = max(1, int(len(sorted_losses) * tail_prob))
        return sorted_losses[:n].mean()

    drm_data    = {sl: _load_pnl(sl) for sl in scoring_labels}
    static_data = _load_pnl("static")

    all_arrays = list(drm_data.values()) + [static_data]
    x_min  = min(np.percentile(a, 0.1)  for a in all_arrays)
    x_max  = max(np.percentile(a, 99.9) for a in all_arrays)
    x_grid = np.linspace(x_min, x_max, 1000)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    plt.subplots_adjust(left=0.25)

    for sl, color in zip(scoring_labels, drm_colors):
        arr      = drm_data[sl]
        cvar_val = -1 * _cvar(arr)
        mean_val = np.mean(arr)

        ax.hist(-1 * arr, bins=100, range=(x_min, x_max),
                density=True, color=color, alpha=0.18, linewidth=0)
        ax.plot(x_grid, gaussian_kde(-1 * arr)(x_grid),
                color=color, linewidth=1.8, label=sl)
        ax.axvline(cvar_val, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
        ax.axvline(mean_val, color=color, linestyle=":",  linewidth=1.2, alpha=0.90)

    cvar_val = -1 * _cvar(static_data)
    mean_val = np.mean(static_data)
    ax.hist(-1 *  static_data, bins=100, range=(x_min, x_max),
            density=True, color=STATIC_COLOR, alpha=0.15, linewidth=0)
    ax.plot(x_grid, gaussian_kde(-1 * static_data)(x_grid),
            color=STATIC_COLOR, linewidth=2.2, label="static")
    ax.axvline(cvar_val, color=STATIC_COLOR, linestyle="--", linewidth=1.2)
    ax.axvline(mean_val, color=STATIC_COLOR, linestyle=":",  linewidth=1.2)
    _style_ax(ax)
    ax.set_xlabel(r"Normalized Terminal Hedging Error", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    _save(fig, save_dir, "pnl_distributions")
    plt.show()


# ---------------------------------------------------------------------------
# 3b. Risk distributions per scoring function, across alpha levels
# ---------------------------------------------------------------------------

def plot_risk_distributions_by_scoring(
    scoring_key:  str,
    data_dir:     str   = "content/data",
    save_dir:     str   = ".",
    tail_prob:    float = 0.05,
    alpha_labels: list  = None,
) -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "axes.titlesize":   FONT_TITLE,
        "axes.labelsize":   FONT_LABEL,
        "xtick.labelsize":  FONT_TICK,
        "ytick.labelsize":  FONT_TICK,
        "legend.fontsize":  FONT_LEGEND,
        "axes.facecolor":   "#f7f7f7",
    })

    if alpha_labels is None:
        alpha_labels = ALPHA_LABELS

    def _load_pnl(alpha_label, key):
        pnl = _load(data_dir, alpha_label, key, "terminal_pnl.npy")
        d0  = _load_shared(data_dir, alpha_label, "deriv_prices.npy")[:, 0]
        return -pnl / d0

    def _cvar(losses):
        sorted_losses = np.sort(losses)[::-1]
        n = max(1, int(len(sorted_losses) * tail_prob))
        return sorted_losses[:n].mean()

    data = {al: _load_pnl(al, scoring_key) for al in alpha_labels}

    all_arrays = list(data.values())
    x_min  = min(np.percentile(a, 0.1)  for a in all_arrays)
    x_max  = max(np.percentile(a, 99.9) for a in all_arrays)
    x_grid = np.linspace(x_min, x_max, 1000)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    plt.subplots_adjust(left=0.25)

    for al in alpha_labels:
        arr      = data[al]
        color    = ALPHA_COLOR.get(al, "#888888")
        cvar_val = -1 * _cvar(arr)
        mean_val = np.mean(arr)

        ax.hist(-1 * arr, bins=100, range=(x_min, x_max),
                density=True, color=color, alpha=0.18, linewidth=0)
        ax.plot(x_grid, gaussian_kde(-1 * arr)(x_grid),
                color=color, linewidth=1.8, label=ALPHA_DISPLAY.get(al, al))
        ax.axvline(cvar_val, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
        ax.axvline(mean_val, color=color, linestyle=":",  linewidth=1.2, alpha=0.90)

    _style_ax(ax)
    ax.set_xlabel(r"Normalized Terminal Hedging Error", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.set_title(SCORING_DISPLAY.get(scoring_key, scoring_key), fontsize=FONT_TITLE, color="#444444")
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    _save(fig, save_dir, f"pnl_distributions_{scoring_key}")
    plt.show()


def plot_all_risk_distributions_by_scoring(
    data_dir:     str   = "content/data",
    save_dir:     str   = ".",
    tail_prob:    float = 0.05,
    scoring_keys: list  = None,
    alpha_labels: list  = None,
) -> None:
    if scoring_keys is None:
        scoring_keys = SCORING_KEYS
    for sk in scoring_keys:
        plot_risk_distributions_by_scoring(
            sk, data_dir=data_dir, save_dir=save_dir,
            tail_prob=tail_prob, alpha_labels=alpha_labels,
        )


# ---------------------------------------------------------------------------
# 3c. Terminal PnL statistics table
# ---------------------------------------------------------------------------

def compute_terminal_stats(
    data_dir:     str  = "content/data",
    save_dir:     str  = ".",
    alpha_labels: list = None,
    alpha_vals:   dict = None,
    model_keys:   list = None,
) -> pd.DataFrame:
    if alpha_labels is None:
        alpha_labels = ALPHA_LABELS
    if alpha_vals is None:
        alpha_vals = {"alpha925": 0.925, "alpha95": 0.95, "alpha975": 0.975, "alpha99": 0.99}

    def _cvar(x, alpha):
        cutoff = np.quantile(x, alpha)
        tail   = x[x >= cutoff]
        return float(tail.mean()) if len(tail) > 0 else float(cutoff)

    rows = []
    for alpha_label in alpha_labels:
        keys_this = (
            model_keys if model_keys is not None
            else _scoring_keys_for_alpha(alpha_label) + ["static"]
        )
        for sk in keys_this:
            path_pnl = os.path.join(data_dir, alpha_label, sk, "terminal_pnl.npy")
            if not os.path.exists(path_pnl):
                print(f"  MISSING: {alpha_label}/{sk}")
                continue

            pnl    = np.load(path_pnl)
            losses = -pnl

            row = {"alpha": alpha_label, "model": sk,
                   "mean": float(np.mean(pnl)), "std": float(np.std(pnl))}
            for a_label, a_val in alpha_vals.items():
                row[f"CVaR_{a_label}"] = _cvar(losses, a_val)
            rows.append(row)

    df_stats = pd.DataFrame(rows)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "terminal_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    return df_stats


# ---------------------------------------------------------------------------
# 3d. Tracking error over time
# ---------------------------------------------------------------------------

def plot_tracking_error(
    alpha_label:    str   = "alpha95",
    data_dir:       str   = "content/data",
    save_dir:       str   = ".",
    scoring_labels: list  = None,
    n_paths:        int   = 1000,
    normalize:      bool  = True,
) -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "axes.titlesize":   FONT_TITLE,
        "axes.labelsize":   FONT_LABEL,
        "xtick.labelsize":  FONT_TICK,
        "ytick.labelsize":  FONT_TICK,
        "legend.fontsize":  FONT_LEGEND,
        "axes.facecolor":   "#f7f7f7",
    })

    if scoring_labels is None:
        scoring_labels = _scoring_keys_for_alpha(
            alpha_label, base_keys=["rational", "arcsinh", "arctan", "arcsin", "power03", "log"]
        )

    drm_colors = [SCORING_COLOR.get(sl, "#888888") for sl in scoring_labels]

    def _load_series(key):
        portval = _load(data_dir, alpha_label, key, "portfolio_values.npy")[:n_paths]  # [n, T]
        if normalize:
            d0 = _load_shared(data_dir, alpha_label, "deriv_prices.npy")[:n_paths, 0]   # shared across keys
            portval = portval / d0[:, None]
        return portval.mean(axis=0)  # [T]

    T = _load(data_dir, alpha_label, scoring_labels[0], "portfolio_values.npy").shape[1]
    t_axis = np.arange(T) / 252

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)

    for sl, color in zip(scoring_labels, drm_colors):
        ax.plot(t_axis, _load_series(sl), color=color, linewidth=1.6, label=sl)

    ax.plot(t_axis, _load_series("static"), color=STATIC_COLOR,
            linewidth=2.2, linestyle="--", label="static")

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    _style_ax(ax)
    ax.set_xlabel("Time (Years)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean Normalized Tracking Error", fontsize=FONT_LABEL)
    ax.set_title(ALPHA_DISPLAY.get(alpha_label, alpha_label), fontsize=FONT_TITLE, color="#444444")
    ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    _save(fig, save_dir, f"tracking_error_{alpha_label}")
    plt.show()


def plot_all_tracking_error(
    data_dir:       str  = "content/data",
    save_dir:       str  = ".",
    alpha_labels:   list = None,
    scoring_labels: list = None,
    n_paths:        int  = 1000,
    normalize:      bool = True,
) -> None:
    """Convenience wrapper: one tracking-error plot per alpha level."""
    if alpha_labels is None:
        alpha_labels = ALPHA_LABELS
    for al in alpha_labels:
        plot_tracking_error(
            alpha_label=al, data_dir=data_dir, save_dir=save_dir,
            scoring_labels=scoring_labels, n_paths=n_paths, normalize=normalize,
        )


# ---------------------------------------------------------------------------
# 4. Scoring curvature 
# ---------------------------------------------------------------------------

def plot_scoring_curvature(
    score_fn,
    save_dir: str = ".",
    N: int = 10_000,
    half: float = 2.0,
    alpha_ds: float = 0.95,
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    np.random.seed(42)
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#BA7517", "#D4537E"]

    def _score_np(a1, a2, y, alpha, C):
        return score_fn(
            torch.tensor(a1, dtype=torch.float32),
            torch.tensor(a2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            alpha, C,
        ).numpy()

    def _score_np_fn(fn, a1, a2, y, alpha, C):
        return fn(
            torch.tensor(a1, dtype=torch.float32),
            torch.tensor(a2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            alpha, C,
        ).numpy()

    alphas = [0.10, 0.30, 0.50, 0.70, 0.85, 0.95]
    y_base = np.random.normal(0, 1, N)
    C_base = abs(y_base.min())

    y1 = np.random.normal(5, 1, N)
    y2 = np.random.normal(25, 1, N)

    # NEW: 3x2 grid (dropped the fixed-VaR/fixed-CVaR row)
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # --- row 0: sweep alpha, fixed score_fn -------------------------------
    for alpha, color in zip(alphas, colors):
        ev = np.quantile(y_base, alpha)
        ec = y_base[y_base >= ev].mean()

        grid = np.linspace(ev - half, ev + half, 300)
        sc = np.array([_score_np(np.full(N, a), np.full(N, a), y_base, alpha, C_base).mean() for a in grid])
        oi = np.argmin(sc)
        axes[0, 0].plot(grid, sc, color=color, lw=1.6, alpha=0.85, label=f"alpha={alpha:.2f}")
        axes[0, 0].scatter(grid[oi], sc[oi], color=color, s=60, zorder=5)

        grid = np.linspace(ec - half, ec + half, 300)
        sc = np.array([_score_np(np.full(N, ev), np.full(N, a), y_base, alpha, C_base).mean() for a in grid])
        oi = np.argmin(sc)
        axes[0, 1].plot(grid, sc, color=color, lw=1.6, alpha=0.85, label=f"alpha={alpha:.2f}")
        axes[0, 1].scatter(grid[oi], sc[oi], color=color, s=60, zorder=5)

    # --- row 1: relative shift, two well-separated normals -----------------
    rel_grid = np.linspace(-3.5, 3.5, 300)
    for y_samp, color, label in [
        (y1, "#378ADD", r"$\mathcal{N}(5,1)$"),
        (y2, "#D85A30", r"$\mathcal{N}(25,1)$"),
    ]:
        C = abs(y_samp.min())
        ev = np.quantile(y_samp, alpha_ds)
        ec = y_samp[y_samp >= ev].mean()

        sc = np.array([_score_np(np.full(N, ev + a), np.full(N, ev + a), y_samp, alpha_ds, C).mean() for a in rel_grid])
        sc -= sc.min()
        oi = np.argmin(sc)
        axes[1, 0].plot(rel_grid, sc, color=color, lw=1.6, alpha=0.85, label=label)
        axes[1, 0].scatter(rel_grid[oi], sc[oi], color=color, s=60, zorder=5)

        sc = np.array([_score_np(np.full(N, ev), np.full(N, ec + a), y_samp, alpha_ds, C).mean() for a in rel_grid])
        sc -= sc.min()
        oi = np.argmin(sc)
        axes[1, 1].plot(rel_grid, sc, color=color, lw=1.6, alpha=0.85, label=label)
        axes[1, 1].scatter(rel_grid[oi], sc[oi], color=color, s=60, zorder=5)

    axes[1, 0].axvline(0, color="k", lw=0.8, linestyle="--", alpha=0.4)
    axes[1, 1].axvline(0, color="k", lw=0.8, linestyle="--", alpha=0.4)

    # --- row 2: overlay scoring functions on N(0,1), excluding power05/07/exp
    C_shift = -y_base.min()
    ev_ds = np.quantile(y_base, alpha_ds)
    ec_ds = y_base[y_base >= ev_ds].mean()

    excluded = {"power05", "power07", "exponential"}
    fns_to_plot = {k: v for k, v in SCORE_FUNCTIONS.items() if k not in excluded}

    cmap = plt.get_cmap("tab10")
    fn_colors = [cmap(i % 10) for i in range(len(fns_to_plot))]

    for (name, fn), color in zip(fns_to_plot.items(), fn_colors):
        grid = np.linspace(ev_ds - half, ev_ds + half, 300)
        sc = np.array([
            _score_np_fn(fn, np.full(N, a), np.full(N, a), y_base, alpha_ds, C_shift).mean()
            for a in grid
        ])
        sc -= sc.min()
        oi = np.argmin(sc)
        axes[2, 0].plot(grid, sc, color=color, lw=1.6, alpha=0.85, label=name)
        axes[2, 0].scatter(grid[oi], sc[oi], color=color, s=60, zorder=5)

    for (name, fn), color in zip(fns_to_plot.items(), fn_colors):
        grid = np.linspace(ec_ds - half, ec_ds + half, 300)
        sc = np.array([
            _score_np_fn(fn, np.full(N, ev_ds), np.full(N, a), y_base, alpha_ds, C_shift).mean()
            for a in grid
        ])
        sc -= sc.min()
        oi = np.argmin(sc)
        axes[2, 1].plot(grid, sc, color=color, lw=1.6, alpha=0.85, label=name)
        axes[2, 1].scatter(grid[oi], sc[oi], color=color, s=60, zorder=5)

    axes[2, 0].axvline(ev_ds, color="k", lw=0.8, linestyle="--", alpha=0.4)
    axes[2, 1].axvline(ec_ds, color="k", lw=0.8, linestyle="--", alpha=0.4)

    # --- shared styling ------------------------------------------------
    panel_xlabels = [
        (axes[0, 0], r"$\mathfrak{a}_1$"),
        (axes[0, 1], r"$\mathfrak{a}_2$"),
        (axes[1, 0], r"$\mathfrak{a}_1$"),
        (axes[1, 1], r"$\mathfrak{a}_2$"),
        (axes[2, 0], r"$\mathfrak{a}_1$"),
        (axes[2, 1], r"$\mathfrak{a}_2$"),
    ]
    for ax, xlabel in panel_xlabels:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color="#444444")
        ax.set_ylabel(r"$\mathrm{s}(\mathfrak{a}_1,\mathfrak{a}_2)$", fontsize=FONT_LABEL, color="#444444")
        _style_ax(ax)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ydata = np.concatenate([l.get_ydata() for l in ax.lines])
        finite = ydata[np.isfinite(ydata)]
        p5, p95 = np.percentile(finite, [5, 95])
        margin = (p95 - p5) * 0.4
        ax.set_ylim(p5 - margin, p95 + margin)

    alpha_handles = [mlines.Line2D([], [], color=c, lw=2, label=f"alpha={a:.2f}") for a, c in zip(alphas, colors)]
    axes[0, 0].legend(handles=alpha_handles, fontsize=FONT_LEGEND_SMALL, frameon=False)
    axes[0, 1].legend(handles=alpha_handles, fontsize=FONT_LEGEND_SMALL, frameon=False)

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.legend(fontsize=FONT_LEGEND_SMALL, frameon=False)

    fn_handles = [mlines.Line2D([], [], color=c, lw=2, label=n) for n, c in zip(fns_to_plot.keys(), fn_colors)]
    axes[2, 0].legend(handles=fn_handles, fontsize=FONT_LEGEND_SMALL, frameon=False, ncol=2)
    axes[2, 1].legend(handles=fn_handles, fontsize=FONT_LEGEND_SMALL, frameon=False, ncol=2)

    plt.tight_layout()
    _save(fig, save_dir, "scoring_curvature_full")
    plt.show()
# ---------------------------------------------------------------------------
# 5. Extreme trajectory
# ---------------------------------------------------------------------------
def plot_extreme_trajectory(
    alpha_label:      str   = "alpha95",
    data_dir:         str   = "content/data",
    env                     = None,
    save_dir:         str   = ".",
    n_paths:          int   = 1000,
    selection_window: int   = 20,
    gamma_scan_paths: int   = 500,
    gamma_chunk_size: int   = 256,
) -> None:
    """
    Selects the hardest-to-hedge path using realized gamma P&L in the last
    `selection_window` days: score = sum_t Gamma_t * (dBasket_t)^2, where
    Gamma is the actual second derivative of the priced basket option w.r.t.
    a uniform shift along basket_weights (computed via autograd through
    env._price_derivative_batch).
    """
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    # Bigger fonts for this 3x2 grid (LaTeX shrinks these a lot at print size)
    EXTREME_LABEL  = 15
    EXTREME_LEGEND = 13
    EXTREME_TICK   = 13

    T          = env.T_days
    timesteps  = np.arange(T)     / 252
    t_full     = np.arange(T + 1) / 252

    # env.basket_weights is system.weights
    weights = env.basket_weights.cpu().numpy()

    corr_pairs  = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    corr_labels = [f"{ASSET_NAMES[i]}-{ASSET_NAMES[j]}" for i, j in corr_pairs]

    scoring_keys_this    = _scoring_keys_for_alpha(alpha_label)
    scoring_display_this = {k: SCORING_DISPLAY[k] for k in scoring_keys_this}
    colors_scoring = [SCORING_COLOR.get(sk, "#888888") for sk in scoring_keys_this]

    colors_assets = list(plt.cm.Set1(np.linspace(0, 0.8, 4)))
    colors_corr   = list(plt.cm.tab10(np.linspace(0, 1.0, 6)))

    S_paths    = _load_shared(data_dir, alpha_label, "S_paths.npy")[:n_paths]        # [n, T+1, 4]
    h_paths    = _load_shared(data_dir, alpha_label, "h_paths.npy")[:n_paths]        # [n, T+1, 4]
    R_paths    = _load_shared(data_dir, alpha_label, "R_paths.npy")[:n_paths]        # [n, T+1, 4, 4]
    deriv_all  = _load_shared(data_dir, alpha_label, "deriv_prices.npy")[:n_paths]   # [n, T(+1)]
    basket_np  = (S_paths * weights).sum(axis=-1)                                       # [n, T+1]

    # ── select hardest-to-hedge path via realized gamma P&L, last selection_window days
    n_scan  = min(n_paths, gamma_scan_paths)
    device  = env.device
    S_scan  = torch.tensor(S_paths[:n_scan], dtype=torch.float32, device=device)
    h_scan  = torch.tensor(h_paths[:n_scan], dtype=torch.float32, device=device)
    R_scan  = torch.tensor(R_paths[:n_scan], dtype=torch.float32, device=device)

    window_start = max(0, T - selection_window)
    t_range      = list(range(window_start, T))
    gamma_window = _compute_basket_gamma_window(
        env, S_scan, h_scan, R_scan, t_range, chunk_size=gamma_chunk_size
    )  # [n_scan, len(t_range)]

    basket_scan      = basket_np[:n_scan]
    moves            = basket_scan[:, window_start + 1:T + 1] - basket_scan[:, window_start:T]
    gamma_pnl_score  = (gamma_window * moves ** 2).sum(axis=1)   # [n_scan]

    hard_idx = int(np.argmax(gamma_pnl_score))
    print(f"Hard trajectory (max realized gamma P&L, last {selection_window}d): "
          f"idx={hard_idx}  score={gamma_pnl_score[hard_idx]:.6f}  "
          f"(basket start={basket_scan[hard_idx, 0]:.2f}  end={basket_scan[hard_idx, -1]:.2f}, "
          f"moneyness_end={basket_scan[hard_idx, -1] / env.K:.3f})")

    basket_hard = basket_np[hard_idx]
    S_hard      = S_paths[hard_idx]
    h_hard      = h_paths[hard_idx]
    vol_hard    = np.sqrt(252 * h_hard)
    R_hard      = R_paths[hard_idx]
    corr_hard   = np.stack([R_hard[:, i, j] for i, j in corr_pairs], axis=1)
    deriv_hard  = deriv_all[hard_idx]
    t_deriv     = t_full if deriv_hard.shape[0] == T + 1 else timesteps

    all_actions      = {}
    all_port_deltas  = {}

    for sk in scoring_keys_this:
        actions = _load(data_dir, alpha_label, sk, "actions.npy")[:, :n_paths, :]
        all_actions[sk]     = actions[:, hard_idx, :]
        all_port_deltas[sk] = (all_actions[sk] * weights).sum(axis=-1)   # [T]

    actions_s = _load(data_dir, alpha_label, "static", "actions.npy")[:, :n_paths, :]
    all_actions["static"]     = actions_s[:, hard_idx, :]
    all_port_deltas["static"] = (all_actions["static"] * weights).sum(axis=-1)   # [T]

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))

    ax_delta, ax_deriv = axes[0, 0], axes[0, 1]
    lower_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]

    # ── portfolio delta (basket_weights . actions), static vs dynamic ──
    for sk, color in zip(scoring_keys_this, colors_scoring):
        ax_delta.plot(timesteps, all_port_deltas[sk],
                      color=color, linewidth=1.6, alpha=0.85)
    ax_delta.plot(timesteps, all_port_deltas["static"],
                  color="black", linewidth=2.4, linestyle="--", alpha=0.95)
    ax_delta.set_xlabel("Time (Years)", fontsize=EXTREME_LABEL)
    ax_delta.set_ylabel("Weighted Aggregate Position", fontsize=EXTREME_LABEL)
    ax_delta.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    _style_ax(ax_delta, ticksize=EXTREME_TICK)

    # ── derivative price along the hard trajectory ──
    ax_deriv.plot(t_deriv, deriv_hard, color="#2e6f95", linewidth=1.8)
    ax_deriv.set_xlabel("Time (Years)", fontsize=EXTREME_LABEL)
    ax_deriv.set_ylabel("Basket Option Price", fontsize=EXTREME_LABEL)
    _style_ax(ax_deriv, ticksize=EXTREME_TICK)

    ax = lower_axes[0]
    ax.plot(t_full, basket_hard, color="#c0392b", linewidth=1.8)
    ax.set_xlabel("Time (Years)", fontsize=EXTREME_LABEL)
    ax.set_ylabel("Basket Value", fontsize=EXTREME_LABEL)
    _style_ax(ax, ticksize=EXTREME_TICK)

    ax = lower_axes[1]
    for i, (name, color) in enumerate(zip(ASSET_NAMES, colors_assets)):
        ax.plot(t_full, S_hard[:, i], color=color, linewidth=1.4, label=name)
    ax.set_xlabel("Time (Years)", fontsize=EXTREME_LABEL)
    ax.set_ylabel("Stock Price", fontsize=EXTREME_LABEL)
    ax.legend(fontsize=EXTREME_LEGEND, frameon=False)
    _style_ax(ax, ticksize=EXTREME_TICK)

    ax = lower_axes[2]
    for i, (name, color) in enumerate(zip(ASSET_NAMES, colors_assets)):
        ax.plot(t_full, vol_hard[:, i], color=color, linewidth=1.4, label=name)
    ax.set_xlabel("Time (Years)", fontsize=EXTREME_LABEL)
    ax.set_ylabel("Ann. Volatility", fontsize=EXTREME_LABEL)
    ax.legend(fontsize=EXTREME_LEGEND, frameon=False)
    _style_ax(ax, ticksize=EXTREME_TICK)

    ax = lower_axes[3]
    for k, (label, color) in enumerate(zip(corr_labels, colors_corr)):
        ax.plot(t_full, corr_hard[:, k], color=color, linewidth=1.2, label=label)
    ax.set_xlabel("Time (Years)", fontsize=EXTREME_LABEL)
    ax.set_ylabel("Correlation", fontsize=EXTREME_LABEL)
    ax.legend(fontsize=EXTREME_LEGEND, frameon=False, ncol=2)
    _style_ax(ax, ticksize=EXTREME_TICK)

    scoring_name_map = {
        "arcsin": "arcsin", "arcsinh": "arcsinh", "arctan": "arctan",
        "log": "log", "power03": "power", "rational": "rational",
    }
    legend_handles = [
        mlines.Line2D([], [], color=c, lw=2.2, alpha=0.85,
                      label=scoring_name_map.get(sk, sk))
        for sk, c in zip(scoring_display_this.keys(), colors_scoring)
    ] + [mlines.Line2D([], [], color="black", lw=2.6, linestyle="--", label="static")]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(legend_handles),
        frameon=False, fontsize=EXTREME_LEGEND, bbox_to_anchor=(0.5, -0.015),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06)

    _save(fig, save_dir, "actions_hard_traj_4x2")
    plt.show()
# ---------------------------------------------------------------------------
# 6. Dynamic risk comparison
# ---------------------------------------------------------------------------


def plot_dynamic_risk_comparison(
    alpha_label:             str = "alpha95",
    data_dir:                str = "content/data",
    all_models:              dict = None,
    all_norms:               dict = None,
    env                          = None,
    CriticVaRExcess              = None,
    CriticCVaRShared             = None,
    nested_ckpt_dynamic_path:str = "NestedCritics/critic_checkpoint_shared_Epoch6.pt",
    nested_ckpt_static_path: str = None,
    nested_actor_scoring_key:str = "log",
    nested_hidden_dim:       int = 512,
    nested_n_layers:         int = 3,
    nested_time_embed_dim:   int = 64,
    save_dir:                str = ".",
    n_paths:                 int = 1000,
    risk_threshold:          int = 95,
    n_groups:                int = 18,
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    device     = env.device
    T          = env.T_days
    group_size = T // n_groups

    scoring_display = {
        "rational": "rational", "arcsinh": "arcsinh", "log": "log",
        "arcsin":   "arcsin",   "arctan":  "arctan",  "power03": "power",
    }
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(scoring_display) + 1, 2)))

    def _load_shared_nested(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        critic = CriticCVaRShared(
            env.state_dim, T,                       # n_groups=1 -> single "group" spans all T
            hidden_dim=nested_hidden_dim,
            n_layers=nested_n_layers,
            time_embed_dim=nested_time_embed_dim,
            device=device,
        )
        critic.load_state_dict(ckpt["critics"][0])   # n_groups=1 -> one entry
        critic.to(device).eval()
        return critic

    nested_critic_dynamic = _load_shared_nested(nested_ckpt_dynamic_path)
    nested_critic_static  = _load_shared_nested(nested_ckpt_static_path) if nested_ckpt_static_path else None

    fig, ax = plt.subplots(figsize=(13, 4.5))
    t_axis  = np.arange(T) / 252

    for (sk, label), color in zip(scoring_display.items(), colors):
        if sk not in all_models.get(alpha_label, {}):
            print(f"  Skipping {sk}: not found in all_models['{alpha_label}']")
            continue

        states_np = _load(data_dir, alpha_label, sk, "states.npy")[:, :n_paths, :]

        raw_critics = all_models[alpha_label][sk].get("critics")
        if raw_critics is None:
            print(f"  Skipping {sk}: no 'critics' entry in all_models['{alpha_label}']['{sk}']")
            continue

        critics = []
        for c in raw_critics:
            fresh = CriticVaRExcess(
                env.state_dim, group_size,
                head_dim_var=128, head_dim_excess=128,
            )
            fresh.load_state_dict(c.state_dict())
            fresh.to(device).eval()
            critics.append(fresh)

        b_values = all_norms[alpha_label][sk]["b_values"]
        states_t = torch.tensor(states_np, dtype=torch.float32, device=device)

        cvar_grid = np.zeros((n_paths, T))
        with torch.no_grad():
            for g in range(n_groups):
                gs, ge = g * group_size, (g + 1) * group_size
                v, e = critics[g](states_t[gs:ge])
                cvar_grid[:, gs:ge] = (
                    (v + e).squeeze(-1) + b_values[n_groups - 1 - g]
                ).T.cpu().numpy()

        ax.plot(t_axis, cvar_grid.mean(axis=0),
                label=label, color=color, linewidth=1.4, alpha=0.75, zorder=3)

    # ---- nested (shared) dynamic overlay -- states from the actor that trained it ----
    states_d_np = _load(data_dir, alpha_label, nested_actor_scoring_key, "states.npy")[:, :n_paths, :]
    states_d    = torch.tensor(states_d_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        nested_d = nested_critic_dynamic.forward_chunked(states_d, chunk=16384)  # [T, n_paths]

    ax.plot(t_axis, nested_d.mean(dim=1).cpu().numpy(),
            color=colors[-1], linewidth=1.6, linestyle="--", zorder=2,
            label=f"Nested - {scoring_display.get(nested_actor_scoring_key, nested_actor_scoring_key)}")

    # ---- static overlay, only if provided ----
    if nested_critic_static is not None:
        states_s_np = _load(data_dir, alpha_label, "static", "states.npy")[:, :n_paths, :]
        states_s    = torch.tensor(states_s_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            nested_s = nested_critic_static.forward_chunked(states_s, chunk=16384)
        ax.plot(t_axis, nested_s.mean(dim=1).cpu().numpy(),
                color="black", linewidth=1.6, linestyle="--", zorder=2,
                label="Nested - Static")

    ax.legend(frameon=False, loc="upper right", fontsize=FONT_LEGEND)
    ax.set_xlabel("Time (years)", fontsize=FONT_LABEL)
    ax.set_ylabel("Dynamic CVaR Risk", fontsize=FONT_LABEL)
    _style_ax(ax)

    plt.tight_layout()
    _save(fig, save_dir, "DynamicRiskComparison")
    plt.show()
# ---------------------------------------------------------------------------
# 7. Validation Dynamic
# ---------------------------------------------------------------------------


def plot_validation_risk(
    logs_root: str   = "content/DeepHedging/training_run_logs",
    alpha:     float = 0.95,
    save_dir:  str   = ".",
    metric:    str   = "CVaR_95%",
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    def _parse(path, key):
        iters, vals, cur = [], [], None
        with open(path) as f:
            for line in f:
                m = re.search(r"Iter (\d+):", line)
                if m:
                    cur = int(m.group(1))
                c = re.search(rf"{key}=([-\d.]+)", line)
                if c and cur is not None:
                    iters.append(cur); vals.append(float(c.group(1)))
        return iters, vals

    alpha_label = f"alpha_{alpha}"
    _remap = lambda k: k.replace("bounded", "").replace("shift", "modified").strip()
    log_files = {_remap(k): v for k, v in _find_training_logs(logs_root, alpha_label)}
    if not log_files:
        print(f"No training.log files found for {alpha_label} under {logs_root}")
        return

    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))[:len(log_files)]
    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, path), color in zip(log_files.items(), colors):
        iters, vals = _parse(path, metric)
        ax.plot(iters, vals, marker="o", markersize=3, lw=1.8,
                label=label, color=color, alpha=0.9)

    ax.set_xlabel("Epoch", fontsize=FONT_LABEL, color="#444444")
    ax.set_ylabel("Validation Dynamic CVaR Risk", fontsize=FONT_LABEL, color="#444444")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper right", labelcolor="#444444")

    plt.tight_layout()
    _save(fig, save_dir, f"validation_risk_alpha{alpha}")
    plt.show()

# ---------------------------------------------------------------------------
# 8. Initial dynamic CVaR
# ---------------------------------------------------------------------------

def plot_initial_dynamic_cvar(
    log_files:               dict,
    alpha_label:             str = "alpha95",
    data_dir:                str = "content/data",
    env                          = None,
    CriticCVaR                   = None,
    nested_ckpt_shift_path:  str = "NestedCritics/critic_checkpoint_Epoch4.pt",
    nested_ckpt_noshift_path:str = "NestedCritics/critic_checkpoint_noshift_Epoch4.pt",
    save_dir:                str = ".",
    n_groups:                int = 18,
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    device     = env.device
    group_size = env.T_days // n_groups

    def _parse_vearly(path):
        iters, vals, cur = [], [], None
        with open(path) as f:
            for line in f:
                m = re.search(r"Iter (\d+):", line)
                if m:
                    cur = int(m.group(1))
                c = re.search(r"V_early=([-\d.]+)", line)
                if c and cur is not None:
                    iters.append(cur); vals.append(float(c.group(1)))
        return iters, vals

    def _nested_t0(ckpt_path, states_t0):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        critics = torch.nn.ModuleList([
            CriticCVaR(env.state_dim, group_size, hidden_dim=64, head_dim=64, device=device)
            for _ in range(n_groups)
        ])
        for g in range(n_groups):
            critics[g].load_state_dict(ckpt["critics"][g])
            critics[g].to(device).eval()

        with torch.no_grad():
            states_in = states_t0.unsqueeze(0).expand(group_size, -1, -1)
            v0 = critics[0](states_in)
            val = v0.mean().item()
        return val

    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))[[0, 1]]
    color_noshift, color_shift = colors

    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, path), color in zip(log_files.items(), colors):
        iters, vals = _parse_vearly(path)
        ax.plot(iters, vals, marker="o", markersize=3, lw=1.8,
                label=label, color=color, alpha=0.9)

    states_shift   = torch.tensor(
        _load(data_dir, alpha_label, "power03", "states.npy")[0],
        dtype=torch.float32, device=device
    )
    states_noshift = torch.tensor(
        _load(data_dir, alpha_label, "static", "states.npy")[0],
        dtype=torch.float32, device=device
    )

    nested_shift   = _nested_t0(nested_ckpt_shift_path,   states_shift)
    nested_noshift = _nested_t0(nested_ckpt_noshift_path, states_noshift)

    ax.axhline(nested_noshift, color=color_noshift, lw=1.8, linestyle="--", label="Nested", zorder=5)
    ax.axhline(nested_shift,   color=color_shift,   lw=1.8, linestyle="--", label="Nested", zorder=5)

    ax.set_xlabel("Epoch", fontsize=FONT_LABEL, color="#444444")
    ax.set_ylabel("Training Initial Dynamic CVaR Risk", fontsize=FONT_LABEL, color="#444444")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper right", labelcolor="#444444")

    plt.tight_layout()
    _save(fig, save_dir, "InitialDynamicCVaR")
    plt.show()


def plot_actor_critic_loss(
    logs_root: str   = "content/DeepHedging/training_run_logs",
    alpha:     float = 0.95,
    save_dir:  str   = ".",
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    def _parse(path, key):
        iters, vals, cur = [], [], None
        with open(path) as f:
            for line in f:
                m = re.search(r"Iter (\d+):", line)
                if m:
                    cur = int(m.group(1))
                c = re.search(rf"{key}=([-\d.]+)", line)
                if c and cur is not None:
                    iters.append(cur); vals.append(float(c.group(1)))
        return iters, vals

    alpha_label = f"alpha_{alpha}"
    log_files = dict(_find_training_logs(logs_root, alpha_label))
    if not log_files:
        print(f"No training.log files found for {alpha_label} under {logs_root}")
        return

    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))[:len(log_files)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, (loss_key, ylabel) in zip(axes, [
        ("critic_loss", "Mean Critic Loss"),
        ("actor_loss",  "Mean Actor Loss"),
    ]):
        for (label, path), color in zip(log_files.items(), colors):
            iters, vals = _parse(path, loss_key)
            ax.plot(iters, vals, marker="o", markersize=3, lw=1.8,
                    label=label, color=color, alpha=0.9)
        ax.set_xlabel("Epoch", fontsize=FONT_LABEL, color="#444444")
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL, color="#444444")
        _style_ax(ax)
        ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper right", labelcolor="#444444")

    plt.tight_layout()
    _save(fig, save_dir, f"ALCL_alpha{alpha}")
    plt.show()


def plot_mean_action(
    logs_root: str   = "content/DeepHedging/training_run_logs",
    alpha:     float = 0.95,
    save_dir:  str   = ".",
) -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    def _parse(path):
        iters, vals, cur = [], [], None
        with open(path) as f:
            for line in f:
                m = re.search(r"Iter (\d+):", line)
                if m:
                    cur = int(m.group(1))
                c = re.search(r"mean_action=([-\d.]+)", line)
                if c and cur is not None:
                    iters.append(cur); vals.append(float(c.group(1)))
        return iters, vals

    alpha_label = f"alpha_{alpha}"
    _remap = lambda k: k.replace("bounded", "").replace("shift", "modified").strip()
    log_files = {_remap(k): v for k, v in _find_training_logs(logs_root, alpha_label)}
    if not log_files:
        print(f"No training.log files found for {alpha_label} under {logs_root}")
        return

    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))[:len(log_files)]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, path), color in zip(log_files.items(), colors):
        iters, vals = _parse(path)
        ax.plot(iters, vals, marker="o", markersize=3, lw=1.8,
                label=label, color=color, alpha=0.9)

    ax.set_xlabel("Epoch", fontsize=FONT_LABEL, color="#444444")
    ax.set_ylabel("Mean Action", fontsize=FONT_LABEL, color="#444444")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper right", labelcolor="#444444")

    plt.tight_layout()
    _save(fig, save_dir, f"mean_action_alpha{alpha}")
    plt.show()


def plot_state_sensitivity(
    alpha_label: str = "alpha95",
    scoring_key: str = "arcsin",
    data_dir:    str = "content/data",
    env              = None,
    save_dir:    str = ".",
    t_index:     int = 0,
    n_paths:     int = 20000,
    top_k:       int = 6,
) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    n_assets = env.n_assets
    corr_pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
    names = (
        [f"Moneyness_{ASSET_NAMES[i]}"  for i in range(n_assets)] + ["tau"] +
        [f"Vol_{ASSET_NAMES[i]}"        for i in range(n_assets)] +
        [f"PrevAction_{ASSET_NAMES[i]}" for i in range(n_assets)] +
        [f"Corr_{ASSET_NAMES[i]}-{ASSET_NAMES[j]}" for i, j in corr_pairs]
    )

    states  = _load(data_dir, alpha_label, scoring_key, "states.npy")[t_index][:n_paths]
    actions = _load(data_dir, alpha_label, scoring_key, "actions.npy")[t_index][:n_paths]

    assert actions.shape[1] == n_assets

    # Only analyze the first asset's hedge position
    asset_delta = actions[:, 0]

    rows = []
    for j, name in enumerate(names):
        xj = states[:, j]

        if np.ptp(xj) < 1e-10:
            rho, pval, abs_rho = np.nan, np.nan, -1.0
        else:
            rho, pval = spearmanr(xj, asset_delta)
            abs_rho = abs(rho)

        rows.append({
            "state_variable": name,
            "spearman_rho": rho,
            "p_value": pval,
            "abs_rho": abs_rho
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("abs_rho", ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(
        os.path.join(
            save_dir,
            f"state_sensitivity_{alpha_label}_{scoring_key}_t{t_index}_asset0.csv"
        ),
        index=False
    )

    top = df[df["abs_rho"] >= 0].head(top_k)

    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.6 * nrows)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, top.iterrows()):
        j = names.index(row["state_variable"])
        order = np.argsort(states[:, j])

        ax.scatter(
            states[order, j],
            asset_delta[order],
            s=4,
            alpha=0.25,
            color="#4C8FE8",
            rasterized=True,
            linewidths=0
        )

        star = "*" if row["p_value"] < 0.05 else ""

        ax.set_title(
            f"{row['state_variable']}   "
            + r"$\rho$="
            + f"{row['spearman_rho']:.3f}{star}",
            fontsize=FONT_TITLE_SMALL,
            color="#444444"
        )
        ax.set_xlabel(row["state_variable"], fontsize=FONT_LABEL_SMALL, color="#444444")
        ax.set_ylabel("Asset 0 Delta", fontsize=FONT_LABEL_SMALL, color="#444444")
        _style_ax(ax)

    for ax in axes[len(top):]:
        ax.axis("off")

    plt.tight_layout()

    _save(
        fig,
        save_dir,
        f"state_sensitivity_{alpha_label}_{scoring_key}_t{t_index}_asset0"
    )

    plt.show()

    return df
def plot_scoring_curvature_normal(
    save_dir: str = ".",
    N: int = 100_000,
    alpha: float = 0.95,
    C: float = 15.0,
    half: float = 0.5,
    n_grid: int = 300,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    np.random.seed(42)

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------
    y = np.random.normal(0.0, 1.0, N)

    # ---------------------------------------------------------
    # True VaR and CVaR for the sampled distribution
    # ---------------------------------------------------------
    var_true = np.quantile(y, alpha)
    cvar_true = y[y >= var_true].mean()

    # ---------------------------------------------------------
    # Six scoring functions used in the experiments
    # ---------------------------------------------------------
    scoring_functions = {
        "log": SCORE_FUNCTIONS["log"],
        "power03": SCORE_FUNCTIONS["power03"],
        "rational": SCORE_FUNCTIONS["rational"],
        "arctan": SCORE_FUNCTIONS["arctan"],
        "arcsin": SCORE_FUNCTIONS["arcsin"],
        "arcsinh": SCORE_FUNCTIONS["arcsinh"],
    }

    # Convert Y to torch once.
    y_tensor = torch.tensor(y, dtype=torch.float32)

    def expected_score(score_fn, a1, a2):
        """
        Monte Carlo estimate of E[s(a1, a2, Y)].
        """
        a1_tensor = torch.full(
            (N,),
            float(a1),
            dtype=torch.float32,
        )

        a2_tensor = torch.full(
            (N,),
            float(a2),
            dtype=torch.float32,
        )

        with torch.no_grad():
            scores = score_fn(
                a1_tensor,
                a2_tensor,
                y_tensor,
                alpha,
                C,
            )

        return scores.mean().item()

    # ---------------------------------------------------------
    # Grids around the true optima
    # ---------------------------------------------------------
    a1_grid = np.linspace(
        var_true - half,
        var_true + half,
        n_grid,
    )

    a2_grid = np.linspace(
        cvar_true - half,
        cvar_true + half,
        n_grid,
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    # ---------------------------------------------------------
    # Plot 1: curvature around VaR
    # ---------------------------------------------------------
    for name, score_fn in scoring_functions.items():

        expected_scores = np.array([
            expected_score(
                score_fn,
                a1,
                cvar_true,
            )
            for a1 in a1_grid
        ])

        axes[0].plot(
            a1_grid,
            expected_scores,
            lw=1.8,
            label=name,
        )

    # ---------------------------------------------------------
    # Plot 2: curvature around CVaR
    # ---------------------------------------------------------
    for name, score_fn in scoring_functions.items():

        expected_scores = np.array([
            expected_score(
                score_fn,
                var_true,
                a2,
            )
            for a2 in a2_grid
        ])

        axes[1].plot(
            a2_grid,
            expected_scores,
            lw=1.8,
            label=name,
        )

    # ---------------------------------------------------------
    # Mark theoretical optima
    # ---------------------------------------------------------
    axes[0].axvline(
        var_true,
        color="k",
        linestyle="--",
        lw=0.9,
        alpha=0.5,
    )

    axes[1].axvline(
        cvar_true,
        color="k",
        linestyle="--",
        lw=0.9,
        alpha=0.5,
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    axes[0].set_xlabel(
        r"$\mathfrak{a}_1$",
        fontsize=FONT_LABEL,
    )

    axes[1].set_xlabel(
        r"$\mathfrak{a}_2$",
        fontsize=FONT_LABEL,
    )

    axes[0].set_ylabel(
        r"$\mathbb{E}[s(\mathfrak{a}_1,"
        r"\operatorname{CVaR}_{\alpha}(Y),Y)]$",
        fontsize=FONT_LABEL,
    )

    axes[1].set_ylabel(
        r"$\mathbb{E}[s(\operatorname{VaR}_{\alpha}(Y),"
        r"\mathfrak{a}_2,Y)]$",
        fontsize=FONT_LABEL,
    )

    axes[0].set_title(
        r"Curvature around $\operatorname{VaR}_{\alpha}$",
        fontsize=FONT_TITLE,
    )

    axes[1].set_title(
        r"Curvature around $\operatorname{CVaR}_{\alpha}$",
        fontsize=FONT_TITLE,
    )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    for ax in axes:
        _style_ax(ax)
        ax.legend(
            fontsize=FONT_LEGEND,
            frameon=False,
        )
        ax.grid(False)

    fig.suptitle(
        rf"$Y\sim\mathcal{{N}}(0,1)$, "
        rf"$\alpha={alpha}$, $C={C}$",
        fontsize=FONT_SUPTITLE,
    )

    plt.tight_layout()

    _save(
        fig,
        save_dir,
        "scoring_curvature_normal",
    )

    plt.show()


# ---------------------------------------------------------------------------
# 9. Static risk vs. maturity (SRM baseline vs. DRM per scoring function)
# ---------------------------------------------------------------------------

def _load_actor_checkpoint(path, state_dim, action_dim, Actor, MLPActorStatic,
                            hidden_dim=256, device="cuda"):
    state_dict = torch.load(path, map_location="cpu", weights_only=False)

    try:
        actor = Actor(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, fixed_std=1e-3,
        )
        actor.load_state_dict(state_dict)
    except RuntimeError:
        actor = MLPActorStatic(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        )
        actor.load_state_dict(state_dict)

    actor.eval()
    return actor.to(device)


def _read_meta_hidden_dim(actor_path, default=256):
    """Infers hidden_dim from the checkpoint's first shared-layer weight shape."""
    try:
        sd = torch.load(actor_path, map_location="cpu", weights_only=False)
        w = sd.get("shared.0.weight")
        if w is not None:
            return int(w.shape[0])
    except Exception as e:
        print(f"  [_read_meta_hidden_dim] falling back to default: {e}")
    return default


def _resolve_drm_actor_path(project_root, alpha_label, scoring_fn):
    direct = os.path.join(
        project_root, "dynamicriskmodels", f"alpha{alpha_label}", scoring_fn, "actor.pt"
    )
    return direct if os.path.isfile(direct) else None


def _evaluate_policy_across_maturities(actor, env, HedgingEnv, maturities_days,
                                        batch_size, alpha, seed, label=""):
    """
    Evaluate one policy across a collection of maturities, each with a
    fresh HedgingEnv built off env's shared settings. Returns
    {T: {mean_loss, std_loss, VaR, CVaR, CVaR_95}}.
    """
    results = {}

    for T in maturities_days:
        sub_env = HedgingEnv(
            simulator=env.simulator,
            system=env.system,
            K=env.K,
            T_days=T,
            S0=env.S0,
            r=env.r_daily.item() if torch.is_tensor(env.r_daily) else env.r_daily,
            transaction_cost=env.transaction_cost,
        )

        with torch.no_grad():
            out = sub_env.simulate_batch(
                actor, batch_size, alpha=alpha, deterministic=True, seed=seed,
            )

        # terminal_pnl is the 10th element of the simulate_batch tuple
        # (states, actions, log_probs, costs, next_states, dones,
        #  portfolio_values, derivative_values, PnL, terminal_pnl,
        #  S_paths, h_paths, R_paths, Q_paths)
        terminal_pnl = out[9]
        losses_np = (-terminal_pnl).cpu().numpy()

        def _cvar(x, a):
            cutoff = np.quantile(x, a)
            tail   = x[x >= cutoff]
            return float(tail.mean()) if len(tail) > 0 else float(cutoff)

        VaR     = float(np.quantile(losses_np, alpha))
        CVaR    = _cvar(losses_np, alpha)
        CVaR_95 = _cvar(losses_np, 0.95)

        results[T] = {
            "mean_loss": float(losses_np.mean()),
            "std_loss":  float(losses_np.std()),
            "VaR":       VaR,
            "CVaR":      CVaR,
            "CVaR_95":   CVaR_95,
        }

        print(f"  [{label}] T={T}: mean_loss={results[T]['mean_loss']:.4f}  "
              f"CVaR_{int(alpha * 100)}={CVaR:.4f}  CVaR_95={CVaR_95:.4f}")

        del sub_env, out
        torch.cuda.empty_cache()

    return results


def plot_static_risk_across_maturities(
    env,
    Actor,
    MLPActorStatic,
    HedgingEnv,
    project_root:      str   = "content/DeepHedging",
    alpha_label:       str   = "95",
    alpha:             float = 0.95,
    scoring_keys:      list  = None,
    save_dir:          str   = ".",
    batch_size:        int   = 10_000,
    seed:              int   = 42,
    maturities_days:   list  = None,
    srm_hidden_dim:    int   = 256,
) -> dict:
    """
    Static CVaR_alpha vs. maturity: SRM baseline against one DRM actor per scoring function.
    """
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "axes.titlesize":   FONT_TITLE,
        "axes.labelsize":   FONT_LABEL,
        "xtick.labelsize":  FONT_TICK,
        "ytick.labelsize":  FONT_TICK,
        "legend.fontsize":  FONT_LEGEND,
        "axes.facecolor":   "#f7f7f7",
    })

    device = env.device
    if scoring_keys is None:
        scoring_keys = SCORING_KEYS
    if maturities_days is None:
        maturities_days = list(range(252, 0, -1))

    print(f"Evaluating alpha={alpha}")

    srm_path = _resolve_srm_actor_path(project_root, alpha)
    assert srm_path is not None, (
        f"No SRM checkpoint found for alpha={alpha} under "
        f"{project_root}/staticriskmodels"
    )
    actor_srm = _load_actor_checkpoint(
        srm_path, env.state_dim, env.action_dim, Actor, MLPActorStatic,
        hidden_dim=srm_hidden_dim, device=device,
    )
    srm_results = _evaluate_policy_across_maturities(
        actor_srm, env, HedgingEnv, maturities_days,
        batch_size=batch_size, alpha=alpha, seed=seed, label="SRM",
    )
    del actor_srm
    torch.cuda.empty_cache()

    drm_results_by_fn = {}
    for sk in scoring_keys:
        drm_path = _resolve_drm_actor_path(project_root, alpha_label, sk)
        if drm_path is None:
            print(f"  [skip] no actor.pt found for alpha={alpha_label}, scoring_fn={sk}")
            continue

        drm_hidden = _read_meta_hidden_dim(drm_path, default=256)
        actor_drm  = _load_actor_checkpoint(
            drm_path, env.state_dim, env.action_dim, Actor, MLPActorStatic,
            hidden_dim=drm_hidden, device=device,
        )
        drm_results_by_fn[sk] = _evaluate_policy_across_maturities(
            actor_drm, env, HedgingEnv, maturities_days,
            batch_size=batch_size, alpha=alpha, seed=seed, label=f"DRM-{sk}",
        )
        del actor_drm
        torch.cuda.empty_cache()

    os.makedirs(save_dir, exist_ok=True)
    out = {
        "alpha": alpha,
        "maturities_days": maturities_days,
        "DRM_by_scoring_fn": drm_results_by_fn,
        "SRM": srm_results,
    }
    json_path = os.path.join(save_dir, f"static_risk_comparison_alpha{alpha_label}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {json_path}")

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)

    for sk, results in drm_results_by_fn.items():
        cvar_vals = [results[t]["CVaR"] for t in maturities_days]
        ax.plot(maturities_days, cvar_vals, color=SCORING_COLOR.get(sk, "#888888"),
                linewidth=1.6, alpha=0.85, marker="o", markersize=4, label=sk)

    srm_cvar = [srm_results[t]["CVaR"] for t in maturities_days]
    ax.plot(maturities_days, srm_cvar, color=STATIC_COLOR, linewidth=2.2,
            linestyle="--", marker="o", markersize=4, alpha=0.95, label="static")

    _style_ax(ax)
    ax.set_xlabel("Maturity (Trading Days)", fontsize=FONT_LABEL)
    ax.set_ylabel(rf"Static $\mathrm{{CVaR}}_{{{int(alpha * 100)}\%}}$", fontsize=FONT_LABEL)
    ax.invert_xaxis()  # longer maturity on left, shorter on right
    ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    plt.tight_layout()
    _save(fig, save_dir, f"static_risk_comparison_alpha{alpha_label}")
    plt.show()

    return out


# ---------------------------------------------------------------------------
# 10. Training curves (actor/critic loss + CVaR_95) split by scoring group
# ---------------------------------------------------------------------------
_ITER_RE   = re.compile(
    r"Iter\s+(\d+):\s*critic_loss=([-+eE0-9.]+)\s*,\s*actor_loss=([-+eE0-9.]+)"
)
_CVAR95_RE = re.compile(r"CVaR_95%\s*=\s*([-+eE0-9.]+)")


def _parse_training_log(path):
    actor_iters, actor_vals   = [], []
    critic_iters, critic_vals = [], []
    cvar_iters, cvar_vals     = [], []

    current_iter = None
    for line in path.read_text().splitlines():
        m = _ITER_RE.search(line)
        if m:
            current_iter = int(m.group(1))
            critic_iters.append(current_iter); critic_vals.append(float(m.group(2)))
            actor_iters.append(current_iter);  actor_vals.append(float(m.group(3)))
            continue
        m = _CVAR95_RE.search(line)
        if m and current_iter is not None:
            cvar_iters.append(current_iter); cvar_vals.append(float(m.group(1)))

    return {
        "actor":  (actor_iters, actor_vals),
        "critic": (critic_iters, critic_vals),
        "cvar":   (cvar_iters, cvar_vals),
    }


def _truncate_before(iters, vals, skip_before):
    if skip_before <= 0:
        return iters, vals
    kept = [(it, v) for it, v in zip(iters, vals) if it >= skip_before]
    if not kept:
        return iters, vals
    its, vs = zip(*kept)
    return list(its), list(vs)


def _plot_training_panel(ax, entries, metric_key, skip_iters_before,
                          label_fs=FONT_LABEL, legend_fs=FONT_LEGEND, tick_fs=FONT_TICK):
    for label, path in sorted(entries):
        parsed = _parse_training_log(path)
        iters, vals = parsed[metric_key]
        if not iters:
            print(f"  ! {path} yielded no '{metric_key}' data -- skipping")
            continue
        iters, vals = _truncate_before(iters, vals, skip_iters_before)
        ax.plot(iters, vals, marker="o", markersize=3, linewidth=1.4,
                color=SCORING_COLOR.get(label, "#888888"), label=label)
    ax.set_xlabel("Epoch", fontsize=label_fs, color="#444444")
    _style_ax(ax, ticksize=tick_fs)
    ax.legend(loc="best", frameon=False, fontsize=legend_fs)


def plot_training_curves_by_scoring(
    logs_root:         str,
    alpha_label:       str  = "alpha_0.95",
    save_dir:          str  = ".",
    group_a:           set  = None,
    skip_iters_before: int  = 20,
) -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "axes.titlesize":   FONT_TITLE,
        "axes.labelsize":   FONT_LABEL,
        "xtick.labelsize":  FONT_TICK,
        "ytick.labelsize":  FONT_TICK,
        "legend.fontsize":  FONT_LEGEND,
        "axes.facecolor":   "#f7f7f7",
    })

    TRAIN_LABEL  = 15
    TRAIN_LEGEND = 13
    TRAIN_TICK   = 13

    if group_a is None:
        group_a = {"log", "arcsinh", "power03"}

    runs = list(_find_training_logs(logs_root, alpha_label))
    if not runs:
        print(f"No training.log files found for {alpha_label}")
        return

    entries_a = [e for e in runs if e[0] in group_a]
    entries_b = [e for e in runs if e[0] not in group_a]

    fig, axes = plt.subplots(3, 2, figsize=(15, 16), dpi=150)

    _plot_training_panel(axes[0, 0], entries_a, "actor", skip_iters_before,
                          label_fs=TRAIN_LABEL, legend_fs=TRAIN_LEGEND, tick_fs=TRAIN_TICK)
    axes[0, 0].set_ylabel("Actor Loss", fontsize=TRAIN_LABEL, color="#444444")
    _plot_training_panel(axes[0, 1], entries_b, "actor", skip_iters_before,
                          label_fs=TRAIN_LABEL, legend_fs=TRAIN_LEGEND, tick_fs=TRAIN_TICK)

    _plot_training_panel(axes[1, 0], entries_a, "critic", skip_iters_before,
                          label_fs=TRAIN_LABEL, legend_fs=TRAIN_LEGEND, tick_fs=TRAIN_TICK)
    axes[1, 0].set_ylabel("Critic Loss", fontsize=TRAIN_LABEL, color="#444444")
    _plot_training_panel(axes[1, 1], entries_b, "critic", skip_iters_before,
                          label_fs=TRAIN_LABEL, legend_fs=TRAIN_LEGEND, tick_fs=TRAIN_TICK)

    _plot_training_panel(axes[2, 0], entries_a, "cvar", skip_iters_before,
                          label_fs=TRAIN_LABEL, legend_fs=TRAIN_LEGEND, tick_fs=TRAIN_TICK)
    axes[2, 0].set_ylabel(r"Static $\mathrm{CVaR}_{95\%}$", fontsize=TRAIN_LABEL, color="#444444")
    _plot_training_panel(axes[2, 1], entries_b, "cvar", skip_iters_before,
                          label_fs=TRAIN_LABEL, legend_fs=TRAIN_LEGEND, tick_fs=TRAIN_TICK)

    plt.tight_layout()
    _save(fig, save_dir, f"losses_cvar_{alpha_label}")
    plt.show()
# ---------------------------------------------------------------------------
# 6b. Dynamic risk table -- mean critic-estimated risk at fixed-month
#      increments, normalized by each path's initial derivative price
# ---------------------------------------------------------------------------
def compute_fixed_price_comparison(
    all_models:   dict,
    all_norms:    dict,
    data_dir:     str   = "content/data",
    save_dir:     str   = ".",
    device:       str   = None,
    alpha_labels: list  = None,
    scoring_keys: list  = None,
) -> pd.DataFrame:
    """
    Price comparison at the FIXED initial condition (S_0=[100,100,100,100],
    h_0=unconditional variance) -- the '_shared_paths' rollouts, not the
    stochastic-initial-condition validation paths used elsewhere.

    For every (alpha, scoring_fn), reconstructs the price the DRM agent
    implies at t=0:
        DRM_price = critic_risk_t0(Z_0 - B) + B_0
    and, on the SAME fixed paths, the price the static (SRM) agent implies:
        SRM_price = CVaR_alpha(-terminal_pnl_static) + B_0
    B_0 is the mean initial basket/derivative price along the fixed paths
    """
    if alpha_labels is None:
        alpha_labels = ALPHA_LABELS
    if scoring_keys is None:
        scoring_keys = SCORING_KEYS
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    alpha_vals = {"alpha925": 0.925, "alpha95": 0.95, "alpha975": 0.975, "alpha99": 0.99}

    def _cvar(losses, alpha):
        cutoff = np.quantile(losses, alpha)
        tail   = losses[losses >= cutoff]
        return float(tail.mean()) if len(tail) > 0 else float(cutoff)

    rows = []
    for alpha_label in alpha_labels:
        alpha_val  = alpha_vals[alpha_label]
        shared_dir = os.path.join(data_dir, alpha_label, "_shared_paths")

        deriv_path = os.path.join(shared_dir, "deriv_prices.npy")
        if not os.path.exists(deriv_path):
            print(f"  ! missing fixed shared paths for {alpha_label} ({deriv_path}) "
                  f"-- re-run generate_data.py's fixed-path block. Skipping.")
            continue
        B0_mean = float(np.load(deriv_path)[:, 0].mean())
        print(f"average price: {B0_mean}, path 0 price: {float(np.load(deriv_path)[0, 0])}")
        # --- SRM / static price on the fixed paths -------------------------
        static_pnl_path = os.path.join(shared_dir, "static", "terminal_pnl.npy")
        if os.path.exists(static_pnl_path):
            static_losses = -np.load(static_pnl_path)
            srm_price     = _cvar(static_losses, alpha_val) + B0_mean
        else:
            print(f"  ! missing fixed static rollout for {alpha_label} "
                  f"({static_pnl_path}) -- SRM_price will be NaN.")
            srm_price = float("nan")

        # --- DRM price per scoring function --------------------------------
        for sk in scoring_keys:
            bundle      = all_models.get(alpha_label, {}).get(sk)
            states_path = os.path.join(shared_dir, sk, "states.npy")
            if bundle is None or not os.path.exists(states_path):
                print(f"  [skip] {alpha_label}/{sk}: missing model or fixed rollout states")
                continue

            states_t0 = np.load(states_path)[0]  # t=0 slice, [n_paths, state_dim]
            states_t0 = torch.tensor(states_t0, dtype=torch.float32, device=device)

            b_values = all_norms[alpha_label][sk]["b_values"]
            critics  = [c.to(device).eval() for c in bundle["critics"]]
            n_groups = len(critics)

            # t=0 -> group 0, local_t=0, reverse-indexed bias b_values[n_groups-1]
            # (same critic-indexing convention as compute_dynamic_risk_table)
            with torch.no_grad():
                v, e = critics[0].forward_single_head(states_t0, 0)
                risk_t0 = (v + e).reshape(-1) + b_values[n_groups - 1]

            drm_price = float(risk_t0.mean().cpu()) + B0_mean

            rows.append({
                "alpha":      alpha_label,
                "scoring_fn": sk,
                "DRM_price":  drm_price,
                "SRM_price":  srm_price,
            })

            for c in critics:
                c.to("cpu")
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows computed -- check the missing-file warnings above.")
        return df

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "fixed_price_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    pivot = df.pivot_table(index="alpha", columns="scoring_fn", values="DRM_price")
    pivot = pivot.reindex(index=alpha_labels, columns=scoring_keys)
    pivot["SRM"] = df.groupby("alpha")["SRM_price"].first().reindex(alpha_labels)
    print("\nDRM price by (alpha x scoring_fn), SRM price for reference "
          "(fixed S0=[100,100,100,100], h0=unconditional variance):")
    print(pivot)

    return df
