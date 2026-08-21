"""
scripts/train_dynamic.py
Usage:
    python scripts/train_dynamic.py
    python scripts/train_dynamic.py --config cfgs/configDynamicRisk.yaml
    python scripts/train_dynamic.py --config cfgs/configDynamicRisk.yaml --device cpu
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import norm, t as t_dist


def set_global_seeds(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Matches notebook calibration run: warn (don't hard-fail) on ops without
    # a deterministic kernel, rather than allowing full nondeterminism.
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic-risk hedging agent")
    parser.add_argument("--config", default="cfgs/configDynamicRisk.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_str = args.device or cfg.get("device", "cuda")
    device = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    if device != device_str:
        print(f"CUDA not available — falling back to CPU")

    set_global_seeds(cfg.get("seed", 42))

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # full precision, no TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from src.data.calibration import (download_returns, fit_all_garch,
                                      compute_garch_residuals, fit_dcc_parameters)
    from src.envs.DCCGARCH import DCCGARCHSimulator
    from src.envs.HedgingEnv import HedgingEnv
    from src.agents.DynamicRiskTraining import train_rl_hedging
    from src.basket_pricing.basket import BasketOptionValuationSystem

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    market_cfg = cfg["market"]

    garch_params_cfg = cfg.get("garch_params")
    garch_params = (
        {int(k): v for k, v in garch_params_cfg.items()}
        if garch_params_cfg is not None else None
    )

    dcc_cfg   = cfg["dcc"]
    nu_Q      = dcc_cfg["nu_Q"]
    dcc_alpha = dcc_cfg.get("dcc_alpha")
    dcc_beta  = dcc_cfg.get("dcc_beta")
    Q_bar_cfg = dcc_cfg.get("Q_bar")
    Q_bar     = np.asarray(Q_bar_cfg, dtype=float) if Q_bar_cfg is not None else None

    fully_prefit = (
        garch_params is not None
        and dcc_alpha is not None
        and dcc_beta is not None
        and Q_bar is not None
    )
    if fully_prefit:
        print(f"All parameters calibrated")
    else:
        from datetime import datetime
        end_date = (datetime.fromisoformat(market_cfg["end_date"])
                    if market_cfg.get("end_date") else None)

        returns_matrix, rf, _ = download_returns(
            tickers=market_cfg["tickers"],
            years_back=market_cfg["years_back"],
            end_date=end_date,
        )

        if garch_params is None:
            print("garch_params not provided in config — calibrating from data "
                  "(same DE-based procedure as the notebook: fit_all_garch)...")
            garch_params = fit_all_garch(
                returns_matrix, rf, list(market_cfg["tickers"].keys())
            )

        if dcc_alpha is not None and dcc_beta is not None and Q_bar is None:
            print("Using DCC alpha/beta/nu_Q from config; computing Q_bar at "
                  "the configured nu_Q (t-copula correlation of the GARCH "
                  "residuals).")
            garch_residuals = compute_garch_residuals(returns_matrix, rf, garch_params)
            U_gauss = np.clip(norm.cdf(garch_residuals), 1e-10, 1 - 1e-10)
            U_t = t_dist.ppf(U_gauss, df=nu_Q)
            Q_bar = np.corrcoef(U_t, rowvar=False)
        elif dcc_alpha is None or dcc_beta is None or Q_bar is None:
            print("Re-fitting DCC parameters (alpha, beta, nu_Q, Q_bar) from "
                  "historical data...")
            garch_residuals = compute_garch_residuals(returns_matrix, rf, garch_params)
            Q_bar, dcc_alpha, dcc_beta, nu_Q = fit_dcc_parameters(garch_residuals)

    print(f"\nCalibration: nu_Q={nu_Q:.4f}, dcc_alpha={dcc_alpha:.6f}, dcc_beta={dcc_beta:.6f}")

    # ------------------------------------------------------------------
    # Pricing system
    # ------------------------------------------------------------------
    system = BasketOptionValuationSystem.load(cfg["pricing"]["system_path"])

    # ------------------------------------------------------------------
    # Simulator + environment
    # ------------------------------------------------------------------
    env_cfg = cfg["env"]

    simulator = DCCGARCHSimulator(
        params=garch_params,
        Q_bar=Q_bar,
        dcc_alpha=dcc_alpha,
        dcc_beta=dcc_beta,
        nu_Q=nu_Q,
        r_daily=market_cfg["r_daily"],
        device=device,
    )

    env = HedgingEnv(
        simulator=simulator,
        system=system,
        K=env_cfg["K"],
        T_days=env_cfg["T_days"],
        S0=env_cfg["S0"],
        r=env_cfg["r"],
        transaction_cost=env_cfg["transaction_cost"]
    )
    print(f"✓ Environment ready | state_dim={env.state_dim} action_dim={env.action_dim} device={env.device}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    tr  = cfg["training"]
    log_dir = Path(tr["log_dir"]) / f"scoring_{tr['scoring_fn']}" / f"alpha_{tr['alpha']}"
    log_dir.mkdir(parents=True, exist_ok=True)

    actor, critic_backwards, run_dir = train_rl_hedging(
        env=env,
        actor_dim        = cfg["actor"]["hidden_dim"],
        head_dim_var     = cfg["critic"]["head_dim_var"],
        head_dim_excess  = cfg["critic"]["head_dim_excess"],
        K                = tr["K"],
        K_2              = tr["K_2"],
        K_initial        = tr["K_initial"],
        K_star           = tr["K_star"],
        K_target         = tr["K_target"],
        K_final          = tr.get("K_final", tr["K_star"]),
        B0               = tr["B0"],
        B1               = tr["B1"],
        B2               = tr["B2"],
        alpha            = tr["alpha"],
        scoring_fn       = tr["scoring_fn"],
        n_groups         = tr["n_groups"],
        n_increment      = tr["n_increment"],
        eta_theta        = tr["eta_theta"],
        min_lr_actor     = tr["min_lr_actor"],
        critic_reset_lr  = tr["critic_reset_lr"],
        C_shift = tr["C_shift"],
        b_shift          = tr["b_shift"],
        critic_grad_clip    = tr["critic_grad_clip"],
        log_interval     = tr["log_interval"],
        log_dir          = str(log_dir),
        resume_checkpoint= tr.get("resume_checkpoint"),
    )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    model_path = run_dir / f"dynamic_final_scoring_{tr['scoring_fn']}_alpha_{tr['alpha']}.pt"
    torch.save({
        "actor":           actor.state_dict(),
        "critic_backwards":[cb.state_dict() for cb in critic_backwards],
        "state_dim":       env.state_dim,
        "action_dim":      env.action_dim,
        "alpha":           tr["alpha"],
        "scoring_fn":      tr["scoring_fn"],
        "config": {
            "K": env.K, "T_days": env.T_days,
            "n_assets": env.n_assets,
            "transaction_cost": env.transaction_cost,
        },
    }, model_path)
    print(f"✓ Final model saved to {model_path}")

    summary = {"scoring_fn": tr["scoring_fn"], "alpha": tr["alpha"],
               "model_path": str(model_path), "run_dir": str(run_dir)}
    with open(Path(tr["log_dir"]) / "dynamic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
