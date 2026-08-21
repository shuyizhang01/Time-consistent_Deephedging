"""
scripts/generate_basket_data.py
================================
One-time pipeline: fetch market data -> calibrate GARCH/DCC -> generate GPU
Monte Carlo training rows for the basket-option call surrogate.

Uses the internals in src/basket_pricing/basket_training.py. Does NOT touch
src/basket_pricing/basket.py (the lean inference class loaded by the RL envs).

Usage:
    python scripts/generate_basket_data.py \
        --data_file src/basket_pricing/artifacts/basket_training_data_50k_paths.csv \
        --metadata_file src/basket_pricing/artifacts/basket_system_metadata.pkl \
        --n_samples 100000 --n_paths 100000 --checkpoint_dir checkpoints
"""
import argparse
import pickle
from datetime import datetime

import numpy as np

from src.basket_pricing.basket_training import (
    BasketOptionValuationSystem,
    fetch_yahoo_historical_returns,
    fetch_yahoo_spot_prices,
    fetch_daily_risk_free_rate,
    generate_training_data_dcc,
)

ASSET_NAMES = ["JPM", "BAC", "WFC", "C"]


def main():
    parser = argparse.ArgumentParser(description="Generate basket surrogate training data")
    parser.add_argument("--data_file", type=str,
                         default="src/basket_pricing/artifacts/basket_training_data_50k_paths.csv")
    parser.add_argument("--metadata_file", type=str,
                         default="src/basket_pricing/artifacts/basket_system_metadata.pkl")
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--n_paths", type=int, default=100_000)
    parser.add_argument("--years_back", type=int, default=10)
    parser.add_argument("--end_date", type=str, default="2025-12-20",
                         help="YYYY-MM-DD, historical-data cutoff")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    print(f"Assets: {ASSET_NAMES}")

    # Step 1: historical returns for DCC calibration
    print("\nStep 1: fetching historical returns...")
    historical_returns, ret_index = fetch_yahoo_historical_returns(
        ASSET_NAMES, end_date, years_back=args.years_back
    )
    print(f"  returns shape: {historical_returns.shape}  "
          f"range: {ret_index[0].date()} - {ret_index[-1].date()}")

    # Step 2: risk-free rate, aligned to the same trading-day index
    print("\nStep 2: fetching risk-free rate (^IRX)...")
    start_date = end_date.replace(year=end_date.year - args.years_back)
    r_daily = fetch_daily_risk_free_rate(start_date, end_date, ret_index)
    print(f"  mean daily CC rate: {r_daily:.8f}  (~{r_daily * 252 * 100:.3f}% annualized)")

    # Step 3: spot prices (for dollar-weighting only)
    print("\nStep 3: fetching spot prices...")
    spot_prices = fetch_yahoo_spot_prices(ASSET_NAMES, end_date)
    for name, px in zip(ASSET_NAMES, spot_prices):
        print(f"  {name}: ${px:.2f}")

    # Step 4: GARCH calibration + basket weights
    print("\nStep 4: calibrating GARCH models...")
    system = BasketOptionValuationSystem(n_assets=len(ASSET_NAMES), weight_type="dollar")
    system.calibrate_system(asset_names=ASSET_NAMES, spot_prices=spot_prices)

    # Step 5: DCC calibration
    print("\nStep 5: calibrating DCC parameters...")
    garch_params_dict = {i: system.garch_models[i].get_params() for i in range(len(ASSET_NAMES))}
    system.calibrate_dcc(historical_returns, garch_params_dict, r_daily, asset_names=ASSET_NAMES)

    # Step 6: GPU Monte Carlo data generation
    print("\nStep 6: generating training data on GPU...")
    X, call_prices = generate_training_data_dcc(
        raw_weights=system.weights,
        garch_models=system.garch_models,
        Q_bar=system.Q_bar,
        dcc_alpha=system.dcc_alpha,
        dcc_beta=system.dcc_beta,
        nu_base=system.nu,
        r=r_daily,
        n_samples=args.n_samples,
        n_paths=args.n_paths,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Step 7: persist data + metadata
    print("\nStep 7: saving training data + metadata...")
    import os
    import pandas as pd

    n_assets = len(ASSET_NAMES)
    n_corr = n_assets * (n_assets - 1) // 2
    column_names = (
        [f"S{i}" for i in range(n_assets)] + [f"h{i}" for i in range(n_assets)]
        + ["T", "log_moneyness"] + [f"R_t_{i}" for i in range(n_corr)]
    )
    df = pd.DataFrame(X, columns=column_names)
    df["call_price"] = call_prices
    os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
    df.to_csv(args.data_file, index=False)
    print(f"  data -> {args.data_file}  shape={df.shape}")
    print(f"  call price stats: min={call_prices.min():.4f} max={call_prices.max():.2f} "
          f"mean={call_prices.mean():.4f}")

    metadata = {
        "n_assets": n_assets,
        "asset_names": ASSET_NAMES,
        "weights": system.weights,
        "weight_type": "dollar",
        "S0_initial": system.S0_initial,
        "price_normalization_factors": system.price_normalization_factors,
        "garch_params": [m.get_params() for m in system.garch_models],
        "Q_bar": system.Q_bar,
        "dcc_alpha": system.dcc_alpha,
        "dcc_beta": system.dcc_beta,
        "nu": system.nu,
        "r_daily": r_daily,
        "input_dim": X.shape[1],
    }
    os.makedirs(os.path.dirname(args.metadata_file), exist_ok=True)
    with open(args.metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  metadata -> {args.metadata_file}")
    print("\nDone. Next: python scripts/train_basket_surrogate.py")


if __name__ == "__main__":
    main()
