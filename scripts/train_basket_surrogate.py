"""
scripts/train_basket_surrogate.py
==================================
One-time pipeline: load the CSV produced by generate_basket_data.py + its
metadata pickle, train the call-price surrogate net, and save the final
system artifact consumed by src/basket_pricing/basket.py at inference time.

Usage:
    python scripts/train_basket_surrogate.py \
        --data_file src/basket_pricing/artifacts/basket_training_data_50k_paths.csv \
        --metadata_file src/basket_pricing/artifacts/basket_system_metadata.pkl \
        --model_file src/basket_pricing/artifacts/basket_system_dcc_surrogate.pkl \
        --epochs 1000 --batch_size 512
"""
import argparse
import pickle

import numpy as np
import pandas as pd

from src.basket_pricing.basket_training import BasketOptionValuationSystem


def main():
    parser = argparse.ArgumentParser(description="Train the basket call-price surrogate")
    parser.add_argument("--data_file", type=str,
                         default="src/basket_pricing/artifacts/basket_training_data_50k_paths.csv")
    parser.add_argument("--metadata_file", type=str,
                         default="src/basket_pricing/artifacts/basket_system_metadata.pkl")
    parser.add_argument("--model_file", type=str,
                         default="src/basket_pricing/artifacts/basket_system_dcc_surrogate.pkl")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    # Step 1: metadata (calibrated GARCH/DCC params + basket weights)
    print("Step 1: loading system metadata...")
    with open(args.metadata_file, "rb") as f:
        metadata = pickle.load(f)

    # Step 2: reconstruct the (untrained) system from metadata
    print("Step 2: reconstructing valuation system...")
    system = BasketOptionValuationSystem(n_assets=metadata["n_assets"], weight_type=metadata["weight_type"])
    system.asset_names = metadata["asset_names"]
    system.weights = metadata["weights"]
    system.S0_initial = metadata["S0_initial"]
    system.price_normalization_factors = metadata["price_normalization_factors"]
    system.Q_bar = metadata["Q_bar"]
    system.dcc_alpha = metadata["dcc_alpha"]
    system.dcc_beta = metadata["dcc_beta"]
    system.nu = metadata["nu"]
    for i, params in enumerate(metadata["garch_params"]):
        model = system.garch_models[i]
        model.omega, model.alpha, model.beta = params["omega"], params["alpha"], params["beta"]
        model.gamma, model.lambda_ = params["gamma"], params["lambda"]
        model.h_unconditional = params["h_unconditional"]
    print(f"  reconstructed with {metadata['n_assets']} assets, weights={system.weights}")

    # Step 3: load training data, recover time-value target (call - discounted intrinsic)
    print(f"\nStep 3: loading training data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    call_prices = df["call_price"].values
    X = df.drop(columns=["call_price"]).values
    print(f"  {len(df)} samples, {X.shape[1]} features")
    print(f"  call price range: ${call_prices.min():.4f} - ${call_prices.max():.2f}")

    r_daily = metadata["r_daily"]
    S0_cols = [f"S{i}" for i in range(metadata["n_assets"])]
    basket_now = (df[S0_cols].values * 100.0) @ system.weights  # undo /100 moneyness normalization
    T_days = np.maximum(1, (df["T"].values * 252).astype(int))
    disc = np.exp(-r_daily * T_days)
    discounted_intrinsic = np.maximum(basket_now - 100.0 * disc, 0.0)  # K == 100 for every row
    time_value = np.maximum(call_prices - discounted_intrinsic, 0.0)

    # Step 4: train the surrogate net
    print("\nStep 4: training the call surrogate...")
    system.call_model, system.scaler_call = system.train_surrogate(
        X, time_value, epochs=args.epochs, batch_size=args.batch_size,
    )
    system.training_data = {"input_dim": X.shape[1], "n_samples": len(X)}

    # Step 5: save the trained system (this is the artifact basket.py loads)
    print(f"\nStep 5: saving trained system to {args.model_file}...")
    import os
    os.makedirs(os.path.dirname(args.model_file), exist_ok=True)
    system.save(args.model_file)
    print("\nDone.")


if __name__ == "__main__":
    main()


