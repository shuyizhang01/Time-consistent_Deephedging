"""
scripts/generate_data.py

One-time data generation script. Run this before any plotting.

Usage
-----
    python scripts/generate_data.py --data-dir "data_path"

Or, run via runpy.run_path(..., init_globals={"DATA_DIR": ...})
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.envs.HedgingEnv        import HedgingEnv
from src.envs.DCCGARCH           import DCCGARCHSimulator
from src.agents.DynamicRiskModel import Actor

BATCH_SIZE   = 10000
SEED         = 42
SCORING_KEYS = ["arcsin", "arcsinh", "arctan", "log", "power03", "rational"]
ALPHA_LABELS = ["alpha925", "alpha95", "alpha975", "alpha99"]
FIXED_ALPHA  = "alpha95"

DEFAULT_DATA_DIR = "content/data"

DATA_DIR = globals().get("DATA_DIR")

if DATA_DIR is None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args, _ = parser.parse_known_args()
    DATA_DIR = args.data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for alpha in all_models:
    for sk in all_models[alpha]:
        all_models[alpha][sk]["actor"] = (
            all_models[alpha][sk]["actor"]
            .to(device)
            .eval()
        )

for alpha in all_static_models:
    all_static_models[alpha]["actor"] = (
        all_static_models[alpha]["actor"]
        .to(device)
        .eval()
    )


def save_simulation(out, save_dir, save_paths=True):
    os.makedirs(save_dir, exist_ok=True)

    (states, actions, log_probs, costs, next_states, dones,
     portval, deriv_prices, PnL, terminal_pnl,
     S_paths, h_paths, R_paths) = out

    np.save(f"{save_dir}/states.npy",           states.cpu().numpy())
    np.save(f"{save_dir}/actions.npy",           actions.cpu().numpy())
    np.save(f"{save_dir}/portfolio_values.npy",  portval.cpu().numpy())
    np.save(f"{save_dir}/terminal_pnl.npy",      terminal_pnl.cpu().numpy())

    if save_paths:
        np.save(f"{save_dir}/deriv_prices.npy", deriv_prices.cpu().numpy())
        np.save(f"{save_dir}/S_paths.npy",      S_paths.cpu().numpy())
        np.save(f"{save_dir}/h_paths.npy",      h_paths.cpu().numpy())
        np.save(f"{save_dir}/R_paths.npy",      R_paths.cpu().numpy())

    pnl_np = terminal_pnl.cpu().numpy()
    print(f"    saved to {save_dir}")
    print(f"    terminal P&L: mean={pnl_np.mean():.4f}  CVaR95={-np.percentile(pnl_np, 5):.4f}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Generating FIXED paths (S0=100 all assets, h0=unconditional var)")
    print(f"  seed={SEED}  batch={BATCH_SIZE}")
    print(f"{'='*60}")

    fixed_S, fixed_h, fixed_R, _ = env._generate_all_paths_gpu(
        BATCH_SIZE, seed=SEED, randomize_init=False
    )
    with torch.no_grad():
        fixed_d = env._price_all_episodes_batched(
            fixed_S, fixed_h, fixed_R, chunk_size=16_384
        )

    fixed_S_np, fixed_h_np = fixed_S.cpu().numpy(), fixed_h.cpu().numpy()
    fixed_R_np, fixed_d_np = fixed_R.cpu().numpy(), fixed_d.cpu().numpy()

    for alpha_label in ALPHA_LABELS:
        shared_dir = os.path.join(DATA_DIR, alpha_label, "_shared_paths")
        os.makedirs(shared_dir, exist_ok=True)
        np.save(f"{shared_dir}/S_paths.npy",      fixed_S_np)
        np.save(f"{shared_dir}/h_paths.npy",      fixed_h_np)
        np.save(f"{shared_dir}/R_paths.npy",      fixed_R_np)
        np.save(f"{shared_dir}/deriv_prices.npy", fixed_d_np)
        print(f"    fixed paths saved to {shared_dir}")

        for sk in SCORING_KEYS:
            print(f"  fixed rollout {alpha_label} / {sk} ...")
            actor    = all_models[alpha_label][sk]["actor"]
            save_dir = os.path.join(shared_dir, sk)
            out = env.rollout_from_paths(
                actor, fixed_S, fixed_h, fixed_R, fixed_d,
                deterministic=True,
            )
            save_simulation(out, save_dir, save_paths=False)
            del out
            torch.cuda.empty_cache()

        print(f"  fixed rollout {alpha_label} / static ...")
        static_actor = all_static_models[alpha_label]["actor"]
        save_dir     = os.path.join(shared_dir, "static")
        out = env.rollout_from_paths(
            static_actor, fixed_S, fixed_h, fixed_R, fixed_d,
            deterministic=True,
        )
        save_simulation(out, save_dir, save_paths=False)
        del out
        torch.cuda.empty_cache()

    del fixed_S, fixed_h, fixed_R, fixed_d, fixed_S_np, fixed_h_np, fixed_R_np, fixed_d_np
    torch.cuda.empty_cache()

    for alpha_label in ALPHA_LABELS:
        print(f"\n{'='*60}")
        print(f"  Generating STOCHASTIC paths")
        print(f"  alpha={alpha_label}  seed={SEED}  batch={BATCH_SIZE}")
        print(f"{'='*60}")

        stoch_S, stoch_h, stoch_R, _ = env._generate_all_paths_gpu(
            BATCH_SIZE, seed=SEED, randomize_init=True
        )
        with torch.no_grad():
            stoch_d = env._price_all_episodes_batched(
                stoch_S, stoch_h, stoch_R, chunk_size=16_384
            )

        alpha_dir = os.path.join(DATA_DIR, alpha_label)
        os.makedirs(alpha_dir, exist_ok=True)
        np.save(f"{alpha_dir}/S_paths.npy",      stoch_S.cpu().numpy())
        np.save(f"{alpha_dir}/h_paths.npy",      stoch_h.cpu().numpy())
        np.save(f"{alpha_dir}/R_paths.npy",      stoch_R.cpu().numpy())
        np.save(f"{alpha_dir}/deriv_prices.npy", stoch_d.cpu().numpy())

        for sk in SCORING_KEYS:
            print(f"  {alpha_label} / {sk}")
            actor    = all_models[alpha_label][sk]["actor"]
            save_dir = os.path.join(DATA_DIR, alpha_label, sk)
            out = env.rollout_from_paths(actor, stoch_S, stoch_h, stoch_R, stoch_d, deterministic=True)
            save_simulation(out, save_dir, save_paths=False)
            del out
            torch.cuda.empty_cache()

        print(f"\n  {alpha_label} / static")
        static_actor = all_static_models[alpha_label]["actor"]
        save_dir     = os.path.join(DATA_DIR, alpha_label, "static")
        out = env.rollout_from_paths(
            static_actor, stoch_S, stoch_h, stoch_R, stoch_d,
            deterministic=True,
        )
        save_simulation(out, save_dir, save_paths=False)
        del out
        torch.cuda.empty_cache()

        del stoch_S, stoch_h, stoch_R, stoch_d
        torch.cuda.empty_cache()

    print(f"\nDone. All data saved to {DATA_DIR}/")
