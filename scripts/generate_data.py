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
WHICH_CHOICES = ["fixed", "stochastic", "fixed_stochastic_policy"]

DATA_DIR = globals().get("DATA_DIR")
WHICH    = globals().get("WHICH")

if DATA_DIR is None or WHICH is None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--which", default=None, choices=WHICH_CHOICES)
    args, _ = parser.parse_known_args()
    if DATA_DIR is None:
        DATA_DIR = args.data_dir
    if WHICH is None:
        WHICH = args.which

RUN_FIXED                   = WHICH is None or WHICH == "fixed"
RUN_STOCHASTIC               = WHICH is None or WHICH == "stochastic"
RUN_FIXED_STOCHASTIC_POLICY  = WHICH is None or WHICH == "fixed_stochastic_policy"

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


def save_simulation(out, save_dir, save_paths=True, Q_paths=None):
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
        if Q_paths is not None:
            np.save(f"{save_dir}/Q_paths.npy", Q_paths.cpu().numpy())

    pnl_np = terminal_pnl.cpu().numpy()
    print(f"    saved to {save_dir}")
    print(f"    terminal P&L: mean={pnl_np.mean():.4f}  CVaR95={-np.percentile(pnl_np, 5):.4f}")


def generate_fixed_paths():
    print(f"\n{'='*60}")
    print(f"  Generating FIXED paths (S0=100 all assets, h0=unconditional var)")
    print(f"  seed={SEED}  batch={BATCH_SIZE}")
    print(f"{'='*60}")
    S, h, R, Q = env._generate_all_paths_gpu(
        BATCH_SIZE, seed=SEED, randomize_init=False, nested=True
    )
    with torch.no_grad():
        d = env._price_all_episodes_batched(S, h, R, chunk_size=16_384)
    return S, h, R, Q, d


def generate_stochastic_paths():
    print(f"\n{'='*60}")
    print(f"  Generating STOCHASTIC paths")
    print(f"  seed={SEED}  batch={BATCH_SIZE}")
    print(f"{'='*60}")
    S, h, R, Q = env._generate_all_paths_gpu(
        BATCH_SIZE, seed=SEED, randomize_init=True, nested=True
    )
    with torch.no_grad():
        d = env._price_all_episodes_batched(S, h, R, chunk_size=16_384)
    return S, h, R, Q, d


if __name__ == "__main__":
    print(f"  running WHICH={WHICH!r} (None means: run all three sets)")

    need_fixed = RUN_FIXED or RUN_FIXED_STOCHASTIC_POLICY
    if need_fixed:
        fixed_S, fixed_h, fixed_R, fixed_Q, fixed_d = generate_fixed_paths()
        fixed_S_np, fixed_h_np = fixed_S.cpu().numpy(), fixed_h.cpu().numpy()
        fixed_R_np, fixed_d_np = fixed_R.cpu().numpy(), fixed_d.cpu().numpy()
        fixed_Q_np = fixed_Q.cpu().numpy()

    if RUN_FIXED:
        for alpha_label in ALPHA_LABELS:
            shared_dir = os.path.join(DATA_DIR, alpha_label, "_shared_paths")
            os.makedirs(shared_dir, exist_ok=True)
            np.save(f"{shared_dir}/S_paths.npy",      fixed_S_np)
            np.save(f"{shared_dir}/h_paths.npy",      fixed_h_np)
            np.save(f"{shared_dir}/R_paths.npy",      fixed_R_np)
            np.save(f"{shared_dir}/Q_paths.npy",      fixed_Q_np)
            np.save(f"{shared_dir}/deriv_prices.npy", fixed_d_np)
            print(f"    fixed paths saved to {shared_dir}")

            for sk in SCORING_KEYS:
                print(f"  fixed rollout {alpha_label} / {sk} (deterministic) ...")
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

    if RUN_FIXED_STOCHASTIC_POLICY:
        for alpha_label in ALPHA_LABELS:
            shared_dir = os.path.join(DATA_DIR, alpha_label, "_shared_paths")
            os.makedirs(shared_dir, exist_ok=True)

            if not RUN_FIXED:
                np.save(f"{shared_dir}/S_paths.npy",      fixed_S_np)
                np.save(f"{shared_dir}/h_paths.npy",      fixed_h_np)
                np.save(f"{shared_dir}/R_paths.npy",      fixed_R_np)
                np.save(f"{shared_dir}/Q_paths.npy",      fixed_Q_np)
                np.save(f"{shared_dir}/deriv_prices.npy", fixed_d_np)
                print(f"    fixed paths saved to {shared_dir}")

            for sk in SCORING_KEYS:
                print(f"  fixed rollout {alpha_label} / {sk} (stochastic policy, std=1e-3) ...")
                actor    = all_models[alpha_label][sk]["actor"]
                save_dir = os.path.join(shared_dir, sk, "stochastic_policy")
                torch.manual_seed(SEED)
                out = env.rollout_from_paths(
                    actor, fixed_S, fixed_h, fixed_R, fixed_d,
                    deterministic=False,
                )
                save_simulation(out, save_dir, save_paths=False)
                del out
                torch.cuda.empty_cache()

            print(f"  fixed rollout {alpha_label} / static (stochastic policy, std=1e-3) ...")
            static_actor = all_static_models[alpha_label]["actor"]
            save_dir     = os.path.join(shared_dir, "static", "stochastic_policy")
            torch.manual_seed(SEED)
            out = env.rollout_from_paths(
                static_actor, fixed_S, fixed_h, fixed_R, fixed_d,
                deterministic=False,
            )
            save_simulation(out, save_dir, save_paths=False)
            del out
            torch.cuda.empty_cache()

    if need_fixed:
        del fixed_S, fixed_h, fixed_R, fixed_Q, fixed_d
        del fixed_S_np, fixed_h_np, fixed_R_np, fixed_Q_np, fixed_d_np
        torch.cuda.empty_cache()

    if RUN_STOCHASTIC:
        for alpha_label in ALPHA_LABELS:
            stoch_S, stoch_h, stoch_R, stoch_Q, stoch_d = generate_stochastic_paths()

            alpha_dir = os.path.join(DATA_DIR, alpha_label)
            os.makedirs(alpha_dir, exist_ok=True)
            np.save(f"{alpha_dir}/S_paths.npy",      stoch_S.cpu().numpy())
            np.save(f"{alpha_dir}/h_paths.npy",      stoch_h.cpu().numpy())
            np.save(f"{alpha_dir}/R_paths.npy",      stoch_R.cpu().numpy())
            np.save(f"{alpha_dir}/Q_paths.npy",      stoch_Q.cpu().numpy())
            np.save(f"{alpha_dir}/deriv_prices.npy", stoch_d.cpu().numpy())

            for sk in SCORING_KEYS:
                print(f"  {alpha_label} / {sk} (deterministic)")
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

            del stoch_S, stoch_h, stoch_R, stoch_Q, stoch_d
            torch.cuda.empty_cache()

    print(f"\nDone. All data saved to {DATA_DIR}/")
