## This repository contains the Python code to replicate the experiments in the paper [*Insights on Time-consistent Deep Hedging under Elicitable Dynamic Risk Measures*](https://arxiv.org/abs/2609.02014) by Shuyi Zhang `szhang48@terpmail.umd.edu` and Frederic Godin `frederic.godin@concordia.ca`.
## Overview

We study a deep hedging framework in which the hedging agent offsets the risk of a European basket call option in a time-consistent fashion by minimizing the dynamic CVaR of costs in a high-dimensional setting. We employ a conditionally elicitable actor-critic RL algorithm and exploit the translation invariance of spectral risk measures to shift the estimation of dynamic risk into more desirable regions of the scoring function for CVaR. We apply this framework to a basket option on 4 underlying assets evolving under a DCC-GARCH(1, 1) model across 1-year time grid with daily increments, and compare the trained dynamic risk (time-consistent) models against a static risk (precommitment) model across 4 confidence levels (92.5%, 95%, 97.5%, 99%) and 6 possible strictly-consistent scoring functions. 

Novel findings:
(1) We demonstrate that the conditional elicitability actor-critic RL framework can derive effective
dynamic risk hedging policies in higher-dimensional settings, an aspect that has received limited
attention in the existing literature. 
(2) We find that high tail-sensitivity scoring functions such as the
canonical logarithmic or fractional-power characterizations yield RL-based dynamic risk estimates
that are the most accurate relative to critics trained using a nested approach benchmark when
sufficient extreme-tail samples are available at high confidence levels. 
(3) Conversely, scoring functions producing saturating gradients can provide enhanced performance and improved stability for
extreme risk levels for which tail observations are very sparse. Our results further reveal that while
a precommitment static risk objective is the optimal choice when mitigating terminal hedging risk
of basket options, deploying a dynamic risk objective yields a hedging policy with consistently
lower hedging risk over shorter horizons, and a monotonic decrease in risk as maturity shortens.
This is in contrast to results from the static risk policy, whose hedging risk increases as maturity decreases.

## Repository Structure

```
.
├── DynamicRisk/          # Dynamic risk models
├── StaticRisk/           # Static risk models
├── src/                  # Shared source code (neural network architecture, environments, hedging pipeline)
├── cfgs/                 # Configuration files
├── scripts/              # Scripts to launch training runs
├── logs/                 # Example output logs
├── Figures_Tables.ipynb  # Figures and tables from the paper
└── requirements.txt

## Running the Experiments

From the repository root, install the required dependencies:

```bash
pip install -r requirements.txt
```

First, navigate to the cloned repository directory. Then run the desired experiment using the provided configuration files as follows:

```bash
PYTHONPATH="path_to_cloned_repo" \
     python scripts/train_dynamic.py \
    --config cfgs/configDynamicRisk.yaml \
    --device cuda
```

```bash
PYTHONPATH="path_to_cloned_repo" \
     python scripts/train_static.py \
    --config cfgs/configStaticRisk.yaml \
    --device cuda
```

The risk level, scoring function, and other experiment settings can be modified in the corresponding configuration files in `cfgs/`. The figures/plots in the paper and the code to generate them can be seen in figures-tables.ipynb. Training logs for the dynamic risk models can be seen in `training_run_logs/`, and the dynamic risk and static risk model weights can be seen in `dynamicriskmodels/` and `staticriskmodels/` respectively.
