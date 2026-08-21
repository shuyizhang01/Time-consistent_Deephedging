This repository contains the Python code to replicate the experiments in the paper *Insights on Time-consistent Deep Hedging under Elicitable Dynamic Risk Measures* by Shuyi Zhang `shuyizhang834@gmail.com` and Frederic Godin `frederic.godin@concordia.ca`.
---

## Overview

We study a deep hedging framework in which the agent minimizes risk through a time-consistent dynamic CVaR objective rather than a static one. The critic estimates dynamic risk using the conditional elicitability of CVaR, which, in the literature, is shown to improve training efficiency. The actor is updated via policy gradient against the risk estimated by the critic. We apply this framework to a basket option on 4 underlying assets evolving under a DCC-GARCH model across a daily, 1-year time grid, and compare the trained dynamic risk models against a static risk model across 4 risk thresholds (92.5%, 95%, 97.5%, 99%) and 6 scoring function choices under the elicitability approach for the dynamic risk models.


## Repository Structure

```
.
├── DynamicRisk/          # Dynamic risk models
├── StaticRisk/           # Static risk models
├── NestedCritics/        # Nested simulation estimation of dynamic risk
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
