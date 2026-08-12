# Human Activity Recognition using Smartphone Sensor Data
## Optimizer Comparison with Feed-Forward Neural Networks

## Setup
1. Place dataset in `data/UCI HAR Dataset/`
2. `pip install -r requirements.txt`
3. `python src/main.py`

## What it does
Trains a fixed feed-forward architecture (Dense128-Dense64-Dense6) with 9
different optimization algorithm configurations across three groups:
- Part I: Batch GD, Mini-batch SGD, Stochastic GD
- Part II: GD, GD+Momentum, GD+Nesterov Momentum
- Part III: AdaGrad, RMSProp, Adam

Outputs: results/metrics.csv, results/figures/*.png (loss curves, confusion
matrices, group comparisons a-d, final ranking).


every time you open a new terminal to work on this project, you must 
re-run venv\Scripts\activate first — otherwise Windows defaults back to your 
3.14 install and you'll get the same error.
