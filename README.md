Sapiens
=============================

Quick start
-----------

```bash
# 1) create a fresh environment
conda create -n umi python=3.10 && conda activate umi
pip install -r requirements.txt          # includes nautilus-trader, torch, ccxt

# 2) drop your *.csv files into ./data
#    └── IMPORTANT: place the 10-year Treasury-yield file at data/DGS10.csv

# 3) edit configs/umi.yaml to taste (freq, costs, risk, …)

# 4) single-machine back-test
python run_backtest.py --cfg configs/umi.yaml

# 5) cluster back-test (SLURM example)
sbatch slurm/backtest.sbatch
```

## Project Layout

```
trading/                         ← project root
├── data/                        ← your raw *.csv files (including DGS10.csv)
│   └── …                        (unchanged – loader stays CSV-only)
├── configs/
│   └── umi.yaml                 ← same keys as before + model.name
├── algos/                       ← all strategy/ML code lives here
│   ├── engine/                  ← re-usable, model-agnostic helpers
│   │   ├── __init__.py
│   │   ├── data_loader.py       # CsvBarLoader (dynamic-universe aware)
│   │   ├── execution.py         # CommissionModelBps, SlippageModelBps
│   │   ├── optimizer.py         # MaxSharpeRatioOptimizer (rf-aware)
│   │   ├── analyzers.py         # EquityCurve, Orders, PnL split
│   │   └── utils.py             # beta calc, timer helpers, etc.
│   ├── models/
│   │   ├── umi/                 ← **copy your whole UMIModel tree here**
│   │   │   ├── __init__.py
│   │   │   └── …                (no edits needed)
│   │   └── __init__.py          # future models live next to /umi
│   ├── strategy.py              # Strategy subclass – loads model by name
│   └── __init__.py
├── run_backtest.py              # CLI wrapper (engine + strategy)
├── slurm/
│   └── backtest.sbatch          # HPC template (array shards)
└── README.md
```

## Live trading 
- Connectors for CCXT (crypto spot) and Interactive Brokers are initialised only when the engine mode is `LIVE`

- Hyper-parameter tuning is normally disabled live (tune_hparams: false);
SQLite storage remains but is unused.

## Risk controls

| YAML key                 | Purpose                                           |    
| ------------------------ | ------------------------------------------------- | 
| `risk.drawdown_pct`      | Hard kill-switch vs. NAV peak (liquidate)         | 
| `risk.trailing_stop_pct` | Per-position trailing stop (high-water/low-water) |    
| `risk.target_vol_annual` | 𝜎-targeting (weights scaled by realised vol)      | 
| `risk.max_weight_abs`    | Hard cap on absolute position size as a fraction of NAV:  abs(wᵢ)  ≤ x% NAV                |   
| `risk.max_weight_rel`    | Hard cap on relative gross exposure:  max( wᵢ ) ≤ pct of gross exp.      |
| Liquidity caps           | ≤ `max_adv_pct` × ADV per rebalance               |

## Execution
- Orders are split into execution.twap_slices child slices (TWAP).

- Up to execution.parallel_orders orders are sent concurrently via asyncio.

- Commission/slippage models: fee_bps, spread_bps in YAML.


## Analysis notebook
After a run finishes, open notebooks/umi_analysis.ipynb (or .py) and run all cells. It will automatically detect the newest run directory and plot:

- saved hyper-parameters

- parameter inventory & model size

- Optuna trial history (if any)

- training-loss curves

- prediction vs. truth for a user-chosen ticker

- cumulative NAV curve from equity_curve.csv


## Treasury yield (risk-free)
A file named DGS10.csv (FRED 10-year constant-maturity Treasury yield) is parsed automatically, forward-filled on the business-day grid, and never counted as a tradeable instrument. The latest value feeds the Maximum-Sharpe-Ratio optimiser as risk_free.


## Additional Notes ....
6 What max_weight_abs and max_weight_rel mean in the code
max_weight_abs
Hard cap on absolute position size as a fraction of NAV.
Example: 0.03 ⇒ never let |w_i| > 3 % of portfolio value.

max_weight_rel
Hard cap on relative gross exposure.
After gross scaling, if any single name would exceed
max_weight_rel × gross_leverage, all weights are rescaled so the
biggest one sits exactly at that threshold.

Combined with the existing ADV cap they guarantee:

Position not too large versus portfolio (max_weight_abs, max_weight_rel)

Trade not too large versus available liquidity (adv_lookback, max_adv_pct)


