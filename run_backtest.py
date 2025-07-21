#!/usr/bin/env python3
"""
CLI helper — spins up a Nautilus Trader BacktestEngine and runs GenericLongShortStrategy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.data import BacktestDataConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestRunConfig, BacktestEngineConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model import Bar


from algos.strategy import BacktestLongShortStrategy
from algos.engine.hparam_tuner import OptunaHparamsTuner, split_hparam_cfg
from algos.engine.data_loader import CsvBarLoader
from models.umi import UMIModel
from model.utils import freq2barspec


def main():
    # loads configuration file
    cfg_path = Path("configs/config.yaml")  # CHANGED: Fixed config filename
    cfg = yaml.safe_load(cfg_path.read_text())

    # pulls dirs from config
    logs_root     = Path(cfg.get("logs_dir", "logs"))  # CHANGED: Fixed key name
    model_name    = cfg["model_name"]
    
    # pulls defaults and hparams search space from cfg file
    defaults, search_space = split_hparam_cfg(cfg["hparams"])

    # -----------------------------------------
    # OPTIONAL Hyper-parameter optimisation
    # -----------------------------------------
    if cfg["training"]["tune_hparams"]:
        print("[HPO] Optuna search started …")

        loader     = CsvBarLoader(Path(cfg["data_dir"]), freq=cfg["freq"])
        data_dict  = loader._frames

        # CHANGED: Create clock function to pass to model
        def clock_fn():
            return pd.Timestamp.utcnow()

        fixed = dict(
            freq        = cfg["freq"],
            feature_dim = len(next(iter(data_dict.values())).columns),
            window_len  = cfg["window_len"],
            pred_len    = cfg["pred_len"],
            end_train   = cfg["train_end"],
            end_valid   = cfg["valid_end"],
            batch_size  = cfg["training"]["batch_size"],
            retrain_delta = pd.DateOffset(days=cfg["training"]["retrain_delta"]),
            dynamic_universe_mult = cfg["dynamic_universe_mult"],
            data_dir    = Path(cfg["data_dir"]),
            clock_fn    = clock_fn,  # TODO: Added clock function
            bar_spec    = , freq2barspec(cfg["freq"]) # TODO: doublecheck if necessary
            **defaults,
        )

        # CHANGED: Removed eval_fn parameter as it's not needed
        tuner = OptunaHparamsTuner(
            model_name  = model_name,
            logs_dir    = logs_root,
            model_cls   = UMIModel,
            train_dict  = data_dict,
            search_space= search_space,
            defaults    = defaults,  # CHANGED: Added defaults parameter
            fixed_kwargs= fixed,
            fit_kwargs  = dict(n_epochs=cfg["training"]["n_epochs"]),
            study_name  = f"{model_name}_hpo",  # CHANGED: Added study name
            n_trials    = cfg["training"]["n_trials"],
        )
        best = tuner.optimize()
        cfg["hparams"] = {**defaults, **best["params"]}
    else:
        cfg["hparams"] = defaults

    # --- engine bootstrap --------------------------------------------
    start = pd.Timestamp(cfg["backtest_start"], tz="UTC")
    end = pd.Timestamp(cfg["backtest_end"], tz="UTC")
    
    
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            strategies=[GenericLongShortStrategy(config=cfg)],
        ),
        data_configs=[
            BacktestDataConfig(
                catalog_path=":memory:",
                data_cls= Bar
                start_time=start,
                end_time=end,
                bar_spec=freq2barspec(cfg["freq"])
            )
        ],
        venues=[Venue(
            name="SIM",
            book_type="L1_MBP"     # bars are inluded in L1 market-by-price
            account_type="CASH",
            base_currency=cfg["currency"],
            starting_balances=str(cfg["initial_cash"])+" "+str(cfg["currency"]),
            bar_adaptive_high_low_ordering=False,  # Enable adaptive ordering of High/Low bar prices
            )],
        run_config=BacktestRunConfig(
            engine_id="001",
            run_id="001",
        ),
        cache=CacheConfig(
        #tick_capacity=10_000,  # Store last 10,000 ticks per instrument
        bar_capacity=cfg["engine"]["cache"]["bar_capacity"],    # Store last 5,000 bars per bar type: 5000 days ~ 13.5 years
        ),
        fill_model=FillModel(
            prob_fill_on_limit=0.2,    # Chance a limit order fills when price matches (applied to bars/trades/quotes + L1/L2/L3 order book)
            prob_slippage=0.5,         # Chance of 1-tick slippage (applied to bars/trades/quotes + L1 order book only)
            random_seed=None,          # Optional: Set for reproducible results
        )
    )

    # --- Load data into engine ---------------------------------------
    # loader = CsvBarLoader(Path(cfg["data_dir"]), freq=cfg["freq"])
    # for bar in loader.bar_iterator():
    #     engine.add_data(bar)

    # --- run ----------------------------------------------------------
    print("[run_backtest] starting")
    engine.run()
    print("[run_backtest] finished")

    # --- Generate reports --------------------------------------------
    engine.generate_report()


if __name__ == "__main__":
    main()