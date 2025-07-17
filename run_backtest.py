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
from nautilus_trader.config import BacktestRunConfig, BacktestEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import Venue


from algos.strategy import GenericLongShortStrategy
from algos.engine.hparam_tuner import OptunaHparamsTuner, split_hparam_cfg
from algos.engine.data_loader import CsvBarLoader
from models.umi import UMIModel


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
            clock_fn    = clock_fn,  # CHANGED: Added clock function
            bar_type    = loader.bar_type,  # CHANGED: Added bar_type
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
    
    # CHANGED: Updated Nautilus Trader configuration
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            strategies=[GenericLongShortStrategy(config=cfg)],
        ),
        data_configs=[
            BacktestDataConfig(
                catalog_path=":memory:",
                start_time=start,
                end_time=end,
                instrument_ids=[],  # Will be populated by strategy
            )
        ],
        venues=[Venue("SIM")],
        run_config=BacktestRunConfig(
            engine_id="001",
            run_id="001",
        ),
    )

    # --- Load data into engine ---------------------------------------
    # CHANGED: Load data using the loader
    loader = CsvBarLoader(Path(cfg["data_dir"]), freq=cfg["freq"])
    for bar in loader.bar_iterator():
        engine.add_data(bar)

    # --- run ----------------------------------------------------------
    print("[run_backtest] starting")
    engine.run()
    print("[run_backtest] finished")

    # --- Generate reports --------------------------------------------
    engine.generate_report()


if __name__ == "__main__":
    main()