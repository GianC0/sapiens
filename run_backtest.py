#!/usr/bin/env python3
"""
CLI helper — spins up a Nautilus Trader BacktestEngine and runs UMIStrategy.

Examples
--------
```bash
python run_backtest.py --cfg configs/umi.yaml --gpus 4 --cpus 32
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.common.component import TestClock
from nautilus_trader.persistence.catalog import ParquetDataCatalog as Catalog
from algos.engine.hparam_tuner import OptunaHparamsTuner
from algos.engine.hparam_tuner import split_hparam_cfg
from algos.engine.data_loader import CsvBarLoader

from algos.strategy import UMIStrategy



def main():

    # loads configuration file
    cfg_path = Path("configs/umi.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    # pulls dirs from config
    logs_root     = Path(cfg.get("logs_root", "logs"))
    model_name    = cfg["model_name"]
    default_root  = logs_root / "models" / model_name    # ← “regular” training


    # pulls defaults and hparams search space fromn cfg file
    defaults, search_space = split_hparam_cfg(cfg["hparams"])

     # -----------------------------------------
     # OPTIONAL Hyper-parameter optimisation
     # -----------------------------------------
    if cfg["training"]["tune_hparams"]:
        print("[HPO] Optuna search started …")

        loader     = CsvBarLoader(Path(cfg["data_dir"]), freq=cfg["freq"])
        data_dict  = loader._frames

        fixed = dict(
            freq        = cfg["freq"],
            feature_dim = len(next(iter(data_dict.values())).columns),
            window_len  = cfg["window_len"],
            pred_len    = cfg["pred_len"],
            end_train   = cfg["train_end"],
            end_valid   = cfg["valid_end"],
            batch_size  = cfg["training"]["batch_size"],
            retrain_delta = int(cfg["training"]["retrain_delta"]),
            dynamic_universe_mult = cfg["dynamic_universe_mult"],
            data_dir    = Path(cfg["data_dir"]),
            model_dir   = logs_root / "optuna" / "models" / model_name / "trials",
            **defaults,
        )

        tuner = OptunaHparamsTuner(
            logs_root   = logs_root,
            model_name  = model_name,
            model_cls   = UMIModel,
            train_dict  = data_dict,
            eval_fn     = quick_score,
            search_space= search_space,
            fixed_kwargs= fixed,
            fit_kwargs  = dict(warm_start=False,
                            n_epochs=cfg["training"]["n_epochs"]),
            n_trials    = cfg["training"]["n_trials"],
            study_name  = cfg["training"]["study_name"],
        )
        best = tuner.optimize()       # -------------- returns dict!
        cfg["hparams"]        = {**defaults, **best["params"]}
        cfg["pretrained_dir"] = str(best["model_dir"])    # ← used later
    else:
        cfg["hparams"] = defaults


    # --- engine bootstrap --------------------------------------------
    start = pd.Timestamp(cfg["backtest_start"], tz="UTC")
    end = pd.Timestamp(cfg["backtest_end"],   tz="UTC")
    currency = str(cfg["currency"])
    starting_cash = float(cfg.get("initial_cash", 10_000))
    clock = TestClock(start=start, end=end)
    catalog = Catalog(":memory:")

    engine = BacktestEngine(clock=clock, catalog=catalog, base_currency=currency, starting_cash=starting_cash )

    strat = UMIStrategy(clock, catalog, cfg_path)
    engine.add_strategy(strat)

    # --- run ----------------------------------------------------------
    print("[run_backtest] starting")
    engine.run()
    print("[run_backtest] finished")


if __name__ == "__main__":
    main()
