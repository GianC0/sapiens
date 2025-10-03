"""
Backtest Hyperparameter Tuner
==============================

Orchestrates two-phase hyperparameter optimization:
1. Model hyperparameters (using model.initialize())  
2. Strategy hyperparameters (using full backtest)

Each phase gets its own Optuna study and MLflow experiment.
"""

from matplotlib.pyplot import bar
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.config import BacktestDataConfig, CacheConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.backtest.node import BacktestEngineConfig, BacktestRunConfig
from nautilus_trader.backtest.config import BacktestRunConfig, BacktestVenueConfig
from nautilus_trader.model.enums import OmsType
from nautilus_trader.config import ImportableStrategyConfig, LoggingConfig
from nautilus_trader.backtest.node import BacktestNode
from pathlib import Path
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import pandas_market_calendars as market_calendars
from math import exp
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple
from unittest import loader
import optuna
from optuna.storages import RDBStorage
import pandas as pd
from sqlalchemy import Engine
import torch.nn as nn
import mlflow
from mlflow import MlflowClient
from xlrd import Book
import yaml
from datetime import datetime, time
import importlib
import logging
logger = logging.getLogger(__name__)

from algos.engine.data_loader import CsvBarLoader
from models.utils import freq2pdoffset, yaml_safe, freq2barspec



class OptunaHparamsTuner:
    """
    Two-phase hyperparameter tuner for model + strategy optimization.
    
    Phase 1: Optimize model hyperparameters using model.initialize()
    Phase 2: Optimize strategy hyperparameters using best model from Phase 1
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        run_timestamp: str,
        run_dir: Path = Path("logs/backtests/"),
        seed: int = 2025,
    ):
        """
        Initialize the backtest hyperparameter tuner.
        
        Args:
            cfg: Configuration dictionary
            run_timestamp: timestamp of this run to use for IDs and paths
            run_dir: Root directory for logs
            seed: Seed for reproducibility of hp tuning runs
        """
        # Load configuration
        self.config = cfg

        # Log Path
        self.run_timestamp = run_timestamp
        self.run_dir = run_dir
        
        
        # Parse configuration sections
        self.model_config = self.config.get('MODEL', {})
        self.strategy_config = self.config.get('STRATEGY', {})
        
        # Extract params and hparams for each component
        self.model_params = self.model_config.get('PARAMS', {})
        self.model_hparams = self.model_config.get('HPARAMS', {})
        self.strategy_params = self.strategy_config.get('PARAMS', {})
        self.strategy_hparams = self.strategy_config.get('HPARAMS', {})
        
        # Sync shared params
        self.model_params["freq"] = self.strategy_params["freq"]
        self.model_params["calendar"] = self.strategy_params["calendar"]

        # TODO: Centralized Setup correct types (Dates and offsets). Now done in specific model/strategy
        

        # Split hparams into defaults and search spaces
        self.model_defaults, self.model_search = self._split_hparam_cfg(self.model_hparams)
        self.strategy_defaults, self.strategy_search = self._split_hparam_cfg(self.strategy_hparams)
        self.seed = seed

        # --- Data Loader and Data Augmentation setup -------------------------------------------
        self.loader = CsvBarLoader(cfg=self.strategy_params, venue_name=self.strategy_params["venue_name"], columns_to_load=self.model_params["features_to_load"], adjust = self.model_params["adjust"])
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Optimization Best Parameters 
        self.best_model_params_flat = {}
        self.best_model_path = None
        self.best_strategy_params_flat = {}
        self.results = {}

        # TODO: ensure that when no optim is run, then this still works
    
    def _setup_mlflow(self):
        """Setup MLflow tracking with hierarchical experiments."""
        mlflow.set_tracking_uri("file:logs/mlruns")
        
        # Create parent experiment for this optimization run
        model_name = self.model_params.get('model_name', 'Unknown')
        strategy_name = self.strategy_params.get('strategy_name', 'Unknown')
        exp_name = f"Backtest-{model_name}-{strategy_name}"
        mlflow.set_experiment(exp_name)
        
        # Start parent run
        self.parent_run = mlflow.start_run(
            run_name=f"{self.run_timestamp}",
            nested=False
        )
        
        # Log configuration
        mlflow.log_params({
            "model_params": self.model_params,
            "strategy_params": self.strategy_params,
        })
    
    def _split_hparam_cfg(self, hp_cfg: dict) -> Tuple[dict, dict]:
        """
        Split hyperparameter config into defaults and search space.
        
        Returns:
            defaults: Dict of default values
            search_space: Dict of Optuna search configurations
        """
        defaults, search = {}, {}
        for k, v in hp_cfg.items():
            if isinstance(v, dict) and "default" in v:
                defaults[k] = v["default"]
                if "optuna" in v:
                    search[k] = v["optuna"]
            else:
                # Scalar value = fixed default
                defaults[k] = v
        return defaults, search
    
    def _suggest_params(self, trial: optuna.Trial, defaults: dict, search_space: dict) -> dict:
        """Convert search space definition to Optuna suggestions."""
        params = {}
        
        for name, default_value in defaults.items():
            if search_space and name in search_space:
                cfg = search_space[name]
                opt_t = cfg["optuna_type"]
                
                if opt_t == "low-high":
                    params[name] = trial.suggest_float(name, cfg["low"], cfg["high"])
                elif opt_t == "log_low-high":
                    params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], log=True)
                elif opt_t == "int_low-high":
                    params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
                elif opt_t == "categorical":
                    params[name] = trial.suggest_categorical(name, cfg["choices"])
                else:
                    raise ValueError(f"Unknown optuna_type '{opt_t}' for parameter '{name}'")
            else:
                # Use default value as categorical (fixed)
                params[name] = trial.suggest_categorical(name, [default_value])
        
        return params
    
    def optimize_model(self) -> Dict[str, Any]:
        """
        Phase 1: Optimize model hyperparameters.
        
        Returns:
            Dictionary with best hyperparameters and model directory
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 1: MODEL HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*60}\n")
        
        # Create MLflow experiment for model optimization
        with mlflow.start_run(
            run_name="Model_Optimization",
            nested=True,
            parent_run_id=self.parent_run.info.run_id
        ) as model_opt_run:
            
            # Setup Optuna study for model
            model_study_name = f"model_{self.model_params['model_name']}"
            model_name = self.model_params['model_name']
            model_db_path = self.run_dir / "model_hpo.db"
            storage = RDBStorage(f"sqlite:///{model_db_path}")
            
            study = optuna.create_study(
                study_name=model_study_name,
                storage=storage,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                load_if_exists=True,
            )

            # Initialize Model Class
            mod = importlib.import_module(f"models.{model_name}.{model_name}")
            ModelClass = getattr(mod, model_name, None) or getattr(mod, "Model")
            if ModelClass is None:
                raise ImportError(f"Could not find model class in models.{model_name}")

            
            # Define objective function for model
            def model_objective(trial: optuna.Trial) -> float:
                trial_hparams = self._suggest_params(trial, self.model_defaults, self.model_search)
                
                # setting train_offset and model_dir to model flatten config dictionary
                # adding train_offset here to avoid type overwite later  (pandas.offset -> str)
                train_offset = trial_hparams["train_offset"]
                model_dir = self.run_dir / "model" /  f"trial_{trial.number}"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_params_flat  = self._add_dates(self.model_params, train_offset, to_strategy=False ) # merged dictionary
                model_params_flat["model_dir"] = model_dir

                # init data timestamps for data loader
                # Initialize calendar. NOTE: calendar and freq already added un OptunaHparamTuner __init__
                
                start = model_params_flat["train_start"]
                end = model_params_flat["valid_end"]
                data_dict = self._get_data(start=start, end=end)

                # Start MLflow run for this trial
                with mlflow.start_run(
                    run_name=f"{model_name}_trial_{trial.number}",
                    nested=True,
                    parent_run_id=model_opt_run.info.run_id
                ) as trial_run:
                    # Log trial parameters
                    mlflow.log_params(trial_hparams)
                    mlflow.log_param("train_offset", train_offset)
                    mlflow.log_param("trial_number", trial.number)
                    
                    # storing the model yaml config for reproducibility
                    with open(model_dir / 'config.yaml', 'w', encoding="utf-8") as f:
                        yaml.dump(yaml_safe(model_params_flat), f, sort_keys=False)
                    mlflow.log_artifact(str(model_dir / 'config.yaml'))
                    
                    
                    # Initialize and train model TODO: ensure train data has validation set as well
                    # leave this order to make train_offset be overwritten by flat params type
                    model = ModelClass(**(trial_hparams | model_params_flat) )
                    # Validation is run automatically if valid_end is set
                    score = model.initialize(data_dict)
                    
                    # Log results
                    epochs_metrics = model._epoch_logs
                    for m in epochs_metrics:
                        epoch = m.pop("epoch")
                        mlflow.log_metrics(m, step=epoch)
                    mlflow.log_metric("best_validation_loss", score)
                    mlflow.pytorch.log_model( model , name = model_name) # type: ignore
                    
                    # Store model directory in trial
                    trial.set_user_attr("model_path", str(model_dir / "init.pt"))
                    trial.set_user_attr("best_model_params_flat", yaml_safe(trial_hparams | model_params_flat) )
                    trial.set_user_attr("best_model_hparams", trial_hparams)
                    trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                    
                    logger.info(f"  Trial {trial.number}: loss = {score:.6f}")
                    
                return score
            
            # Run optimization
            # TODO: define case when no hp tuning is needed
            logger.info(f"Running {self.model_params["n_trials"]} model trials...")

            # Handle case without optimization
            n_trials = self.model_params.get("n_trials",1)
            if self.model_params["tune_hparams"] is False or n_trials < 1:
                n_trials = 1
            study.optimize(model_objective, n_trials=n_trials)
            
            # Get best trial
            best_trial = study.best_trial
            self.best_model_params_flat = best_trial.user_attrs["best_model_params_flat"]
            # quick and dirty fix
            self.best_model_params_flat["train_offset"] = self.best_model_params_flat["train_offset"]

            self.best_model_path = Path(best_trial.user_attrs["model_path"])
            
            # Log best results
            mlflow.log_param("best_hparams", best_trial.user_attrs["best_model_hparams"] )
            
            if best_trial is None:
                raise Exception("Could not compute hp optimization of model")
            
            mlflow.log_metric("best_model_loss", best_trial.value) # type: ignore
            
            logger.info(f"\nBest model trial: {best_trial.number}")
            logger.info(f"Best model loss: {best_trial.value:.6f}")
            logger.info(f"Best model hparams: {best_trial.user_attrs["best_model_hparams"]}")
        
        return {
            "hparams": self.best_model_params_flat,
            "model_path": self.best_model_path,
        }

    def optimize_strategy(self, ) -> Dict[str, Any]:
        """
        Phase 2: Optimize strategy hyperparameters using best model.
        
        Returns:
            Dictionary with best strategy hyperparameters and performance metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 2: STRATEGY HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*60}\n")
        
        if self.best_model_params_flat is None or self.best_model_path is None:
            raise ValueError("Must run optimize_model() before optimize_strategy()")
        
        strategy_name = self.strategy_params.get('strategy_name', 'TopKStrategy')

        try:
            # Try to import from algos module
            strategy_module = importlib.import_module(f"algos.{strategy_name}")
            StrategyClass = getattr(strategy_module, strategy_name, None)
            
            if StrategyClass is None:
                # Try without redundant naming (e.g., algos.TopKStrategy.Strategy)
                StrategyClass = getattr(strategy_module, "Strategy", None)
            
            if StrategyClass is None:
                raise ImportError(f"Could not find strategy class in algos.{strategy_name}")
                
        except ImportError as e:
            logger.error(f"Failed to import strategy {strategy_name}: {e}")
            raise

        # Create MLflow experiment for strategy optimization
        with mlflow.start_run(
            run_name="Strategy_Optimization",
            nested=True,
            parent_run_id=self.parent_run.info.run_id
        ) as strategy_opt_run:
            
            # Setup Optuna study for strategy
            strategy_study_name = f"strategy_{self.strategy_params['strategy_name']}"
            strategy_db_path = self.run_dir / "strategy_hpo.db"
            storage = RDBStorage(f"sqlite:///{strategy_db_path}")

            # Parse Objectives and directions. Structure: obj:direction
            objectives, directions = zip(*self.strategy_params['objectives'].items())
            
            # TODO: ensure propoer direction depending on all the metrics
            study = optuna.create_study(
                study_name=strategy_study_name,
                storage=storage,
                directions=list(directions),  # follow the specific directions given in config for each metric
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                load_if_exists=True,
            )
            
            # Define objective function for strategy
            def strategy_objective(trial: optuna.Trial) -> Tuple:
                trial_hparams = self._suggest_params(trial, self.strategy_defaults, self.strategy_search)

                offset = self.best_model_params_flat["train_offset"]
                strategy_params_flat = self._add_dates(self.strategy_params, offset, to_strategy=True ) | trial_hparams
                strategy_params_flat["catalog_path"] = str(self.run_dir / "catalog")
                self.best_model_params_flat["train_offset"] = freq2pdoffset(offset)
                
                # Start MLflow run for this trial
                with mlflow.start_run(
                    run_name=f"strategy_trial_{trial.number}",
                    nested=True,
                    parent_run_id=strategy_opt_run.info.run_id
                ) as trial_run:
                    # Log trial parameters
                    strategy_params_path = self.run_dir / "strategy"/ f"strategy_optimization_trial{trial.number}.yaml" # necessary for nautilus trader
                    strategy_params_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(strategy_params_path, 'w') as f:
                        yaml.dump(yaml_safe(strategy_params_flat), f)
                    mlflow.log_params(trial_hparams)
                    mlflow.log_param("trial_number", trial.number)
                    mlflow.log_param("phase", "strategy_optimization")
                    mlflow.log_param("model_path", str(self.best_model_path))
                    


                    # Run backtest with these strategy parameters
                    metrics = self._backtest(
                        model_params_flat= self.best_model_params_flat,  # type: ignore
                        strategy_params_flat = strategy_params_flat,
                        start = strategy_params_flat["valid_start"],
                        end = strategy_params_flat["valid_end"],
                    )
                    
                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Use Portfolio return as objective
                    scores = tuple(metrics[obj] for obj in objectives )
                    
                    # Store trial attributes
                    trial.set_user_attr("metrics", metrics)
                    trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                    trial.set_system_attr("strategy_params_path", strategy_params_path)
                    
                    logger.info(f"  Trial {trial.number}: Scores = {scores:.4f}")
                    
                    return scores
            
            # Run optimization
            n_trials = self.strategy_params.get("n_trials", 1)
            if self.strategy_params.get("tune_hparams", True) and n_trials > 0:
                logger.info(f"Running {n_trials} strategy trials...")
                study.optimize(strategy_objective, n_trials=n_trials)
            else:
                logger.info("Strategy hyperparameter tuning disabled, using defaults")
                # Run single trial with defaults
                study.optimize(strategy_objective, n_trials=1)
            
            # Get best trial
            best_trial = study.best_trial
            self.best_strategy_params_flat = self.strategy_params | best_trial.params
            
            # Log best results
            mlflow.log_params({
                f"best_strategy_{k}": v for k, v in best_trial.params.items()
            })
            mlflow.log_metric("best_strategy_sharpe", best_trial.value) # type: ignore
            
            logger.info(f"\nBest strategy trial: {best_trial.number}")
            logger.info(f"Best strategy Sharpe: {best_trial.value:.4f}")
            logger.info(f"Best strategy hparams: {best_trial.params}")
            
            # Save strategy optimization results
            strategy_results = {
                "best_trial": best_trial.number,
                "best_sharpe": best_trial.value,
                "best_hparams": best_trial.params,
                "best_metrics": best_trial.user_attrs.get("metrics", {}),
                "all_trials": [
                    {
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "state": str(t.state),
                    }
                    for t in study.trials
                ]
            }
            
            results_path = self.run_dir / "strategy_optimization_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(yaml_safe(strategy_results), f)
            mlflow.log_artifact(str(results_path))
        
        return {
            "strategy_params_path": best_trial.user_attrs.get("strategy_params_path"),
            "hparams": best_trial.params,
            "metrics": best_trial.user_attrs.get("metrics", {})
        }
    
    def get_best_config_flat(self, ) -> Dict[str, Any]:
        config = {"MODEL": self.best_model_params_flat , "STRATEGY": self.best_strategy_params_flat }

        return config

    def _backtest(
        self,
        model_params_flat: dict,
        strategy_params_flat: dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ) -> Dict[str, float]:
        """
        Run a full backtest with given model and strategy hyperparameters.
        
        Returns:
            Dictionary of performance metrics
        """
        
        # init config (flat)
        config = {
            'MODEL': model_params_flat ,
            'STRATEGY': strategy_params_flat ,
        }

        # Add data starting from t = - (windows_len + 1)
        bars = []
        data_load_start = strategy_params_flat["data_load_start"]
        for bar_or_feature in self.loader.bar_iterator():
            if isinstance(bar_or_feature, Bar):
                bar_ts = pd.Timestamp(bar_or_feature.ts_event, unit='ns', tz='UTC')
                if data_load_start <= bar_ts <= end:
                    bars.append(bar_or_feature)
        
        # Error Handling
        if len(bars) == 0:
            logger.warning(f"No bars found in period {start} to {end}")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
            }

        catalog = ParquetDataCatalog(path = strategy_params_flat["catalog_path"],  fs_protocol="file")
        catalog.write_data(list(self.loader.instruments.values()))
        catalog.write_data(bars)


        # Train model on Train + Valid with best HP
        # (DONE THROUGH THE STRATEGY)
        backtest_cfg = self._produce_backtest_config(config, start, end)


        node = BacktestNode(configs=[backtest_cfg])


        # Run backtest
        node.run()

        engine = node.get_engines()
        venue = Venue(self.strategy_params["venue_name"])

        # Calculate metrics using Nautilus built-in analyzer
        metrics = self._compute_metrics(engine=engine, venue=venue, strategy_params_flat=strategy_params_flat)
        
        node.dispose()
        
        return metrics

    def _compute_metrics(self, engine: BacktestEngine, venue: Venue, strategy_params_flat: Dict) -> Dict:
        # Calculate metrics using Nautilus built-in analyzer
        metrics = {}

        # Get portfolio analyzer for comprehensive metrics
        portfolio_analyzer = engine.analyzer
        
        # Get account for final balance
        account = engine.cache.account_for_venue(venue)
        
        try:
            # Get portfolio statistics from analyzer
            stats = portfolio_analyzer.get_portfolio_stats()
            
            # TODO: decide what to store/analyze
            # check: https://github.com/nautechsystems/nautilus_trader/tree/develop/nautilus_trader/analysis/statistics
            # check: https://nautilustrader.io/docs/latest/concepts/reports

            # Extract key metrics
            metrics['sharpe_ratio'] = stats.get('sharpe_ratio', 0.0) if stats else 0.0
            metrics['sortino_ratio'] = stats.get('sortino_ratio', 0.0) if stats else 0.0
            metrics['calmar_ratio'] = stats.get('calmar_ratio', 0.0) if stats else 0.0
            metrics['max_drawdown'] = stats.get('max_drawdown', 0.0) if stats else 0.0
            
            # Get returns statistics
            returns_stats = portfolio_analyzer.get_returns_stats() if hasattr(portfolio_analyzer, 'get_returns_stats') else {}
            metrics['total_return'] = returns_stats.get('total_return', 0.0) if returns_stats else 0.0
            metrics['annual_return'] = returns_stats.get('annual_return', 0.0) if returns_stats else 0.0
            metrics['daily_vol'] = returns_stats.get('daily_vol', 0.0) if returns_stats else 0.0
            
            # Get trade statistics
            trade_stats = portfolio_analyzer.get_trade_stats() if hasattr(portfolio_analyzer, 'get_trade_stats') else {}
            metrics['num_trades'] = trade_stats.get('total_trades', 0) if trade_stats else 0
            metrics['win_rate'] = trade_stats.get('win_rate', 0.0) if trade_stats else 0.0
            metrics['avg_win'] = trade_stats.get('avg_win', 0.0) if trade_stats else 0.0
            metrics['avg_loss'] = trade_stats.get('avg_loss', 0.0) if trade_stats else 0.0
            metrics['profit_factor'] = trade_stats.get('profit_factor', 0.0) if trade_stats else 0.0

            logger.info(f"Metrics loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not get analyzer metrics: {e}, calculating manually")
            
            # Fallback to manual calculation if analyzer methods not available
            positions = list(engine.cache.positions_closed())
            num_trades = len(positions)
            
            if num_trades > 0:
                winning_trades = sum(1 for p in positions if p.realized_pnl.as_double() > 0)
                win_rate = winning_trades / num_trades
                
                # Calculate PnLs
                pnls = [p.realized_pnl.as_double() for p in positions]
                avg_pnl = sum(pnls) / len(pnls) if pnls else 0
                
                # Calculate returns from equity curve
                initial_balance = float(strategy_params_flat['initial_cash'])
                final_balance = float(account.balance_total(strategy_params_flat["currency"]).as_double())
                total_return = (final_balance / initial_balance) - 1 if initial_balance > 0 else 0
                
                # Simple Sharpe calculation
                if len(pnls) > 1:
                    returns = [(pnls[i] / initial_balance) for i in range(len(pnls))]
                    if len(returns) > 0 and all(r == returns[0] for r in returns):
                        sharpe_ratio = 0.0  # All returns are the same
                    else:
                        import numpy as np
                        returns_array = np.array(returns)
                        sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252) if np.std(returns_array) > 0 else 0
                else:
                    sharpe_ratio = 0.0
                    
                metrics = {
                    'sharpe_ratio': sharpe_ratio,
                    'total_return': total_return,
                    'max_drawdown': 0.0,  # Would need equity curve to calculate
                    'win_rate': win_rate,
                    'num_trades': num_trades,
                    'avg_pnl': avg_pnl,
                    'final_balance': final_balance,
                }
            else:
                # No trades executed
                initial_balance = float(strategy_params_flat['initial_cash'])
                final_balance = float(account.balance_total(strategy_params_flat["currency"]).as_double())
                
                metrics = {
                    'sharpe_ratio': 0.0,
                    'total_return': (final_balance / initial_balance) - 1 if initial_balance > 0 else 0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'final_balance': final_balance,
                }

        return metrics
    
    def _add_dates(self, cfg, offset, to_strategy = False) -> Dict:
        # TODO: to verify this is also working for strategy
        # take some params from strategy

        
        
        # trial HPARAMS -> model PARAMS 
        # Calculate proper date ranges
        backtest_start = pd.Timestamp(self.strategy_params["backtest_start"], tz="UTC")
        backtest_end = pd.Timestamp(self.strategy_params["backtest_end"], tz="UTC")
        train_offset = freq2pdoffset(offset)
        valid_split =  self.strategy_params["valid_split"]

        
        calendar = market_calendars.get_calendar(self.model_params["calendar"])
        days_range = calendar.schedule(start_date= ( backtest_start - train_offset ), end_date=backtest_start)
        timestamps = market_calendars.date_range(days_range, frequency=self.model_params["freq"]).normalize()
        
        # compute traing and validation dates
        valid_offset = (timestamps[-1] - timestamps[0]) / (1 - valid_split) * valid_split
        train_start = backtest_start - train_offset - valid_offset
        train_end = train_start + train_offset
        valid_start = train_end + pd.Timedelta("1ns")
        valid_end = backtest_start - pd.Timedelta("1ns")
        
        

        # This should always be true even if validation is not run
        assert backtest_end > backtest_start, f"Backtest start {backtest_start} must come before backtest end {backtest_end}"
        
        # Update config with correct data types
        cfg["train_start"] = train_start
        cfg["train_offset"] = train_offset
        cfg["train_end"] = train_end
        cfg["valid_start"] = valid_start
        cfg["valid_end"] = valid_end
        cfg["backtest_start"] = backtest_start
        cfg["backtest_end"] = backtest_end

        if to_strategy:
            # compute data load date for validation: t = valid_start - (window_len + 1)
            days_range = calendar.schedule(start_date=train_start, end_date=valid_start)
            timestamps = market_calendars.date_range(days_range, frequency=self.model_params["freq"]).normalize()
            # start date including the initial window data needed for prediting at time t=0 of walk-frwd
            data_load_start = timestamps[-(self.best_model_params_flat["window_len"] + 1)]
            cfg["data_load_start"] = data_load_start
        

        return cfg
    
    def _get_data(self, start, end) -> Dict[str, pd.DataFrame]:

        # Create data dictionary for selected stocks
        calendar = market_calendars.get_calendar(self.model_params["calendar"])
        days_range = calendar.schedule(start_date=start, end_date=end)
        timestamps = market_calendars.date_range(days_range, frequency=self.model_params["freq"]).normalize()

        # init train+valid data
        data = {}
        for ticker in self.loader.universe:
            if ticker in self.loader._frames:
                df = self.loader._frames[ticker]
                # Get data up to validation end for initial training
                # TODO: this has to change in order to manage data from different timezones/market hours
                df.index = df.index.normalize()
                data[ticker] = df.reindex(timestamps).dropna()
                #logger.info(f"  {ticker}: {len(data[ticker])} bars")
        return data
    
    def _produce_backtest_config(self, backtest_cfg, start, end) -> BacktestRunConfig:

        # Initialize Venue configs
        venue_configs = [
            BacktestVenueConfig(
                name=backtest_cfg["STRATEGY"]["venue_name"],
                book_type="L1_MBP",         # bars are inluded in L1 market-by-price
                oms_type = backtest_cfg["STRATEGY"]["oms_type"],
                account_type=backtest_cfg["STRATEGY"]["account_type"],
                base_currency=backtest_cfg["STRATEGY"]["currency"],
                starting_balances=[str(backtest_cfg["STRATEGY"]["initial_cash"])+" "+str(backtest_cfg["STRATEGY"]["currency"])],
                bar_adaptive_high_low_ordering=False,  # Enable adaptive ordering of High/Low bar prices,
                
            ),
        ]
        bar_spec = freq2barspec(backtest_cfg["STRATEGY"]["freq"])
        data_configs=[
            BacktestDataConfig(
                catalog_path=backtest_cfg["STRATEGY"]["catalog_path"],
                data_cls=Bar,
                start_time=pd.Timestamp.isoformat(start),
                end_time=pd.Timestamp.isoformat(end),
                bar_spec=bar_spec,
                #instrument_ids =  [inst.id.value for inst in self.loader.instruments.values()],
                bar_types = [BarType(instrument_id=inst.id, bar_spec=bar_spec) for inst in self.loader.instruments.values()],
            )
        ]

        # Initialize Strategy
        strategy_name = self.strategy_params['strategy_name']
        model_name = self.model_params["model_name"]
        try:
            # Try to import from algos module
            strategy_module = importlib.import_module(f"algos.{strategy_name}")
            StrategyClass = getattr(strategy_module, strategy_name, None)
            if StrategyClass is None:
                # Try without redundant naming (e.g., algos.TopKStrategy.Strategy)
                StrategyClass = getattr(strategy_module, "Strategy", None)
            if StrategyClass is None:
                raise ImportError(f"Could not find strategy class in algos.{strategy_name}")
        except ImportError as e:
            logger.error(f"Failed to import strategy {strategy_name}: {e}")
            raise
        #strategy = StrategyClass(config=backtest_cfg)
        # TODO: add the fees from transactions and costs from config
        config=BacktestRunConfig(
                engine=BacktestEngineConfig(
                    trader_id=f"Backtest-{model_name}-{strategy_name}",
                    strategies=[ImportableStrategyConfig(
                        strategy_path=f"algos.{strategy_name}:{strategy_name}",
                        config_path = f"algos.{strategy_name}:{strategy_name}Config",
                        config = {"config":yaml_safe(backtest_cfg)},
                    )],
                    cache=CacheConfig(
                        bar_capacity=backtest_cfg["STRATEGY"].get("engine", {}).get("cache", {}).get("bar_capacity", 4096)
                        ),
                    logging=LoggingConfig(log_level="INFO"),
                    # TODO: fix fill model issue
                    #fill_model=FillModel(
                    #    prob_fill_on_limit=backtest_cfg["STRATEGY"].get("costs", {}).get("prob_fill_on_limit", 0.2),
                    #    prob_slippage=backtest_cfg["STRATEGY"].get("costs", {}).get("prob_slippage", 0.2),
                    #    random_seed=self.seed,
                    #    ),
                    ),
                data=data_configs,
                venues=venue_configs,
            )

        return config