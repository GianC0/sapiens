"""
Backtest Hyperparameter Tuner
==============================

Orchestrates two-phase hyperparameter optimization:
1. Model hyperparameters (using model.initialize())  
2. Strategy hyperparameters (using full backtest)

Each phase gets its own Optuna study and MLflow experiment.
"""

from copy import Error
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.identifiers import Venue, InstrumentId
from nautilus_trader.config import BacktestDataConfig, CacheConfig, ImportableStrategyConfig, ImportableExecAlgorithmConfig, LoggingConfig, ImportableFeeModelConfig, ImportableFillModelConfig, RiskEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, FeeModel
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.backtest.node import BacktestEngineConfig
#from nautilus_trader.common.component import RiskEngine   TODO: fix
from nautilus_trader.backtest.config import BacktestRunConfig, BacktestVenueConfig
from nautilus_trader.model.enums import OmsType
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm, TWAPExecAlgorithmConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.model.data import TradeTick
from nautilus_trader.core.nautilus_pyo3 import CurrencyType
from nautilus_trader.analysis.tearsheet import create_tearsheet
from nautilus_trader.analysis import TearsheetConfig

import numpy as np
import os
import pandas_market_calendars as market_calendars
from math import exp
from pathlib import Path
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, List
from unittest import loader
import optuna
from optuna.storages import RDBStorage
import optuna.visualization as optuna_viz
from optuna.importance import get_param_importances
import pandas as pd
from sqlalchemy import Engine
import torch.nn as nn
import re
import mlflow
from mlflow import MlflowClient
import yaml
from datetime import datetime, time
import importlib
import logging
logger = logging.getLogger(__name__)

from engine.logging_config import get_mlflow_uri
from models.utils import freq2pdoffset, yaml_safe, freq2barspec, freq2bartype
from engine.databento_loader import DatabentoTickLoader
from engine.ModelGenerator import ModelGenerator



class OptunaHparamsTuner:
    """
    Two-phase hyperparameter tuner for model + strategy optimization.
    
    Phase 1: Optimize model hyperparameters using model.initialize()
    Phase 2: Optimize strategy hyperparameters using best model from Phase 1
    """
    
    def __init__(
        self,
        catalog: ParquetDataCatalog,
        sapiens_config: Dict[str, Any],
        strategy_config: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
        run_dir: Path = Path(".runs/"),
        log_dir: Path = Path(".logs/"),
        optimization_id: str = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        """
        Initialize the backtest hyperparameter tuner.
        
        Args:
            sapiens_config: Main Sapiens configuration
            catalog: Data catalog
            strategy_config: Stratey configuration .yaml with PARAMS and HPARAMS
            strategy_config: Stratey configuration .yaml with PARAMS and HPARAMS
            run_dir: Runs directory
        """
        # Load configuration
        self.sapiens_config = sapiens_config

        # Log Path
        self.run_dir = run_dir
        self.logs_dir = log_dir

        # Generate model if needed
        if not model_config:
            model_config = self.generate_model()


        # Combine configs
        self.config = {
            'MODEL': model_config,
            'STRATEGY': strategy_config
        }
        
        # Extract sections
        self.model_params = self.config['MODEL']['PARAMS']
        self.model_hparams = self.config['MODEL'].get('HPARAMS', {})
        self.strategy_params = self.config['STRATEGY']['PARAMS']
        self.strategy_hparams = self.config['STRATEGY'].get('HPARAMS', {})
        
        # Sync shared params
        self.model_params["freq"] = self.strategy_params["freq"]
        self.model_params["calendar"] = self.strategy_params["calendar"]

        # TODO: Centralized Setup correct types (Dates and offsets). Now done in specific model/strategy
        

        # Split hparams into defaults and search spaces
        self.model_defaults, self.model_search = self._split_hparam_cfg(self.model_hparams)
        self.strategy_defaults, self.strategy_search = self._split_hparam_cfg(self.strategy_hparams)
        self.seed = self.sapiens_config["seed"]

        # --- ParquetDataCatalog and Data Augmentation setup -------------------------------------------
        self.catalog = catalog
        self.instrument_ids = list(set(inst.id for inst in self.catalog.instruments()))
    
        # Setup MLflow and Generate unique optimization ID NOTE: mlflow path set in logging_config)
        self.mlflow_client = MlflowClient()
        #NOTE: bug in mlflow dir creates this even if useless
        os.rmdir(run_dir.parent / "mlruns")
        self.optimization_id = optimization_id


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
    
    def generate_model(self) -> Dict[str, Any]:
        """
        Generate model via DeepCode if configured.
        
        Returns:
            model_config dictionary 
        """
        model_name = self.sapiens_config["SAPIENS_MODEL"]['model_name']
        
        logger.info(f"Generating new model: {model_name}")
        
        gen_cfg = self.sapiens_config["SAPIENS_MODEL"]['generation']
        gen_cfg['claude_model'] = gen_cfg['claude_model']
        
        generator = ModelGenerator(gen_cfg)
        
        model_dir = generator.generate_model(
            source_type=gen_cfg['source_type'],
            source_path=gen_cfg['source_path'],
            model_name=model_name
        )

        # Load model config from model folder
        model_config_path = Path(f"models/{model_name}/model_config.yaml")
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))["MODEL"]
        
        logger.info(f"Generated model at: {model_dir}")

        return model_config
        


    def optimize_model(self, ) -> Dict[str, Any]:
        """
        Phase 1: Optimize model hyperparameters.
        
        Returns:
            Dictionary with best hyperparameters and model directory
        """

        # Setup run path
        model_name = self.model_params["model_name"]
        study_path = self.run_dir / "Models" / model_name

        # Create/Load MLflow Model experiment for optimization
        parent_run_id = self._get_model_parent_run_id()

        # Setup Optuna study for model
        study_path.mkdir(parents=True, exist_ok=True)
        storage = RDBStorage(f"sqlite:///{study_path / "model_hpo.db"}")
        study = optuna.create_study(
            study_name=model_name,
            storage=storage,
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            load_if_exists=True,
        )

        # save old best trial to update the "is_best_model" tag at the end
        try:
            old_best_trial = study.best_trial
        except ValueError:
            old_best_trial = None

        # Initialize Model Class
        mod = importlib.import_module(f"models.{model_name}.{model_name}")
        ModelClass = getattr(mod, model_name, None) or getattr(mod, "Model")
        if ModelClass is None:
            raise ImportError(f"Could not find model class in models.{model_name}")

        
        # Define objective function for model
        def model_objective(trial: optuna.Trial) -> float:
            logging.info(f"\n{"#"*26}\n### New Model Trial: {trial.number} ###\n{"#"*26}\n")
            trial_hparams = self._suggest_params(trial, self.model_defaults, self.model_search)
            
            # setting train_offset and model_dir to model flatten config dictionary
            # adding train_offset here to avoid type overwite later  (pandas.offset -> str)
            train_offset = trial_hparams["train_offset"]
            retrain_offset = trial_hparams["retrain_offset"]
            n_retrains_on_valid = trial_hparams["n_retrains_on_valid"]
            inference_window = trial_hparams["inference_window"]
            model_dir = study_path /  f"trial_{trial.number}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_params_flat  = self._add_dates(
                                        self.model_params,
                                        train_offset, 
                                        retrain_offset, 
                                        n_retrains_on_valid, 
                                        inference_window, 
                                        to_strategy=False) 
            model_params_flat["model_dir"] = model_dir
            model_params_flat = trial_hparams | model_params_flat

            # init data timestamps for data loader
            # Initialize calendar. NOTE: calendar and freq already added un OptunaHparamTuner __init__
            
            # Retrieve CSV Bar data from loader for training
            start = model_params_flat["train_start"]
            end = model_params_flat["valid_end"]
            data_dict = self.get_ohlcv_data_from_catalog(
                frequency = self.model_params["freq"], 
                start=start, 
                end=end, 
                instrument_ids=self.instrument_ids, 
                with_risk_free_ticker=False,
                )

            # Start MLflow run for this trial
            mlflow.set_tracking_uri(get_mlflow_uri())
            mlflow.set_experiment("Models")
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
                parent_run_id=parent_run_id
            ) as trial_run:
                
                # Log trial parameters and tags
                mlflow.log_params(trial_hparams)
                mlflow.log_param("trial_number", trial.number)
                mlflow.set_tags({
                    "optimization_id": self.optimization_id,
                    "model_name": model_name,
                    "phase": "model_hpo"
                })
                
                # storing the model yaml config for reproducibility
                with open(model_dir / 'model_config.yaml', 'w', encoding="utf-8") as f:
                    yaml.dump(yaml_safe(model_params_flat), f, sort_keys=False)
                mlflow.log_artifact(str(model_dir / 'model_config.yaml'))
                
                # leave this order to make train_offset be overwritten by flat params type
                model = ModelClass(**(model_params_flat) )
                
                # all assets have same number of bars is ensured by get_data()
                total_bars = len(next(iter(data_dict.values())))

                # Validation is run automatically if valid_end is set
                # TODO: data_dict contains valid and train data because model is splitting it internally: 
                # this should be separated into 2 model calls: .initialize(train_start -> train_end) and .validate(valid_start, valid_end)
                score = model.initialize(data = data_dict, total_bars = total_bars)
                
                # Log best validation loss (with early-stopping)
                mlflow.log_metric("best_validation_loss", score)

                # Log per-epoch metrics
                epochs_metrics = model._epoch_logs
                for m in epochs_metrics:
                    epoch = m.pop("epoch")
                    mlflow.log_metrics(m, step=epoch)
                
                # Store model directory in trial
                trial.set_user_attr("model_path", str(model_dir / "init.pt"))
                trial.set_user_attr("model_params_flat", yaml_safe(trial_hparams | model_params_flat) )
                trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                
                logger.info(f"  Trial {trial.number}: loss = {score:.6f}")
                
            return score
        
        # Run optimization
        # TODO: define case when no hp tuning is needed
        

        # Handle case with and without optimization
        n_trials = self.sapiens_config["SAPIENS_MODEL"]["optimization"]["n_trials"]
        if self.sapiens_config["SAPIENS_MODEL"]["optimization"]["tune_hparams"] is True and n_trials >= 1:
            logger.info(f"Running {self.sapiens_config["SAPIENS_MODEL"]["optimization"]["n_trials"]} model trials...")
            study.optimize(model_objective, n_trials=n_trials)
        else:
            logger.info(f"Hyper-parameter tuning disabled. Taking best trial for {self.sapiens_config["SAPIENS_MODEL"]["model_name"]} from database...")
        
        # Remove best model tag from old best trial
        if old_best_trial:
            try:
                self.mlflow_client.delete_tag(run_id = old_best_trial.user_attrs["mlflow_run_id"], key = "is_best_model")
            except:
                logger.warning("Last best Trial was interrupted and not updated in mlflow")
        # Add the best model tag to the new best trial
        try:
            best_trial = study.best_trial

            if len(study.trials) > 1:
                fig_importance = optuna_viz.plot_param_importances(study)
            else:
                fig_importance = None
                logger.warning("Insufficient trials for parameter importance")

        except ValueError:
            raise Exception(f"Model {model_name} has never been been through hyper-parameters tuning")
        
        self.mlflow_client.set_tag(best_trial.user_attrs["mlflow_run_id"], "is_best_model", "true")

        
        logger.info(f"Best model trial: {best_trial.number}")
        logger.info(f"Best trial hparams: {best_trial.params}")
        logger.info(f"Best model loss: {best_trial.value:.6f}")
        
        
        return {
            "model_path": best_trial.user_attrs["model_path"],
            "models_params_flat": best_trial.user_attrs["model_params_flat"],
            "mlflow_run_id": best_trial.user_attrs["mlflow_run_id"], 
            "param_importance_fig": fig_importance,
        }

    def optimize_strategy(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 2: Optimize strategy hyperparameters using best model.
        
        Returns:
            Dictionary with best strategy hyperparameters and performance metrics
        """
        
        # If model_name = None then best model overall ( any family )
        best_model_params_flat = self.get_best_model_params_flat(model_name)
        if best_model_params_flat is None:
            raise ValueError("Must run optimize_model() before optimize_strategy()")
        best_model_name = best_model_params_flat["model_name"]
        best_model_mlflow_run_id = self.get_best_model_mlflow_run_id(model_name = best_model_name)


        # Setup run path
        strategy_name = self.strategy_params['strategy_name']
        study_path = self.run_dir / "Strategies" / f"{strategy_name}_{best_model_name}"

        # Create/Load MLflow Strategy experiment for optimization
        parent_run_id = self._get_strategy_parent_run_id(model_name = best_model_name)
        
        # Setup Optuna study for strategy
        study_path.mkdir(parents=True, exist_ok=True)
        storage = RDBStorage(f"sqlite:///{study_path / "strategy_hpo.db" }")

        # Parse Objectives and directions. Structure: obj:direction
        objectives, directions = zip(*self.sapiens_config["SAPIENS_STRATEGY"]["optimization"]['objectives'].items())
        primary_obj_index = objectives.index(self.sapiens_config["SAPIENS_STRATEGY"]["optimization"]["primary_objective"])
        
        study = optuna.create_study(
            study_name= f"{strategy_name}_{best_model_name}",
            storage=storage,
            directions=list(directions),  # follow the specific directions given in config for each metric
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            load_if_exists=True,
        )

        # save old best trial to update the "is_best_model" tag at the end
        old_best_trials = study.best_trials
        if len(old_best_trials) > 0 and study.directions[primary_obj_index] == optuna.study.StudyDirection.MINIMIZE:
            old_best_trial = min(old_best_trials, key=lambda t: t.values[primary_obj_index])
        elif len(old_best_trials) > 0 and study.directions[primary_obj_index] == optuna.study.StudyDirection.MAXIMIZE:
            old_best_trial = max(old_best_trials, key=lambda t: t.values[primary_obj_index])
        else:
            old_best_trial = None
            logging.info("No previous old best trial of strategy hpo")
        
        
        # Define objective function for strategy
        def strategy_objective(trial: optuna.Trial) -> Tuple:
            logging.info(f"\n{"#"*29}\n### New Strategy Trial: {trial.number} ###\n{"#"*29}\n")
            trial_hparams = self._suggest_params(trial, self.strategy_defaults, self.strategy_search)

            strategy_params_flat = self._add_dates(
                self.strategy_params.copy(), 
                best_model_params_flat["train_offset"], 
                best_model_params_flat["retrain_offset"],
                best_model_params_flat["n_retrains_on_valid"],
                best_model_params_flat["inference_window"],
                to_strategy=True, ) | trial_hparams
            
            # Start MLflow run for this trial
            mlflow.set_tracking_uri(get_mlflow_uri())
            mlflow.set_experiment("Strategies")
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
                parent_run_id=parent_run_id
            ) as trial_run:
                
                trial_path = study_path / f"trial_{trial.number}"
                trial_config_path = trial_path / "strategy_config.yaml"        # necessary for nautilus trader
                trial_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Log trial parameters
                mlflow.log_params(trial_hparams)
                mlflow.log_param("trial_number", trial.number)
                mlflow.set_tags({
                    "optimization_id": self.optimization_id,
                    "strategy_name": self.strategy_params['strategy_name'],
                    "model_name": best_model_name,
                    "model_hpo_run_id": best_model_mlflow_run_id,
                    "phase": "strategy_hpo"
                })
                
                # Storing strategy params for reproducibility
                with open(trial_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_safe(strategy_params_flat), f)
                mlflow.log_artifact(str(trial_config_path))

                # Run backtest with these strategy parameters
                results_path, metrics = self._backtest(
                    model_params_flat= best_model_params_flat,
                    strategy_params_flat = strategy_params_flat,
                    start = strategy_params_flat["valid_start"],
                    end = strategy_params_flat["valid_end"],
                    path = trial_path,
                )
                
                # Log all metrics and results
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                mlflow.log_artifact(local_path=results_path)
                
                
                # Use Portfolio return as objective
                scores = tuple(metrics[obj] for obj in objectives)
                scores_dict = {obj: metrics[obj] for obj in metrics if obj in objectives}

                # Store trial attributes
                trial.set_user_attr("metrics", metrics)
                trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                trial.set_user_attr("trial_config_path", str(trial_config_path))
                trial.set_user_attr("results_path", results_path)
                
                logger.info(f"  Trial {trial.number}: Scores = {scores_dict}")
                
                return scores
        
        # Run optimization
        n_trials = self.sapiens_config["SAPIENS_STRATEGY"]["optimization"]["n_trials"]
        if self.sapiens_config["SAPIENS_STRATEGY"]["optimization"]["tune_hparams"] and n_trials > 0:
            logger.info(f"Running hyper-parameter tuning with {n_trials} strategy trials...")
            study.optimize(strategy_objective, n_trials=n_trials)
        else:
            logger.info(f"Hyper-parameter tuning for strategy disabled. Taking best trial for {self.sapiens_config["SAPIENS_STRATEGY"]["strategy_name"]} from database...")

        
        # Remove best model tag from old best trial
        if old_best_trial:
            try:
                self.mlflow_client.delete_tag(run_id = old_best_trial.user_attrs["mlflow_run_id"], key = "is_best_strategy_for_model")
            except:
                logger.warning("Last best Trial was interrupted and not updated in mlflow")


        # Get best trial based on primary metric
        best_trials = study.best_trials
        try:
            logger.info(f"Considering best trial for {objectives[primary_obj_index]} as the overall multi-objective best trial")
            if study.directions[primary_obj_index] == optuna.study.StudyDirection.MINIMIZE:
                best_trial = min(best_trials, key=lambda t: t.values[primary_obj_index])
            else:
                best_trial = max(best_trials, key=lambda t: t.values[primary_obj_index])
            if best_trial is None:
                raise ValueError("Could not compute hp optimization of strategy")
            
            if len(study.trials) > 1:
                # Generate plot for visualization
                fig_importance = optuna_viz.plot_param_importances(study,target=lambda t: t.values[primary_obj_index])
            else:
                fig_importance = None
                logger.warning("Insufficient trials for parameter importance")
        except Error:
            raise Exception(f"Straegy {strategy_name} has never been been through hyper-parameters tuning")
        
        


        # Set best trial tag in MLflow for plotting
        best_strategy_model_run_id = best_trial.user_attrs["mlflow_run_id"]
        self.mlflow_client.set_tag(best_strategy_model_run_id, "is_best_strategy_for_model", "true")
        self.mlflow_client.set_tag(best_strategy_model_run_id, "primary_objective", objectives[primary_obj_index])
        
        logger.info(f"Best strategy trial: {best_trial.number}")
        logger.info(f"Best strategy hparams: {best_trial.params}")
        logger.info(f"Best strategy {objectives[primary_obj_index]}: {best_trial.values[primary_obj_index]:.4f}")
        

        
        return {
            "best_params_path": best_trial.user_attrs.get("trial_config_path"),
            "hparams": best_trial.params,
            "metrics": best_trial.user_attrs.get("metrics", {}),
            "mlflow_run_id": best_strategy_model_run_id,
            "param_importance_fig": fig_importance,
            "results_path": best_trial.user_attrs.get("results_path", str(study_path / f"trial{best_trial.number}" / "results.html" )  ),
        }
    
    def run_final_backtest(self, 
                           backtest_start: str, 
                           backtest_end: str, 
                           strategy_hpo_run_id: str,
                           optimization_id: str = '',
                           ) -> Tuple[str, Dict]:
        """Run final backtest with strategy run_id containing the best execution for strategy-model combo"""
        # Setup experiment
        exp = self.mlflow_client.get_experiment_by_name("Backtests")
        if exp is None:
            exp_path = Path(self.run_dir / "mlflow")
            exp_path.mkdir(parents=True,exist_ok=True)
            exp_id = self.mlflow_client.create_experiment("Backtests", artifact_location=str(exp_path))
        else:
            exp_id = exp.experiment_id
        
        # Retrieve strategy and model flat params
        model_name = self.mlflow_client.get_run(run_id = strategy_hpo_run_id).data.tags.get("model_name")
        strategy_name = self.mlflow_client.get_run(run_id = strategy_hpo_run_id).data.tags.get("strategy_name")
        full_flat = self.get_best_full_params_flat(model_name=model_name, strategy_name = strategy_name)
        model_params_flat = full_flat["MODEL"]
        strategy_params_flat = full_flat["STRATEGY"]
        
        tags = {
                "strategy_name": strategy_name,
                "model_name": model_name,
                "strategy_hpo_run_id": strategy_hpo_run_id,
                "phase": "backtest"
            }
        if optimization_id:
            tags["optimization_id"] = optimization_id

        logger.info(f"Running final backtest from {backtest_start} to {backtest_end}")
        run = self.mlflow_client.create_run(
            experiment_id=exp_id, 
            tags=tags, 
            run_name=f"Backtest_{strategy_name}_{model_name}"
        )
        run_id = run.info.run_id
        
        # Log backtest params
        self.mlflow_client.log_param( run_id= run_id, key = "config", value = full_flat )
        self.mlflow_client.log_param( run_id= run_id, key = "backtest_start", value = backtest_start )
        self.mlflow_client.log_param( run_id= run_id, key = "backtest_end", value = backtest_end )

            
        # Save and log optimized config
        backtest_path = self.run_dir / "Backtests" / f"{strategy_name}_{model_name}"
        backtest_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimized_config_flat_path = backtest_path / f"{timestamp}_config.yaml"
        with open(optimized_config_flat_path, 'w') as f:
            yaml.dump(yaml_safe(full_flat), f)
        self.mlflow_client.log_artifact( run_id= run_id, local_path = str(optimized_config_flat_path) )
        
        # Run backtest
        
        results_path, final_metrics = self._backtest(
            model_params_flat=model_params_flat,
            strategy_params_flat=strategy_params_flat,
            start=pd.Timestamp(backtest_start),
            end=pd.Timestamp(backtest_end),
            path = backtest_path,
        )

        # Log all metrics and results
        for metric_name, metric_value in final_metrics.items():
            self.mlflow_client.log_metric(run_id=run_id, key=metric_name, value=metric_value)
        self.mlflow_client.log_artifact(run_id=run_id, local_path=results_path)

        return results_path, final_metrics



    def _backtest(
        self,
        model_params_flat: dict,
        strategy_params_flat: dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        path: Path,
        ) -> Tuple[str, Dict[str, float]]:
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


        # Train model on Train + Valid with best HP
        # (DONE THROUGH THE STRATEGY)
        # TODO: fix this in live trading as it will be possible to call on_historical_data() from databento client
        logger.warning(f"Moving Start Date: {start} -> {strategy_params_flat["data_load_start"]} to include also training data ")
        start = strategy_params_flat["data_load_start"]
        
        backtest_cfg = self._produce_backtest_config(config, start, end)


        node = BacktestNode(configs=[backtest_cfg])
        


        # Run backtest
        node.run()

        engine = node.get_engines()[0]
        venue = Venue(self.strategy_params["venue_name"])

        # Calculate metrics using Nautilus built-in analyzer
        metrics = self._compute_metrics(engine=engine, venue=venue)
        
        benchm_ticker = strategy_params_flat["benchmark_ticker"]
        data_dict = self.get_ohlcv_data_from_catalog(
                        frequency=strategy_params_flat["freq"],
                        start=start,
                        end=end,
                        instrument_ids=[InstrumentId.from_str(benchm_ticker)],
                    )            
        benchmark_returns = data_dict[benchm_ticker]['close'].pct_change().fillna(0)

        output_path = str(path / "results.html")
        create_tearsheet(
            engine=engine,
            output_path=output_path,
            benchmark_returns=benchmark_returns,
            benchmark_name=benchm_ticker,
            config = TearsheetConfig(
                theme="plotly_white",
                include_benchmark=True,
                chart_args={"bars_with_fills": {"bar_types": [freq2bartype(iid, strategy_params_flat["freq"]) for iid in self.instrument_ids ]}}
                ),
        )

        node.dispose()
        
        return output_path, metrics

    def _compute_metrics(self, engine: BacktestEngine, venue: Venue) -> Dict:
        """
        Compute comprehensive metrics and collect time-series data.
        Returns  scalar metrics charting.
        """
        metrics = {}

        # Get portfolio, trader, and account
        portfolio = getattr(engine, "portfolio", None)
        portfolio_analyzer = getattr(portfolio, "analyzer", None) if portfolio is not None else None
        trader = getattr(engine, "trader", None)
        #account = engine.cache.account_for_venue(venue)


        # ===== EXTRACT PORTFOLIO STATISTICS =====
        stats_general = {}
        returns_stats = {}
        pnl_stats = {}
        
        # Stored as json so to run Offline Analysis
        if portfolio_analyzer is not None:
            # Get general performance statistics
            stats_general = portfolio_analyzer.get_performance_stats_general()
            #mlflow.log_table(stats_general, "stats_general.json")
            # Get and store returns statistics
            returns_stats = portfolio_analyzer.get_performance_stats_returns()
            #mlflow.log_table(returns_stats, "returns_stats.json")
            # Get and store PnL statistics
            pnl_stats = portfolio_analyzer.get_performance_stats_pnls()
            #mlflow.log_table(pnl_stats, "pnl_stats.json")
            # Get and store Returns Series
            #returns = portfolio_analyzer.returns()
            #mlflow.log_table(returns, "returns.json")


        # ===== LOG ALL REPORTS IN MLFLOW =====
        if trader is not None:
            # 1. All Orders Report
            orders_report = trader.generate_orders_report()
            mlflow.log_table(orders_report, "orders.json")
            
            # 2. Order Fills Report (one row per each filled order)
            order_fills_report = trader.generate_order_fills_report()
            mlflow.log_table(order_fills_report, "order_fills.json")

            # 3. Fills Report (one row per each individual fill event) 
            fills_report = trader.generate_fills_report()
            mlflow.log_table(fills_report, "fills.json")
                
            # 4. Positions Report (includes snapshots for NETTING OMS)
            positions_report = trader.generate_positions_report()
            mlflow.log_table(positions_report, "positions.json")

            # 5. Account Report (balance changes over time)
            account_report = trader.generate_account_report(venue)
            mlflow.log_table(account_report, "account.json")



        # Combine all stats and log numeric values
        all_stats = {**stats_general, **returns_stats, **pnl_stats}
        all_stats = {k: float(v) for k, v in all_stats.items() if isinstance(v, (int, float))}

        # Extract Returns metrics
        metrics['returns_volatility'] = all_stats.get('Returns Volatility (252 days)', 0.0)
        metrics['avg_return'] = all_stats.get('Average (Return)', 0.0)
        metrics['avg_loss_pct'] = all_stats.get('Average Loss (Return)', 0.0)
        metrics['avg_win_pct'] = all_stats.get('Average Win (Return)', 0.0)
        metrics['sharpe_ratio'] = all_stats.get('Sharpe Ratio (252 days)', 0.0)
        metrics['sortino_ratio'] = all_stats.get('Sortino Ratio (252 days)' , 0.0)
        metrics['profit_factor'] = all_stats.get('Profit Factor', 0.0)
        metrics['risk_return_ratio'] = all_stats.get('Risk Return Ratio', 0.0)

        # Extract PnL metrics
        metrics['total_pnl'] = all_stats.get('PnL (total)', 0.0)
        metrics['total_pnl_pct'] = all_stats.get('PnL% (total)', 0.0)
        metrics['max_drawdown'] = all_stats.get('Max Drawdown', 0.0)
        metrics['max_winner'] = all_stats.get('Max Winner', 0.0)
        metrics['avg_winner'] = all_stats.get('Avg Winner', 0.0)
        metrics['min_winner'] = all_stats.get('Min Winner', 0.0)
        metrics['min_loser'] = all_stats.get('Min Loser', 0.0)
        metrics['avg_loser'] = all_stats.get('Avg Loser', 0.0)
        metrics['max_loser'] = all_stats.get('Max Loser', 0.0)
        metrics["expectancy"] = all_stats.get('Expectancy', 0.0)
        metrics["win_rate"] = all_stats.get('Win Rate', 0.0)    

        logger.info(f"Extracted {len(metrics)} metrics successfully")

        return metrics

    
    def _add_dates(self, cfg, train_ofst, retrain_ofst, retrains_on_valid, inference_window, to_strategy = False) -> Dict:
        
        # trial HPARAMS -> model PARAMS 
        # Calculate proper date ranges
        backtest_start = pd.Timestamp(self.sapiens_config["backtest_start"], tz="UTC").normalize()
        backtest_end = pd.Timestamp(self.sapiens_config["backtest_end"], tz="UTC").normalize()
        train_offset = freq2pdoffset(train_ofst)
        retrain_offset = freq2pdoffset(retrain_ofst)
        n_retrains_on_valid = int(retrains_on_valid)
        
        calendar = market_calendars.get_calendar(self.model_params["calendar"])

        # compute traing and validation dates
        valid_offset = retrain_offset * (1 + n_retrains_on_valid)
        valid_end = backtest_start - pd.Timedelta("1ns")
        valid_start = valid_end - valid_offset

        train_start = backtest_start - valid_offset - train_offset
        train_end = valid_start - pd.Timedelta("1ns")
        
        # Calculate number of bars in inference window
        
        window_len = int(pd.Timedelta(inference_window) / pd.Timedelta(self.model_params["freq"]) )
        assert window_len > 1, f"Inference window {inference_window} < sampling frequency {self.model_params["freq"]}"

        # This should always be true even if validation is not run
        assert backtest_end > backtest_start, f"Backtest start {backtest_start} must come before backtest end {backtest_end}"
        
        # Update config with correct data types
        cfg["train_start"] = train_start
        cfg["train_offset"] = train_offset
        cfg["train_end"] = train_end
        
        cfg["valid_start"] = valid_start
        cfg["valid_offset"] = valid_offset
        cfg["valid_end"] = valid_end
        
        cfg["retrain_offset"] = retrain_offset
        cfg["valid_split"] = float( (valid_end - valid_start) / ( valid_end - train_start) )  # = valid / (valid + train) 
        cfg["window_len"] = window_len  # Store computed value for reference 

        cfg["backtest_start"] = backtest_start
        cfg["backtest_end"] = backtest_end

        if to_strategy:
            # Compute data load start for initial window
            # Need window_len bars before valid_start
            #window_len = self.get_best_model_params_flat(model_name=model_name)["window_len"]
            
            # Calculate using period count, not time offset
            pre_valid_range = calendar.schedule(start_date=train_start, end_date=valid_start)
            pre_valid_timestamps = market_calendars.date_range(pre_valid_range, frequency=self.model_params["freq"])
            
            if len(pre_valid_timestamps) >= window_len:
                data_load_start = pre_valid_timestamps[-window_len]
            else:
                # If not enough history, use train_start
                logger.warning(f"Insufficient history for window_len={window_len}, using train_start")
                data_load_start = train_start
            
            cfg["data_load_start"] = data_load_start

        

        return cfg
    
    
    def _produce_backtest_config(self, backtest_cfg, start, end) -> BacktestRunConfig:

        fee_model = backtest_cfg["STRATEGY"]["fee_model"]["name"]
        
        # Setup full strategy config        
        currency_code = backtest_cfg["STRATEGY"]["currency"]
        if currency_code == "USD":
            currency = Currency(
                code='USD', precision=3, iso4217=840,
                name='United States dollar', currency_type=CurrencyType.FIAT
            )
        elif currency_code == "EUR":
            currency = Currency(
                code='EUR', precision=3, iso4217=978,
                name='Euro', currency_type=CurrencyType.FIAT
            )
        
        # Initialize Venue configs
        venue_configs = [
            BacktestVenueConfig(
                name=backtest_cfg["STRATEGY"]["venue_name"],
                book_type="L1_MBP",         # bars are inluded in L1 market-by-price
                oms_type = backtest_cfg["STRATEGY"]["oms_type"],
                account_type=backtest_cfg["STRATEGY"]["account_type"],
                base_currency=currency.code,
                starting_balances=[str(backtest_cfg["STRATEGY"]["initial_cash"])+" "+currency.code],
                bar_adaptive_high_low_ordering=False,  # Enable adaptive ordering of High/Low bar prices,
                fill_model=ImportableFillModelConfig(
                        fill_model_path = "nautilus_trader.backtest.models:FillModel",
                        config_path = "nautilus_trader.backtest.config:FillModelConfig",
                        config = {
                            "prob_fill_on_limit" : backtest_cfg["STRATEGY"]["costs"]["prob_fill_on_limit"],
                            "prob_slippage" : backtest_cfg["STRATEGY"]["costs"]["prob_slippage"],
                            "random_seed" : self.seed,
                            },

                ),
                fee_model=ImportableFeeModelConfig(
                    fee_model_path = f"engine.fees.{fee_model}:{fee_model}",
                    config_path = f"engine.fees.{fee_model}:{fee_model}Config",
                    config = { "config" : {
                                    "commission_per_unit": Money(float(backtest_cfg["STRATEGY"]["fee_model"]["commission_per_unit"]) , currency),  
                                    "min_commission": Money( backtest_cfg["STRATEGY"]["fee_model"]["min_commission"] , currency),
                                    "max_commission_pct": Decimal(backtest_cfg["STRATEGY"]["fee_model"]["max_commission_pct"])
                                }
                    },
                ),
            ),
        ]
        if "catalog_path" not in backtest_cfg["STRATEGY"]:
            raise ValueError("catalog_path must be set in strategy config")
        data_configs=[
            BacktestDataConfig(
                catalog_path=backtest_cfg["STRATEGY"]["catalog_path"],
                data_cls=TradeTick,
                start_time=pd.Timestamp(start),
                end_time=pd.Timestamp(end),
                instrument_ids=[iid.value for iid in self.instrument_ids],
                #bar_spec=bar_spec,
                #bar_types = [freq2bartype(instrument_id=iid, frequency= backtest_cfg["STRATEGY"]["freq"]) for iid in self.instrument_ids],
                )
            ]

        # Initialize Strategy
        strategy_name = self.strategy_params['strategy_name']
        model_name = self.model_params["model_name"]
        # TODO:  add risk engine
        config=BacktestRunConfig(
                engine=BacktestEngineConfig(
                    trader_id=f"Backtest-{model_name}-{strategy_name}",
                    strategies=[ImportableStrategyConfig(
                        strategy_path=f"strategies.{strategy_name}.{strategy_name}:{strategy_name}",
                        config_path = f"strategies.SapiensStrategy:SapiensStrategyConfig",
                        config = {"config":yaml_safe(backtest_cfg)},
                    )],
                    exec_algorithms=[ ImportableExecAlgorithmConfig(
                        exec_algorithm_path="nautilus_trader.examples.algorithms.twap:TWAPExecAlgorithm",
                        config_path="nautilus_trader.examples.algorithms.twap:TWAPExecAlgorithmConfig",
                        config={}  # Empty config or you can pass TWAP-specific params here
                    )],
                    risk_engine=RiskEngineConfig(
                        bypass=False,
                        max_order_submit_rate="100/00:00:01",
                        max_notional_per_order={},
                    ),

                    cache=CacheConfig(
                        tick_capacity=backtest_cfg["STRATEGY"].get("engine", {}).get("cache", {}).get("tick_capacity", 50000),
                        bar_capacity=backtest_cfg["STRATEGY"].get("engine", {}).get("cache", {}).get("bar_capacity", 4096)
                        ),
                    logging=LoggingConfig(
                        log_level=self.sapiens_config["logs_level"],           # or "WARNING" to reduce noise
                        log_level_file=self.sapiens_config["logs_level"],      # file can have more detail
                        log_file_format=None,                                  # or "json"
                        log_directory=str(self.logs_dir / "nautilus"),         # where to save log files
                        log_file_name="nautilus_trader.log",                   # filename prefix
                        bypass_logging=False,                                  # set True to disable all logging
                        log_colors=True,
                    )
                ),
                data=data_configs,
                venues=venue_configs,
            )

        return config


    def _get_model_parent_run_id(self) -> str:
        """Get or create parent run for model family in 'Models' experiment."""
        exp = self.mlflow_client.get_experiment_by_name("Models")
        if exp is None:
            exp_path = Path(self.run_dir / "mlflow")
            exp_path.mkdir(parents=True,exist_ok=True)
            exp_id = self.mlflow_client.create_experiment("Models", artifact_location=str(exp_path))
        else:
            exp_id = exp.experiment_id

        # Search for existing parent run for this model family
        model_name = self.model_params["model_name"]
        filter_str = f"tags.model_name = '{model_name}' AND tags.is_parent = 'true'"
        runs = self.mlflow_client.search_runs([exp_id], filter_string=filter_str, max_results=1)
        
        if runs:
            return runs[0].info.run_id
        else:
            # Create new parent run
            tags = {
                "is_parent": "true", 
                "model_name": model_name, 
                "optimization_id": self.optimization_id, 
                "phase": "model_hpo" 
            }
            run = self.mlflow_client.create_run(
                experiment_id=exp_id, 
                tags=tags, 
                run_name=model_name 
            )
            run_id = run.info.run_id

            # Log model family params
            for k , p in self.model_params.items():
                self.mlflow_client.log_param( run_id= run_id, key = k, value = p )

            return run_id

    def _get_strategy_parent_run_id(self, model_name : str) -> str:
        """Get or create parent run for strategy+model combo in 'Strategies' experiment."""
        exp = self.mlflow_client.get_experiment_by_name("Strategies")
        if exp is None:
            exp_path = Path(self.run_dir / "mlflow")
            exp_path.mkdir(parents=True,exist_ok=True)
            exp_id = self.mlflow_client.create_experiment("Strategies", artifact_location=str(exp_path))
        else:
            exp_id = exp.experiment_id
        
        strategy_name = self.strategy_params['strategy_name']
        run_name = f"{strategy_name}_{model_name}"
        
        # Search for existing parent for this combo
        filter_str = (f"tags.strategy_name = '{strategy_name}' AND "
                    f"tags.model_name = '{model_name}' AND "
                    f"tags.is_parent = 'true'")
        runs = self.mlflow_client.search_runs([exp_id], filter_string=filter_str, max_results=1)
        
        if runs:
            return runs[0].info.run_id
        
        else:
            best_model_run_id = self.get_best_model_mlflow_run_id(model_name=model_name)
            best_model_params_flat = self.get_best_model_params_flat(model_name=model_name)

            # Create new parent run
            tags = {
                "optimization_id": self.optimization_id,
                "is_parent": "true",
                "strategy_name": strategy_name,
                "model_name": model_name,
                "best_model_run_id": best_model_run_id,
                "phase": "strategy_hpo"
            }
            run = self.mlflow_client.create_run(
                experiment_id=exp_id,
                tags=tags,
                run_name=run_name
            )

            run_id = run.info.run_id

            # Retrieve the best model params for model_name family



            # Log Strategy family params
            self.mlflow_client.log_param(run_id = run_id, key = "best_model_params_flat", value = best_model_params_flat)
            for k , p in self.strategy_params.items():
                self.mlflow_client.log_param( run_id= run_id, key = k, value = p )

            return run.info.run_id

    def _filter_to_market_hours(
        self,
        df: pd.DataFrame,
        schedule: pd.DataFrame,
        instrument_id: str
    ) -> pd.DataFrame:
        """
        Filter tick data to only include regular market hours.
        
        Args:
            df: DataFrame with tick data (index must be timezone-aware)
            schedule: Market calendar schedule with 'market_open' and 'market_close'
            instrument_id: Instrument identifier for logging
        
        Returns:
            Filtered DataFrame containing only ticks during market hours
        """
        # Get market open/close times in UTC
        market_opens = schedule['market_open'].dt.tz_convert('UTC')
        market_closes = schedule['market_close'].dt.tz_convert('UTC')
        
        # Create mask for market hours
        mask = pd.Series(False, index=df.index)
        for open_time, close_time in zip(market_opens, market_closes):
            mask |= (df.index >= open_time) & (df.index <= close_time)
        
        df_filtered = df[mask]
        
        # Log filtering results
        n_excluded = len(df) - len(df_filtered)
        if n_excluded > 0:
            pct_excluded = (n_excluded / len(df)) * 100
            logger.info(
                f"{instrument_id}: {len(df):,} total ticks → "
                f"{len(df_filtered):,} in market hours "
                f"({n_excluded:,} pre/after-hours excluded, {pct_excluded:.1f}%)"
            )
        
        return df_filtered

    # Add new method to OptunaHparamsTuner class
    def get_ohlcv_data_from_catalog(
        self,
        frequency: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        instrument_ids: Optional[List[InstrumentId]] = None,
        with_risk_free_ticker: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate trade ticks from catalog into OHLCV bars.
        
        Args:
            frequency: Pandas frequency string (e.g., '1D', '1H', '5T')
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            instrument_ids: List of instrument IDs objects. 
                        If None, uses all instruments in catalog.
        
        Returns:
            Dictionary mapping symbol strings to OHLCV DataFrames with columns:
            ['open', 'high', 'low', 'close', 'volume']
        """
        # Get all instruments if not specified
        # Remove risk free asset from training data as it will not be used for forecasting
        if instrument_ids is None:
            instruments = self.catalog.instruments(instrument_type=TradeTick)
            instrument_ids = [inst.instrument_id for inst in instruments ]
        if not with_risk_free_ticker:
            instrument_ids = [iid for iid in instrument_ids if iid.value != self.strategy_params["risk_free_ticker"] ]
    
        # Create unified date range using market calendar
        calendar = market_calendars.get_calendar(self.model_params["calendar"])
        schedule = calendar.schedule(start_date=start, end_date=end)
        unified_index = market_calendars.date_range(schedule, frequency=frequency)

        data_dict = {}
        
        for iid in instrument_ids:
            try:
                # Query trade ticks from catalog
                ticks = self.catalog.trade_ticks(
                    instrument_ids=[iid],
                    start=start,
                    end=end
                )
                
                if not ticks or len(ticks) == 0:
                    logger.warning(f"No ticks found for {iid}")
                    continue
                
                # Vectorized conversion: pre-allocate arrays
                n_ticks = len(ticks)
                timestamps = np.empty(n_ticks, dtype='int64')
                prices = np.empty(n_ticks, dtype=np.float64)
                sizes = np.empty(n_ticks, dtype=np.float64)
                
                # Batch extract attributes
                for i, tick in enumerate(ticks):
                    timestamps[i] = tick.ts_event
                    prices[i] = float(tick.price)
                    sizes[i] = float(tick.size)
                
                # Create DataFrame with timezone-aware index
                df = pd.DataFrame(
                    {'price': prices, 'size': sizes},
                    index=pd.to_datetime(timestamps, unit='ns', utc=True)
                )

                # Filter to market hours
                df_filtered = self._filter_to_market_hours(df, schedule, str(iid))
                
                if len(df_filtered) == 0:
                    logger.warning(f"No ticks in market hours for {iid}")
                    continue
                
                df_filtered = df_filtered.copy()

                # Find which bar each tick belongs to
                bar_indices = np.searchsorted(unified_index, df_filtered.index, side='right') - 1
                
                # Clip to valid range
                bar_indices = np.clip(bar_indices, 0, len(unified_index) - 1)
                
                # Assign bar timestamps
                
                df_filtered['bar_time'] = unified_index[bar_indices]
                
                # Group by bar and aggregate
                ohlcv_df = df_filtered.groupby('bar_time').agg({
                    'price': ['first', 'max', 'min', 'last'],
                    'size': 'sum'
                })
                
                # Flatten columns
                ohlcv_df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Reindex to ensure all periods present (even with no trades)
                ohlcv_df = ohlcv_df.reindex(unified_index)
                
                # Forward-fill OHLC, zero-fill volume
                ohlcv_df[['open', 'high', 'low', 'close']] = ohlcv_df[['open', 'high', 'low', 'close']].ffill()
                ohlcv_df['volume'] = ohlcv_df['volume'].fillna(0)
                
                if ohlcv_df['close'].isna().all():
                    logger.warning(f"No valid data for {iid} after aggregation")
                    continue
                
                data_dict[iid.value] = ohlcv_df
                
            except Exception as e:
                logger.error(f"Error processing {iid}: {e}")
                continue
        
        # Validate alignment
        lengths = {iid: len(df) for iid, df in data_dict.items()}
        if len(set(lengths.values())) > 1:
            logger.error(f"❌ Misaligned: {lengths}")
            raise ValueError("Instruments have different bar counts")
        
        logger.info(f"All {len(data_dict)} instruments aligned: {list(lengths.values())[0]} bars")
        
        return data_dict
    
    def get_best_model_params_flat(self, model_name: Optional[str]) -> Dict:
        """Returns flat params dictionary of model_name familiy (if specified) or best model overall (any family)"""
        
        exp = self.mlflow_client.get_experiment_by_name("Models")
        if exp is None:
            raise Error(f"No model optimization has been run yet")
        
        # Specific to a model_name (if specified)
        if model_name:
            model_runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.model_name = '{model_name}' AND tags.is_best_model = 'true'",
                max_results=1
                )
        # Return best_model_params_flat of best model overall ( any family )
        else:
            model_runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.is_best_model = 'true'",
                order_by=["metrics.best_validation_loss ASC"],
                max_results=1
            )

        if not model_runs:
            raise Error(f"No model optimization has been run yet for family: {model_name}")
        
        model_config_path = self.mlflow_client.download_artifacts(model_runs[0].info.run_id, "model_config.yaml")
        model_params_flat = {}
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_params_flat = yaml.safe_load(f.read())

        return model_params_flat
    
    def get_best_model_mlflow_run_id(self, model_name: Optional[str]) -> str:
        """Returns the mlflow best run_id of the specified model_name (if specified) or the run_id of the best model overall (any family) """
        
        exp = self.mlflow_client.get_experiment_by_name("Models")
        if exp is None:
            raise Error(f"No model optimization has been run yet")
        
        # Specific to a model_name (if specified)
        if model_name:
            model_runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.model_name = '{model_name}' AND tags.is_best_model = 'true'",
                max_results=1
                )
            if not model_runs:
                raise Error(f"No model optimization has been run yet for family: {model_name}")
            best_model_mlflow_run_id = model_runs[0].info.run_id
            return best_model_mlflow_run_id

        # Return best_model_mlflow_run_id of best model overall ( any family )
        model_runs = self.mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.is_best_model = 'true'",
            order_by=["metrics.best_validation_loss ASC"],
            max_results=1
        )
        if not model_runs:
            raise Error(f"No model optimization has been run yet for family: {model_name}")

        best_model_mlflow_run_id = model_runs[0].info.run_id
        return best_model_mlflow_run_id
    
    def get_best_full_params_flat(self, primary_objective_with_direction: Dict[str,str] = {"total_pnl_pct" : "maximize"}, strategy_name: Optional[str] = None, model_name: Optional[str] = None):
        """ Returns best full flat params dictionary (MODEL & STRATEGY) filtering by
            strategy_nam (if specified), best model name (if specified) or best 
            strategy + model combo overall.
            
            Args:
                primary_objective_with_direction: Dictionary with Strategy optimization primary objective and direction (maximize, minimize)
                strategy_name: Filter by specific strategy (optional)
                model_name: Filter by specific model (optional)
            
            Returns:
                Dict with "MODEL" and "STRATEGY" keys containing best flat params
           """
        
        # Set experiments for mlflow
        strategy_exp = self.mlflow_client.get_experiment_by_name("Strategies")
        model_exp = self.mlflow_client.get_experiment_by_name("Models")
        if strategy_exp is None or model_exp is None:
            raise ValueError(f"Required experiments not found. \nStrategy exp: {strategy_exp} \nModel exp: {model_exp}" )
        
        # Built filter of strategy and model
        filter_parts = ["tags.is_best_strategy_for_model = 'true'"]
        if strategy_name:
            filter_parts.append(f"tags.strategy_name = '{strategy_name}'")
        if model_name:
            filter_parts.append(f"tags.model_name = '{model_name}'")
        filter_string = " AND ".join(filter_parts)

        # TODO: ensure all strategies have the primary obj in common (total_pnl_pct)
        # Matching optuna ordering flag with mlflow search order
        objective, direction = next(iter(primary_objective_with_direction.items()))
        if direction not in ("maximize", "minimize"):
            raise Error(f"Direction should be either 'maximize' or 'maximize'... it was instead {direction}")
        if direction == "maximize":
            search_order = "DESC"
        else:
            search_order = "ASC"
        
        # Finding best model and strategy runs from filters 
        strategy_runs = self.mlflow_client.search_runs(
            experiment_ids=[strategy_exp.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{objective} {search_order}"],
            max_results=1
        )
        if not strategy_runs:
            raise ValueError(f"No matching strategy run found for filters: {filter_string}")
        
        best_strategy_run = strategy_runs[0]
        model_hpo_run_id = best_strategy_run.data.tags.get("model_hpo_run_id")
        if not model_hpo_run_id:
            raise ValueError("Strategy run missing model_hpo_run_id tag")
        
        best_relative_model_run = self.mlflow_client.get_run(model_hpo_run_id)
        
        # Load model params flat for the best model
        model_config_path = self.mlflow_client.download_artifacts(best_relative_model_run.info.run_id, "model_config.yaml")
        model_params_flat = {}
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_params_flat = yaml.safe_load(f.read())

        # Load strategy params flat for the best strategy
        strategy_config_path = self.mlflow_client.download_artifacts(best_strategy_run.info.run_id, "strategy_config.yaml")
        strategy_params_flat = {}
        with open(strategy_config_path, "r", encoding="utf-8") as f:
            strategy_params_flat = yaml.safe_load(f.read())
        
        full_config = {
            "MODEL": model_params_flat ,
            "STRATEGY": strategy_params_flat
        }
    
        return full_config
    
    def get_strategy_hpo_matrix(self, metric: str = "total_pnl_pct") -> pd.DataFrame:
        """
        Generate matrix: strategies × models with best HP tuning metrics.
        
        Args:
            metric: Metric name to display in matrix
        
        Returns:
            DataFrame with strategies as rows, models as columns
        """
        exp = self.mlflow_client.get_experiment_by_name("Strategies")
        if not exp:
            raise Error(f"No strategies optimization has been run yet")

        runs = self.mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.is_best_strategy_for_model = 'true'",
        )
        
        data = []
        for run in runs:
            data.append({
                'strategy_name': run.data.tags.get('strategy_name', 'Unknown'),
                'model_name': run.data.tags.get('model_name', 'Unknown'),
                metric: run.data.metrics.get(metric, None)
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        matrix = df.pivot_table(
            index='strategy_name',
            columns='model_name',
            values=metric,
            #aggfunc='first'
        )
        return matrix

    def get_final_backtest_matrix(self, metric: str = "total_pnl_pct") -> pd.DataFrame:
        """
        Generate matrix: strategies × models with final backtest metrics.
        
        Args:
            metric: Metric name to display in matrix
        
        Returns:
            DataFrame with strategies as rows, models as columns
        """
        exp = self.mlflow_client.get_experiment_by_name("Backtests")
        if exp is None:
            logger.warning("No Final_Backtests experiment found")
            return pd.DataFrame()
        
        runs = self.mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
        )
        
        data = []
        for run in runs:
            data.append({
                'strategy_name': run.data.tags.get('strategy_name', 'Unknown'),
                'model_name': run.data.tags.get('model_name', 'Unknown'),
                metric: run.data.metrics.get(metric, None)
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        matrix = df.pivot_table(
            index='strategy_name',
            columns='model_name',
            values=metric,
#            aggfunc='first'
        )
        return matrix