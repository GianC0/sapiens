"""
Backtest Hyperparameter Tuner
==============================

Orchestrates two-phase hyperparameter optimization:
1. Model hyperparameters (using model.initialize())  
2. Strategy hyperparameters (using full backtest)

Each phase gets its own Optuna study and MLflow experiment.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple
import optuna
from optuna.storages import RDBStorage
import pandas as pd
import torch.nn as nn
import mlflow
from mlflow import MlflowClient
import yaml
from datetime import datetime
import importlib

from nautilus_trader.backtest.engine import BacktestEngine
from algos.strategy import LongShortStrategy
from algos.engine.data_loader import CsvBarLoader


class BacktestHparamsTuner:
    """
    Two-phase hyperparameter tuner for model + strategy optimization.
    
    Phase 1: Optimize model hyperparameters using model.initialize()
    Phase 2: Optimize strategy hyperparameters using best model from Phase 1
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        model_train_data : Dict[str, pd.DataFrame],
        strategy_train_data :  Dict[str, pd.DataFrame],
        logs_dir: Path = Path("logs/"),
        n_model_trials: Optional[int] = None,
        n_strategy_trials: Optional[int] = None,
        seed: int = 2025,
    ):
        """
        Initialize the backtest hyperparameter tuner.
        
        Args:
            cfg: Configuration dictionary
            model_train_data: Train data for hp optimization of model
            strategy_train_data: Train data for hp optimization of strategy
            logs_dir: Root directory for logs
            n_model_trials: Override for model HP trials (None = use config)
            n_strategy_trials: Override for strategy HP trials (None = use config)
            seed: Seed for reproducibility of hp tuning runs
        """
        # Load configuration
        self.config = cfg
        
        # Parse configuration sections
        self.model_config = self.config.get('MODEL', {})
        self.strategy_config = self.config.get('STRATEGY', {})
        
        # Extract params and hparams for each component
        self.model_params = self.model_config.get('PARAMS', {})
        self.model_hparams = self.model_config.get('HPARAMS', {})
        self.strategy_params = self.strategy_config.get('PARAMS', {})
        self.strategy_hparams = self.strategy_config.get('HPARAMS', {})
        
        # TODO: setup specific for UMIModel, to generalize/remove for future
        

        # Split hparams into defaults and search spaces
        self.model_defaults, self.model_search = self._split_hparam_cfg(self.model_hparams)
        self.strategy_defaults, self.strategy_search = self._split_hparam_cfg(self.strategy_hparams)
        self.seed = seed

        # Data setup
        self.model_train_data = model_train_data
        self.strategy_train_data = strategy_train_data
        
        # Trial counts
        self.n_model_trials = n_model_trials or self.model_params.get('n_trials', 20)
        self.n_strategy_trials = n_strategy_trials or self.strategy_params.get('n_trials', 10)
        
        # Setup directories
        self.logs_dir = Path(logs_dir).expanduser().resolve()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.logs_dir / "backtest_optimization" / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config for reproducibility
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Results storage
        self.best_model_hparams = None
        self.best_model_dir = None
        self.best_strategy_hparams = None
        self.results = {}
    
    def _setup_mlflow(self):
        """Setup MLflow tracking with hierarchical experiments."""
        mlflow.set_tracking_uri("file:logs/mlruns")
        
        # Create parent experiment for this optimization run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.parent_experiment_name = f"Backtest_Optimization_{timestamp}"
        mlflow.set_experiment(self.parent_experiment_name)
        
        # Start parent run
        self.parent_run = mlflow.start_run(
            run_name=f"Full_Optimization_{timestamp}",
            nested=False
        )
        
        # Log configuration
        mlflow.log_artifact(str(self.run_dir / "config.yaml"))
        mlflow.log_params({
            "n_model_trials": self.n_model_trials,
            "n_strategy_trials": self.n_strategy_trials,
            "model_name": self.model_params.get('model_name', 'Unknown'),
            "strategy_name": self.strategy_params.get('strategy_name', 'Unknown'),
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
        print(f"\n{'='*60}")
        print("PHASE 1: MODEL HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}\n")
        
        # Create MLflow experiment for model optimization
        with mlflow.start_run(
            run_name="Model_Optimization",
            nested=True
        ):
            
            # Setup Optuna study for model
            model_study_name = f"model_{self.model_params['model_name']}"
            model_db_path = self.run_dir / "model_hpo.db"
            storage = RDBStorage(f"sqlite:///{model_db_path}")
            
            study = optuna.create_study(
                study_name=model_study_name,
                storage=storage,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                load_if_exists=True,
            )
            
            # Define objective function for model
            def model_objective(trial: optuna.Trial) -> float:
                trial_hparams = self._suggest_params(trial, self.model_defaults, self.model_search)
                
                # Start MLflow run for this trial
                with mlflow.start_run(
                    run_name=f"model_trial_{trial.number}",
                    nested=True
                ) as trial_run:
                    # Log trial parameters
                    mlflow.log_params(trial_hparams)
                    mlflow.log_param("trial_number", trial.number)
                    mlflow.log_param("phase", "model_optimization")
                    
                    # Initialize and train model
                    model_dir, score = self._train_model(trial_hparams, trial.number, self.model_train_data)
                    
                    # Log results
                    mlflow.log_metric("validation_loss", score)
                    mlflow.log_artifact(str(model_dir / "init.pt"))
                    
                    # Store model directory in trial
                    trial.set_user_attr("model_dir", str(model_dir))
                    trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
                    
                    print(f"  Trial {trial.number}: loss = {score:.6f}")
                    
                return score
            
            # Run optimization
            print(f"Running {self.n_model_trials} model trials...")
            study.optimize(model_objective, n_trials=self.n_model_trials)
            
            # Get best trial
            best_trial = study.best_trial
            self.best_model_hparams = best_trial.params
            self.best_model_dir = Path(best_trial.user_attrs["model_dir"])
            
            # Log best results
            mlflow.log_params({
                f"best_model_{k}": v for k, v in self.best_model_hparams.items()
            })
            
            if best_trial is None:
                raise Exception("Could not compute hp optimization of model")
            
            mlflow.log_metric("best_model_loss", best_trial.value)
            
            print(f"\nBest model trial: {best_trial.number}")
            print(f"Best model loss: {best_trial.value:.6f}")
            print(f"Best model hparams: {self.best_model_hparams}")
            
            # Save model optimization results
            model_results = {
                "best_trial": best_trial.number,
                "best_loss": best_trial.value,
                "best_hparams": self.best_model_hparams,
                "model_dir": str(self.best_model_dir),
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
            
            results_path = self.run_dir / "model_optimization_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(model_results, f)
            mlflow.log_artifact(str(results_path))
        
        return {
            "hparams": self.best_model_hparams,
            "model_dir": self.best_model_dir,
            "validation_loss": best_trial.value
        }
    
    def optimize_strategy(self) -> Dict[str, Any]:
        """
        Phase 2: Optimize strategy hyperparameters using best model.
        
        Returns:
            Dictionary with best strategy hyperparameters and performance metrics
        """
        print(f"\n{'='*60}")
        print("PHASE 2: STRATEGY HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}\n")
        
        if self.best_model_hparams is None or self.best_model_dir is None:
            raise ValueError("Must run optimize_model() before optimize_strategy()")
        
        # Create MLflow experiment for strategy optimization
        with mlflow.start_run(
            run_name="Strategy_Optimization",
            nested=True
        ):
            
            # Setup Optuna study for strategy
            strategy_study_name = f"strategy_{self.strategy_params['strategy_name']}"
            strategy_db_path = self.run_dir / "strategy_hpo.db"
            storage = RDBStorage(f"sqlite:///{strategy_db_path}")
            
            study = optuna.create_study(
                study_name=strategy_study_name,
                storage=storage,
                direction="maximize",  # Maximize strategy performance (e.g., Portfolio return)
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                load_if_exists=True,
            )
            
            # Define objective function for strategy
            def strategy_objective(trial: optuna.Trial) -> float:
                trial_hparams = self._suggest_params(trial, self.strategy_defaults, self.strategy_search)
                
                # Start MLflow run for this trial
                with mlflow.start_run(
                    run_name=f"strategy_trial_{trial.number}",
                    nested=True
                ):
                    # Log trial parameters
                    mlflow.log_params(trial_hparams)
                    mlflow.log_param("trial_number", trial.number)
                    mlflow.log_param("phase", "strategy_optimization")
                    mlflow.log_param("model_dir", str(self.best_model_dir))
                    
                    # Run backtest with these strategy parameters
                    metrics = self._run_backtest(
                        model_dir=self.best_model_dir,
                        model_hparams=self.best_model_hparams,
                        strategy_hparams=trial_hparams,
                        trial_number=trial.number
                    )
                    
                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Use Sharpe ratio as objective (or another metric)
                    score = metrics.get('sharpe_ratio', 0.0)
                    
                    print(f"  Trial {trial.number}: Sharpe = {score:.4f}")
                    
                return score
            
            # Run optimization
            print(f"Running {self.n_strategy_trials} strategy trials...")
            study.optimize(strategy_objective, n_trials=self.n_strategy_trials)
            
            # Get best trial
            best_trial = study.best_trial
            self.best_strategy_hparams = best_trial.params
            
            # Log best results
            mlflow.log_params({
                f"best_strategy_{k}": v for k, v in self.best_strategy_hparams.items()
            })
            mlflow.log_metric("best_strategy_sharpe", best_trial.value)
            
            print(f"\nBest strategy trial: {best_trial.number}")
            print(f"Best strategy Sharpe: {best_trial.value:.4f}")
            print(f"Best strategy hparams: {self.best_strategy_hparams}")
            
            # Save strategy optimization results
            strategy_results = {
                "best_trial": best_trial.number,
                "best_sharpe": best_trial.value,
                "best_hparams": self.best_strategy_hparams,
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
                yaml.dump(strategy_results, f)
            mlflow.log_artifact(str(results_path))
        
        return {
            "hparams": self.best_strategy_hparams,
            "sharpe_ratio": best_trial.value
        }
    
    def _train_model(self, hparams: dict, trial_number: int, train_data: Dict[str, pd.DataFrame]) -> Tuple[Path, float]:
        """
        Train a model with given hyperparameters.
        
        Returns:
            (model_dir, validation_loss)
        """
        # Import model class dynamically
        model_name = self.model_params['model_name']
        mod = importlib.import_module(f"models.{model_name}.{model_name}")
        ModelClass = getattr(mod, model_name, None) or getattr(mod, "Model")
        
        if ModelClass is None:
            raise ImportError(f"Could not find model class in models.{model_name}")
        
        # Prepare model parameters
        model_params = {
            **self.model_params,
            "model_dir": self.run_dir / "models" / f"trial_{trial_number}",
        }
        
        
        # Initialize and train model
        model = ModelClass(**model_params, **hparams)
        validation_loss = model.initialize(train_data)
        
        return model_params["model_dir"], validation_loss
    
    def _run_backtest(
        self,
        model_dir: Path,
        model_hparams: dict,
        strategy_hparams: dict,
        trial_number: int
    ) -> Dict[str, float]:
        """
        Run a full backtest with given model and strategy hyperparameters.
        
        Returns:
            Dictionary of performance metrics
        """
        # This is a simplified version - you'll need to adapt to your actual backtest engine

        
        # Create modified config with strategy hparams
        config = self.config.copy()
        config['MODEL']['HPARAMS'] = {k: v for k, v in model_hparams.items()}
        config['STRATEGY']['HPARAMS'] = {k: v for k, v in strategy_hparams.items()}
        config['MODEL']['PARAMS']['model_dir'] = str(model_dir)
        config['MODEL']['PARAMS']['tune_hparams'] = False  # Skip tuning, use provided params
        
        # Run backtest (simplified - adapt to your setup)
        # You'll need to implement the actual backtest execution here
        # For now, returning dummy metrics
        
        # TODO: Implement actual backtest execution
        # engine = BacktestEngine(...)
        # strategy = LongShortStrategy(config=config)
        # engine.run()
        # metrics = engine.get_performance_metrics()
        
        # Dummy metrics for illustration
        import random
        metrics = {
            'sharpe_ratio': random.uniform(0.5, 2.5),
            'total_return': random.uniform(-0.1, 0.3),
            'max_drawdown': random.uniform(-0.2, -0.05),
            'win_rate': random.uniform(0.4, 0.6),
        }
        
        return metrics
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """
        Run the complete two-phase optimization process.
        
        Returns:
            Dictionary with all optimization results
        """
        try:
            # Phase 1: Model optimization
            model_results = self.optimize_model()
            
            # Phase 2: Strategy optimization
            strategy_results = self.optimize_strategy()
            
            # Combine results
            self.results = {
                "model": model_results,
                "strategy": strategy_results,
                "run_dir": str(self.run_dir),
                "config_path": str(self.config_path),
            }
            
            # Save final results
            final_results_path = self.run_dir / "optimization_results.yaml"
            with open(final_results_path, 'w') as f:
                yaml.dump(self.results, f)
            
            # Log to MLflow
            mlflow.log_artifact(str(final_results_path))
            mlflow.log_params({
                "final_model_dir": str(self.best_model_dir),
                "optimization_complete": True,
            })
            
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Results saved to: {self.run_dir}")
            print(f"MLflow experiment: {self.parent_experiment_name}")
            
            return self.results
            
        finally:
            # End parent MLflow run
            mlflow.end_run()
    
    def get_best_config(self) -> dict:
        """
        Generate a config file with the best hyperparameters found.
        
        Returns:
            Complete configuration with optimized hyperparameters
        """
        if not self.results:
            raise ValueError("Must run optimization before getting best config")
        
        # Create optimized config
        optimized_config = self.config.copy()
        
        # Update model hyperparameters
        for param, value in self.best_model_hparams.items():
            if param in optimized_config['MODEL']['HPARAMS']:
                optimized_config['MODEL']['HPARAMS'][param] = {
                    'default': value,
                    # Remove optuna section since we have the best value
                }
        
        # Update strategy hyperparameters
        for param, value in self.best_strategy_hparams.items():
            if param in optimized_config['STRATEGY']['HPARAMS']:
                optimized_config['STRATEGY']['HPARAMS'][param] = {
                    'default': value,
                    # Remove optuna section since we have the best value
                }
        
        # Disable hyperparameter tuning flags
        optimized_config['MODEL']['PARAMS']['tune_hparams'] = False
        optimized_config['STRATEGY']['PARAMS']['tune_hparams'] = False
        
        # Save optimized config
        optimized_config_path = self.run_dir / "optimized_config.yaml"
        with open(optimized_config_path, 'w') as f:
            yaml.dump(optimized_config, f)
        
        print(f"Optimized config saved to: {optimized_config_path}")
        
        return optimized_config