from __future__ import annotations
from pathlib import Path
from typing   import Callable, Dict, Any
import optuna
from optuna.storages import RDBStorage
import pandas as pd
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path


class OptunaHparamsTuner:
    """
    Generic Optuna wrapper.
    """

    def __init__(
        self,
        *,
        model_name: str,
        ModelClass: nn.Module,                   # class  – NOT an instance
        start: pd.Timestamp,                     # train start
        end: pd.Timestamp,                       # validation end
        logs_dir: str | Path = "logs",           # logs root directory 
        universe_dataframe: Dict[str, Any],              # {ticker: DataFrame, ...}
        model_params: Dict[str, Any],
        defaults: Dict[str, Any],                # default hp values
        search_space: Optional[Dict[str, Dict]], # parsed from YAML
        n_trials: int,
        log: Any,                                # logger
        direction: str = "minimize",
        sampler: optuna.samplers.TPESampler | None = None,
    ):
        self.ModelClass   = ModelClass
        self.universe_dataframe  = universe_dataframe
        self.search_space = search_space
        self.model_params  = model_params
        self.defaults    = defaults
        self.n_trials    = n_trials
        self.log = log

        # directories storing models and hp tuning data
        start_str = start.strftime('%Y-%m-%d_%X')
        end_str = end.strftime('%Y-%m-%d_%X')
        base_dir   = Path(logs_dir).expanduser().resolve() / "optuna" / f"{start_str}_{end_str}"/ model_name
        self.model_params["model_dir"] = base_dir

        study_name = model_name.lower()
        db_path = base_dir / f"{model_name}_hpo.db"                              # trials database
        self.hp_log    = base_dir / "hp_trials.csv"                              # logs/optuna/models/<model>/hp_trials.csv
        storage = RDBStorage(f"sqlite:///{db_path}")

        self.study = optuna.create_study(
            study_name     = study_name,
            storage        = storage,
            direction      = direction,
            sampler        = sampler,
            load_if_exists = True,
        )

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Translate YAML search-space into Optuna API calls."""
        params = {}
        
        # Process all parameters
        for name, default_value in self.defaults.items():
            if self.search_space and name in self.search_space:
                # Use Optuna to suggest value
                cfg = self.search_space[name]
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
                # Use default value - Python preserves the type from YAML parsing
                params[name] = trial.suggest_categorical(name, [default_value])
        
        return params

    def _objective(self, trial: optuna.Trial) -> float:
        trial_hparams = self._suggest(trial)

        model = self.ModelClass(**self.model_params, **trial_hparams)
        score = model.initialize(self.universe_dataframe)

        # keep a pointer to the weights folder
        trial.set_user_attr("model_dir", str(model.model_dir))

        # ------------- append to hp_trials.csv -----------------------
        rec = {"trial": trial.number, "loss": score, **trial_hparams}
        pd.DataFrame([rec]).to_csv(
            self.hp_log,
            mode   ="a",
            header = not self.hp_log.exists(),
            index  = False,
        )
        self.log.info("New HPO trial: {rec}")
        return float(score)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def optimize(self) -> dict:
        self.study.optimize(self._objective, n_trials=self.n_trials)
        best   = self.study.best_trial
        self.log.info(f"Best hparams: {best.params} \nBest model path: {best.user_attrs["model_dir"]}")
        return {
            "hparams"    : best.params,
            "model_dir" : Path(best.user_attrs["model_dir"]),   # ← contains init.pt
        }


# utility function to collect hparams and their default values from config.yaml
def split_hparam_cfg(hp_cfg: dict):
    """
    Returns
    -------
    defaults      : {name: value}        (always passed to the model)
    search_space  : {name: optuna_dict}  (only when tuning is on)
    """
    defaults, search = {}, {}
    for k, v in hp_cfg.items():
        if isinstance(v, dict) and "default" in v:        # new structure
            defaults[k] = v["default"]
            if "optuna" in v:
                search[k] = v["optuna"]
        else:                                             # old scalar style
            defaults[k] = v
    return defaults, search
