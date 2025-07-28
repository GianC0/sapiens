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
        ModelClass: nn.Model,                    # class  – NOT an instance
        start: str,                              # train start
        end: str,                                # validation end
        logs_dir: str | Path = "logs",           # logs root directory 
        train_dict: Dict[str, Any],              # {ticker: DataFrame, ...}
        model_params: Dict[str, Any],
        defaults: Dict[str, Any],                # default hp values
        search_space: Optional[Dict[str, Dict]], # parsed from YAML
        n_trials: int,
        direction: str = "minimize",
        sampler: optuna.samplers.TPESampler | None = None,
    ):
        self.ModelClass   = ModelClass
        self.train_dict  = train_dict
        self.search_space = search_space
        self.model_params  = model_params
        self.defaults    = defaults
        self.n_trials    = n_trials

        # directories storing models and hp tuning data

        self.base_dir   = Path(logs_dir).expanduser().resolve() / "optuna" / f"{start}--{end}"/ model_name

        study_name = model_name.lower()
        db_path = self.base_dir / f"{model_name}_hpo.db"                              # trials database
        self.hp_log    = self.base_dir / "hp_trials.csv"                              # logs/optuna/models/<model>/hp_trials.csv
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
        # ensure every trial lands inside <logs_root>/optuna/<start>-<end>/<model_name>/
        

        hp_id = f"lamic{trial_hparams.get('lambda_ic',0):.3f}_"\
            + f"lamsync{trial_hparams.get('lambda_sync',0):.3f}_"\
            + f"lamrankic{trial_hparams.get('lambda_rankic',0):.3f}"\
            + f"syncthres{trial_hparams.get('sync_thr',0):.3f}"
        time = pd.Timestamp.utcnow().strftime('%Y-%m-%d %X')
        model_dir = (self.base_dir / self.model_params["freq"] / f"{time}_{hp_id}").resolve()   
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model_params["model_dir"] = model_dir

        model = self.ModelClass(**self.model_params, **trial_hparams)
        model.initialize(self.train_dict)
        score = model._eval()

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
        return float(score)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def optimize(self) -> dict:
        self.study.optimize(self._objective, n_trials=self.n_trials)
        best   = self.study.best_trial
        return {
            "params"    : best.params,
            "model_dir" : Path(best.user_attrs["model_dir"]),   # ← contains best.pt
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
