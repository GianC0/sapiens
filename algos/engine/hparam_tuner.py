from __future__ import annotations
from pathlib import Path
from typing   import Callable, Dict, Any
import optuna
from optuna.storages import RDBStorage
import pandas as pd


class OptunaHparamsTuner:
    """
    Generic Optuna wrapper.

    Parameters
    ----------
    model_cls      : a class (not an instance) with signature  model_cls(**fixed_kwargs, **trial_params)
                     and a .fit(train_dict, **fit_kwargs) method returning a scalar score
                     (lower is better unless `direction="maximize"` is passed).
    train_dict     : {ticker: pd.DataFrame}  –- the data shown to every trial.
                     For pure ML models this is commonly `model._eval(loader_valid)`.
    search_space   : Dict[str, Dict]         –- see YAML example below.
    fixed_kwargs   : kwargs required by the model but *not* tuned.
    fit_kwargs     : kwargs forwarded verbatim to model.fit().
    """

    def __init__(
        self,
        *,
        model_name: str,
        logs_dir: str | Path = "logs",           # logs root directory 
        model_cls,                               # class  – NOT an instance
        train_dict: Dict[str, Any],              # {ticker: DataFrame}
        search_space: Dict[str, Dict],           # parsed from YAML
        defaults: Dict[str, Any],                # default hp values
        fixed_kwargs: Dict[str, Any],
        fit_kwargs: Dict[str, Any] ,
        study_name: str ,
        n_trials: int,
        direction: str = "minimize",
        sampler: optuna.samplers.TPESampler | None = None,
    ):
        self.model_cls   = model_cls
        self.train_dict  = train_dict
        self.search_space = search_space
        self.defaults    = defaults
        self.fit_kwargs  = fit_kwargs                # warm n_epochs and warm_start
        self.fixed_kwargs  = fixed_kwargs
        self.n_trials    = n_trials

        # directories storing models and hp tuning data

        self.base_dir   = Path(logs_dir).expanduser().resolve() / "optuna" / model_name
        self.trials_dir = self.base_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)

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
        for name, cfg in self.search_space.items():
            t = cfg["type"]
            if t == "float":
                params[name] = trial.suggest_float(name, cfg["low"], cfg["high"])
            elif t == "log_float":
                params[name] = trial.suggest_float(
                    name, cfg["low"], cfg["high"], log=True
                )
            elif t == "int":
                params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
            elif t == "categorical":
                params[name] = trial.suggest_categorical(name, cfg["choices"])
            else:
                raise ValueError(f"Unknown param type {t} for {name}")
        return params

    def _objective(self, trial: optuna.Trial) -> float:
        trial_params = self._suggest(trial)
        # ensure every trial lands inside <logs_root>/optuna/models/<name>/trials/
        model_dir_base        = self.trials_dir
        model = self.model_cls(model_dir=model_dir_base, **self.fixed_kwargs,
                             **trial_params)
        model.fit(self.train_dict, **self.fit_kwargs)
        score = model._eval()

        # keep a pointer to the weights folder
        trial.set_user_attr("model_dir", str(model.model_dir))

        # ------------- append to hp_trials.csv -----------------------
        rec = {"trial": trial.number, "loss": score, **trial_params}
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
