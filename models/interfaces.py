"""
Shared typing protocol: any model which implements this interface can
be dropped into UMIStrategy without editing the strategy code.
"""
from __future__ import annotations

from typing import Protocol, Dict, Any, Union, runtime_checkable
import pandas as pd
from nautilus_trader.cache.cache import Cache   # NT built-in cache ¹

DataDict = Dict[str, pd.DataFrame]
DataSource = Union[DataDict, Cache]             # accepted everywhere


@runtime_checkable
class MarketModel(Protocol):
    """
    The minimum contract – **no inheritance required**.
    A class *quacks* if it has the following public members.
    """

    # two public attributes readable by the caller
    L: int          # look-back window in bars (≥ 1)
    pred_len: int   # forecast horizon in bars

    # -------- mandatory methods --------
    def fit(self, data: DataSource, *, n_epochs: int = 1, **kw) -> None:
        """(Re)train the model on `data`."""
        ...

    def update(self, data: DataSource, **kw) -> None:
        """Light-weight maintenance (e.g. decide if re-training is needed)."""
        ...

    def predict(self, data: DataSource, **kw) -> Dict[str, float]:
        """
        Return next-bar return forecasts *per ticker*.
        Keys must be exactly the tickers present in `data`.
        """
        ...

    # -------- optional persistence helpers --------
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, sd: Dict[str, Any]) -> None: ...
