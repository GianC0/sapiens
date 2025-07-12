"""
Helpers that extract a `{ticker: DataFrame}` snapshot from the
Nautilus Trader runtime `Cache`.
"""
from __future__ import annotations
from typing import Dict, List
import pandas as pd
from nautilus_trader.cache.cache import Cache
from nautilus_trader.model.data.bar import BarType   # public NT API


def cache_to_dict(
    cache: Cache,
    tickers: List[str],
    lookback: int,
    bar_type: BarType,
) -> Dict[str, pd.DataFrame]:
    """
    Return a *brand-new* `{ticker: DataFrame}` containing the last
    `lookback` bars for each `ticker`.

    The DataFrame columns exactly match the CSV loader used before:
    ``["Open", "High", "Low", "Close", "Volume"]``
    and the index is a timezone-aware `pd.DatetimeIndex`.

    Parameters
    ----------
    cache     : Nautilus Trader in-memory database (already populated).
    tickers   : Symbols we want – preserves the slot order the model uses.
    lookback  : How many bars per symbol (typically L + pred_len + 1).
    bar_type  : `BarType` instance (e.g. DAILY, 15MIN) the strategy trades.

    Notes
    -----
    * Missing symbols are silently skipped – the caller decides what to do.
    * Volume comes from `bar.volume` which is 0 for FX pairs but present
      for stocks/crypto; no extra checks needed downstream.
    """
    out: Dict[str, pd.DataFrame] = {}

    # The cache stores *all* symbols in a single ring per BarType.
    # We have to filter by `bar.instrument.symbol` to build per-ticker frames.
    for sym in tickers:
        rows = []
        ts_index = []

        for bar in cache.bars(bar_type, limit=lookback):
            if bar.instrument.symbol != sym:
                continue
            rows.append([bar.open, bar.high, bar.low, bar.close, bar.volume])
            ts_index.append(bar.ts_event.to_pydatetime())  # UTC datetime

        if rows:  # skip empty symbols
            df = pd.DataFrame(
                rows,
                index=pd.DatetimeIndex(ts_index, tz="UTC"),
            )
            out[sym] = df

    return out
