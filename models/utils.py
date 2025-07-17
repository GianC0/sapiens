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
    * This function now looks for FeatureBarData objects first,
      falling back to regular Bar objects if features aren't available.
    * Volume comes from `bar.volume` which is 0 for FX pairs but present
      for stocks/crypto; no extra checks needed downstream.
    """
    out = {}
    for sym in tickers:
        # CHANGED: Try to get FeatureBarData first
        feature_data = []
        bars_data = []
        
        # Get all data objects for this symbol
        for obj in cache.data():
            if hasattr(obj, 'instrument_id') and obj.instrument_id.symbol == sym:
                if isinstance(obj, FeatureBarData):
                    feature_data.append(obj)
                    
        # If we have feature data, use it to build the full DataFrame
        if feature_data:
            feature_data = sorted(feature_data, key=lambda x: x.ts_event)[-lookback:]
            
            # Build DataFrame from features
            rows = []
            for fd in feature_data:
                rows.append(fd.features)
                
            if rows:
                df = pd.DataFrame(
                    rows,
                    index=pd.DatetimeIndex(
                        [pd.Timestamp(fd.ts_event, unit='ns', tz='UTC') for fd in feature_data]
                    )
                )
                out[sym] = df
        else:
            # Fallback to regular bars if no feature data
            bars = []
            for bar in cache.bars(bar_type):
                if bar.instrument_id.symbol == sym:
                    bars.append(bar)
                    
            bars = sorted(bars, key=lambda x: x.ts_event)[-lookback:]
            
            if bars:
                df = pd.DataFrame({
                    'Open': [b.open for b in bars],
                    'High': [b.high for b in bars],
                    'Low': [b.low for b in bars],
                    'Close': [b.close for b in bars],
                    'Volume': [b.volume for b in bars],
                }, index=pd.DatetimeIndex(
                    [pd.Timestamp(b.ts_event, unit='ns', tz='UTC') for b in bars]
                ))
                out[sym] = df
                
    return out
