"""
Helpers that extract a `{ticker: DataFrame}` snapshot from the
Nautilus Trader runtime `Cache`.
"""
from __future__ import annotations
from typing import Dict, List
import pandas as pd
from nautilus_trader.cache.cache import Cache
from nautilus_trader.model.data.bar import BarType   # public NT API
import torch

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
    lookback  : How many bars per symbol (typically L + pred_len + 1). Use 0 for all.
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


# --------------------------------------------------------------------------- #
# 1. Dataset that yields sliding windows ready for the factor blocks          #
# --------------------------------------------------------------------------- #
class SlidingWindowDataset(Dataset):
    """
    Yields tuples:
      prices_seq : (L+1, I)
      feat_seq   : (L+1, I, F)
      target     : (I, 1)   -- next-bar return (t = L)  [optional]
    """
    def __init__(
        self,
        panel: torch.Tensor,      # (T, I, F) F includes "close" at close_idx position
        active: torch.Tensor,     # (I, T) bool mask for active stocks
        window_len: int,
        pred_len: int,
        close_idx: int = 3,  # usually "close" is at index 3
        with_target: bool = True,
    ):
        self.active = active  
        self.panel = panel
        self.L = window_len
        self.pred = pred_len
        self.close_idx = close_idx
        self.with_target = with_target

    def __len__(self):
        return self.panel.size(0) - self.L - self.pred + 1

    def __getitem__(self, idx):
        seq = self.panel[idx : idx + self.L + 1]             # (L+1,I,F)
        prices = seq[..., self.close_idx]                    # (L+1,I)
        a = self.active[:, idx + self.L]             # (I) activity at t = L  

        if self.with_target:
            tgt_close = self.panel[idx + self.L : idx + self.L + self.pred,
                                    :, self.close_idx]       # (pred,I)
            ret = (tgt_close[-1] - tgt_close[0]) / (tgt_close[0] + 1e-8)
            ret = ret.unsqueeze(-1)                          # (I,1)
            return prices, seq, ret, a
        return prices, seq, a
