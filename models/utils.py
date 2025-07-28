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
import re
from pandas.tseries.frequencies import to_offset
from nautilus_trader.model.data.bar import BarSpecification           # Specification object
from nautilus_trader.model.c_enums.bar_aggregation import BarAggregation
from nautilus_trader.model.c_enums.price_type import PriceType        # BID / ASK / MID / LAST …

# def cache_to_dict(
#     cache: Cache,
#     tickers: List[str],
#     lookback: int,
#     bar_type: BarType,
#     ) -> Dict[str, pd.DataFrame]:
#     """
#     Return a *brand-new* `{ticker: DataFrame}` containing the last
#     `lookback` bars for each `ticker`.

#     The DataFrame columns exactly match the CSV loader used before:
#     ``["Open", "High", "Low", "Close", "Volume"]``
#     and the index is a timezone-aware `pd.DatetimeIndex`.

#     Parameters
#     ----------
#     cache     : Nautilus Trader in-memory database (already populated).
#     tickers   : Symbols we want – preserves the slot order the model uses.
#     lookback  : How many bars per symbol (typically L + pred_len + 1). Use 0 for all.
#     bar_type  : `BarType` instance (e.g. DAILY, 15MIN) the strategy trades.

#     Notes
#     -----
#     * This function now looks for FeatureBarData objects first,
#       falling back to regular Bar objects if features aren't available.
#     * Volume comes from `bar.volume` which is 0 for FX pairs but present
#       for stocks/crypto; no extra checks needed downstream.
#     """
#     out = {}
#     for sym in tickers:
#         feature_data = []
        
#         # Get all data objects for this symbol
#         for obj in cache.data():
#             if hasattr(obj, 'instrument_id') and obj.instrument_id.symbol == sym:
#                 if isinstance(obj, FeatureBarData):
#                     feature_data.append(obj)
                    
#         # If we have feature data, use it to build the full DataFrame
#         if feature_data:
#             feature_data = sorted(feature_data, key=lambda x: x.ts_event)[-lookback:]
            
#             # Build DataFrame from features
#             rows = []
#             for fd in feature_data:
#                 rows.append(fd.features)
                
#             if rows:
#                 df = pd.DataFrame(
#                     rows,
#                     index=pd.DatetimeIndex(
#                         [pd.Timestamp(fd.ts_event, unit='ns', tz='UTC') for fd in feature_data]
#                     )
#                 )
#                 out[sym] = df
#         else:
#             # Fallback to regular bars if no feature data
#             bars = []
#             for b in cache.bars(bar_type):
#                 if b.instrument_id.symbol == sym:
#                     bars.append(b)
                    
#             bars = sorted(bars, key=lambda x: x.ts_event)[-lookback:]
            
#             if bars:
#                 df = pd.DataFrame({
#                     'Open': [b.open for b in bars],
#                     'High': [b.high for b in bars],
#                     'Low': [b.low for b in bars],
#                     'Close': [b.close for b in bars],
#                     'Volume': [b.volume for b in bars],
#                 }, index=pd.DatetimeIndex(
#                     [pd.Timestamp(b.ts_event, unit='ns', tz='UTC') for b in bars]
#                 ))
#                 out[sym] = df
                
#     return out


def freq2barspec(freq: str,
                    price_type: PriceType = PriceType.LAST
                   ) -> BarSpecification:
    """
    Convert frequency strings such as '1m', '15S', '1M' into a NautilusTrader
    BarSpecification(step, aggregation, price_type).

    Parameters
    ----------
    freq : str
        Time-based frequency string (integer step + unit code).
        Minute/month ambiguity follows pandas conventions:
        - lower-case  'm'  → minutes
        - upper-case  'M'  → months
    price_type : PriceType, default PriceType.LAST
        BID / ASK / MID / LAST – whatever you need for the bar stream.

    Returns
    -------
    BarSpecification
        The bar spec you can feed straight into `BarType`, strategy subscriptions,
        or catalog loaders.

    Raises
    ------
    ValueError
        If the syntax is invalid, the unit is unsupported, or step ≤ 0.
    """
    
    # ------------------------------ mapping table ------------------------------ #
    # Key = canonical unit-code from the user's string (case-sensitive where needed)
    # Value = (BarAggregation enum, is_uppercase_sensitive)
    _BAR_AGG_MAP = {
        "ms":  BarAggregation.MILLISECOND,
        "s":   BarAggregation.SECOND,
        "sec": BarAggregation.SECOND,
        "min": BarAggregation.MINUTE,     # preferred spelling (“1min”)
        "m":   BarAggregation.MINUTE,     # lower-case m = minute   (pandas-style “1m”)
        "h":   BarAggregation.HOUR,
        "d":   BarAggregation.DAY,
        "w":   BarAggregation.WEEK,
        "M":   BarAggregation.MONTH,      # UPPER-case M = month    (same distinction as pandas)
    }

    _FREQ_RE = re.compile(r"^\s*([+-]?\d+)\s*([a-zA-Z]+)\s*$")

    m = _FREQ_RE.fullmatch(freq)
    if not m:
        raise ValueError(f"Bad frequency syntax: {freq!r} (expected e.g. '15min', '4H')")

    step = int(m.group(1))
    if step <= 0:
        raise ValueError("Bar step must be a positive integer.")

    unit = m.group(2)
    # Minute/month case-sensitivity: keep original case so 'M' ≠ 'm'
    key = unit if unit == "M" else unit.lower()

    if key not in _BAR_AGG_MAP:
        raise ValueError(f"Unsupported bar unit {unit!r}")

    aggregation = _BAR_AGG_MAP[key]
    return BarSpecification(step, aggregation, price_type)



def freq2pdoffset(freq_str: str):
    """
    Convert 'count + unit' strings to a pandas DateOffset.

    Parameters
    ----------
    freq_str : str
        Examples: '1B', '15min', '0.5h', '2W', '3MS'

    Returns
    -------
    pandas.tseries.offsets.BaseOffset
        Suitable wherever pandas expects a frequency/offset.

    Raises
    ------
    ValueError
        If the string cannot be parsed or the unit is unsupported.
    """

    # ---- configurable map of user-friendly “unit codes” → pandas offset aliases
    _FREQ_MAP = {
        # sub-day (post-2.2 recommended lower-case spellings)
        "ns": "ns",            # nanoseconds
        "us": "us",            # microseconds
        "ms": "ms",            # milliseconds
        "s":  "s",             # seconds
        "min": "min",          # minutes
        "h":   "h",            # hours
        # daily & above
        "d":  "D",             # calendar days
        "B":  "B",             # business days
        "w":  "W",             # weeks (Mon-Sun anchored via suffix – see tips below)
        "M":  "M",             # month-end
        "MS": "MS",            # month-start
        "Q":  "Q",             # quarter-end
        "QS": "QS",            # quarter-start
        "Y":  "A",             # year-end (alias 'A' = 'Y')
        "YS": "AS"             # year-start
    }

    _unit_re = re.compile(r"^\s*([+-]?\d+(?:\.\d*)?)\s*([a-zA-Z]+)\s*$")
    m = _unit_re.fullmatch(freq_str)
    if not m:
        raise ValueError(f"Bad frequency syntax: {freq_str!r}")

    count, unit = m.groups()
    unit = unit.lower() if unit not in ("B", "M", "MS", "Q", "QS", "Y", "YS") else unit  # preserve case where required

    if unit not in _FREQ_MAP:
        raise ValueError(f"Unknown frequency unit {unit!r}")

    offset_alias = f"{count}{_FREQ_MAP[unit]}"
    return to_offset(offset_alias)





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
