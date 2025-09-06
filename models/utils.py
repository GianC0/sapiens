"""
Helpers that extract a `{ticker: DataFrame}` snapshot from the
Nautilus Trader runtime `Cache`.
"""
from __future__ import annotations
from sqlite3.dbapi2 import Timestamp
from typing import Dict, List
import pandas as pd
import numpy as np
from nautilus_trader.cache.cache import Cache
import torch
from torch.utils.data import Dataset
import re
import json
from pandas.tseries.frequencies import to_offset
from nautilus_trader.model.data import BarSpecification, BarType, BarAggregation          
from nautilus_trader.core.rust.model import PriceType       # Prince type =  BID / ASK / MID / LAST …


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
        "D":  "D",             # calendar days
        "B":  "B",             # business days
        "w":  "W",             # weeks (Mon-Sun anchored via suffix – see tips below)
        "M":  "ME",            # month-end
        "ME":  "ME",           # month-end
        "BME": "BME",          # business month-end
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
    #unit = unit.lower if unit not in ("B", "M", "MS","ME", "BME", "Q", "QS", "Y", "YS") else unit  # preserve case where required

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
      seq : (L+1, I, F)    -- sequence of features for all instruments (L+1 timestamps)
      prices   : (L+1, I)  -- prices at each timestamp in the window
      ret     : (I, 1)     -- next-bar return (t = L)  [optional]
      active_mask : (I,)   -- True if instrument is active at last timestamp in window
    """
    def __init__(
        self,
        panel: torch.Tensor,      # (T, I, F) F includes "close" at target_idx position
        window_len: int,
        pred_len: int,
        target_idx: int = 3,  # usually "close" is at index 3
        with_target: bool = True,
    ):
        self.panel = panel
        self.L = window_len
        self.pred = pred_len
        self.target_idx = target_idx
        self.with_target = with_target
        self.n_windows = max(0, self.panel.size(0) - self.L - (self.pred if with_target else 0))

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):

        if idx >= self.n_windows:
            raise IndexError(f"Index {idx} out of range for {self.n_windows} windows")

        seq = self.panel[idx : idx + self.L + 1]             # (L+1,I,F)
        prices = seq[..., self.target_idx]                   # (L+1,I)
        ret = None                                         # instrument return is NaN if this is walk-farward
        # Compute active mask for this window (based on last timestamp in window)
        active_mask = ~torch.all(torch.isnan(seq[-1]), dim=1)  # (I,)
        if self.with_target:
            tgt_close = self.panel[idx + self.L : idx + self.L + self.pred,
                                    :, self.target_idx]       # (pred,I)
            ret = (tgt_close[-1] - tgt_close[0]) / (tgt_close[0] + 1e-8)  # (I,)
        return prices, seq, ret, active_mask


def build_input_tensor( 
    data: Dict[str, pd.DataFrame],
    timestamps: pd.DatetimeIndex,
    feature_dim: int,
    split_valid_timestamp: pd.Timestamp,
    device: torch.device,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, pd.DatetimeIndex]:
    """
    Efficiently build aligned tensor from dict of DataFrames.
    
    Args:
        data: Dict mapping ticker to DataFrame with features
        timestamps: DatetimeIndex of all timestamps
        universe: List of tickers to include (in order)
        feature_dim: Number of features per instrument (e.g. 5 for OHLCV)
        split_valid_timestamp: Timestamp to split training and validation data
        
    Returns 
        tuple for training and validation:
        - tensor: (T, I, F) tensor with NaN for missing data
        - mask: (I,) boolean mask for last timestamp only (True = active)
        
        
    Raises:
        ValueError: If universe is empty or all timestamps are NaN
    """
    
    universe = list(data.keys())

    if not universe:
        raise ValueError("Universe cannot be empty")
    
    # Pre-allocate array for efficiency
    F = feature_dim
    T = len(timestamps)
    I = len(universe)
    
    # Use float32 to save memory (matches PyTorch default)
    tensor_array = torch.full((T, I, F), float('nan'), dtype=torch.float32, device=device)
    
    # Fill in data for each instrument
    for i, ticker in enumerate(universe):
        assert data[ticker].shape == (T,F)  #not correct during last batch.
        # Copy values into pre-allocated array
        tensor_array[:, i, :] = torch.tensor(data[ticker].values, dtype=torch.float32, device=device)
    
    # Validate that we have at least some data
    if torch.isnan(tensor_array).all():
        raise ValueError("All data is NaN - no valid observations")
    
    # Check for completely empty timestamps
    empty_timestamps = torch.isnan(tensor_array).all(dim=2).all(dim=1)  # (T,)
    if torch.any(empty_timestamps):
        n_empty = torch.sum(empty_timestamps)
        raise ValueError(
            f"Found {n_empty} completely empty timestamps. "
        )
    
    
    # move to torch tensor
    train_idx = torch.as_tensor(timestamps <= split_valid_timestamp, dtype=torch.bool, device=device)

    # Instrument is active if it has ALL non-NaN value at last timestamp
    # train tensor and mask
    train_tensor = tensor_array[train_idx]  # (T_train, I, F)
    train_mask = ~torch.all(torch.isnan(train_tensor[-1, :, :]), dim=1)  # (I,)
    
    # valid tensor and mask
    valid_tensor = torch.tensor(0)
    valid_mask = torch.tensor(0)
    # Create train/valid split. if train_end == valid_end -> this is an update() call, or no validation 
    if split_valid_timestamp < timestamps[-1]:
        valid_tensor = tensor_array[~train_idx]  # (T - T_train, I, F)
        valid_mask = ~torch.all(torch.isnan(valid_tensor[-1, :, :]), dim=1)  # (I,)

    return (train_tensor, train_mask), (valid_tensor, valid_mask)

def build_pred_tensor(    
    data: Dict[str, pd.DataFrame],
    timestamps: pd.DatetimeIndex,
    feature_dim: int,
    device: torch.device,
    ) -> Tuple[torch.Tensor, torch.BoolTensor, pd.DatetimeIndex]:
    """
    Efficiently build aligned tensor from dict of DataFrames.
    
    Args:
        data: Dict mapping ticker to DataFrame with features
        timestamps: DatetimeIndex of all timestamps
        universe: List of tickers to include (in order)
        feature_dim: Number of features per instrument (e.g. 5 for OHLCV)
        split_valid_timestamp: Timestamp to split training and validation data
        
    Returns 
        tuple for training and validation:
        - tensor: (T, I, F) tensor with NaN for missing data
        - mask: (I,) boolean mask for last timestamp only (True = active)
        
        
    Raises:
        ValueError: If universe is empty or all timestamps are NaN
    """

    universe = list(data.keys())


    if not universe:
        raise ValueError("Universe cannot be empty")
    
    # Pre-allocate array for efficiency
    F = feature_dim
    T = len(timestamps)
    I = len(universe)
    
    # Use float32 to save memory (matches PyTorch default)
    tensor_array = torch.full((T, I, F), float('nan'), dtype=torch.float32, device=device)
    
    # Fill in data for each instrument
    for i, ticker in enumerate(universe):
        # Reindex to common timestamps (automatically fills NaN for missing)
        assert data[ticker].shape == (T,F), f"tiker dataframe size mismatch. Expected ({T},{F}), got {data[ticker].shape}"
        # Copy values into pre-allocated array
        tensor_array[:, i, :] = torch.tensor(data[ticker].values, dtype=torch.float32, device=device)
    
    # Validate that we have at least some data
    if torch.isnan(tensor_array).all():
        raise ValueError("All data is NaN - no valid observations")
    
    # Check for completely empty timestamps
    empty_timestamps = torch.isnan(tensor_array).all(dim=2).all(dim=1)  # (T,)
    if torch.any(empty_timestamps):
        n_empty = torch.sum(empty_timestamps)
        raise ValueError(
            f"Found {n_empty} completely empty timestamps. "
        )
    

    # train tensor and mask
    pred_tensor = tensor_array  # (T, I, F)
    # Instrument is active if it has ALL non-NaN value at last timestamp
    pred_mask = ~torch.all(torch.isnan(pred_tensor[-1, :, :]), dim=1)  # (I,)
    return pred_tensor, pred_mask

def yaml_safe(obj):
    """Return a plain Python structure safe for yaml.dump by JSON round-tripping.

    Non-JSON-serializable objects are replaced with obj.freqstr when present,
    otherwise with str(obj). This is small and readable but **lossy** for some types.
    Useful for pandas.Timedelta attributes
    """
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "freqstr", str(o))))
