"""
CsvBarLoader
============

Streams BOTH
    • `Bar`                 – open, high, low, close, volume
    • `FeatureBarData`      – *all* numeric columns present in the CSV

into Nautilus Trader’s engine / cache.  The model will later read those
feature objects; if they are missing (e.g. user supplied only OHLCV files),
the adapter automatically falls back to the plain Bar fields.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
from nautilus_trader.model.data import Bar, BarType
#from nautilus_trader.model import Data  # base class for custom data


# ------------------------------------------------------------------ #
# Custom data class
# ------------------------------------------------------------------ #
class FeatureBarData(Data):
    """
    Carries the *entire* numeric feature vector for one instrument at one
    timestamp.  Stored in the cache so that the model can rebuild the same
    tensor it saw during training.
    """

    __slots__ = ("_instrument", "_ts_event", "_ts_init", "_features")

    def __init__(self, instrument: str, ts_event: int, ts_init: int, features: np.ndarray):
        # CHANGED: Call Data.__init__ with proper timestamp parameters
        self._ts_event = ts_event
        self._ts_init = ts_init
        self._instrument = instrument
        self._features = features          # 1-D numpy array

    # -- properties required by Nautilus ---------------------------------
    @property
    def instrument_id(self):
        return self._instrument

    @property
    def ts_event(self) -> int:
        return self._ts_event

    @property
    def ts_init(self) -> int:
        return self._ts_init

    # -- convenience -----------------------------------------------------
    @property
    def features(self) -> np.ndarray:
        return self._features


# ------------------------------------------------------------------ #
# Loader
# ------------------------------------------------------------------ #
class CsvBarLoader:
    def __init__(
        self,
        root: Path,
        freq: str,
        tz: str = "UTC",
        universe: Optional[List[str]] = None,
    ):
        self._root = Path(root).expanduser().resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(self._root)

        self.freq     = freq
        self.tz       = tz
        self.bar_type = BarType.from_string(freq.upper())

        stock_dir = self._root / "stocks"
        bond_dir  = self._root / "bonds"
        self._stock_files = sorted(stock_dir.glob("*.csv"))
        self._rf_file     = bond_dir / "DGS10.csv"

        self._universe = (
            universe
            if universe is not None
            else [p.stem.upper() for p in self._stock_files]
        )

        # parse CSVs once to numeric frames
        self._frames: Dict[str, pd.DataFrame] = {
            p.stem.upper(): self._parse_stock_csv(p) for p in self._stock_files
        }
        self._yield_series: Optional[pd.Series] = (
            self._parse_rf_csv(self._rf_file) if self._rf_file.exists() else None
        )

    # ------------------------------------------------------------------ #
    # public ----------------------------------------------------------------
    # ------------------------------------------------------------------ #
    @property
    def universe(self) -> List[str]:
        return self._universe

    @property
    def rf_series(self) -> Optional[pd.Series]:
        return self._yield_series

    def bar_iterator(self) -> Iterator[Bar]:
        """
        Yield **two** events per symbol/time-stamp:
            1. FeatureBarData   (all numeric columns)
            2. Bar              (OHLCV only)

        They share the VERY SAME ts_event so the cache keeps them aligned.
        """
        all_ts = sorted({ts for f in self._frames.values() for ts in f.index})

        for ts in all_ts:
            epoch_ns = int(ts.tz_localize(self.tz).value)
            ts_init = epoch_ns  # Use same as event time for simplicity

            for sym in self._universe:
                df = self._frames[sym]
                if ts not in df.index:
                    continue        # missing bar – universe padding

                row            = df.loc[ts]
                numeric_vector = row.values.astype(np.float32)

                # ---- 1) full-feature object -------------------------
                yield FeatureBarData(
                    instrument=sym,
                    ts_event=epoch_ns,
                    ts_init=ts_init,
                    features=numeric_vector,
                )

                # ---- 2) traditional Bar -----------------------------
                # Missing columns default to 0.0
                def _get(col): return float(row[col]) if col in row else 0.0
                # Fixed Bar constructor parameters
                yield Bar(
                    bar_type=self.bar_type,
                    instrument_id=sym,
                    open=_get("Open"),
                    high=_get("High"),
                    low=_get("Low"),
                    close=_get("Adj Close") if "Adj Close" in row else _get("Close"),
                    volume=_get("Volume"),
                    ts_event=epoch_ns,
                    ts_init=ts_init,
                )
    # ------------------------------------------------------------------ #
    # parsing helpers
    # ------------------------------------------------------------------ #
    def _parse_stock_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # -- adjusted close --------------------------------------------
        cols_lc = {c.lower(): c for c in df.columns}
        if "adj close" in cols_lc:
            df.rename(columns={cols_lc["adj close"]: "Adj Close"}, inplace=True)
        else:
            df = self._compute_adj_close(df)

        return df.select_dtypes(include=[np.number])   # keep ALL numeric columns

    @staticmethod
    def _compute_adj_close(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index().copy()
        factors = np.ones(len(df))
        div  = df.get("Dividends", pd.Series(0.0, index=df.index)).fillna(0.0)
        splt = df.get("Stock Splits", pd.Series(0.0, index=df.index)).fillna(0.0)

        factor = 1.0
        for i in range(len(df) - 1, -1, -1):
            c = df.iloc[i]["Close"]
            factor *= (1.0 - div.iat[i] / c) / (1.0 + splt.iat[i])
            factors[i] = factor
        df["Adj Close"] = df["Close"] * factors
        return df

    @staticmethod
    def _parse_rf_csv(path: Path) -> pd.Series:
        rf = pd.read_csv(path, parse_dates=["observation_date"])
        rf.rename(columns={"observation_date": "Date"}, inplace=True)
        rf.set_index("Date", inplace=True)
        rf["DGS10"] = pd.to_numeric(rf["DGS10"], errors="coerce") / 100.0
        return rf["DGS10"].asfreq("B").ffill()
