"""
CsvBarLoader
============

Reads OHLCV CSVs (plus optional “DGS10.csv”) and streams Nautilus Trader
`BarData` into the engine.  The loader is *frequency-agnostic* and honours the
`freq` string passed at construction (e.g. "1D", "15m", "2h", …).

Special rules
-------------
* A file named **DGS10.csv** is treated as 10-year US Treasury yields, *never*
  added to the tradeable universe.  The series is forward-filled onto the same
  business-day grid and published as `YieldCurveData`.
* All other CSVs become tradeable instruments; numeric columns automatically
  become features for UMIModel.

Usage
-----
loader = CsvBarLoader(
data_dir=Path("data"),
freq="15m",
tz="UTC",
)
engine.add_data_source(loader)

"""


from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Iterator

from nautilus_trader.data import Bar, BarType
from nautilus_trader.common.clock import LiveClock, SimulationClock
from nautilus_trader.common.data import DataCatalog


class CsvBarLoader:
    def __init__(
        self,
        data_dir: Path,
        freq: str,
        tz: str = "UTC",
        universe: Optional[List[str]] = None,
    ):
        self._dir = Path(data_dir).expanduser().resolve()
        if not self._dir.is_dir():
            raise FileNotFoundError(self._dir)

        self.freq = freq
        self.tz = tz
        self._bar_type = BarType.from_string(freq.upper())

        self._universe = (
            universe
            if universe is not None
            else sorted(
                p.stem.upper()
                for p in self._dir.glob("*.csv")
                if p.stem.upper() != "DGS10"
            )
        )

        self._yield_series: Optional[pd.Series] = None
        self._frames: Dict[str, pd.DataFrame] = {}
        self._load_all()

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    @property
    def universe(self) -> List[str]:
        return self._universe

    @property
    def rf_series(self) -> Optional[pd.Series]:
        return self._yield_series

    def data_catalog(self) -> DataCatalog:
        cat = DataCatalog()
        for sym in self._universe:
            cat.add_instrument(sym)  # String ID is fine
        return cat

    def bar_iterator(self) -> Iterator[Bar]:
        """Yields `Bar` objects globally sorted by timestamp ascending."""
        # Build multi-asset index
        idx = pd.Index(sorted({ts for f in self._frames.values() for ts in f.index}))
        idx = idx.tz_localize(self.tz)

        for ts in idx:
            for sym in self._universe:
                df = self._frames[sym]
                if ts not in df.index:
                    continue                   # missing bar – universe padding
                row = df.loc[ts]
                yield Bar(
                    instrument=sym,
                    bar_type=self._bar_type,
                    ts_epoch_ns=int(ts.value),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _load_all(self):
        for csv in self._dir.glob("*.csv"):
            name = csv.stem.upper()
            if name == "DGS10":
                self._yield_series = self._load_rf(csv)
                continue
            self._frames[name] = self._load_ohlcv(csv)

    def _load_ohlcv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        if "Date" not in df.columns:
            raise ValueError(f"{path} missing Date column")

        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # If Adj Close exists, rename to Close
        lower = df.columns.str.lower()
        if "adj close" in lower or "adj_close" in lower:
            df.rename(
                columns=lambda c: "Close"
                if c.lower() in ["adj close", "adj_close"]
                else c,
                inplace=True,
            )

        # Otherwise build one from dividends/splits if present
        elif {"dividends", "stock splits"}.issubset(lower):
            df = self._compute_adjusted_close(df)
            df.rename(columns={"Adj Close": "Close"}, inplace=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if "Close" not in numeric_cols:
            raise ValueError(f"{path} missing Close price")

        return df[numeric_cols]

    @staticmethod
    def _compute_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index().copy()
        adj_factor = 1.0
        factors = []
        for i in range(len(df) - 1, -1, -1):
            c, div, split = df.iloc[i][["Close", "Dividends", "Stock Splits"]]
            if split and split != 0:
                adj_factor /= 1.0 + split
            if div and div != 0:
                adj_factor *= (c - div) / c
            factors.append(adj_factor)
        df["Adj Close"] = df["Close"] * list(reversed(factors))
        return df

    def _load_rf(self, path: Path) -> pd.Series:
        rf = pd.read_csv(path, parse_dates=["observation_date"])
        rf.rename(columns={"observation_date": "Date"}, inplace=True)
        rf.set_index("Date", inplace=True)
        rf["DGS10"] = pd.to_numeric(rf["DGS10"], errors="coerce") / 100.0
        return rf["DGS10"].asfreq("B").fillna(method="ffill")