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
import torch
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.c_enums import AssetClass, CurrencyType
from nautilus_trader.model.currencies import USD,EUR
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from ....models.utils import freq2barspec
#from nautilus_trader.model import Data  # base class for custom data


# ------------------------------------------------------------------ #
# Custom data class
# ------------------------------------------------------------------ #
class FeatureBarData:
    """
    Carries the *entire* numeric feature vector for one instrument at one
    timestamp.  Stored in the cache so that the model can rebuild the same
    tensor it saw during training.
    """

    __slots__ = ("_instrument", "_ts_event", "_ts_init", "_features")

    def __init__(self, instrument: str, ts_event: int, ts_init: int, features: np.ndarray):
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
        cfg: Dict,
        venue_name: str="SIM",
        tz: str = "UTC",
        universe: Optional[List[str]] = None,
    ):
        root = Path(cfg["data_dir"])
        freq = cfg["freq"]
        self._root = Path(root).expanduser().resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(self._root)

        self.freq     = freq
        self.tz       = tz
        self.venue = Venue(venue_name)
        self.cfg = cfg

        stock_dir = self._root / "stocks"
        bond_dir  = self._root / "bonds"
        self._stock_files = sorted(stock_dir.glob("*.csv"))
        self._rf_file     = bond_dir / "DGS10.csv"

        self._universe: List[str] = (
            universe
            if universe is not None
            else [p.stem.upper() for p in self._stock_files]
        )   

        # parse CSVs once to numeric frames
        self._frames: Dict[str, pd.DataFrame] = {}    
        self._instruments: Dict[str, Equity] = {}
        
        for p in self._stock_files:
            symbol = p.stem.upper()
            if symbol in self._universe:
                df = self._parse_stock_csv(p)
                self._frames[symbol] = df
                # Create instrument for each stock
                self._instruments[symbol] = self._create_equity_instrument(symbol)
        self._yield_series: Optional[pd.Series] = (
            self._parse_rf_csv(self._rf_file) if self._rf_file.exists() else None
        )

    def _create_equity_instrument(self, symbol: str) -> Equity:
        """Create an Equity instrument for the given symbol."""
        instrument_id = InstrumentId(
            symbol=Symbol(symbol),
            venue=self.venue
        )
        
        # TODO: double check price precision.
        return Equity(
            instrument_id=instrument_id,
            native_symbol=Symbol(symbol),
            currency=self.cfg["currency"],
            price_precision=2,  # Standard for US equities
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_int(1),
            # margin_init=Money(0, USD),  # No margin requirement for cash account
            # margin_maint=Money(0, USD),
            # maker_fee=Money(0, USD),    # Fees handled by commission model
            # taker_fee=Money(0, USD),
            ts_event=0,
            ts_init=0,
        )
    # ------------------------------------------------------------------ #
    # public ----------------------------------------------------------------
    # ------------------------------------------------------------------ #
    @property
    def universe(self) -> List[str]:
        return self._universe

    @property
    def instruments(self) -> Dict[str, Equity]:
        """Return all created instruments."""
        return self._instruments

    @property
    def rf_series(self) -> Optional[pd.Series]:
        return self._yield_series

    def bar_iterator(self) -> Iterator[Bar]:
        """
        Yield **two** events per symbol/time-stamp:
            1. FeatureBarData   (all numeric columns)
            2. Bar              (OHLCV with adjusted close)

        They share the VERY SAME ts_event so the cache keeps them aligned.
        """
        all_ts = sorted({ts for f in self._frames.values() for ts in f.index})

        for ts in all_ts:
            epoch_ns = int(ts.tz_localize(self.tz).value)
            ts_init = epoch_ns  # Use same as event time for simplicity

            for sym in self._universe:
                if sym not in self._frames:
                    continue
                    
                df = self._frames[sym]
                if ts not in df.index:
                    continue        # missing bar – universe padding

                row = df.loc[ts]
                numeric_vector = row.values.astype(np.float32)
                
                instrument = self._instruments[sym]

                # ---- 1) full-feature object -------------------------
                yield FeatureBarData(
                    instrument=sym,
                    ts_event=epoch_ns,
                    ts_init=ts_init,
                    features=numeric_vector,
                )

                # ---- 2) traditional Bar with ADJUSTED CLOSE ---------
                # Always use adjusted close as the close price
                def _get(col): return float(row[col]) if col in row else 0.0
                
                yield Bar(
                    bar_type=BarType(
                        instrument_id=instrument.id,
                        bar_spec= freq2barspec(self.cfg["freq"])),
                    instrument_id=instrument.id,
                    open=Price.from_str(f"{_get('Open'):.2f}"),
                    high=Price.from_str(f"{_get('High'):.2f}"),
                    low=Price.from_str(f"{_get('Low'):.2f}"),
                    close=Price.from_str(f"{_get('Adj Close'):.2f}"),  # Always use adjusted close
                    volume=Quantity.from_int(int(_get('Volume'))),
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

        # Ensure we have adjusted close - compute it if not present
        cols_lc = {c.lower(): c for c in df.columns}
        if "adj close" in cols_lc:
            df.rename(columns={cols_lc["adj close"]: "Adj Close"}, inplace=True)
        else:
            df = self._compute_adj_close(df)

        # Verify we have the adjusted close column
        if "Adj Close" not in df.columns:
            raise ValueError(f"Failed to compute adjusted close for {path.stem}")

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
    
    def build_universe_dataframe(
        self,
        universe: List[str],
        end_time: pd.Timestamp,
        train_offset: pd.DateOffset,
        freq: str = "1B",
        feature_dim: int = 5,
    ) -> (torch.Tensor, torch.Tensor, List[pd.Timestamp]):
        """
        Build aligned universe DataFrame as tensor.
        
        Returns:
            panel: torch.Tensor of shape (T, I, F)
            active_mask: torch.Tensor of shape (T, I) indicating active instruments
            timestamps: List of timestamps for each T
        """
        I = len(universe)
        F = feature_dim
        
        # Build time index
        end_time = pd.Timestamp(end_time, tz='UTC')
        start_time = end_time - train_offset
        timestamps = pd.date_range(end=end_time, start=start_time, freq=freq, tz=self.tz)
        T = len(timestamps)

        
        # Initialize tensors
        panel = torch.zeros(T, I, F, dtype=torch.float32)
        active_mask = torch.zeros(T, I, dtype=torch.bool)
        
        # Fill data for each instrument in universe order
        for i, symbol in enumerate(universe):
            if symbol not in self._frames:
                continue
                
            df = self._frames[symbol]
            
            for t, ts in enumerate(timestamps):
                # Find closest available data point (forward fill logic)
                mask = df.index <= ts
                if mask.any():
                    # Get most recent data up to this timestamp
                    idx = df.index[mask][-1]
                    row = df.loc[idx]
                    
                    # Check if data is stale (more than 5 business days old)
                    days_old = (ts - idx).days
                    if days_old <= 7:  # Allow up to 1 week of staleness
                        panel[t, i, :] = torch.tensor(row.values, dtype=torch.float32)
                        active_mask[t, i] = True
        
        return panel, active_mask, timestamps.tolist()
