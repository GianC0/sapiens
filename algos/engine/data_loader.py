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
import logging
import numpy as np
import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USD,EUR
from models.utils import freq2barspec
#from nautilus_trader.model import Data  # base class for custom data

logger = logging.getLogger(__name__)


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
        columns_to_load: str = "candles", 
        adjust: bool = False,
        benchmark: str = "SPY"
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

        stock_dir = self._root / "stocks" / cfg["calendar"]
        self._stock_files = sorted(stock_dir.glob("*.csv"))
        benchmarks_file = self._root / "benchmarks" / "bond_etfs.csv"

        self._universe: List[str] = (
            universe
            if universe is not None
            else [p.stem.upper() for p in self._stock_files]
        )   

        # parse CSVs once to numeric frames
        self._frames: Dict[str, pd.DataFrame] = {}    
        self._instruments: Dict[str, Equity] = {}
        
        for path in self._stock_files:
            symbol = path.stem.upper()
            if symbol in self._universe:
                df = self._parse_stock_csv(path, adjust = adjust)
                if columns_to_load == "candles":
                    available_cols = [col for col in ["Open","High","Low","Close","Volume"] if col in df.columns]
                    self._frames[symbol] = df[available_cols]
                else:
                    self._frames[symbol] = df
                # Create instrument for each stock
                self._instruments[symbol] = self._create_equity_instrument(symbol)
        
        # Load benchmark data
        
        self._benchmark_data = None
        self._risk_free_df = None
        if benchmarks_file.exists():
            df = pd.read_csv(benchmarks_file, parse_dates=['Date'])
            #df.rename(columns={"dt": "Date"}, inplace=True)
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            # Risk-free rate from US Treasury 3-months yield column (already in %)
            #self._risk_free_df = df[['us3m']] / 100.0  # Double brackets → DataFrame
            #self._risk_free_df.rename(columns={"us3m": "risk_free"}, inplace=True)
            #self._risk_free_df.sort_index(inplace=True)
            
            # Risk-free rate from US Treasury 3-months by following SGOV ETF index.
            self._risk_free_df = df[['SGOV']].copy() # Double brackets → DataFrame

            # Ensure time order
            self._risk_free_df.sort_index(inplace=True)

            # Convert to daily returns (percentage change)
            self._risk_free_df['return'] = self._risk_free_df['SGOV'].pct_change()
            self._risk_free_df.rename(columns={"SGOV": "risk_free"}, inplace=True)
            self._risk_free_df.sort_index(inplace=True)

            
            # Benchmark using S&P500
            if benchmark == "SPY":
                self._benchmark_data = df[['sp500', 'sp500_volume']].copy()
                self._benchmark_data = self._benchmark_data.rename(columns={"sp500": "Benchmark", 'sp500_volume': 'Volume'})
                self._benchmark_returns = self._benchmark_data['Benchmark'].pct_change()
            
            # Benchmark using DowJones Industrial Average
            elif benchmark == "DJIA":
                self._benchmark_data = df[['djia', 'djia_volume']].copy()
                self._benchmark_data = self._benchmark_data.rename(columns={"djia": "Benchmark", 'djia_volume': 'Volume'})
                self._benchmark_returns = self._benchmark_data['sp500'].pct_change()

            # Defaulting to S&P500
            else:
                logger.warning(f"{benchmark} is not a valid benchmark... Using SPY instead")
                self._benchmark_data = df[['sp500', 'sp500_volume']].copy()
                self._benchmark_data = self._benchmark_data.rename(columns={"sp500": "Benchmark", 'sp500_volume': 'Volume'})
                self._benchmark_returns = self._benchmark_data['Benchmark'].pct_change()
            
            self._benchmark_data.sort_index(inplace=True)
            self._benchmark_returns.sort_index(inplace=True)




    def _create_equity_instrument(self, symbol: str) -> Equity:
        """Create an Equity instrument for the given symbol."""
        instrument_id = InstrumentId(
            symbol=Symbol(symbol),
            venue=self.venue
        )
        
        assert self.cfg["currency"] in (USD,EUR)

        # TODO: double check price precision.
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
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
    def benchmark_returns(self) -> Optional[pd.Series]:
        return self._benchmark_returns
    
    @property
    def instruments(self) -> Dict[str, Equity]:
        """Return all created instruments."""
        return self._instruments

    @property
    def risk_free_df(self) -> Optional[pd.DataFrame]:
        """Return DataFrame with column risk_free"""
        return self._risk_free_df

    def bar_iterator(self) -> Iterator[Bar]:
        """
        Yield **two** events per symbol/time-stamp:
            1. FeatureBarData   (all numeric columns)
            2. Bar              (OHLCV with adjusted close)

        They share the VERY SAME ts_event so the cache keeps them aligned.
        """
        all_ts = sorted({ts for f in self._frames.values() for ts in f.index})

        for ts in all_ts:
            ts_init = pd.to_datetime([ts]).astype(int).item()

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
                    ts_event=ts_init,
                    ts_init=ts_init,
                    features=numeric_vector,
                )

                # ---- 2) traditional Bar with ADJUSTED CLOSE ---------
                # Always use adjusted close as the close price
                def _get(col): return float(row[col]) if col in row else 0.0
                
                yield Bar(
                    bar_type=BarType(
                        instrument_id=instrument.id,
                        bar_spec= freq2barspec(self.cfg["freq"])
                        ),
                    open=Price.from_str(f"{_get('Open'):.2f}"),
                    high=Price.from_str(f"{_get('High'):.2f}"),
                    low=Price.from_str(f"{_get('Low'):.2f}"),
                    close=Price.from_str(f"{_get('Close'):.2f}"),  # Always use adjusted close
                    volume=Quantity.from_int(int(_get('Volume'))),
                    ts_event=ts_init,
                    ts_init=ts_init,
                )
    
    def bars(self) -> List[Bar]:
        """
        Returns List of either FeatureBarData or Bar :
            1. FeatureBarData   (all numeric columns)
            2. Bar              (OHLCV with adjusted close)

        They share the VERY SAME ts_event so the cache keeps them aligned.
        """
        all_ts = sorted({ts for f in self._frames.values() for ts in f.index})

        for ts in all_ts:
            ts_init = pd.to_datetime([ts]).astype(int).item()

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
                    ts_event=ts_init,
                    ts_init=ts_init,
                    features=numeric_vector,
                )

                # ---- 2) traditional Bar with ADJUSTED CLOSE ---------
                # Always use adjusted close as the close price
                def _get(col): return float(row[col]) if col in row else 0.0
                
                yield Bar(
                    bar_type=BarType(
                        instrument_id=instrument.id,
                        bar_spec= freq2barspec(self.cfg["freq"])
                        ),
                    open=Price.from_str(f"{_get('Open'):.2f}"),
                    high=Price.from_str(f"{_get('High'):.2f}"),
                    low=Price.from_str(f"{_get('Low'):.2f}"),
                    close=Price.from_str(f"{_get('Close'):.2f}"),  # Always use adjusted close
                    volume=Quantity.from_int(int(_get('Volume'))),
                    ts_event=ts_init,
                    ts_init=ts_init,
                )
        ret
    # ------------------------------------------------------------------ #
    # parsing helpers
    # ------------------------------------------------------------------ #
    def _parse_stock_csv(self, path: Path, adjust: bool) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Adjust OHLC on stock splits and dividends
        if adjust:
            df = self._compute_adj_ohlc(df)

        return df.select_dtypes(include=[np.number])   # keep ALL numeric columns

    @staticmethod
    def _compute_adj_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index().copy()
        factors = np.ones(len(df))
        div  = df.get("Dividends", pd.Series(0.0, index=df.index)).fillna(0.0)
        splt = df.get("Stock Splits", pd.Series(0.0, index=df.index)).fillna(0.0)

        factor = 1.0
        for i in range(len(df) - 1, -1, -1):
            c = df.iloc[i]["Close"]
            factor *= (1.0 - div.iat[i] / c) / (1.0 + splt.iat[i])
            factors[i] = factor
        df["Open"] = df["Open"] * factors
        df["High"] = df["High"] * factors
        df["Low"] = df["Low"] * factors
        df["Close"] = df["Close"] * factors


        return df

    @staticmethod
    def _parse_rf_csv(path: Path) -> pd.Series:
        rf = pd.read_csv(path, parse_dates=["observation_date"])
        rf.rename(columns={"observation_date": "Date"}, inplace=True)
        rf.set_index("Date", inplace=True)
        rf["DGS10"] = pd.to_numeric(rf["DGS10"], errors="coerce") / 100.0
        return rf["DGS10"].asfreq("B").ffill()
    
