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
import pandas_market_calendars as market_calendars
from decimal import Decimal
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USD,EUR
from models.utils import freq2barspec
from algos.engine.data_augmentation import FeatureBar
#from nautilus_trader.model import Data  # base class for custom data
import json
import pyarrow as pa
from nautilus_trader.serialization.arrow import serializer

logger = logging.getLogger(__name__)

# Define schema
FEATURE_BAR_SCHEMA = pa.schema([
    ("bar_type", pa.string()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("ts_event", pa.int64()),
    ("ts_init", pa.int64()),
    ("is_revision", pa.bool_()),
    ("features", pa.string()),
])

# Encoder: FeatureBar → Arrow table
def encode_feature_bar(obj: FeatureBar):
    data = {
        "bar_type": [str(obj.bar_type)],
        "open": [float(obj.open)],
        "high": [float(obj.high)],
        "low": [float(obj.low)],
        "close": [float(obj.close)],
        "volume": [float(obj.volume)],
        "ts_event": [obj.ts_event],
        "ts_init": [obj.ts_init],
        "is_revision": [obj.is_revision],
        "features": [json.dumps(obj.features)],
    }
    table = pa.table(data, schema=FEATURE_BAR_SCHEMA)
    # Convert table to RecordBatch (Nautilus expects RecordBatch)
    return table.to_batches()[0]

# Decoder: Arrow table → list of FeatureBar
def decode_feature_bar(table: pa.Table):
    bars = []
    for row in range(table.num_rows):
        features = json.loads(table["features"][row].as_py())
        bars.append(
            FeatureBar(
                bar_type=table["bar_type"][row].as_py(),
                open=table["open"][row].as_py(),
                high=table["high"][row].as_py(),
                low=table["low"][row].as_py(),
                close=table["close"][row].as_py(),
                volume=table["volume"][row].as_py(),
                ts_event=table["ts_event"][row].as_py(),
                ts_init=table["ts_init"][row].as_py(),
                is_revision=table["is_revision"][row].as_py(),
                features=features,
            )
        )
    return bars

# Register serializer
serializer.register_arrow(
    data_cls=FeatureBar,
    schema=FEATURE_BAR_SCHEMA,
    encoder=encode_feature_bar,
    decoder=decode_feature_bar
)

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

        benchmarks_file = self._root / "benchmarks" / "benchmarks.csv"
        benchmark = cfg["benchmark_ticker"]
        
        risk_free_ticker = cfg["risk_free_ticker"]
        risk_free_path = self._root / "etf"/ f"{risk_free_ticker}.csv"   # risk free taken as returns of etf following T-Bills 

        self._universe: List[str] = (
            universe
            if universe is not None
            else [p.stem.upper() for p in self._stock_files]
        )   

        # parse CSVs once to numeric frames
        self._frames: Dict[str, pd.DataFrame] = {}    
        self._instruments: Dict[str, Equity] = {}
        
        # load stocks
        for path in self._stock_files:
            symbol = path.stem.upper()
            if symbol in self._universe: 
                df = self._parse_stock_csv(path, adjust = adjust,  has_tz=True)
                if columns_to_load == "candles":
                    available_cols = [col for col in ["Open","High","Low","Close","Volume"] if col in df.columns]
                    self._frames[symbol] = df[available_cols]
                else:
                    self._frames[symbol] = df
                # Create instrument for each stock
                self._instruments[symbol] = self._create_equity_instrument(symbol)
        
        # Include SGOV or risk_free ETF as tradable instrument
        df_rf = self._parse_stock_csv(risk_free_path, adjust=adjust, has_tz=False)
        if risk_free_path.exists() and risk_free_ticker not in self._frames:
            if columns_to_load == "candles":
                available_cols = [col for col in ["Open","High","Low","Close","Volume"] if col in df_rf.columns]
                self._frames[risk_free_ticker] = df_rf[available_cols]
            else:
                self._frames[risk_free_ticker] = df_rf
            self._instruments[risk_free_ticker] = self._create_equity_instrument(risk_free_ticker)
        self._universe.append(risk_free_ticker)
        
        # Create risk free to use for sharpe ratio
        self._risk_free_df = (
            df_rf[["Adj Close"]]                   # keep only Close col
            .rename(columns={"Adj Close": "risk_free"})
            .pct_change()
            .fillna(0.0)
            )
        
        
        # Load benchmark data
        
        self._benchmark_data = None
        
        if benchmarks_file.exists():
            df = pd.read_csv(benchmarks_file, parse_dates=['dt'])
            df.rename(columns={"dt": "Date"}, inplace=True)
            df.index = pd.to_datetime(df.index) 
            df.index = df.index.tz_localize('America/New_York').tz_convert('UTC')

            
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

    def bar_iterator(self, start_time: Optional[pd.Timestamp] = None, 
                    end_time: Optional[pd.Timestamp] = None,
                    symbols: Optional[List[str]] = None) -> Iterator[Bar]:
        """
        Efficiently yield Bar and FeatureBarData events for each symbol.
        
        Args:
            start_time: Optional start timestamp filter
            end_time: Optional end timestamp filter  
            symbols: Optional list of symbols to iterate (defaults to universe)
            
        Yields:
            FeatureBarData and Bar objects in chronological order per symbol
        """
        # Pre-compute bar_spec once
        bar_spec = freq2barspec(self.cfg["freq"])
        
        # Determine symbols to process
        symbols_to_process = symbols if symbols else self._universe
        
        # Pre-compile column indices for faster access
        OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for symbol in symbols_to_process:
            if symbol not in self._frames:
                continue
                
            df = self._frames[symbol]
            instrument = self._instruments[symbol]
            bar_type = BarType(
                instrument_id=instrument.id,
                bar_spec=bar_spec
            )
            
            # Apply time filters if provided
            if start_time or end_time:
                mask = pd.Series(True, index=df.index)
                if start_time:
                    mask &= (df.index >= start_time)
                if end_time:
                    mask &= (df.index <= end_time)
                df_filtered = df[mask]
            else:
                df_filtered = df
            
            if df_filtered.empty:
                continue
            
            # Pre-calculate column indices for OHLCV
            col_indices = {}
            for col in OHLCV_COLS:
                if col in df_filtered.columns:
                    col_indices[col] = df_filtered.columns.get_loc(col)
            
            # Convert timestamps once for this symbol
            timestamps_ns = df_filtered.index.astype(np.int64)
            
            # Get numpy array for faster iteration
            values_array = df_filtered.values.astype(np.float32)
            
            # Iterate through rows efficiently
            for i, (_, ts_ns) in enumerate(zip(df_filtered.index, timestamps_ns)):
                row_values = values_array[i]
                
                # 1) Yield FeatureBarData with all numeric features
                if self.cfg.get("columns_to_load") != "candles":
                    yield FeatureBarData(
                        instrument=symbol,
                        ts_event=ts_ns,
                        ts_init=ts_ns,
                        features=row_values
                    )
                
                # 2) Yield Bar with OHLCV
                # Direct array access using pre-calculated indices
                open_val = row_values[col_indices.get('Open', 0)] if 'Open' in col_indices else 0.0
                high_val = row_values[col_indices.get('High', 0)] if 'High' in col_indices else 0.0
                low_val = row_values[col_indices.get('Low', 0)] if 'Low' in col_indices else 0.0
                close_val = row_values[col_indices.get('Close', 0)] if 'Close' in col_indices else 0.0
                volume_val = row_values[col_indices.get('Volume', 0)] if 'Volume' in col_indices else 0.0
                
                yield FeatureBar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{open_val:.3f}"),
                    high=Price.from_str(f"{high_val:.3f}"),
                    low=Price.from_str(f"{low_val:.3f}"),
                    close=Price.from_str(f"{close_val:.3f}"),
                    volume=Quantity.from_int(int(volume_val)),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )


    def get_data(self, calendar, frequency, start, end) -> Dict[str, pd.DataFrame]:

        # Create data dictionary for selected stocks
        cal = market_calendars.get_calendar(calendar)
        days_range = cal.schedule(start_date=start, end_date=end)
        timestamps = market_calendars.date_range(days_range, frequency=frequency)

        total_bars = 0

        # init train+valid data
        data = {}
        for ticker in self._universe:
            if ticker in self._frames:
                df = self._frames[ticker]
                # Ensure timezone compatibility
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                elif df.index.tz != timestamps.tz:
                    df.index = df.index.tz_convert(timestamps.tz)
                # re-indexing breaks different time-zones
                #data[ticker] = df.reindex(timestamps).dropna()
                df = df[(df.index >= start) & (df.index <= end)]
                # assume all instruments have same bars as first instrumemnt loaded.
                if total_bars == 0:
                    total_bars = len(df.index)
                # if asset does not have same number of bars for the same period, then skip it
                elif len(df.index) < total_bars:
                    continue

                data[ticker] = df

        return data

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #
    def _parse_stock_csv(self, path: Path, adjust: bool, has_tz = True) -> pd.DataFrame:
        df = pd.read_csv(path) 
        if has_tz:
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df.set_index("Date", inplace=True, ) 
            df.sort_index(inplace=True)
            df.index = df.index.tz_convert('UTC')
        else:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True, ) 
            df.sort_index(inplace=True)
            df.index = df.index.tz_localize('America/New_York').tz_convert('UTC')

        # Adjust OHLC on stock splits and dividends
        if adjust:
            df = self._compute_adj_ohlc(df)

        return df.select_dtypes(include=[np.number])   # keep ALL numeric columns

    @staticmethod
    def _compute_adj_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute adjusted OHLC prices using either:
        1. Adj Close / Close ratio if both columns exist
        2. Dividends and Stock Splits if available
        """
        df = df.sort_index().copy()
        
        # Method 1: If Adj Close exists, use it to compute factors
        if "Adj Close" in df.columns and "Close" in df.columns:
            # Calculate adjustment factors from Adj Close / Close ratio
            factors = df["Adj Close"] / df["Close"]
            factors = factors.fillna(1.0)  # Handle any NaN values
            
            # Apply factors to OHLC
            df["Open"] = df["Open"] * factors
            df["High"] = df["High"] * factors
            df["Low"] = df["Low"] * factors
            df["Close"] = df["Adj Close"]  # Use Adj Close as the adjusted close
            
            logger.info(f"Using Adj Close column for adjustment factors")
        
        # Method 2: Fall back to dividends and splits calculation
        else:
            factors = np.ones(len(df))
            div = df.get("Dividends", pd.Series(0.0, index=df.index)).fillna(0.0)
            splt = df.get("Stock Splits", pd.Series(0.0, index=df.index)).fillna(0.0)
            
            factor = 1.0
            for i in range(len(df) - 1, -1, -1):
                c = df.iloc[i]["Close"]
                if c > 0:  # Avoid division by zero
                    factor *= (1.0 - div.iat[i] / c) / (1.0 + splt.iat[i])
                factors[i] = factor
            
            # Apply factors to OHLC
            df["Open"] = df["Open"] * factors
            df["High"] = df["High"] * factors
            df["Low"] = df["Low"] * factors
            df["Close"] = df["Close"] * factors
            
            logger.info(f"Using Dividends/Splits for adjustment factors")
        
        return df

    @staticmethod
    def _parse_rf_csv(path: Path) -> pd.Series:
        rf = pd.read_csv(path, parse_dates=["observation_date"])
        rf.rename(columns={"observation_date": "Date"}, inplace=True)
        rf.set_index("Date", inplace=True)
        rf["DGS10"] = pd.to_numeric(rf["DGS10"], errors="coerce") / 100.0
        return rf["DGS10"].asfreq("B").ffill()
    
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
            price_precision=3, 
            price_increment=Price.from_str("0.001"),
            lot_size=Quantity.from_int(1),
            # margin_init=Money(0, USD),  # No margin requirement for cash account
            # margin_maint=Money(0, USD),
            maker_fee=0, 
            taker_fee=0,
            ts_event=0,
            ts_init=0,
        )