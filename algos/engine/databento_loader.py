"""
Databento Trade Tick Loader for Nautilus Trader
Loads DBN files and provides data iteration.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import databento as db
import logging

import pandas as pd
import pandas_market_calendars as market_calendars
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity, Currency
from nautilus_trader.model.data import TradeTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.core.nautilus_pyo3 import CurrencyType

logger = logging.getLogger(__name__)


class DatabentoTickLoader:
    """Loads Databento DBN trade tick files for backtesting."""
    
    def __init__(
        self,
        cfg: Dict,
        venue_name: str = "XNAS",
        tz: str = "UTC",
        universe: Optional[List[str]] = None,
    ):
        self.cfg = cfg
        self.venue = Venue(venue_name)
        self.tz = tz
        
        # Data paths
        self._root = Path(cfg["data_dir"]).expanduser().resolve()
        self.stock_trades_dir = self._root / "stocks" / "trades"
        self.etf_trades_dir = self._root / "etf" / "trades"
        self.benchmark_trades_dir = self._root / "benchmark" / "trades"
        
        if not self.stock_trades_dir.exists():
            raise FileNotFoundError(f"Trades directory not found: {self.stock_trades_dir}")
        
        # Find DBN files
        self._dbn_files = list(self.stock_trades_dir.glob("*.trades.*.dbn.zst"))
        self._dbn_files += list(self.etf_trades_dir.glob("*.trades.*.dbn.zst"))
        
        # Extract symbols from filenames
        self._universe = universe if universe else self._extract_symbols()
        
        # Create instruments
        self._instruments: Dict[str, Equity] = {}
        for symbol in self._universe:
            self._instruments[symbol] = self._create_equity_instrument(symbol)
        
        # Initialize Databento loader
        self._databento_loader = DatabentoDataLoader()
    
    def _extract_symbols(self) -> List[str]:
        """Extract symbols from DBN filenames."""
        symbols = []
        for f in self._dbn_files:
            # Parse: xnas-itch-20240101-20251017.trades.AAPL.dbn.zst
            parts = f.stem.split('.')
            if len(parts) >= 3 and parts[1] == 'trades':
                symbol = parts[2]
                symbols.append(symbol)
        return sorted(set(symbols))
    
    @property
    def universe(self) -> List[str]:
        return self._universe
    
    @property
    def instruments(self) -> Dict[str, Equity]:
        return self._instruments
    
    def load_to_catalog(
        self,
        catalog_path: Path,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> ParquetDataCatalog:
        """
        Load DBN files into Nautilus ParquetDataCatalog.
        
        Args:
            catalog_path: Path to catalog
            start_time: Optional start filter
            end_time: Optional end filter
        
        Returns:
            ParquetDataCatalog instance
        """
        catalog = ParquetDataCatalog(path=str(catalog_path))
        
        # Write instruments
        #catalog.write_data(list(self._instruments.values()))
        
        # Load each DBN file
        for symbol in self._universe:
            dbn_file = self._find_dbn_file(symbol)
            if not dbn_file:
                logger.warning(f"No DBN file found for {symbol}")
                continue
            
            logger.info(f"Loading {symbol} from {dbn_file.name}")
            
            # Load using Databento loader
            trades = self._databento_loader.from_dbn_file(
                path=dbn_file,
                instrument_id=self._instruments[symbol].id,
                as_legacy_cython=False,  # Use Rust implementation
            )
            
            # Filter by time if needed
            if start_time or end_time:
                filtered_trades = []
                for tick in trades:
                    tick_time = pd.Timestamp(tick.ts_event, unit='ns', tz='UTC')
                    if start_time and tick_time < start_time:
                        continue
                    if end_time and tick_time > end_time:
                        break
                    filtered_trades.append(tick)
                trades = filtered_trades
            
            # Write to catalog
            catalog.write_data(trades)
            logger.info(f"Loaded {len(trades)} trades for {symbol}")
        
        return catalog
    
    def _find_dbn_file(self, symbol: str) -> Optional[Path]:
        """Find DBN file for a given symbol."""
        for f in self._dbn_files:
            if f".trades.{symbol}." in f.name:
                return f
        return None
    
    def _create_equity_instrument(self, symbol: str) -> Equity:
        """Create Equity instrument."""
        instrument_id = InstrumentId(
            symbol=Symbol(symbol),
            venue=self.venue
        )
        
        currency = self.cfg["currency"]
        
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=currency,
            price_precision=2,
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_int(1),
            ts_event=0,
            ts_init=0,
        )
    

    def get_ohlcv_data(self, 
                       frequency: str, 
                       start: Optional[pd.Timestamp] = None, 
                       end: Optional[pd.Timestamp] = None):
        """Aggregate DBN data to OHLCV Dict[str, Dataframe]"""
        data_dict = {}
        
        for path in self._dbn_files:

            # Parse: xnas-itch-20240101-20251017.trades.SYMBOL.dbn.zst
            symbol = path.stem.split('.')[2]
            df = db.DBNStore.from_file(path).to_df()
        
            ohlcv_df = df.resample(frequency).agg({
                'price': ['first', 'max', 'min', 'last'],
                'size': 'sum'
            })

            # Flatten the multi-level column index
            ohlcv_df.columns = ['_'.join(col).strip() for col in ohlcv_df.columns.values]
            ohlcv_df.rename(columns={
                'price_first': 'open',
                'price_max': 'high',
                'price_min': 'low',
                'price_last': 'close',
                'size_sum': 'volume'
            }, inplace=True)

            # Filter by period
            filtered_ohlcv_df = ohlcv_df.copy()
            if start is not None:
                filtered_ohlcv_df = filtered_ohlcv_df.loc[(filtered_ohlcv_df.index >= start)]
            if end is not None:
                filtered_ohlcv_df = filtered_ohlcv_df.loc[(filtered_ohlcv_df.index <= end)]

            # Add to data dictionary
            data_dict[symbol] = filtered_ohlcv_df

        return data_dict