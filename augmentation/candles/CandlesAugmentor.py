from augmentation.SapiensAugmentor import SapiensAugmentor
import pandas as pd
import talib
from pathlib import Path
import yaml
from typing import Dict, Any
import numpy as np
from models.utils import freq2bartype
from nautilus_trader.model.identifiers import InstrumentId
import pickle
import logging 

logger = logging.getLogger(__name__)

class CandlesAugmentor(SapiensAugmentor):

    def __init__(self,
                cfg: Dict[str,Any],
                augment_modality: str = "train",
                stream_cache = None):
        
        super().__init__(cfg, augment_modality)

        self.technical_indicators = cfg.get("technical_indicators", True)
        #self.timeperiod = cfg.get("techinds_timeperiod", 14)
        #self.source = talib if self.modality == "train" else talib.stream


        self.technical_indicators_list = []
        #self.statistical_features_list = []
        #self.raw_features_list = []

        #Store reference to strategy's cache
        #self.cache = strategy_cache
        self.freq = cfg.get("freq", "15min")

        # NEW: Rolling buffers per instrument (stores last N bars)
        self.max_lookback = cfg.get("techinds_timeperiod", 14)  # e.g., 50 bars
        self.buffers = {}  # {instrument_id: deque of bars}

                    # Store for later use
        self.normalization_params = {}
    
    @property
    def source(self):
        """Dynamically return talib or talib.stream based on modality."""
        return talib if self.modality == "train" else talib #.stream TODO: if changing to storing TAs

    
    def save_state(self, path: Path) -> None:
        """
        Save augmentor state to disk (excludes unpicklable objects).
        
        Args:
            path: File path to save state (e.g., model_dir / "augmentor_state.pkl")
        """
        state = {
            'class_name': self.__class__.__name__,
            'module_path': self.__class__.__module__,
            'cfg': self.cfg,
            'modality': self.modality,
            'technical_indicators': self.technical_indicators,
            'technical_indicators_list': self.technical_indicators_list,
            'freq': self.freq,
            'max_lookback': self.max_lookback,
            # Add normalization and other state params when implemented
            'normalization_params': self.normalization_params,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved augmentor state to {path}")

    @classmethod
    def load_state(cls, path: Path, stream_cache=None, modality: str = "stream") -> 'CandlesAugmentor':
        """
        Load augmentor from saved state.
        
        Args:
            path: File path to load state from
            stream_cache: Nautilus cache for streaming mode (required for inference)
            modality: Override modality ("train" or "stream")
        
        Returns:
            Reconstructed CandlesAugmentor instance
        """
        if not path.exists():
            raise FileNotFoundError(f"Augmentor state not found: {path}")
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Reconstruct augmentor with saved config
        augmentor = cls(
            cfg=state['cfg'],
            augment_modality=modality,
            stream_cache=stream_cache
        )
        
        # Restore state
        augmentor.technical_indicators = state['technical_indicators']
        augmentor.technical_indicators_list = state['technical_indicators_list']
        augmentor.freq = state['freq']
        augmentor.max_lookback = state['max_lookback']
        
        # Restore normalization and other state params when implemented
        if 'normalization_params' in state:
             augmentor.normalization_params = state['normalization_params']
        
        logger.info(f"Loaded augmentor state from {path} in {modality} mode")
        
        return augmentor

    def _check_input(self, df: pd.DataFrame) -> None:
        super()._check_input(df)
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Centralized normalization for all feature types."""
        out = df.copy()
        
        # Collect all features that need normalization
        features_to_normalize = (
            self.technical_indicators_list
            #+ self.statistical_features_list
        )
        
        if not features_to_normalize:
            return out
        
        # Handle NaNs first #TODOSB: temporary, to be avoided with good window handling
        out[features_to_normalize] = out[features_to_normalize].ffill().fillna(0)
        
        if self.modality == "train":
            # TRAINING: Compute and store normalization parameters
            means = out[features_to_normalize].mean()
            stds = out[features_to_normalize].std() + 1e-8
            
            # Store for later use
            self.normalization_params['means'] = means.to_dict()
            self.normalization_params['stds'] = stds.to_dict()
            
            # Apply normalization
            out[features_to_normalize] = (out[features_to_normalize] - means) / stds
            
            logger.info(f"Computed normalization params for {len(features_to_normalize)} features")
            
        elif self.modality == "stream":
            # INFERENCE: Use stored normalization parameters
            if not self.normalization_params['means']:
                raise ValueError(
                    "No normalization parameters found. "
                    "Must run in 'train' mode first to compute parameters."
                )
            
            # Apply stored normalization
            for feature in features_to_normalize:
                if feature in self.normalization_params['means']:
                    mean = self.normalization_params['means'][feature]
                    std = self.normalization_params['stds'][feature]
                    out[feature] = (out[feature] - mean) / std
                else:
                    logger.warning(f"Feature '{feature}' not found in normalization params")
        
        # Safety checks
        if out[features_to_normalize].isna().any().any():
            logger.warning(
                f"NaN values after normalization: "
                f"{out[features_to_normalize].isna().sum()[out[features_to_normalize].isna().sum() > 0].to_dict()}"
            )
            out[features_to_normalize] = out[features_to_normalize].fillna(0)
        
        if np.isinf(out[features_to_normalize].values).any():
            logger.warning("Inf values detected, clipping...")
            out[features_to_normalize] = out[features_to_normalize].clip(-10, 10)
        
        return out
    
    def set_freq(self, freq: str):
        """Set frequency for bar retrieval in streaming mode."""
        self.freq = freq



    def augment(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Augment data for all instruments."""

        logger.debug(f"Running {self.__class__.__name__} augmentation, mode={self.modality}")

        augmented = {}
        
        for instrument_id, df in data_dict.items():
            # At inference, retrieve more historical data from cache
            if self.modality == "stream" and self.cache is not None:
                df = self._get_historical_window(instrument_id)
                
                logger.debug(f"Prepended historical data for {instrument_id}, total bars: {len(df)}")
            
            # Now augment with full history available
            augmented[instrument_id] = self._augment(df)
            logger.debug(f"Augmented data for {instrument_id}, final bars: {len(augmented[instrument_id])}")
        
        return augmented

      
    def _get_historical_window(self, instrument_id: str) -> pd.DataFrame:
        """Retrieve historical bars from Nautilus cache and prepend to latest data."""
        
        # Get bar_type for this instrument
        iid = InstrumentId.from_str(instrument_id)
        bar_type = freq2bartype(instrument_id=iid, frequency=self.freq)
        
        # Retrieve bars from cache
        bars = self.cache.bars(bar_type)
        
        logger.debug(f"Retrieved {len(bars)} bars from cache for {instrument_id} at freq {self.freq}")

        if not bars or len(bars) < self.max_lookback:
            raise ValueError(f"Insufficient bars in cache for {instrument_id}")
        
        
        # Take last max_lookback bars
        historical_bars = bars[:self.max_lookback]
        
        # Convert to DataFrame
        historical_df = pd.DataFrame({
            'open': [float(b.open) for b in historical_bars],
            'high': [float(b.high) for b in historical_bars],
            'low': [float(b.low) for b in historical_bars],
            'close': [float(b.close) for b in historical_bars],
            'volume': [float(b.volume) for b in historical_bars]
        }, index=[pd.Timestamp(b.ts_event, unit='ns', tz="UTC") for b in historical_bars])
        
        historical_df.sort_index(inplace=True)
        
        return historical_df
    
    


    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        super()._augment(df) 
        out = df.copy(deep=False)

        if self.modality == "stream":
            #historical = retrieveFromCatalog(symbol).tail(self.timeperiod)
            #df = pd.concat([historical, df]).reset_index(drop=True)
            out = out

        if self.technical_indicators:
            out = self._add_technical_indicators(out)

            #remove initial NaN values introduced by indicators
            #out = out.dropna().reset_index(drop=True)

        
        # Add other features
        #out = self._add_statistical_features(out)
        #out = self._add_time_features(out)
        
        # Normalize (different behavior for train vs stream)
        out = self._normalize_features(out)
        
        return out
            
    def _add_technical_indicators(self, df, timeperiod = 14):

        out = df.copy()

        open, high, low, close, volume = out["open"], out["high"], out["low"], out["close"], out["volume"]

        # Calls to indicators, from https://ta-lib.github.io/ta-lib-python/funcs.html
        dict_indicators = {}
        source = self.source

        #Overlap Studies
        #TODOSB: if %B and BandWidth are highly correlated with volatility features consider reducing redundancy before PCA

        upper, middle, lower = source.BBANDS(close, timeperiod, nbdevup=2, nbdevdn=2)
        dict_indicators['%B'] = (close - lower) / (upper - lower) # %B: position of price within bands
        dict_indicators['BandWidth']= (upper - lower) / middle # BandWidth: relative band width normalized by middle band
        #dict_indicators['DEMA'] = source.DEMA(close, timeperiod) #redundant moving average
        dict_indicators['EMA'] = source.EMA(close, timeperiod) #TODOSB: do not keep a single one
        #dict_indicators['HT_TRENDLINE'] = source.HT_TRENDLINE(close) #redundant moving average
        dict_indicators['KAMA'] = source.KAMA(close, timeperiod)
        #dict_indicators['MA'] = source.MA(close, timeperiod) #generic moving average, SMA standard, redundant
        #dict_indicators['MAMA'], dict_indicators['FAMA'] = source.MAMA(close) #redundant moving average
        #dict_indicators['MIDPOINT'] = source.MIDPOINT(close, timeperiod) #Redundant with SMA or EMA
        #dict_indicators['MIDPRICE'] = source.MIDPRICE(high, low, timeperiod) #Redundant with other volatility features
        dict_indicators['SAR'] = source.SAR(high, low)
        #dict_indicators['SAREXT'] = source.SAREXT(high, low) #Overly complex; adds no clear gain over standard SAR
        #dict_indicators['SMA'] = source.SMA(close, timeperiod) #redundant moving average
        #dict_indicators['T3'] = source.T3(close, timeperiod) #redundant moving average
        dict_indicators['TEMA'] = source.TEMA(close, timeperiod)
        #dict_indicators['TRIMA'] = source.TRIMA(close, timeperiod) #redundant moving average
        #dict_indicators['WMA'] = source.WMA(close, timeperiod) #redundant moving average 

        #Momentum Indicators
        dict_indicators['ADX'] = source.ADX(high, low, close,timeperiod)
        #dict_indicators['ADXR'] = source.ADXR(high, low, close, timeperiod) # redundant with ADX
        dict_indicators['APO'] = source.APO(close, fastperiod=12, slowperiod=26, matype=0)
        #dict_indicators['AROON_down'], dict_indicators['AROON_up'] = source.AROON(high, low, timeperiod) # AROONOSC = Aroon Up − Aroon Down.
        dict_indicators['AROONOSC'] = source.AROONOSC(high, low, timeperiod) #compact AROON
        dict_indicators['BOP'] = source.BOP(open, high, low, close)
        dict_indicators['CCI'] = source.CCI(high, low, close, timeperiod)
        dict_indicators['CMO'] = source.CMO(close, timeperiod)
        #dict_indicators['DX'] = source.DX(high, low, close, timeperiod) #redundant with ADX
        dict_indicators['MACD'], dict_indicators['MACD_signal'], dict_indicators['MACD_hist'] = source.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        #dict_indicators['MACDEXT'], dict_indicators['MACDEXT_signal'], dict_indicators['MACDEXT_hist'] = source.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0) #redundant with MACD
        #dict_indicators['MACFIX'], dict_indicators['MACFIX_signal'], dict_indicators['MACFIX_hist'] = source.MACDFIX(close, signalperiod=9) #redundant with MACD
        dict_indicators['MFI'] = source.MFI(high, low, close, volume, timeperiod)
        #dict_indicators['MINUS_DI'] = source.MINUS_DI(high, low, close, timeperiod) #redundant with ADX
        #dict_indicators['MINUS_DM'] = source.MINUS_DM(high, low, timeperiod) #redundant with ADX
        dict_indicators['MOM'] = source.MOM(close, timeperiod)
        #dict_indicators['PLUS_DI'] = source.PLUS_DI(high, low, close, timeperiod) #redundant with ADX
        #dict_indicators['PLUS_DM'] = source.PLUS_DM(high, low, timeperiod) #redundant with ADX
        #dict_indicators['PPO'] = source.PPO(close, fastperiod=12, slowperiod=26, matype=0) redundant with APO, MACD
        dict_indicators['ROC'] = source.ROC(close, timeperiod)
        #dict_indicators['ROCP'] = source.ROCP(close, timeperiod) #redundant with ROC 
        #dict_indicators['ROCR'] = source.ROCR(close, timeperiod) #redundant with ROC 
        #dict_indicators['ROCR100'] = source.ROCR100(close, timeperiod) #redundant with ROC
        dict_indicators['RSI'] = source.RSI(close, timeperiod)
        dict_indicators['STOCH_slowk'], dict_indicators['STOCH_slowd'] = source.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        #dict_indicators['STOCHF_fastk'], dict_indicators['STOCHF_fastd'] = source.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0) #redundant with STOCH
        #dict_indicators['STOCHRSI_fastk'], dict_indicators['STOCHRSI_fastd'] = source.STOCHRSI(close, timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0) #redundant with RSI and STOCH
        dict_indicators['TRIX'] = source.TRIX(close, timeperiod)    
        #dict_indicators['ULTOSC'] = source.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28) #redundant with RSI and MFI
        #dict_indicators['WILLR'] = source.WILLR(high, low, close, timeperiod) #redundant with RSI and STOCH

        #Volume Indicators
        #dict_indicators['AD'] = source.AD(high, low, close, volume) #redundant with ADOSC
        dict_indicators['ADOSC'] = source.ADOSC(high, low, close, volume) 
        dict_indicators['OBV'] = source.OBV(close, volume)

        #Volatility Indicators
        dict_indicators['ATR']   = source.ATR(high, low, close, timeperiod)
        #dict_indicators['NATR']  = source.NATR(high, low, close, timeperiod) #redundant with ATR
        #dict_indicators['TRANGE'] = source.TRANGE(high, low, close) #redundant with ATR

        #Price Transform Indicators
        #dict_indicators['AVGPRICE']  = source.AVGPRICE(open, high, low, close) #very correlated with OHLCV
        #dict_indicators['MEDPRICE']  = source.MEDPRICE(high, low) #very correlated with OHLCV
        #dict_indicators['TYPPRICE']  = source.TYPPRICE(high, low, close) #very correlated with OHLCV
        #dict_indicators['WCLPRICE']  = source.WCLPRICE(high, low, close) #very correlated with OHLCV
        #add more indicators
        
        for indicator_name, values in dict_indicators.items():
            if self.modality == "train":
                out[indicator_name] = values
            elif self.modality == "stream":
                #out.iloc[-1, out.columns.get_loc(indicator_name)] = values
                out[indicator_name] = values #TODO: if eventually previous TAs are stored, add only last value instead of recomputing
                
        # Store the list after all indicators are added
        self.technical_indicators_list = list(dict_indicators.keys())

        return out     