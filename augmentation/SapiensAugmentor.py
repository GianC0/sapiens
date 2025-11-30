
from __future__ import annotations
import pandas as pd
import abc
import logging 
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SapiensAugmentor(abc.ABC):
    """Abstract base class for all market data augmentors."""

    def __init__(self, 
                 cfg: Dict[str, Any],
                 augment_modality: str = "train",
                 stream_cache = None):

        self.cfg = cfg

        self.market_data_type = cfg.get("features_to_load", "candles")  #OHLCV, orderbook, possibly more
        
        self.pca = cfg.get("pca", False)
        self.pca_var_explained = cfg.get("pca_var_explained", 0.95) #if using var explained to choose n_components


        if augment_modality not in {"train", "stream"}:
            raise ValueError("augment_modality must be 'train' or 'stream'")
        
        self.modality = augment_modality

        self.cache = stream_cache  #Cache for historical data in streaming mode


    @abc.abstractmethod
    def _check_input(self, df: pd.DataFrame) -> None:
        """Validate the DataFrame structure."""
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty DataFrame.")

        pass

    @abc.abstractmethod
    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply augmentation logic and return augmented DataFrame."""
        self._check_input(df)

        pass

    @abc.abstractmethod 
    def _get_historical_window(self, instrument_id: str, latest_df: pd.DataFrame) -> pd.DataFrame:
        """Retrieve historical bars from Nautilus cache and prepend to latest data."""
        pass

    def set_cache(self, stream_cache) -> None:
        """Set or update the cache for streaming mode."""
        self.cache = stream_cache

    def set_modality(self, augment_modality: str) -> None:
        """Set the augmentation modality: 'train' or 'stream'."""
        if augment_modality not in {"train", "stream"}:
            raise ValueError("augment_modality must be 'train' or 'stream'")
        
        self.modality = augment_modality

    def augment(self, data_dict) -> Dict[str, pd.DataFrame]:
        """Standard entry point: validate then augment."""

        logger.debug(f"Running {self.__class__.__name__} augmentation, mode={self.modality}")

        augmented_dict = {}
        for symbol, df in data_dict.items():

            if self.modality == "stream" and self.cache is not None:
                df = self._get_historical_window(symbol, df)


            df_aug = self._augment(df)
                        
            # Store augmented DataFrame
            augmented_dict[symbol] = df_aug

        if self.pca:
            logger.warning("PCA augmentation not yet implemented.")

        return augmented_dict