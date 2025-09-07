"""
Data Augmentation Module for Market Data
=========================================

Provides various augmentation techniques for financial time series:
- Bootstrap resampling
- GARCH residual augmentation
- Mixup on returns
- Latent space augmentation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from arch import arch_model
import logging

logger = logging.getLogger(__name__)


class MarketDataAugmenter:
    """
    Augments market data with various techniques for improved model training.
    """
    
    def __init__(
        self,
        augmentation_config: Dict[str, Any],
        market_data_type: str = "L1",  # L1, L2, L3, tick
        seed: int = 42
    ):
        """
        Initialize the data augmenter.
        
        Args:
            augmentation_config: Configuration for augmentation methods
            market_data_type: Type of market data (L1/L2/L3/tick)
            seed: Random seed for reproducibility
        """
        self.config = augmentation_config
        self.market_data_type = market_data_type
        self.rng = np.random.RandomState(seed)
        
        # Augmentation flags
        self.use_bootstrap = augmentation_config.get("bootstrap", {}).get("enabled", False)
        self.use_garch = augmentation_config.get("garch", {}).get("enabled", False)
        self.use_mixup = augmentation_config.get("mixup", {}).get("enabled", False)
        self.use_latent = augmentation_config.get("latent", {}).get("enabled", False)
        
        # Feature engineering based on market data type
        self.feature_extractors = self._setup_feature_extractors()
        
    def _setup_feature_extractors(self) -> Dict:
        """Setup feature extractors based on market data type."""
        extractors = {}
        
        if self.market_data_type == "L1":
            extractors["spread"] = self._compute_spread
            extractors["mid_price"] = self._compute_mid_price
            extractors["volume_imbalance"] = self._compute_volume_imbalance
            
        elif self.market_data_type == "L2":
            extractors["book_imbalance"] = self._compute_book_imbalance
            extractors["weighted_mid"] = self._compute_weighted_mid_price
            extractors["depth_ratio"] = self._compute_depth_ratio
            
        elif self.market_data_type == "L3":
            extractors["order_flow"] = self._compute_order_flow
            extractors["aggressive_ratio"] = self._compute_aggressive_ratio
            extractors["hidden_liquidity"] = self._estimate_hidden_liquidity
            
        elif self.market_data_type == "tick":
            extractors["tick_rule"] = self._compute_tick_rule
            extractors["volume_clock"] = self._compute_volume_clock
            extractors["trade_intensity"] = self._compute_trade_intensity
            
        return extractors
    
    def augment_data(
        self,
        data: Dict[str, pd.DataFrame],
        mode: str = "train"
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply augmentation to market data.
        
        Args:
            data: Dictionary of DataFrames with market data
            mode: "train" or "valid" - only augment training data
            
        Returns:
            Augmented data dictionary
        """
        if mode != "train":
            return data  # No augmentation for validation/test
            
        augmented_data = {}
        
        for symbol, df in data.items():
            # Add engineered features
            df_aug = self._add_features(df.copy())
            
            # Apply augmentation techniques
            if self.use_bootstrap:
                df_aug = self._bootstrap_augment(df_aug)
                
            if self.use_garch:
                df_aug = self._garch_augment(df_aug)
                
            if self.use_mixup:
                df_aug = self._mixup_augment(df_aug, data)
                
            if self.use_latent:
                df_aug = self._latent_augment(df_aug)
                
            augmented_data[symbol] = df_aug
            
        return augmented_data
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features based on market data type."""
        for name, extractor in self.feature_extractors.items():
            try:
                df[name] = extractor(df)
            except Exception as e:
                logger.warning(f"Failed to extract feature {name}: {e}")
                
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add microstructure features
        df = self._add_microstructure_features(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators."""
        if "Close" in df.columns:
            # Returns
            df["returns"] = df["Close"].pct_change()
            df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
            
            # Moving averages
            for window in [5, 10, 20]:
                df[f"sma_{window}"] = df["Close"].rolling(window).mean()
                df[f"ema_{window}"] = df["Close"].ewm(span=window).mean()
            
            # Volatility
            df["volatility_20"] = df["returns"].rolling(20).std()
            
            # RSI
            df["rsi"] = self._compute_rsi(df["Close"])
            
        if "Volume" in df.columns:
            # Volume indicators
            df["volume_sma_10"] = df["Volume"].rolling(10).mean()
            df["volume_ratio"] = df["Volume"] / df["volume_sma_10"]
            
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        if all(col in df.columns for col in ["High", "Low", "Close"]):
            # Garman-Klass volatility
            df["gk_vol"] = np.sqrt(
                np.log(df["High"] / df["Low"]) ** 2 / 2 -
                (2 * np.log(2) - 1) * np.log(df["Close"] / df["Open"]) ** 2
            )
            
            # Parkinson volatility
            df["park_vol"] = np.sqrt(
                np.log(df["High"] / df["Low"]) ** 2 / (4 * np.log(2))
            )
            
        return df
    
    def _bootstrap_augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply bootstrap resampling with block structure."""
        config = self.config.get("bootstrap", {})
        block_size = config.get("block_size", 20)
        n_samples = config.get("n_samples", 1)
        
        if n_samples <= 1:
            return df
            
        n_blocks = len(df) // block_size
        augmented_dfs = [df]
        
        for _ in range(n_samples - 1):
            # Sample blocks with replacement
            block_indices = self.rng.choice(n_blocks, n_blocks, replace=True)
            
            # Reconstruct series from blocks
            new_data = []
            for idx in block_indices:
                start = idx * block_size
                end = min(start + block_size, len(df))
                new_data.append(df.iloc[start:end])
                
            augmented_df = pd.concat(new_data, ignore_index=True)
            augmented_df.index = df.index[:len(augmented_df)]
            augmented_dfs.append(augmented_df)
            
        return pd.concat(augmented_dfs, ignore_index=True)
    
    def _garch_augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply GARCH-based augmentation."""
        if "returns" not in df.columns:
            return df
            
        config = self.config.get("garch", {})
        n_simulations = config.get("n_simulations", 1)
        
        if n_simulations <= 1:
            return df
            
        # Fit GARCH model
        returns = df["returns"].dropna()
        model = arch_model(returns, vol='Garch', p=1, q=1)
        
        try:
            res = model.fit(disp='off')
            
            # Generate synthetic returns
            augmented_dfs = [df]
            for _ in range(n_simulations - 1):
                simulated = res.forecast(horizon=len(df), method='simulation')
                synthetic_returns = simulated.simulations.values[-1, :, 0]
                
                # Create augmented DataFrame
                aug_df = df.copy()
                aug_df["returns"] = synthetic_returns
                
                # Reconstruct prices from returns
                if "Close" in df.columns:
                    aug_df["Close"] = df["Close"].iloc[0] * (1 + synthetic_returns).cumprod()
                    
                augmented_dfs.append(aug_df)
                
            return pd.concat(augmented_dfs, ignore_index=True)
            
        except Exception as e:
            logger.warning(f"GARCH augmentation failed: {e}")
            return df
    
    def _mixup_augment(
        self,
        df: pd.DataFrame,
        all_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Apply mixup augmentation on returns."""
        config = self.config.get("mixup", {})
        alpha = config.get("alpha", 0.2)
        n_mixups = config.get("n_mixups", 1)
        
        if n_mixups <= 0 or "returns" not in df.columns:
            return df
            
        # Select random partner for mixup
        symbols = list(all_data.keys())
        augmented_dfs = [df]
        
        for _ in range(n_mixups):
            partner_symbol = self.rng.choice(symbols)
            partner_df = all_data[partner_symbol]
            
            if "returns" in partner_df.columns:
                # Sample mixing coefficient from Beta distribution
                lam = self.rng.beta(alpha, alpha)
                
                # Mix returns
                mixed_df = df.copy()
                mixed_returns = lam * df["returns"] + (1 - lam) * partner_df["returns"].iloc[:len(df)]
                mixed_df["returns"] = mixed_returns
                
                augmented_dfs.append(mixed_df)
                
        return pd.concat(augmented_dfs, ignore_index=True)
    
    def _latent_augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply latent space augmentation using PCA."""
        config = self.config.get("latent", {})
        n_components = config.get("n_components", 10)
        noise_level = config.get("noise_level", 0.1)
        n_samples = config.get("n_samples", 1)
        
        if n_samples <= 1:
            return df
            
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data_matrix = df[numeric_cols].fillna(0).values
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, data_matrix.shape[1]))
        latent = pca.fit_transform(data_matrix)
        
        augmented_dfs = [df]
        
        for _ in range(n_samples - 1):
            # Add noise in latent space
            noise = self.rng.normal(0, noise_level, latent.shape)
            augmented_latent = latent + noise
            
            # Transform back
            augmented_data = pca.inverse_transform(augmented_latent)
            
            # Create augmented DataFrame
            aug_df = df.copy()
            aug_df[numeric_cols] = augmented_data
            augmented_dfs.append(aug_df)
            
        return pd.concat(augmented_dfs, ignore_index=True)
    
    # Feature extraction methods
    def _compute_spread(self, df: pd.DataFrame) -> pd.Series:
        """Compute bid-ask spread."""
        if "High" in df.columns and "Low" in df.columns:
            return df["High"] - df["Low"]
        return pd.Series(0, index=df.index)
    
    def _compute_mid_price(self, df: pd.DataFrame) -> pd.Series:
        """Compute mid price."""
        if "High" in df.columns and "Low" in df.columns:
            return (df["High"] + df["Low"]) / 2
        return df.get("Close", pd.Series(0, index=df.index))
    
    def _compute_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume imbalance."""
        if "Volume" in df.columns:
            return df["Volume"] - df["Volume"].rolling(20).mean()
        return pd.Series(0, index=df.index)
    
    def _compute_book_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Compute order book imbalance (L2 data)."""
        # Placeholder - would need actual L2 data
        return pd.Series(0, index=df.index)
    
    def _compute_weighted_mid_price(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume-weighted mid price."""
        if "Close" in df.columns and "Volume" in df.columns:
            return (df["Close"] * df["Volume"]).rolling(10).sum() / df["Volume"].rolling(10).sum()
        return df.get("Close", pd.Series(0, index=df.index))
    
    def _compute_depth_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Compute depth ratio (L2 data)."""
        return pd.Series(0, index=df.index)
    
    def _compute_order_flow(self, df: pd.DataFrame) -> pd.Series:
        """Compute order flow (L3 data)."""
        return pd.Series(0, index=df.index)
    
    def _compute_aggressive_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Compute aggressive order ratio."""
        return pd.Series(0, index=df.index)
    
    def _estimate_hidden_liquidity(self, df: pd.DataFrame) -> pd.Series:
        """Estimate hidden liquidity."""
        return pd.Series(0, index=df.index)
    
    def _compute_tick_rule(self, df: pd.DataFrame) -> pd.Series:
        """Compute tick rule for trade classification."""
        if "Close" in df.columns:
            return np.sign(df["Close"].diff())
        return pd.Series(0, index=df.index)
    
    def _compute_volume_clock(self, df: pd.DataFrame) -> pd.Series:
        """Compute volume clock bars."""
        if "Volume" in df.columns:
            return df["Volume"].cumsum()
        return pd.Series(0, index=df.index)
    
    def _compute_trade_intensity(self, df: pd.DataFrame) -> pd.Series:
        """Compute trade intensity."""
        if "Volume" in df.columns:
            return df["Volume"] / df["Volume"].rolling(20).mean()
        return pd.Series(1, index=df.index)
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))