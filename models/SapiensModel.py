"""
Base class for all Sapiens models.
Provides common interface between PyTorch models and Sapiens strategies.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import pickle
import logging

import torch
import torch.nn as nn
import pandas as pd
from pandas import DataFrame, Timestamp
import pandas_market_calendars as market_calendars
from torch.utils.data import DataLoader

from .utils import SlidingWindowDataset, build_tensor, freq2pdoffset

logger = logging.getLogger(__name__)


class SapiensModel(nn.Module, ABC):
    """
    Base class for market prediction models in Sapiens framework.
    
    Child classes must implement:
    - _build_model(): Model architecture construction
    - _forward_train(batch): Training forward pass returning loss
    - _forward_predict(data): Prediction forward pass returning predictions
    """
    
    def __init__(self, **config):
        super().__init__()
        
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Time parameters
        self.freq = config['freq']
        self.train_start = pd.Timestamp(config['train_start'])
        self.train_end = pd.Timestamp(config['train_end'])
        self.train_offset = config['train_offset']
        self.valid_start = pd.Timestamp(config['valid_start'])
        self.valid_end = pd.Timestamp(config['valid_end'])
        self.valid_split = float(config['valid_split'])
        self.n_retrains_on_valid = int(config.get('n_retrains_on_valid', 0))
        assert self.valid_end >= self.train_end, "Validation end date must be after training end date."
        
        # Model parameters
        self.F = config['feature_dim']
        self.L = config['window_len']
        self.pred_len = config['pred_len']
        self.target_idx = config.get('target_idx', 3)
        
        # Training parameters
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.patience = config['patience']
        self.warm_start = config['warm_start']
        self.warm_training_epochs = config.get('warm_training_epochs', self.n_epochs)
        self.save_backups = config.get('save_backups', False)
        
        # Model directory
        self.model_dir = Path(config['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self._is_initialized = False
        self._global_epoch = 0
        self._epoch_logs = []
        
        # Hyperparameters (override in child class _default_hparams)
        self.hp = self._default_hparams()
    
    @property
    def is_initialized(self) -> bool:
        """Check if model has been initialized."""
        return self._is_initialized
    

    def _default_hparams(self) -> Dict[str, Any]:
        """Default hyperparameters. Override in child class."""
        return {}
    
    @abstractmethod
    def _build_model(self):
        """Build model architecture. Must be implemented by child class."""
        pass
    
    @abstractmethod
    def _forward_train(self, batch: Tuple) -> torch.Tensor:
        """
        Forward pass during training.
        
        Args:
            batch: Tuple from DataLoader (prices_seq, features_seq, targets, active_mask)
            
        Returns:
            loss: Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def _forward_predict(self, data: Dict[str, DataFrame], indexes: int, 
                        active_mask: torch.Tensor) -> Dict[str, float]:
        """
        Forward pass during prediction.
        
        Args:
            data: Dict of ticker -> DataFrame with current data
            indexes: Number of historical bars to use
            active_mask: Boolean mask of active instruments
            
        Returns:
            Dict mapping ticker to prediction
        """
        pass
    
    def initialize(self, data: Dict[str, DataFrame], total_bars: int) -> float:
        """Initial training on historical data."""
        if self._is_initialized:
            logger.warning("Model already initialized, skipping")
            return 0.0
        
        logger.info(f"[{self.__class__.__name__}] Initializing model...")
        
        # Build model architecture
        self._build_model()
        
        # Train with rolling validation if configured
        if self.n_retrains_on_valid > 0:
            best_val = self._train_with_rolling_validation(data, self.n_epochs, total_bars)
        else:
            # Single train/valid split
            full_tensor, _ = build_tensor(data, total_bars, self.F, self._device)
            train_bars = int(total_bars * (1 - self.valid_split))
            train_loader, valid_loader = self._create_loaders(full_tensor, train_bars, total_bars)
            best_val = self._train_loop(train_loader, valid_loader, self.n_epochs)
        
        # Save initial state
        torch.save(self.state_dict(), self.model_dir / "init.pt")
        torch.save(self.state_dict(), self.model_dir / "latest.pt")
        
        self._is_initialized = True
        
        logger.info(f"[initialize] Complete. Best validation: {best_val:.6f}")
        return best_val
    
    def update(self, data: Dict[str, DataFrame], current_time: Timestamp, 
               retrain_start_date: Timestamp, active_mask: torch.Tensor, total_bars: int):
        """Warm-start retraining on fresh data."""
        
        if not self.warm_start:
            logger.info(f"[update] Reinitializing (warm_start=False)")
            self._is_initialized = False
            self.initialize(data=data, total_bars=total_bars)
            return
        
        if not self._is_initialized:
            raise RuntimeError("Cannot update uninitialized model")
        
        # Load latest weights
        latest_path = self.model_dir / "latest.pt"
        if latest_path.exists():
            logger.info(f"[update] Loading weights from {latest_path}")
            self.load_state_dict(torch.load(latest_path, map_location=self._device, weights_only=False))
        else:
            logger.warning(f"[update] No latest.pt found, training from scratch")
        
        # Train without validation
        epochs = self.warm_training_epochs if self.warm_training_epochs else self.n_epochs
        full_tensor, _ = build_tensor(data, total_bars, self.F, self._device)
        train_loader, _ = self._create_loaders(full_tensor, total_bars, None)
        self._train_loop(train_loader, None, epochs)
        
        # Save backup if requested
        if self.save_backups:
            backup_name = f"checkpoint_{current_time.strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(self.state_dict(), self.model_dir / backup_name)
        
        # Save state
        torch.save(self.state_dict(), self.model_dir / "latest.pt")
        logger.info(f"[update] Complete at {current_time}")
    
    def predict(self, data: Dict[str, DataFrame], indexes: int, 
                active_mask: torch.Tensor) -> Dict[str, float]:
        """Generate predictions for current market state."""
        
        if not self._is_initialized:
            logger.error("Model not initialized")
            return {}
        
        return self._forward_predict(data, indexes, active_mask)
    
    def _create_loaders(self, tensor: torch.Tensor, train_end: int, 
                       valid_end: Optional[int] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create train/valid data loaders."""
        train_ds = SlidingWindowDataset(tensor[:train_end], self.L, self.pred_len, self.target_idx)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        valid_loader = None
        if valid_end and valid_end > train_end:
            valid_tensor = tensor[train_end:valid_end]
            if len(valid_tensor) >= self.L + self.pred_len:
                valid_ds = SlidingWindowDataset(valid_tensor, self.L, self.pred_len, self.target_idx)
                valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, valid_loader
    
    def _train_loop(self, train_loader: DataLoader, valid_loader: Optional[DataLoader], 
                    n_epochs: int) -> float:
        """Generic training loop with early stopping."""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            self.train()
            total_loss = 0.0
            for batch in train_loader:
                loss = self._forward_train(batch)
                loss.backward()
                # Optimizer step handled in child class _forward_train
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            val_loss = float('nan')
            if valid_loader:
                self.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch in valid_loader:
                        loss = self._forward_train(batch)
                        total_val_loss += loss.item()
                val_loss = total_val_loss / len(valid_loader)
                
                # Early stopping
                if val_loss < best_val_loss * 0.995:  # 0.5% improvement threshold
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            #self._log_training_metrics(self._global_epoch, avg_train_loss, val_loss)
            self._global_epoch += 1
            
            if epoch % max(1, n_epochs // 10) == 0:
                logger.debug(f"Epoch {epoch}: train={avg_train_loss:.5f}, val={val_loss:.5f}")
        
        return best_val_loss
    
    def _train_with_rolling_validation(self, data: Dict[str, DataFrame], 
                                      n_epochs: int, total_bars: int) -> float:
        """Train with rolling validation across multiple folds."""
        
        full_tensor, _ = build_tensor(data, total_bars, self.F, self._device)
        
        n_folds = 1 + self.n_retrains_on_valid
        train_bars = int(total_bars * (1 - self.valid_split))
        fold_size = (total_bars - train_bars) // n_folds
        
        if fold_size < self.L + self.pred_len:
            logger.warning("Validation fold too small, using single validation")
            train_loader, valid_loader = self._create_loaders(full_tensor, train_bars, total_bars)
            return self._train_loop(train_loader, valid_loader, n_epochs)
        
        best_overall = float('inf')
        
        for fold_idx in range(n_folds):
            train_end = train_bars + fold_idx * fold_size
            valid_end = train_end + fold_size if fold_idx < n_folds - 1 else total_bars
            fold_epochs = n_epochs if fold_idx == 0 else self.warm_training_epochs
            
            logger.info(f"Rolling fold {fold_idx+1}/{n_folds}: train[:{train_end}] valid[{train_end}:{valid_end}]")
            
            train_loader, valid_loader = self._create_loaders(full_tensor, train_end, valid_end)
            fold_best = self._train_loop(train_loader, valid_loader, fold_epochs)
            best_overall = min(best_overall, fold_best)
        
        return best_overall
    
    def _log_training_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        self._epoch_logs.append(metrics)
        
        log_file = self.model_dir / "train_losses.csv"
        pd.DataFrame([metrics]).to_csv(
            log_file, mode="a", header=not log_file.exists(), index=False
        )
    