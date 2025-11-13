"""
UMI Model - refactored to use SapiensModel base class.
"""

from sqlalchemy import over
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, override
import pandas as pd
from pandas import DataFrame

from models.SapiensModel import SapiensModel
from models.utils import SlidingWindowDataset, build_tensor
from .learners import StockLevelFactorLearning, MarketLevelFactorLearning, ForecastingLearning

import logging
logger = logging.getLogger(__name__)


class UMIModel(SapiensModel):
    """
    UMI (Unveiling Market Intelligence) model with multi-stage factor learning.
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        
        # UMI-specific parameters
        self.I = None  # Number of instruments (set during initialization)
        self.training_mode = config.get('training_mode', 'sequential')
        self.pretrain_epochs = config.get('pretrain_epochs', 5)
        
        # Hyperparameters:
        # Update with any hparams passed in config
        hparam_keys = ['lambda_ic', 'lambda_sync', 'lambda_rankic', 'temperature', 
                      'sync_thr', 'weight_decay', 'lr_stage1', 'lr_stage1_ft', 'lr_stage2']
        for key in hparam_keys:
            if key in config:
                self.hp[key] = config[key]

        # UMI-specific components (initialized in _build_model)
        self.stock_factor = None
        self.market_factor = None
        self.forecaster = None
        self._eye = None
    
    @override
    def _default_hparams(self) -> Dict:
        """UMI-specific hyperparameters."""
        return {
            'lambda_ic': 0.5,
            'lambda_sync': 1.0,
            'lambda_rankic': 0.1,
            'temperature': 0.07,
            'sync_thr': 0.6,
            'weight_decay': 0.0,
            'lr_stage1': 1e-3,
            'lr_stage1_ft': 1e-4,
            'lr_stage2': 1e-4,
        }
    
    def _build_model(self):
        """Build UMI architecture components."""
        if self.I is None:
            raise ValueError("Must set self.I before building model")
        
        self._eye = torch.eye(self.I, dtype=torch.float32)
        
        self.stock_factor = StockLevelFactorLearning(
            self.I, lambda_ic=self.hp['lambda_ic']
        ).to(self._device)
        
        self.market_factor = MarketLevelFactorLearning(
            self.I, self.F, window_L=self.L,
            lambda_sync=self.hp['lambda_sync'],
            temperature=self.hp['temperature'],
            sync_threshold=self.hp['sync_thr'],
        ).to(self._device)
        
        self.forecaster = ForecastingLearning(
            self.I, self.F, u_dim=1,
            pred_len=self.pred_len,
            lambda_rankic=self.hp['lambda_rankic'],
        ).to(self._device)
    
    def _forward_train(self, batch: Tuple) -> torch.Tensor:
        """
        UMI training forward pass.
        Note: UMI uses custom multi-stage training, so this is not called directly.
        """
        prices_seq, features_seq, targets, active_mask = batch
        prices_seq = prices_seq.to(self._device)
        features_seq = features_seq.to(self._device)
        targets = targets.to(self._device)
        active_mask = active_mask.to(self._device)
        
        # Stage 1
        u_seq, r_t, m_t, loss_stock, loss_market = self._stage1_forward(
            prices_seq, features_seq, active_mask
        )
        
        # Stage 2
        _, loss_pred, _ = self._stage2_forward(
            features_seq, u_seq, r_t, m_t, targets, active_mask
        )
        
        return loss_stock + loss_market + loss_pred
    
    def _forward_predict(self, data: Dict[str, DataFrame], indexes: int, 
                        active_mask: torch.Tensor) -> Dict[str, float]:
        """UMI prediction forward pass."""
        
        data_tensor, data_mask = build_tensor(data, indexes, self.F, self._device)
        assert torch.equal(data_mask, active_mask), "Active mask mismatch"
        
        dataset = SlidingWindowDataset(data_tensor, self.L, self.pred_len, 
                                      self.target_idx, with_target=False)
        
        if len(dataset) == 0:
            return {}
        
        prices_seq, feat_seq, _, dataset_mask = dataset[0]
        prices_seq = prices_seq.unsqueeze(0).to(self._device)
        feat_seq = feat_seq.unsqueeze(0).to(self._device)
        mask = active_mask.unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            u_seq, r_t, m_t, _, _ = self._stage1_forward(prices_seq, feat_seq, mask)
            preds, _, _ = self._stage2_forward(feat_seq, u_seq, r_t, m_t, None, mask)
            preds = preds.squeeze(0)
        
        tickers = list(data.keys())
        return {ticker: float(preds[i].cpu()) for i, ticker in enumerate(tickers)}
    
    def initialize(self, data: Dict[str, DataFrame], total_bars: int) -> float:
        """Initialize with UMI-specific logic."""
        if self._is_initialized:
            return 0.0
        
        logger.info(f"[UMIModel] Initializing...")
        
        # Set instrument count
        self.I = len(data)
        
        # Build architecture
        self._build_model()
        
        # Train using UMI's custom logic
        if self.n_retrains_on_valid > 0:
            best_val = self._train_with_rolling_validation(data, self.n_epochs, total_bars)
        else:
            full_tensor, _ = build_tensor(data, total_bars, self.F, self._device)
            train_bars = int(total_bars * (1 - self.valid_split))
            train_loader, valid_loader = self._create_loaders(full_tensor, train_bars, total_bars)
            best_val, _ = self._train(train_loader, valid_loader, self.n_epochs)
        
        # Save
        torch.save(self.state_dict(), self.model_dir / "init.pt")
        torch.save(self.state_dict(), self.model_dir / "latest.pt")
        self._is_initialized = True
        
        logger.info(f"[initialize] Complete. Best validation: {best_val:.6f}")
        return best_val
    
    @override
    def _train_with_rolling_validation(self, data: Dict[str, DataFrame], 
                                    n_epochs: int, total_bars: int) -> float:
        """Rolling validation using UMI's custom training."""
        
        full_tensor, _ = build_tensor(data, total_bars, self.F, self._device)
        
        n_folds = 1 + self.n_retrains_on_valid
        train_bars = int(total_bars * (1 - self.valid_split))
        fold_size = (total_bars - train_bars) // n_folds
        
        if fold_size < self.L + self.pred_len:
            logger.warning("Validation fold too small, using single validation")
            train_loader, valid_loader = self._create_loaders(full_tensor, train_bars, total_bars)
            return self._train(train_loader, valid_loader, n_epochs)[0]
        
        best_overall = float('inf')
        
        for fold_idx in range(n_folds):
            train_end = train_bars + fold_idx * fold_size
            valid_end = train_end + fold_size if fold_idx < n_folds - 1 else total_bars
            fold_epochs = n_epochs if fold_idx == 0 else self.warm_training_epochs
            
            logger.info(f"Rolling fold {fold_idx+1}/{n_folds}: train[:{train_end}] valid[{train_end}:{valid_end}]")
            
            train_loader, valid_loader = self._create_loaders(full_tensor, train_end, valid_end)
            fold_best, self._global_epoch = self._train(train_loader, valid_loader, fold_epochs)
            best_overall = min(best_overall, fold_best)
        
        return best_overall
    def _train(self, train_loader, valid_loader, n_epochs: int) -> Tuple[float, int]:
        """UMI-specific multi-stage training."""
        
        # Setup optimizers
        optimizer_stock = torch.optim.AdamW(
            self.stock_factor.parameters(),
            lr=self.hp['lr_stage1'],
            weight_decay=self.hp.get('weight_decay', 0.0)
        )
        
        optimizer_market = torch.optim.AdamW(
            self.market_factor.parameters(),
            lr=self.hp['lr_stage1'],
            weight_decay=self.hp.get('weight_decay', 0.0)
        )
        
        optimizer_forecaster = torch.optim.AdamW(
            self.forecaster.parameters(),
            lr=self.hp['lr_stage2'],
            weight_decay=self.hp.get('weight_decay', 0.0)
        )
        
        best_val = float('inf')
        has_validation = valid_loader is not None
        
        # Dispatch to training mode
        if self.training_mode == "hybrid":
            best_val, self._global_epoch = self._train_hybrid_mode(
                train_loader, valid_loader, has_validation,
                optimizer_stock, optimizer_market, optimizer_forecaster,
                n_epochs, best_val, self._global_epoch
            )
        elif self.training_mode == "sequential":
            best_val, self._global_epoch = self._train_sequential_mode(
                train_loader, valid_loader, has_validation,
                optimizer_stock, optimizer_market, optimizer_forecaster,
                n_epochs, best_val, self._global_epoch
            )
        else:  # "joint"
            best_val, self._global_epoch = self._train_joint_mode(
                train_loader, valid_loader, has_validation,
                optimizer_stock, optimizer_market, optimizer_forecaster,
                n_epochs, best_val, self._global_epoch
            )
        
        return best_val, self._global_epoch



# ---------------- Model Specific Methods ----------------------- #

    def _train_joint_mode(self, train_loader, valid_loader, has_validation,
                        optimizer_stock, optimizer_market, optimizer_forecaster,
                        n_epochs, best_validation_loss, global_epoch) -> Tuple[float, int]:
        """Train all components jointly for n_epochs."""
        
        if logger:
            logger.info(f"[joint] Training all components jointly for {n_epochs} epochs")
        
        # Enable gradients for all components
        self.stock_factor.requires_grad_(True)
        self.market_factor.requires_grad_(True) 
        self.forecaster.requires_grad_(True)
        
        for epoch in range(n_epochs):
            # Training step
            train_loss, loss_stock, loss_market, loss_pred = self._execute_training_epoch(
                train_loader, optimizer_stock, optimizer_market, optimizer_forecaster
            )
            
            # Validation step
            validation_loss = float('nan')
            if has_validation:
                validation_loss = self._execute_validation_epoch(valid_loader)
                best_validation_loss = min(best_validation_loss, validation_loss)
            
            # Logging
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 5) == 0:
                if logger:
                    logger.info(f"Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        return best_validation_loss, global_epoch

    def _train_hybrid_mode(self, train_loader, valid_loader, has_validation,
                        optimizer_stock, optimizer_market, optimizer_forecaster,
                        n_epochs, best_validation_loss, global_epoch) -> Tuple[float, int]:
        """Hybrid training: Stage-1 pretraining + joint fine-tuning."""
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 1: Stage-1 Pretraining
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if logger:
            logger.info(f"[hybrid] Phase 1: Stage-1 pretraining for {self.pretrain_epochs} epochs")
        
        # Disable forecaster gradients
        self.stock_factor.requires_grad_(True)
        self.market_factor.requires_grad_(True)
        self.forecaster.requires_grad_(False)
        
        for epoch in range(self.pretrain_epochs):
            train_loss, loss_stock, loss_market, loss_pred = self._execute_training_epoch(
                train_loader, optimizer_stock, optimizer_market, optimizer_forecaster
            )
            
            validation_loss = float('nan')
            if has_validation:
                validation_loss = self._execute_validation_epoch(valid_loader)
                best_validation_loss = min(best_validation_loss, validation_loss)
            
            if epoch == 0 or (epoch + 1) % max(1, self.pretrain_epochs // 3) == 0:
                if logger:
                    logger.info(f"Pretrain Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 2: Joint Fine-tuning
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if logger:
            logger.info(f"[hybrid] Phase 2: Joint fine-tuning for {n_epochs} epochs")
        
        # Enable all gradients and reduce Stage-1 learning rates
        self.forecaster.requires_grad_(True)
        stage1_finetune_lr = self.hp.get("lr_stage1_ft", self.hp["lr_stage1"])
        
        for param_group in optimizer_stock.param_groups:
            param_group["lr"] = stage1_finetune_lr
        for param_group in optimizer_market.param_groups:
            param_group["lr"] = stage1_finetune_lr
        
        for epoch in range(n_epochs):
            train_loss, loss_stock, loss_market, loss_pred = self._execute_training_epoch(
                train_loader, optimizer_stock, optimizer_market, optimizer_forecaster
            )
            
            validation_loss = float('nan')
            if has_validation:
                validation_loss = self._execute_validation_epoch(valid_loader)
                best_validation_loss = min(best_validation_loss, validation_loss)
            
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 5) == 0:
                if logger:
                    logger.info(f"Finetune Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        return best_validation_loss, global_epoch

    def _train_sequential_mode(self, train_loader, valid_loader, has_validation,
                            optimizer_stock, optimizer_market, optimizer_forecaster,
                            n_epochs, best_validation_loss, global_epoch) -> Tuple[float, int]:
        """Sequential training: Stage-1 with early stopping + Stage-2 only."""
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 1: Stage-1 Training with Early Stopping
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if logger:
            logger.info(f"[sequential] Phase 1: Stage-1 training with early stopping")
        
        # Disable forecaster gradients
        self.stock_factor.requires_grad_(True)
        self.market_factor.requires_grad_(True)
        self.forecaster.requires_grad_(False)
        
        # Early stopping state
        best_stage1_state_stock = None
        best_stage1_state_market = None
        patience_counter = 0
        stage1_best_loss = float('inf')
        
        for epoch in range(n_epochs):
            train_loss, loss_stock, loss_market, loss_pred = self._execute_training_epoch(
                train_loader, optimizer_stock, optimizer_market, optimizer_forecaster
            )
            
            validation_loss = float('nan')
            if has_validation:
                validation_loss = self._execute_validation_epoch(valid_loader)
                
                # Early stopping logic
                improvement_threshold = 0.995  # 0.5% improvement required
                if validation_loss < stage1_best_loss * improvement_threshold:
                    stage1_best_loss = validation_loss
                    best_validation_loss = min(best_validation_loss, validation_loss)
                    patience_counter = 0
                    
                    # Save best Stage-1 weights
                    best_stage1_state_stock = {k: v.cpu().clone() 
                                            for k, v in self.stock_factor.state_dict().items()}
                    best_stage1_state_market = {k: v.cpu().clone() 
                                            for k, v in self.market_factor.state_dict().items()}
                else:
                    patience_counter += 1
                
                # Check early stopping
                if patience_counter >= self.patience:
                    if logger:
                        logger.info(f"[sequential] Early stopping after {patience_counter} epochs without improvement")
                    break
            
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
                if logger:
                    logger.debug(f"Stage1 Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f} | patience {patience_counter}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        # Restore best Stage-1 weights
        if best_stage1_state_stock is not None and best_stage1_state_market is not None and has_validation:
            if logger:
                logger.info("[sequential] Restoring best Stage-1 weights")
            self.stock_factor.load_state_dict(best_stage1_state_stock)
            self.market_factor.load_state_dict(best_stage1_state_market)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 2: Stage-2 Only Training
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if logger:
            logger.info(f"[sequential] Phase 2: Stage-2 training for {n_epochs} epochs")
        
        # Freeze Stage-1 components, enable forecaster
        self.stock_factor.requires_grad_(False)
        self.market_factor.requires_grad_(False)
        self.forecaster.requires_grad_(True)

        # This ensures shared parameter is trained only by MarketForecaster
        for param in self.market_factor.W_iota.parameters():
            param.requires_grad_(False)
        
        # Reinitialize forecaster optimizer
        optimizer_forecaster = torch.optim.AdamW(
            self.forecaster.parameters(),
            lr=self.hp["lr_stage2"],
            weight_decay=self.hp.get("weight_decay", 0.0)
        )
        
        for epoch in range(n_epochs):
            train_loss, loss_stock, loss_market, loss_pred = self._execute_training_epoch(
                train_loader, optimizer_stock, optimizer_market, optimizer_forecaster
            )
            
            validation_loss = float('nan')
            if has_validation:
                validation_loss = self._execute_validation_epoch(valid_loader)
                best_validation_loss = min(best_validation_loss, validation_loss)

            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
                if logger:
                    logger.debug(f"Stage2 Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        return best_validation_loss, global_epoch
    
    def _stage1_forward(self, prices_seq, feat_seq, active_mask):
        """Run stock-level & market-level factor learning *concurrently*."""

        assert active_mask.ndim == 2 , f"active_mask should have 2 dims, got instead {active_mask.ndim}"
        # assert active_mask.shape == (self.batch_size, self.I)  NOT TRUE for last batch with batch_size smaller
        stockIDs = self._eye.to(prices_seq.device)
        if torch.cuda.is_available():
            # ---- GPU: two independent CUDA streams ------------------- #
            stream_s = torch.cuda.Stream()
            stream_m = torch.cuda.Stream()
            with torch.cuda.stream(stream_s):
                u_seq, loss_s, _ = self.stock_factor(prices_seq, active_mask)  
            with torch.cuda.stream(stream_m):
                r_t, m_t, loss_m, _ = self.market_factor(feat_seq, stockIDs, active_mask)
            torch.cuda.synchronize()  # Wait for both streams to finish
            
        else:
            # ---- CPU: sequential execution -------------------------- #
            u_seq, loss_s, _ = self.stock_factor(prices_seq, active_mask)
            r_t, m_t, loss_m, _ = self.market_factor(feat_seq, stockIDs, active_mask)


        # ---- return results ---------------------------------------- #
        return u_seq, r_t, m_t, loss_s, loss_m

    def _stage2_forward(self, feat_seq, u_seq, r_t, m_t, target, active_mask):
        assert active_mask.ndim == 2 , f"active_mask should have 2 dims, got instead {list(active_mask.shape)}"
        # assert active_mask.shape == (self.batch_size, self.I) not correct since batch_size for last batch is smaller
        stockIDs = self._eye.to(feat_seq.device)

        # Get W_iota weight from market factor
        W_iota_weight = self.market_factor.W_iota.weight.detach()  # Shape: (2F, I)
        
        # Control gradient flow based on training phase
        # if not any(p.requires_grad for p in self.market_factor.parameters()):
        #     # Phase 2 (market factor frozen): detach to prevent gradients flowing back to market factor
        #     W_iota_weight = W_iota_weight.detach()
        return self.forecaster(
            W_iota_weight, feat_seq, u_seq, r_t, m_t, stockIDs, target, active_mask
        )

    def _execute_training_epoch(self, train_loader, optimizer_stock, optimizer_market, optimizer_forecaster):
        """Execute one training epoch and return average losses."""
        
        # Set models to training mode
        self.stock_factor.train()
        self.market_factor.train()
        self.forecaster.train()
        
        total_loss = 0.0
        total_loss_stock = 0.0
        total_loss_market = 0.0
        total_loss_pred = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (prices_seq, features_seq, targets, active_mask) in enumerate(train_loader):
            # Move data to device
            prices_seq = prices_seq.to(self._device)
            features_seq = features_seq.to(self._device) 
            targets = targets.to(self._device)
            active_mask = active_mask.to(self._device)
            
            # ═══════════════════════════════════════════════════════════════════════════════
            # Stage-1 Forward Pass (Stock + Market Factors)
            # ═══════════════════════════════════════════════════════════════════════════════
            
            u_seq, r_t, m_t, loss_stock, loss_market = self._stage1_forward(
                prices_seq, features_seq, active_mask
            )
            
            # Stage-1 Backward Pass (if components require gradients)
            stage1_requires_grad = (
                any(p.requires_grad for p in self.stock_factor.parameters()) or
                any(p.requires_grad for p in self.market_factor.parameters())
            )
            
            if stage1_requires_grad:
                optimizer_stock.zero_grad(set_to_none=True)
                optimizer_market.zero_grad(set_to_none=True)
                
                # Parallel backward passes on GPU, sequential on CPU
                if torch.cuda.is_available():
                    stream_stock = torch.cuda.Stream()
                    stream_market = torch.cuda.Stream()
                    with torch.cuda.stream(stream_stock):
                        loss_stock.backward(retain_graph=True)
                        optimizer_stock.step()
                    with torch.cuda.stream(stream_market):
                        loss_market.backward()
                        optimizer_market.step()
                    torch.cuda.synchronize()  # Wait for both streams
                else:
                    loss_stock.backward(retain_graph=True)
                    loss_market.backward()
                    optimizer_stock.step()
                    optimizer_market.step()
                
                
            
            # ═══════════════════════════════════════════════════════════════════════════════
            # Stage-2 Forward Pass (Forecasting)
            # ═══════════════════════════════════════════════════════════════════════════════
            
            if any(p.requires_grad for p in self.forecaster.parameters()):
                predictions, loss_pred, _ = self._stage2_forward(
                    features_seq, u_seq, r_t, m_t, targets, active_mask
                )
                
                optimizer_forecaster.zero_grad(set_to_none=True)
                loss_pred.backward()
                optimizer_forecaster.step()
            else:
                # Forecaster frozen, just compute dummy loss for logging
                loss_pred = torch.tensor(0.0, device=self._device)
            
            # ═══════════════════════════════════════════════════════════════════════════════
            # Accumulate Losses
            # ═══════════════════════════════════════════════════════════════════════════════
            
            batch_total_loss = loss_stock.item() + loss_market.item() + loss_pred.item()
            total_loss += batch_total_loss
            total_loss_stock += loss_stock.item()
            total_loss_market += loss_market.item()
            total_loss_pred += loss_pred.item()
        
        # Return average losses
        return (
            total_loss / num_batches,
            total_loss_stock / num_batches,
            total_loss_market / num_batches, 
            total_loss_pred / num_batches
        )

    def _execute_validation_epoch(self, valid_loader):
        """Execute validation epoch and return average prediction loss."""
        
        # Set models to evaluation mode
        self.stock_factor.eval()
        self.market_factor.eval()
        self.forecaster.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for prices_seq, features_seq, targets, active_mask in valid_loader:
                # Move data to device
                prices_seq = prices_seq.to(self._device)
                features_seq = features_seq.to(self._device)
                targets = targets.to(self._device) 
                active_mask = active_mask.to(self._device)
                
                # Forward pass
                u_seq, r_t, m_t, _, _ = self._stage1_forward(prices_seq, features_seq, active_mask)
                _, loss_pred, _ = self._stage2_forward(features_seq, u_seq, r_t, m_t, targets, active_mask)
                
                total_loss += loss_pred.item()
        
        return total_loss / len(valid_loader)

    def state_dict(self) -> Dict:
        return {
            "I": self.I,
            "F": self.F,
            "L": self.L,
            "stock_factor":  self.stock_factor.state_dict(),
            "market_factor": self.market_factor.state_dict(),
            "forecaster":    self.forecaster.state_dict(),
            "hparams": self.hp,
        }

    def load_state_dict(self, sd: Dict):
        self.I, self.F, self.L = sd["I"], sd["F"], sd["L"]
        self.hp.update(sd["hparams"])
        self._build_model()
        self.stock_factor.load_state_dict(sd["stock_factor"])
        self.market_factor.load_state_dict(sd["market_factor"])
        self.forecaster.load_state_dict(sd["forecaster"])
        self._is_initialized = True
    
    @override
    def _log_training_metrics(self, epoch, loss_stock, loss_market, loss_pred, validation_loss):
        """Log training metrics for the current epoch."""
        
        metrics = {
            'epoch': epoch,
            'train_loss_stock': loss_stock,
            'train_loss_market': loss_market, 
            'train_loss_pred': loss_pred,
            'total_loss': loss_stock + loss_market + loss_pred,
            'valid_loss_pred': validation_loss
        }
        
        self._epoch_logs.append(metrics)

        # Save to CSV file
        log_file = self.model_dir / "train_losses.csv"
        pd.DataFrame([metrics]).to_csv(
            log_file, 
            mode="a", 
            header=not log_file.exists(), 
            index=False
        )