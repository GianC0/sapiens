###############################################################################
# High-level UMI wrapper – engineered for live, back-test                     #
###############################################################################
#
# NOTE
# ----
# • Relies on the low-level blocks
#   (StockLevelFactorLearning, MarketLevelFactorLearning, ForecastingLearning)
# 
#
###############################################################################


import os, math, json, shutil, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
from pandas.tseries.offsets import BaseOffset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pandas import DataFrame, Timestamp
import pandas_market_calendars as market_calendars

from ..utils import SlidingWindowDataset, build_input_tensor, build_pred_tensor, freq2pdoffset
from .learners import StockLevelFactorLearning, MarketLevelFactorLearning, ForecastingLearning

# --------------------------------------------------------------------------- #
# Main model                                                               #
# --------------------------------------------------------------------------- #
class UMIModel(nn.Module):
    """
    Orchestrates factor learning + forecasting, periodic retraining.  Designed for both back-tests and always-on live services.
    """

    # ---------------- constructor ------------------------------------ #
    def __init__(
        self,
        freq: str,
        feature_dim: int,
        window_len: int,
        pred_len: int,
        train_offset: BaseOffset,
        pred_offset: BaseOffset,
        train_end: pd.Timestamp,
        valid_end: pd.Timestamp,
        n_epochs: int = 20,
        batch_size: int = 64,
        patience:int=10,
        pretrain_epochs:int=5,
        training_mode: str="sequential",    # hybrid/sequential:
                                            # hybrid -> stage1 factors for few epochs, then stage2 altogether
                                            # sequential -> stage1 first until early stopping, then stage2
        close_idx: int = 3,  # usually "close" is at index 3
        warm_start : bool= False,
        warm_training_epochs: int = 5,
        save_backups: bool = False,         # Flag for saving backups during walk-forward
        data_dir: Path = Path("data/stocks"),
        model_dir: Path = Path("logs/UMIModel"),
        logger: Optional[Any] = None,
        calendar: str = 'NYSE',

        **hparams,
    ):

        super().__init__()

        
        # hw parameters
        self._device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")         # device for training
        
        # time parameters
        self.market_calendar = market_calendars.get_calendar(calendar)
        self.freq           = freq                                                                  # e.g. "1d", "15m", "1h"  
        self.pred_len    = pred_len                                                                 # number of bars to predict
        self.train_offset   = train_offset                                                          # time window for training
        self.pred_offset    = pred_offset                                                           # time window for prediction
        self.train_end      = pd.Timestamp(train_end)                                               # end of training date
        self.valid_end      = pd.Timestamp(valid_end)                                               # end of validation date
        assert self.valid_end >= self.train_end, "Validation end date must be after training end date."

        self.logger         = logger                                                                # actions logger
        self.save_backups   = save_backups                                                          # whether to save backups during walk-forward
        #self.bt_end         = pd.to_datetime(bt_end, utc=True)                                     # end of back-test date (optional)
        

        # model parameters
        self.F              = feature_dim                                                          # number of features per stock
        self.L              = window_len                                                           # length of the sliding window (L+1 bars)
        self.I              = None                                                                 # number of stocks/instruments. here it is just initialized
        self.n_epochs       = n_epochs                                                             # number of epochs for training
        self.pretrain_epochs = pretrain_epochs                                                     # epochs for Stage-1 pre-training (hybrid mode)
        self.training_mode = training_mode                                                         # "hybrid" or "sequential"
        self.data_dir       = data_dir                                                             # directory where the data is stored      
         
        hp_id =   f"lamic{hparams.get('lambda_ic',0):.3f}_"\
                + f"lamsync{hparams.get('lambda_sync',0):.3f}_"\
                + f"lamrankic{hparams.get('lambda_rankic',0):.3f}"\
                + f"syncthres{hparams.get('sync_thr',0):.3f}"
        time = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        
        self.model_dir = (model_dir / freq / f"{time}_{hp_id}").resolve()                             # model directory with timestamp and hparams
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # stock universe
        self._universe: list[str] = []                                                               # list of ordered instruments the model is trained on
        #self.universe_mult = max(1, int(dynamic_universe_mult))                                     # over-allocation factor for dynamic universe
        
        self.close_idx = close_idx                                                                  # usually "close" is at index 3 in the dataframes
        
        # training parameters
        self._is_initialized = False
        self.batch_size     = batch_size                                                            # batch size for training
        self.patience = patience                                                                    # early stopping patience epochs for any training stage
        self.hp = self._default_hparams()                                                           # hyper-parameters
        self.hp.update(hparams)                                                                     # updated with those defined in the yaml config
        self.warm_start          = warm_start                                                       # warm start when stock is deleted or retrain delta elapsed
        self.warm_training_epochs = warm_training_epochs if warm_start is not None else self.n_epochs
        self._global_epoch = 0                                                                      # global epoch counter for stats
    



    @property
    def is_initialized(self) -> bool:
        """Check if model has been initialized."""
        return self._is_initialized
    
    def initialize(self, data: Dict[str, DataFrame], **kwargs) -> float:
        """
        Initial training on historical data.
         
        - This is the entry point for first training
        - No walk-forward logic
        - Trains from scratch
        - Returns best validation loss
        """
        assert self._is_initialized is False
        if self.logger:
            self.logger.info(f"[{self.__class__.__name__}] Initializing model...")
    
        # Register active mask
        self.I = len(data)
        active_mask = torch.ones(self.I, dtype=torch.bool)  # all stocks are active at initialization
        
        # Register new universe
        self._universe = list(data.keys())

        # Build submodules (learners)
        self._build_submodules()

       
        # Train from scratch
        if self.logger:
            self.logger.info(f"[initialize] Training for {self.n_epochs} epochs...")
        best_val = self._train(data, self.n_epochs, active_mask )
        
        # Save initial state
        torch.save(self.state_dict(), self.model_dir / "init.pt")
        
        # Mark as initialized
        self._is_initialized = True
        
        # Dump metadata
        #self._dump_model_metadata()
        
        if self.logger:
            self.logger.info(f"[initialize] Complete. Best validation: {best_val:.6f}")

        # setting up MLflow logging
        self._setup_mlflow()
        # Log final model to MLflow
        mlflow.pytorch.log_model(self, "model")
        mlflow.log_artifact(str(self.model_dir / "init.pt"))
        mlflow.log_metric("best_validation_loss", best_val)

        return best_val

    def update(self, data: Dict[str, DataFrame], current_time: Timestamp, active_mask: torch.Tensor):
        """
        One-off training (train + valid).
        """
        
        assert torch.numel(active_mask) == self.I, f"Active mask size must match the number of trained instruments ({self.I})"

        # Assert universe
        assert self._universe == list(data.keys()), "Universe error!"

        # set current time for end of training (no validation)
        # the model assumes that input cutoff of old data is done by strategy at current_time - training_offset
        self.train_end = current_time
        self.valid_end = current_time

        if not self.warm_start:
            if self.logger:
                self.logger.info(f"[update] Warm-start disabled. Initialization on most updated window up until: {current_time}")
    
            # Reinitialize with new windows
            self._is_initialized = False
            self.initialize(data)
            return

        assert self._is_initialized

        # Regular warm-start training
        if self.logger:
            self.logger.info(f"[update] Warm-start training...")
        latest_path = self.model_dir / "latest.pt"
        if latest_path.exists():
            if self.logger:
                self.logger.info(f"[update] Loading weights from {latest_path}")
            try:
                self.load_state_dict(torch.load(latest_path, map_location='cpu'))
            except Exception as e:
                if self.logger:
                    self.logger.info(f"[update] Failed to load weights: {e}, training from scratch")
                self._is_initialized = False
                self.initialize(data)
                return
        else:
            raise ImportError("Could not find latest.pt warmed-up model in ", self.model_dir)
        
        # Train
        _ = self._train(data, self.warm_training_epochs, active_mask)
        
        # Save backup if requested
        if self.save_backups:
            backup_name = f"checkpoint_{current_time.strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(self.state_dict(), self.model_dir / backup_name)
            if self.logger:
                self.logger.info(f"[update] Saved backup: {backup_name}")
        
        # Save state
        torch.save(self.state_dict(), self.model_dir / "latest.pt")

        if self.logger:
            self.logger.info(f"[update] Complete. Model up-to-date at: {current_time}")
        
        # Log update metrics
        mlflow.log_metric("update_timestamp", current_time.timestamp())
        mlflow.log_artifact(str(self.model_dir / "latest.pt"))       


    def predict(self, data: Dict[str, DataFrame], current_time: pd.Timestamp, active_mask: torch.Tensor) -> Dict[str, float]:
        """
        Generate predictions for current market state.
        
        Args:
            data_dict: Dict[ticker, DataFrame] with current market data
            current_time: pd.Timestamp indicating the current time
            active_mask: pre-computed mask for active instruments for verification
            
        Returns:
            Dict[ticker, prediction] for active instruments only
        """
        assert self._is_initialized, "Model must be initialized before prediction"
        assert torch.numel(active_mask) == self.I, "Active mask size must match the number of stocks (I)"
        
        # Assert universe
        assert self._universe == list(data.keys()), "Universe error!"

        # Build panel from current data
        days_range = self.market_calendar.schedule(start_date=current_time, end_date=current_time + self.pred_offset)
        timestamps = market_calendars.date_range(days_range, frequency=self.freq).normalize()
        data_tensor, data_mask = build_pred_tensor(data, timestamps, feature_dim=self.F, device=self._device)
        
        # assertion on the universe size
        assert torch.equal(data_mask, active_mask), "Active mask mismatch during prediction"
        
        dataset = SlidingWindowDataset(data_tensor, self.L, self.pred_len, self.close_idx, with_target = False)
        assert len(dataset) == 1, "Prediction dataset should contain exactly one window"
        
        # Get the last window for prediction
        if len(dataset) == 0:
            if self.logger:
                self.logger.warning("Insufficient data for prediction")
            return {}
        
        # Get last window
        prices_seq, feat_seq, _, dataset_mask = dataset[0]
        assert torch.equal(data_mask, dataset_mask), "Dataset mask different from build_pred_tensor input mask during prediction"
        
        
        # Add batch dimension and move to device
        # TODO: ensure that B=1 still works in this case and should not be expanded
        prices_seq = prices_seq.unsqueeze(0).to(self._device)  # (1, L+1, I)
        feat_seq = feat_seq.unsqueeze(0).to(self._device)      # (1, L+1, I, F)
        
        # Use provided mask or dataset mask
        assert active_mask is not None
        mask = active_mask.to(self._device)
        
        
        # Forward pass
        with torch.no_grad():
            # Stage 1: Factor learning
            u_seq, r_t, m_t, _, _ = self._stage1_forward(prices_seq, feat_seq, mask)
            
            # Stage 2: Prediction
            preds, _, _ = self._stage2_forward(feat_seq, u_seq, r_t, m_t, target=None, active_mask=mask)
            
            # Extract predictions for instruments
            preds = preds.squeeze(0)  # Remove batch dimension -> (I,)

            #TODO: ensure to be able to pred for pred_len>1 :  preds = (I, self.pred_len)
        
        # Return predictions as dict for active tickers only
        #tickers = list(data.keys())
        #result = {}
        #for i, ticker in enumerate(tickers):
        #    if i < len(mask) and mask[i]:
        #        result[ticker] = float(preds[i].cpu())

        # Return predictions as dict for ALL tickers
        tickers = list(data.keys())
        result = {}
        for i, ticker in enumerate(tickers):
            result[ticker] = float(preds[i].cpu())
        
        return result

    # ------------------------------------------------------------------ #
    # state-dict helpers (single-process save/load)                       #
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "I": self.I,
            "F": self.F,
            "L": self.L,
            "stock_factor":  self.stock_factor.state_dict(),
            "market_factor": self.market_factor.state_dict(),
            "forecaster":    self.forecaster.state_dict(),
            "hparams": self.hp,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.I, self.F, self.L = sd["I"], sd["F"], sd["L"]
        self.hp.update(sd["hparams"])
        self._build_submodules()
        self.stock_factor.load_state_dict(sd["stock_factor"])
        self.market_factor.load_state_dict(sd["market_factor"])
        self.forecaster.load_state_dict(sd["forecaster"])
        self._is_initialized = True
        


# --------------------------------------------------------------------------- #
# Utilities functions                                                         #
# --------------------------------------------------------------------------- #

    # ---------------- hyper-param defaults and suggest function--------------------------- #
    def _default_hparams(self) -> Dict[str, Any]:
        return dict(
            lambda_ic     = 0.5,  # official UMI paper value for stocks
            lambda_sync   = 1.0,  # official UMI paper value for market 
            lambda_rankic = 0.1,  # official UMI paper value for forecasting
            temperature   = 0.07,
            sync_thr      = 0.6,   # official UMI paper value
            weight_decay  = 0.0,
            lr_stage1      = 1e-3,
            lr_stage1_ft   = 1e-4,
            lr_stage2      = 1e-4,
        )
    
    # ---------------- sub-module builder ----------------------------- #
    def _build_submodules(self):
        I = self.I
        if not hasattr(self, "_eye"):
            self._eye = torch.eye(I, dtype=torch.float32)

        self.stock_factor = StockLevelFactorLearning(
            I, lambda_ic=self.hp["lambda_ic"]
        ).to(self._device)

        self.market_factor = MarketLevelFactorLearning(
            I, self.F, window_L=self.L,
            lambda_sync=self.hp["lambda_sync"],
            temperature=self.hp["temperature"],
            sync_threshold=self.hp["sync_thr"],
        ).to(self._device)

        self.forecaster = ForecastingLearning(
            I, self.F, u_dim=1,
            pred_len=self.pred_len,
            lambda_rankic=self.hp["lambda_rankic"],
        ).to(self._device)
        
        return

    # MLflow logging function
    def _setup_mlflow(self):
        """Setup MLflow tracking for this model instance."""
        mlflow.set_tracking_uri("file:./mlruns")  # Local file store
        mlflow.set_experiment(f"UMI_{self.freq}")
        
        # Start a new run if not already in one
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f"UMI_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log model configuration
        mlflow.log_params({
            "model_name": "UMIModel",
            "freq": self.freq,
            "feature_dim": self.F,
            "window_len": self.L,
            "pred_len": self.pred_len,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "training_mode": self.training_mode,
            "train_end": str(self.train_end),
            "valid_end": str(self.valid_end),
        })
        
        # Log hyperparameters
        for key, value in self.hp.items():
            mlflow.log_param(f"hp_{key}", value)
    # ---------------- metadata on memory size and params ------------- #
    # def _dump_model_metadata(self):
    #     """
    #     Persists:
    #     • param_inventory.csv   – per-tensor shape / #params / bytes
    #     • run_meta.json         – run-time settings + totals
    #     """
        
    #     records, total_p, total_b = [], 0, 0
    #     sd = self.state_dict()
    #     for blk, obj in sd.items():
    #         if isinstance(obj, dict):           # sub-module dict
    #             for n, t in obj.items():
    #                 numel = t.numel(); bytes_ = numel * t.element_size()
    #                 records.append(dict(tensor=f"{blk}.{n}",
    #                                     shape=list(t.shape),
    #                                     numel=numel, bytes=bytes_))
    #                 total_p += numel; total_b += bytes_
    #         else:                               # flat tensor
    #             numel = torch.numel(obj); bytes_ = numel * obj.element_size()
    #             records.append(dict(tensor=blk, shape=list(obj.shape),
    #                                 numel=numel, bytes=bytes_))
    #             total_p += numel; total_b += bytes_

    #     pd.DataFrame(records).to_csv(self.model_dir / "param_inventory.csv",
    #                                 index=False)

    #     # some model params information
    #     some = dict(
    #         freq=self.freq, feature_dim=self.F, window_len=self.L,
    #         pred_len=self.pred_len, train_end=str(self.train_end),
    #         valid_end=str(self.valid_end), n_epochs=self.n_epochs,
    #         #dynamic_universe_mult=self.universe_mult, 
    #         total_params=total_p,
    #         approx_size_mb=round(total_b / 1024 / 1024, 3),
    #     )
    #     # adding hparams
    #     some.update(self.hp)

    #     # storing all params infos
    #     with open(self.model_dir / "all_params.json", "w") as fp:
    #         json.dump(some, fp, indent=2)

    # ----------------------------------------------------------------- #
    # ---------------- training utilities ----------------------------- #
    # ----------------------------------------------------------------- #
    def _train(self, data: Dict[str, DataFrame], n_epochs: int, active_mask: torch.Tensor) -> float:
        """
        Unified training function that handles all three training modes:
        - "joint": Train all components together
        - "hybrid": Pretrain Stage-1, then joint fine-tuning  
        - "sequential": Train Stage-1 to convergence, then Stage-2 only
        
        Returns the best validation loss achieved.
        """
        if self.logger:
            self.logger.info(f"[_train] Starting training with mode: {self.training_mode.capitalize()}")
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1. DATA PREPARATION
        # ═══════════════════════════════════════════════════════════════════════════════

        if self.valid_end < self.train_end:
            raise ValueError(f"Validation end date: {self.valid_end} is earlier that end train end date {self.train_end}")

        # Create common timestamp index. This solves the start= -1 bar problem
        start_date = self.valid_end - self.train_offset + freq2pdoffset(self.freq)
        days_range = self.market_calendar.schedule(start_date= start_date, end_date=self.valid_end)
        timestamps = market_calendars.date_range(days_range, frequency=self.freq).normalize()
        
        assert len(data) > 0
        (train_tensor, train_mask) , (valid_tensor, valid_mask) = build_input_tensor(data=data, timestamps=timestamps, feature_dim=self.F, split_valid_timestamp=self.train_end, device=self._device )

        # empty tensor has size 1 with Null object
        has_validation = torch.numel(valid_tensor) > 1
        if has_validation:
            assert torch.equal(valid_mask, active_mask) , "Active mask mismatch during training"
        else:
            assert torch.equal(train_mask, active_mask), "Active mask mismatch during training"
        # Build training dataset
        train_dataset = SlidingWindowDataset(train_tensor, self.L, self.pred_len, target_idx=self.close_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build validation dataset (if available)
        valid_loader = None
        if has_validation:
            valid_dataset = SlidingWindowDataset(valid_tensor, self.L, self.pred_len, target_idx=self.close_idx)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 2. OPTIMIZERS SETUP
        # ═══════════════════════════════════════════════════════════════════════════════
        
        optimizer_stock = torch.optim.AdamW(
            self.stock_factor.parameters(),
            lr=self.hp["lr_stage1"],
            weight_decay=self.hp.get("weight_decay", 0.0)
        )
        
        optimizer_market = torch.optim.AdamW(
            self.market_factor.parameters(),
            lr=self.hp["lr_stage1"], 
            weight_decay=self.hp.get("weight_decay", 0.0)
        )
        
        optimizer_forecaster = torch.optim.AdamW(
            self.forecaster.parameters(),
            lr=self.hp["lr_stage2"],
            weight_decay=self.hp.get("weight_decay", 0.0)
        )
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 3. TRAINING STATE INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════════════
        
        best_validation_loss = float('inf')
        self._epoch_logs = []
        global_epoch = 0
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 4. TRAINING MODE DISPATCH
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if self.training_mode == "hybrid":
            best_validation_loss = self._train_hybrid_mode(
                train_loader, valid_loader, has_validation,
                optimizer_stock, optimizer_market, optimizer_forecaster,
                n_epochs, best_validation_loss, global_epoch
            )
            
        elif self.training_mode == "sequential":
            best_validation_loss = self._train_sequential_mode(
                train_loader, valid_loader, has_validation,
                optimizer_stock, optimizer_market, optimizer_forecaster,
                n_epochs, best_validation_loss, global_epoch
            )
            
        else:  # "joint" mode (default) - safe fallback
            best_validation_loss = self._train_joint_mode(
                train_loader, valid_loader, has_validation,
                optimizer_stock, optimizer_market, optimizer_forecaster,
                n_epochs, best_validation_loss, global_epoch
            )
        
        if self.logger:
            self.logger.info(f"[_train] Training completed. Best validation loss: {best_validation_loss:.6f}")
        return best_validation_loss


    def _train_joint_mode(self, train_loader, valid_loader, has_validation,
                        optimizer_stock, optimizer_market, optimizer_forecaster,
                        n_epochs, best_validation_loss, global_epoch):
        """Train all components jointly for n_epochs."""
        
        if self.logger:
            self.logger.info(f"[joint] Training all components jointly for {n_epochs} epochs")
        
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
                if self.logger:
                    self.logger.info(f"Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        return best_validation_loss


    def _train_hybrid_mode(self, train_loader, valid_loader, has_validation,
                        optimizer_stock, optimizer_market, optimizer_forecaster,
                        n_epochs, best_validation_loss, global_epoch):
        """Hybrid training: Stage-1 pretraining + joint fine-tuning."""
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 1: Stage-1 Pretraining
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if self.logger:
            self.logger.info(f"[hybrid] Phase 1: Stage-1 pretraining for {self.pretrain_epochs} epochs")
        
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
                if self.logger:
                    self.logger.info(f"Pretrain Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 2: Joint Fine-tuning
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if self.logger:
            self.logger.info(f"[hybrid] Phase 2: Joint fine-tuning for {n_epochs} epochs")
        
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
                if self.logger:
                    self.logger.info(f"Finetune Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        return best_validation_loss


    def _train_sequential_mode(self, train_loader, valid_loader, has_validation,
                            optimizer_stock, optimizer_market, optimizer_forecaster,
                            n_epochs, best_validation_loss, global_epoch):
        """Sequential training: Stage-1 with early stopping + Stage-2 only."""
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 1: Stage-1 Training with Early Stopping
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if self.logger:
            self.logger.info(f"[sequential] Phase 1: Stage-1 training with early stopping")
        
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
                    if self.logger:
                        self.logger.info(f"[sequential] Early stopping after {patience_counter} epochs without improvement")
                    break
            
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
                if self.logger:
                    self.logger.info(f"Stage1 Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f} | patience {patience_counter}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        # Restore best Stage-1 weights
        if best_stage1_state_stock is not None and best_stage1_state_market is not None and has_validation:
            if self.logger:
                self.logger.info("[sequential] Restoring best Stage-1 weights")
            self.stock_factor.load_state_dict(best_stage1_state_stock)
            self.market_factor.load_state_dict(best_stage1_state_market)
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE 2: Stage-2 Only Training
        # ═══════════════════════════════════════════════════════════════════════════════
        
        if self.logger:
            self.logger.info(f"[sequential] Phase 2: Stage-2 training for {n_epochs} epochs")
        
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
            
            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 5) == 0:
                if self.logger:
                    self.logger.info(f"Stage2 Epoch {global_epoch:>3} | train {train_loss:.5f} | "
                            f"val {validation_loss:.5f} | best {best_validation_loss:.5f}")
            
            self._log_training_metrics(global_epoch, loss_stock, loss_market, loss_pred, validation_loss)
            global_epoch += 1
        
        return best_validation_loss


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


    def _log_training_metrics(self, epoch, loss_stock, loss_market, loss_pred, validation_loss):
        """Log training metrics for the current epoch."""
        
        metrics = {
            'epoch': epoch,
            'loss_stock': loss_stock,
            'loss_market': loss_market, 
            'loss_pred': loss_pred,
            'val_pred': validation_loss
        }
        
        self._epoch_logs.append(metrics)
        
        # Log to MLflow
        mlflow.log_metrics({
            'train_loss_stock': loss_stock,
            'train_loss_market': loss_market,
            'train_loss_pred': loss_pred,
            'val_loss': validation_loss if not pd.isna(validation_loss) else 0.0,
            'total_loss': loss_stock + loss_market + loss_pred
        }, step=epoch)


        # Save to CSV file
        log_file = self.model_dir / "train_losses.csv"
        pd.DataFrame([metrics]).to_csv(
            log_file, 
            mode="a", 
            header=not log_file.exists(), 
            index=False
        )
        self._global_epoch += 1
    # ---------------- stage1 / stage2 wrappers ----------------------- #
    def _stage1_forward(self, prices_seq, feat_seq, active_mask):
        """Run stock-level & market-level factor learning *concurrently*."""

        assert active_mask.ndim == 2 , f"active_mask must be of size ({B},{I}); got {list(active_mask.shape)}"
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
        assert active_mask.ndim == 2 , f"active_mask must be of size ({B},{I}); got {list(active_mask.shape)}"
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
