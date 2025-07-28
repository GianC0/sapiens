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

# TODO: CLOCK LOGIC TO BE REMOVED self._clock()

import os, math, json, shutil, datetime as dt
from pathlib import Path
from typing import Dict, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from ..utils import SlidingWindowDataset
from .learners import StockLevelFactorLearning, MarketLevelFactorLearning, ForecastingLearning
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pandas import DataFrame
     

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
        train_end: str,
        valid_end: str,
        #retrain_delta: pd.DateOffset = pd.DateOffset(days=30),
        #dynamic_universe_mult: float | int = 2,
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
        data_dir: Path = Path("data/stocks"),
        model_dir: Path = Path("logs/UMIModel"),
        logger: Optional[Any] = None,
        **hparams,
    ):

        super().__init__()

        
        # hw parameters
        self._device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")         # device for training
        
        # time parameters
        self.freq           = freq                                                                  # e.g. "1d", "15m", "1h"  
        self.pred_len    = pred_len                                                                 # number of bars to predict
        
        #self.retrain_delta  = retrain_delta                                                        # time delta for retraining 
        self.train_end      = pd.to_datetime(train_end, utc=True)                                   # end of training date
        self.valid_end      = pd.to_datetime(valid_end, utc=True)                                   # end of validation date
        self.log         = logger                                                                   # actions logger
        #self.bt_end         = pd.to_datetime(bt_end, utc=True)                                     # end of back-test date (optional)
        #self._last_fit_time = None                                                                 # last time fit() was called
        

        # model parameters
        self.F              = feature_dim                                                          # number of features per stock
        self.L              = window_len                                                           # length of the sliding window (L+1 bars)
        self.n_epochs       = n_epochs                                                             # number of epochs for training
        self.pretrain_epochs = pretrain_epochs                                                     # epochs for Stage-1 pre-training (hybrid mode)
        self.training_mode = training_mode                                                         # "hybrid" or "sequential"
        self.data_dir       = data_dir                                                             # directory where the data is stored      
         
        hp_id =   f"lamic{hparams.get('lambda_ic',0):.3f}_"\
                + f"lamsync{hparams.get('lambda_sync',0):.3f}_"\
                + f"lamrankic{hparams.get('lambda_rankic',0):.3f}"\
                + f"syncthres{hparams.get('sync_thr',0):.3f}"
        time = pd.Timestamp.utcnow().strftime('%Y-%m-%d %X')
        
        self.model_dir = (model_dir / freq / f"{time}_{hp_id}").resolve()                             # model directory with timestamp and hparams
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # stock universe
        #self._universe: list[str] = []                                                              # stable “slot → ticker” mapping
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


    def initialize(self, data: Dict[str, DataFrame], **kwargs) -> None:

        assert self._is_initialized == False
        """
        Initial training on historical data.
         
        - This is the entry point for first training
        - No walk-forward logic
        - Trains from scratch
        """
        self.log.info(f"[{self.__class__.__name__}] Initializing model...")
        
        # Build panel and setup
        panel, active, idx, universe = self._build_panel(data, lookback=0)
        self.I_active = panel.size(1)                                                        # real stocks today
        self.I = max(self.I_active + 1, math.ceil(self.I_active * (1.0 + self.universe_mult)))   # padded size for dynamic universe

        # Register active mask
        pad = self.I - self.I_active
        self.register_buffer("_active_mask",torch.cat([torch.ones(self.I_active), torch.zeros(pad)]).bool())
        
        # Build submodules (learners)
        self._build_submodules()

        # Create train/valid split
        train_mask = idx <= self.train_end
        valid_mask = (idx > self.train_end) & (idx <= self.valid_end)
        
        ds_train = SlidingWindowDataset(panel[train_mask], active[:, train_mask], self.L, self.pred_len, close_idx=self.close_idx)
        ds_valid = SlidingWindowDataset(panel[valid_mask], active[:, valid_mask] ,self.L, self.pred_len, close_idx=self.close_idx)
        loader_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        loader_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=True)
       
        # Train from scratch
        self.log.info(f"[initialize] Training for {self.n_epochs} epochs...")
        best_val = self._train(loader_train, loader_valid, self.n_epochs)
        self._last_val_loss = best_val
        
        # Save initial state
        torch.save(self.state_dict(), self.model_dir / "initial.pt")
        torch.save(self.state_dict(), self.model_dir / "best.pt")
        
        # Mark as initialized
        self._is_initialized = True
        self._last_fit_time = pd.Timestamp.utcnow()
        
        # Dump metadata
        self._dump_model_metadata()
        
        self.log.info(f"[initialize] Complete. Best validation: {best_val:.6f}")

    # ---------------- fit -------------------------------------------- #
    def fit(self, data: DataSource , warm_start: bool = True):
        """
        One-off training (train + valid).
        """

        if not warm_start:
            # Update training windows to latest data
            assert isinstance(data, Cache)
            # Get latest timestamp from cache
            latest_ts = None
            for bar in data.bars(self.bar_type): #TODO: no bar type
                if latest_ts is None or bar.ts_event > latest_ts:
                    latest_ts = bar.ts_event
            
            if latest_ts:
                latest = pd.Timestamp(latest_ts, unit='ns', tz='UTC')
                # Shift windows forward
                train_period = self.valid_end - self.train_end
                self.valid_end = latest
                self.train_end = self.valid_end - train_period
                self.log.info(f"[fit] Updated windows - train_end: {self.train_end}, valid_end: {self.valid_end}")
    
            # Reinitialize with new windows
            self._is_initialized = False
            self.initialize(data)
            return

        # Check if architecture needs updating after insertion. TODO: does not account for insertion and delisting at the same time
        if panel.size(1) > self.I_active:
            self.log.info(f"[fit] Universe expanded, need to reinitialize...")
            self.fit(data, initialize=True)
            return

        # Regular warm-start training
        self.log.info(f"[fit] Warm-start training...")

        # Build panel
        panel, active, idx = self._build_panel(data, lookback=None)   # panel.shape: (T,I,F)



        best_path = self.model_dir / "best.pt"
        if best_path.exists():
            self.log.info(f"[fit] Loading weights from {best_path}")
            try:
                self.load_state_dict(torch.load(best_path, map_location='cpu'))
            except Exception as e:
                self.log.info(f"[fit] Failed to load weights: {e}, training from scratch")
        else:
            raise Exception("Could not find best.pt warmed-up model in ", model_dir)

        # Update training windows for walk-forward
        current_time = self._clock()
        if self._last_fit_time:
            time_shift = current_time - self._last_fit_time
            self.train_end = self.train_end + time_shift
            self.valid_end = self.valid_end + time_shift

        train_mask = idx <= self.train_end
        valid_mask = (idx > self.train_end) & (idx <= self.valid_end)
        
        ds_train = SlidingWindowDataset(
            panel[train_mask], active[:, train_mask],
            self.L, self.pred_len, close_idx=self.close_idx
        )
        ds_valid = SlidingWindowDataset(
            panel[valid_mask], active[:, valid_mask],
            self.L, self.pred_len, close_idx=self.close_idx
        )
        
        # Create loaders
        loader_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        loader_valid = DataLoader(ds_valid, batch_size=self.batch_size, shuffle=True)
        
        # Train
        best_val = self._train(loader_train, loader_valid, self.warm_training_epochs)
        
        # Save state
        torch.save(self.state_dict(), self.model_dir / "best.pt")

        # save training logs
        pd.DataFrame(self._epoch_logs).to_csv(self.model_dir / "train_losses.csv",index=False)
        if self._global_epoch == 0:
            self._dump_model_metadata() # save model metadata

        self._last_fit_time = current_time
        
        self.log.info(f"[fit] Complete. Best validation: {best_val:.6f}")       

    # ------------------------------------------------------------------ #
    # update : decide if retrain_delta elapsed → refit on latest data    #
    # ------------------------------------------------------------------ #
    def update(self, data_dict: Dict[str, DataFrame] ):
        if self._last_fit_time is None:
            raise RuntimeError("Call fit() before update().")

        now = self._clock()        # <── the only place we read “time”

        # ---------- build fresh activity mask at NOW ------------------

        #  Add new tickers to the first free slots
        for k in sorted(data_dict):
            if k not in self._universe:
                if len(self._universe) < self.I:        # capacity check
                    self._universe.append(k)            # reserve slot
                    self.fit(data_dict, warm_start=False, n_epochs=self.n_epochs)               # rebuild with larger I. fit will expand the universe
                    return
                else:
                    self._universe.append(k)              # give the new stock a slot
                    self.log.info(f"[info] universe full. Retraining …")
                    self.fit(data_dict, warm_start=False, n_epochs=self.n_epochs)               # rebuild with larger I. fit will expand the universe
                    return                                # early exit; retrain done

        _, active, idx = _build_panel(data_dict, universe=self._universe)
        new_mask = active[:, -1]                       # (I_active_now) mask for stocks that are live NOW
        latest = idx[-1]
        self.valid_end = latest
        self.train_end = self.valid_end - pd.DateOffset(months=6)
        pad = self.I - new_mask.size(0)
        if pad:
            new_mask = torch.cat([new_mask, torch.zeros(pad, dtype=torch.bool)])

        # ---------- detect newly-delisted stocks ----------------------
        was_live = self._active_mask[: new_mask.size(0)]
        just_delisted = (was_live & (~new_mask)).any().item()   # True if at least one stock has disappeared since the last update, False otherwise.

        # always store most-recent mask for later calls
        with torch.no_grad():
            self._active_mask.copy_(new_mask)

        # ---------- decide whether to retrain -------------------------
        time_elapsed = now >= self._last_fit_time + self.retrain_delta
        if time_elapsed or just_delisted:
            reason = "time window" if time_elapsed else "delisting event"
            self.log.info(f"Retraining triggered : {reason}.")
            self.fit(data_dict, warm_start=self.warm_start, n_epochs=self.warm_training_epochs)

    # ------------------------------------------------------------------ #
    # predict : one inference step                                       #
    # ------------------------------------------------------------------ #
    def predict(self, data_dict: Dict[str, DataFrame],  _retry: bool = False) -> Dict[str, float]:

        panel, active, idx = _build_panel(data_dict, universe=self._universe)   # (T,I_active,F) , (I_active,T)
        assert panel.size(0) >= self.L + 1, "need L+1 bars for inference"

        prices_seq = panel[-self.L-1:, :, self.close_idx]    # (L+1,I)
        feat_seq   = panel[-self.L-1:]                       # (L+1,I,F)
        prices_seq = prices_seq.to(self._device)
        feat_seq   = feat_seq.to(self._device)

        # ---- build 1-D mask for the *current* bar (t = L) -------------
        active_mask = active[:, -1]                          # (I_active) at last time step
        pad = self.I - active_mask.size(0)
        if pad>=0:                                              # keep same length as in training
            active_mask = torch.cat([active_mask, torch.zeros(pad, dtype=torch.bool)])
        else:
            if _retry:
                raise RuntimeError("Universe capacity still insufficient after one retrain.")
            self.log.info("[warn] universe overflow – growing capacity and retraining once.")
            # 2) full re-fit on the larger I
            self.fit(data_dict, warm_start=False, n_epochs=self.n_epochs )
            # 3) recurse – now pad will be ≥ 0
            return self.predict(data_dict, _retry=True)

        active_mask = active_mask.to(self._device).unsqueeze(0)  # → (1, I)

        with torch.no_grad():
            u_seq, r_t, m_t, _ , _ = self._stage1_forward(prices_seq.unsqueeze(0), feat_seq.unsqueeze(0), active_mask=active_mask )
            preds, _, _ = self._stage2_forward(feat_seq.unsqueeze(0), u_seq, r_t, m_t, target=None, active_mask=active_mask)
            preds = preds[:, :self.I_active]
        # return as dict {stock: prediction}
        keys = sorted(data_dict.keys())
        return {k: float(preds[0, i].cpu()) for i, k in enumerate(keys)}

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
        assert self.I == sd["I"] and self.F == sd["F"]



    # --------------------------------------------------------------------------- #
    #  Utility : build a panel tensor from the {stockID: dataframe} dict        #
    # --------------------------------------------------------------------------- #
    def _build_panel(data_dict: Dict[str, DataFrame], lookback: int) -> torch.Tensor:
        """
        Returns training data tensor of shape (T, I, F) sorted by timestamp ascending &
        stocks alphabetically.
        Returns
        tensor : (T, I, F)
        active : (I, T)  bool mask for active stocks
        idx    : DatetimeIndex (UTC)
        """

        
        # Set universe
        universe = sorted(data_dict.keys())
        self.log.info(f"[initialize] Universe: {len(universe)} stocks")
        
        
        # --- harmonise indices -------------------------------------------------
        for k, df in data_dict.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)  

        long = pd.concat([df[self.feature_dim].assign(stock=k) for k, df in data_dict.items()])

        # ---------- proper pivot ---------------------------------------------- #
        df_pivot = long.pivot_table(
            values=self.feature_dim,      # every feature
            index=long.index,             
            columns="stock"
        ).sort_index()

        #  re-index columns ⇢ fill missing tickers with NaN so slots
        if universe is not None:
            df_pivot = df_pivot.reindex( pd.MultiIndex.from_product([self.feature_dim, keys]), axis=1 )

        T, F, I = len(df_pivot), len(self.feature_dim), len(keys)
        values = torch.tensor(df_pivot.values, dtype=torch.float32)
        tensor = values.reshape(T, F, I).permute(0,2,1) # (T,I,F)
        active = torch.isfinite(tensor).all(-1).transpose(0,1) # (I,T)

        return tensor, active, df_pivot.index, universe  

    
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
        if not hasattr(self, "_eye") or self._eye.size(0) != I:
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
            W_iota=self.market_factor.W_iota,
            pred_len=self.pred_len,
            lambda_rankic=self.hp["lambda_rankic"],
        ).to(self._device)
        
        return

    def _log_epoch(self, l_s, l_m, l_p, *, val_pred=None):
        row = dict(epoch=self._global_epoch, loss_stock=l_s, loss_market=l_m,
                loss_pred=l_p, val_pred=val_pred)
        self._epoch_logs.append(row)
        fn = self.model_dir / "train_losses.csv"
        pd.DataFrame([row]).to_csv(fn, mode="a", header=not fn.exists(), index=False)
        self._global_epoch += 1

    # ────────────────────────────────────────────────────────────────
    # helper – write out every file the visual notebook relies on
    # ────────────────────────────────────────────────────────────────
    def _dump_model_metadata(self):
        """
        Persists:
        • param_inventory.csv   – per-tensor shape / #params / bytes
        • run_meta.json         – run-time settings + totals
        """
        
        records, total_p, total_b = [], 0, 0
        sd = self.state_dict()
        for blk, obj in sd.items():
            if isinstance(obj, dict):           # sub-module dict
                for n, t in obj.items():
                    numel = t.numel(); bytes_ = numel * t.element_size()
                    records.append(dict(tensor=f"{blk}.{n}",
                                        shape=list(t.shape),
                                        numel=numel, bytes=bytes_))
                    total_p += numel; total_b += bytes_
            else:                               # flat tensor
                numel = obj.numel(); bytes_ = numel * obj.element_size()
                records.append(dict(tensor=blk, shape=list(obj.shape),
                                    numel=numel, bytes=bytes_))
                total_p += numel; total_b += bytes_

        pd.DataFrame(records).to_csv(self.model_dir / "param_inventory.csv",
                                    index=False)

        # some model params information
        some = dict(
            freq=self.freq, feature_dim=self.F, window_len=self.L,
            pred_len=self.pred_len, train_end=str(self.train_end),
            valid_end=str(self.valid_end), n_epochs=self.n_epochs,
            #dynamic_universe_mult=self.universe_mult, 
            total_params=total_p,
            approx_size_mb=round(total_b / 1024 / 1024, 3),
        )
        # adding hparams
        some.update(self.hp)

        # storing all params infos
        with open(self.model_dir / "all_params.json", "w") as fp:
            json.dump(some, fp, indent=2)

    # ---------------- training utilities ----------------------------- #
    
    # ------------ quick eval helper (MSE on loader) ------------------- #
    def _eval(self, loader: "DataLoader") -> float:
        self.stock_factor.eval()
        self.market_factor.eval()
        self.forecaster.eval()
        tot = 0.0
        with torch.no_grad():
            for p_seq, f_seq, tgt, active_mask in loader:
                p_seq, f_seq, tgt = p_seq.to(self._device), f_seq.to(self._device), tgt.to(self._device)
                u_seq, r_t, m_t, _, _  = self._stage1_forward(p_seq, f_seq, active_mask)
                _, loss, _ = self._stage2_forward(f_seq, u_seq, r_t, m_t, tgt, active_mask)
                tot += loss.item()
        return tot / len(loader)

    # ------------ shared training routine ---------------------------- #
    def _train( self, loader_train: "DataLoader", loader_valid: Optional["DataLoader"], n_epochs: int, ) -> float:
        """ 
        ONE entry-point that now handles all three schedules:

            • self.training_mode == "joint"      – unchanged behaviour
            • self.training_mode == "hybrid"     – k warm-up epochs (Stage-1 only)
                                                then joint fine-tune
            • self.training_mode == "sequential" – Stage-1 with early-stop,
                                                then Stage-2 only
        Returns the best validation MSE seen across *all* phases. 
        """

        # ───────────────────────────── optimisers ──────────────────────────
        opt_s = torch.optim.AdamW(self.stock_factor.parameters(),
                                lr=self.hp["lr_stage1"],
                                weight_decay=self.hp.get("weight_decay", 0))
        opt_m = torch.optim.AdamW(self.market_factor.parameters(),
                                lr=self.hp["lr_stage1"],
                                weight_decay=self.hp.get("weight_decay", 0))
        opt_f = torch.optim.AdamW(self.forecaster.parameters(),
                                lr=self.hp["lr_stage2"],
                                weight_decay=self.hp.get("weight_decay", 0))

        best_val: float = math.inf
        self._epoch_logs = []
        ep = 0                         # global epoch counter for logging

        # ────────────────────────── helper inner loop ─────────────────────
        def _run_epochs(num: int) -> None:
            nonlocal ep, best_val
            for _ in range(num):
                trn, l_s, l_m, l_pred = self._train_epoch(loader_train,
                                                        opt_s, opt_m, opt_f)

                val = float("nan")
                if loader_valid is not None:
                    val = self._eval(loader_valid)
                    best_val = min(best_val, val)

                if ep == 0 or (ep + 1) % max(1, (self.pretrain_epochs + n_epochs) // 5) == 0:
                    self.log.info(f"ep {ep:>3} | train {trn:.5f} | val {val:.5f} | best {best_val:.5f}")

                self._log_epoch(l_s, l_m, l_pred, val_pred=val)
                ep += 1

        # ─────────────────────────── schedule switch ──────────────────────
        if self.training_mode == "hybrid":
            # 1) warm-up – Stage-1 only
            self.log.info(f"[hybrid] warm-up Stage-1  ({self.pretrain_epochs} epochs)")
            self.forecaster.requires_grad_(False)
            _run_epochs(self.pretrain_epochs)

            # 2) joint fine-tune
            self.log.info(f"[hybrid] joint fine-tune  ({n_epochs} epochs)")
            self.forecaster.requires_grad_(True)
            for g in opt_s.param_groups + opt_m.param_groups:
                g["lr"] = self.hp.get("lr_stage1_ft", self.hp["lr_stage1"])
            _run_epochs(n_epochs)

        elif self.training_mode == "sequential":
            # ---------- Stage-1 with early-stopping ----------
            self.log.info("[sequential] Stage-1 until early-stop")
            self.forecaster.requires_grad_(False)

            best_state_s = best_state_m = None
            bad = 0
            for _ in range(n_epochs):
                _, l_s, l_m, l_pred = self._train_epoch(loader_train, opt_s, opt_m, opt_f)
                val = float("nan")
                if loader_valid is not None:
                    val = self._eval(loader_valid)
                    improve = val < best_val * 0.995
                    best_val = min(best_val, val)
                    bad = 0 if improve else bad + 1
                    if improve:
                        best_state_s = {k: v.cpu() for k, v in self.stock_factor.state_dict().items()}
                        best_state_m = {k: v.cpu() for k, v in self.market_factor.state_dict().items()}
                    if bad >= self.patience:
                        self.log.info(f"[sequential] early-stop after {bad} bad rounds")
                        break

                self._log_epoch(l_s, l_m, l_pred, val_pred=val)

            # restore best Stage-1 weights
            if best_state_s:
                self.stock_factor.load_state_dict(best_state_s)
                self.market_factor.load_state_dict(best_state_m)

            # ---------- Stage-2 only ----------
            self.log.info(f"[sequential] Stage-2 fine-tune  ({n_epochs} epochs)")
            self.stock_factor.requires_grad_(False)
            self.market_factor.requires_grad_(False)
            self.forecaster.requires_grad_(True)

            # (re-)initialise optimiser for the head only
            opt_f = torch.optim.AdamW(self.forecaster.parameters(),
                                    lr=self.hp["lr_stage2"],
                                    weight_decay=self.hp.get("weight_decay", 0))
            _run_epochs(n_epochs)

        else:   # "joint" (default behaviour)
            _run_epochs(n_epochs)

        return best_val
 
    def _train_epoch(self, loader, optim_s, optim_m, optim_f):
        self.stock_factor.train(); self.market_factor.train(); self.forecaster.train()
        sum_pred = sum_s = sum_m = 0.0
        n_batches = len(loader)

        for p_seq, f_seq, tgt, active_mask in loader:
            p_seq, f_seq, tgt = p_seq.to(self._device), f_seq.to(self._device), tgt.to(self._device)

            # ───────── Stage-1  (parallel fwd + bwd) ────────────────
            u_seq, r_t, m_t, loss_s, loss_m = self._stage1_forward(p_seq, f_seq, active_mask)

            if any(p.requires_grad for p in self.stock_factor.parameters()) \
            or any(p.requires_grad for p in self.market_factor.parameters()):

                optim_s.zero_grad(set_to_none=True)
                optim_m.zero_grad(set_to_none=True)

                if torch.cuda.is_available():
                    stream_s, stream_m = torch.cuda.Stream(), torch.cuda.Stream()
                    with torch.cuda.stream(stream_s):
                        loss_s.backward(retain_graph=True)
                    with torch.cuda.stream(stream_m):
                        loss_m.backward()
                else:                                  # CPU fallback
                    loss_s.backward(retain_graph=True)
                    loss_m.backward()

                optim_s.step(); optim_m.step()

            # ───────── Stage-2  (only if head is trainable) ─────────
            if any(p.requires_grad for p in self.forecaster.parameters()):
                preds, loss_pred, _ = self._stage2_forward(
                    f_seq, u_seq, r_t, m_t, tgt, active_mask
                )
                optim_f.zero_grad(set_to_none=True)
                loss_pred.backward()
                optim_f.step()
            else:
                preds = torch.zeros(tgt.size(0), tgt.size(1), device=tgt.device)
                loss_pred = torch.tensor(0.0, device=self._device)

            # ───────── bookkeeping ──────────────────────────────────
            sum_pred += loss_pred.item()
            sum_s    += loss_s.item()
            sum_m    += loss_m.item()

        # averages for logging
        return ( (sum_pred + sum_s + sum_m) / n_batches,
                sum_s   / n_batches,
                sum_m   / n_batches,
                sum_pred / n_batches)
    # ---------------- stage1 / stage2 wrappers ----------------------- #
    def _stage1_forward(self, prices_seq, feat_seq, active_mask):
        """Run stock-level & market-level factor learning *concurrently*."""

        assert active_mask.ndim == 2 and active_mask.shape == (self.batch_size, self.I), f"active_mask must be (B, I); got {list(active_mask.shape)}"
        stockIDs = self._eye.to(prices_seq.device)
        if torch.cuda.is_available():
            # ---- GPU: two independent CUDA streams ------------------- #
            stream_s = torch.cuda.Stream()
            stream_m = torch.cuda.Stream()
            with torch.cuda.stream(stream_s):
                u_seq, loss_s, _ = self.stock_factor(prices_seq, active_mask)  
            with torch.cuda.stream(stream_m):
                r_t, m_t, loss_m, _ = self.market_factor(feat_seq, stockIDs, active_mask)
            
        else:
            # ---- CPU fallback: simple Python threads ----------------- #
            def _run_stock(): 
                return self.stock_factor(prices_seq, active_mask)
            def _run_market(): 
                return self.market_factor(feat_seq, stockIDs, active_mask)
            with ThreadPoolExecutor(max_workers=2) as pool:
                fu_s = pool.submit(_run_stock)
                fu_m = pool.submit(_run_market)
                u_seq, loss_s, _ = fu_s.result()
                r_t, m_t, loss_m, _ = fu_m.result()

        # ---- return results ---------------------------------------- #
        return u_seq, r_t, m_t, loss_s, loss_m

    def _stage2_forward(self, feat_seq, u_seq, r_t, m_t, target, active_mask):
        assert active_mask.ndim == 2 and active_mask.shape == (self.batch_size, self.I), f"active_mask must be (B, I); got {list(active_mask.shape)}"
        stockIDs = self._eye.to(feat_seq.device)
        return self.forecaster(
            feat_seq, u_seq, r_t, m_t, stockIDs, target, active_mask
        )



# ------------------------------------------------------------------ #
#  Main guard: lets the file act as CLI for SLURM array jobs         #
# ------------------------------------------------------------------ #
# if __name__ == "__main__":

#     # use fixed random seeds for reproducibility
#     torch.manual_seed(0); np.random.seed(0); random.seed(0)

#     BASE_DIR        = Path(__file__).resolve().parent
#     DEFAULT_DATADIR = BASE_DIR / "data"
#     DEFAULT_MODELDR = BASE_DIR / "models" / "UMI"          #   umi/models/UMI

#     import argparse, json, pandas as pd, os
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data-dir",   default=DEFAULT_DATADIR, help=f"directory with OHLCV data in format (TICKER).csv")
#     ap.add_argument("--freq", default="1B", help="frequency of the data e.g. 1D, 15m, 1h. B means 1 business day")
#     ap.add_argument("--window-len",  type=int, default=60)
#     ap.add_argument("--pred-len",    type=int, default=1, help="horizon in *bars*, e.g. 4 → predict 4 steps ahead")
#     ap.add_argument("--dynamic-universe-mult", type=int, default=2, help="over-allocation factor for dynamic universe")   
#     ap.add_argument("--training-mode", choices=["hybrid","sequential"], default="sequential",
#                     help="hybrid: stage-1 pre-training, then joint training; sequential: stage-1 first, then stage-2")
#     ap.add_argument("--retrain-delta", type=int, default=30, help="delta bars (time units) for retraining, e.g. 30, 300 minutes or days")
#     ap.add_argument("--pretrain-epochs", type=int, default=5, help="Stage-1 epochs before joint training (hybrid)")
#     ap.add_argument("--patience",    type=int, default=10, help="Early stopping patience epochs for any training stage")
#     ap.add_argument("--start-train", default="2018-01-01", help="start date for back-test")
#     ap.add_argument("--end-train",   default="2018-02-28")
#     ap.add_argument("--end-valid",   default="2018-04-30")
#     ap.add_argument("--end-test",    default="2018-04-30", help="optional end date for test set")
#     ap.add_argument("--n-epochs",    type=int, default=1)
#     ap.add_argument("--batch-size",    type=int, default=128)
#     ap.add_argument("--n-trials", type=int, default=20)
#     ap.add_argument("--close-idx", type=int, default=3, help="index of the 'close' column in the dataframes (default: 3)")
#     ap.add_argument("--model-dir", type=str, default=DEFAULT_MODELDR, help="directory where the model is saved")
#     ap.add_argument("--bt-end",  default="2018-04-30", help="backtest end date, if not provided, uses the max date in the data")

    
#     args = ap.parse_args()

#     # ---------- load data -------------------------------------------
#     if args.data_dir is not None:
#         data_path = Path(args.data_dir)
#         if not data_path.is_dir():
#             raise FileNotFoundError(f"{data_path} is not a directory.")

#         data_dict: dict[str, pd.DataFrame] = {}
#         for csv_file in data_path.glob("*.csv"):
#             ticker = csv_file.stem.upper()          # file name → ticker

#             df = pd.read_csv(csv_file)

#             # -- make sure the index is a datetime index the rest of the code expects
#             if "Date" in df.columns:
#                 df["Date"] = pd.to_datetime(df["Date"], utc=True)
#                 df.set_index("Date", inplace=True)
#             else:
#                 raise ValueError(f"{csv_file} must contain a 'Date' column.")

#             # OPTIONAL – bring the classic OHLCV to the front so that
#             #           'Close' really is column #3 as the model defaults expect.
#             wanted = ["Open", "High", "Low", "Close"]
#             reorder = [c for c in wanted if c in df.columns] + [c for c in df.columns if c not in wanted]
#             df = df[reorder]
#             data_dict[ticker] = df
#     else:
#         raise ValueError("No data directory provided. Use --data-dir to specify the path to the data.")

    
#     clock = SimClock(pd.Timestamp(args.valid_end, tz="UTC"))

#     # ---------- build & fit -----------------------------------------
#     # Each SLURM array task sees exactly one GPU thanks to CUDA_VISIBLE_DEVICES
#     model = UMIModel(
#         freq=args.freq,
#         feature_dim=len(next(iter(data_dict.values())).columns),
#         window_len=args.window_len,
#         pred_len=args.pred_len,
#         train_end=args.train_end,
#         valid_end=args.valid_end,
#         n_epochs=args.n_epochs,     # then trains for n_epochs
#         batch_size=args.batch_size,
#         retrain_delta=pd.DateOffset(days=args.retrain_delta),    # to modify if wanted hours or minutes
#         dynamic_universe_mult=args.dynamic_universe_mult,    # 2x over-allocation for dynamic universe
#         training_mode=args.training_mode,
#         pretrain_epochs=args.pretrain_epochs,
#         patience=args.patience,
#         close_idx=( args.close_idx if args.close_idx is not None else list(next(iter(data_dict.values())).columns).index("Close")),
#         model_dir=args.model_dir,
#         data_dir= args.data_dir,
#         bt_end = args.bt_end,
#         clock_fn=clock,              # pass the clock closure
#     )

#     self.log.info(next(iter(data_dict.items()))[1].head())
#     self.log.info("Detected close_idx =", list(next(iter(data_dict.values())).columns).index("Close"))

#     try:
#         model.fit(data_dict, )  
#     except RuntimeError as e:
#         if "out of memory" in str(e).lower():
#             self.log.info("[warn] OOM – retrying with dynamic_universe_mult = 1")
#             raise
#         else:
#             raise
      

#     assert model._last_fit_time is not None   # sanity check

#         # -------------- WALK-FORWARD BACK-TEST --------------------------
#     bt_start = pd.Timestamp(args.start_train, tz="UTC")
#     bt_end   = (pd.Timestamp(args.end_test, tz="UTC")
#                 if args.bt_end else max(df.index.max() for df in data_dict.values()))

#     # initialise clock *on* bt_start − 1 bar
#     clock = SimClock(bt_start - pd.Timedelta(model.freq))
#     model._clock = clock   # hand the mutable clock to the already-fitted model

#     preds, targets = [], []
#     dates = pd.date_range(bt_start, bt_end, freq="B")  # business days only

#     pred_rows, truth_rows = [], []


#     for t in dates:
#         # advance the clock first
#         clock.tick(t)

#         # slice up-to-now (keep only last L+1 bars – faster)
#         now_dict = cut_off(data_dict, t, lookback=model.L + model.pred_len + 1)

#         # === prediction ===
#         y_hat = model.predict(now_dict)


#         step   = pd.tseries.frequencies.to_offset(model.freq) * model.pred_len
#         t_next = t + step                       # first bar the forecast refers to

#         # 1) close[t]  —— last price we *know* at time t
#         close_now = {k: df.loc[t, "Close"] for k, df in now_dict.items()}

#         # 2) close[t_next] —— realise it only if it already exists in the file
#         if all(t_next in df.index for df in data_dict.values()):
#             close_next = {k: df.loc[t_next, "Close"] for k, df in data_dict.items()}

#             # ----- SERIES we will store -----------------------------------
#             truth_close_ser = pd.Series(close_next, name=t)            # real future close
#             truth_ret_ser   = ((truth_close_ser - pd.Series(close_now)) /                 # realised return
#                             pd.Series(close_now)).rename(t)

#             # optional: convert model's return prediction into a
#             # predicted close so the notebook can plot both on the same axis
#             pred_close_ser = (1.0 + pd.Series(y_hat)) * pd.Series(close_now)
#             pred_close_ser.name = t

#             # ----- book-keeping -------------------------------------------
#             pred_rows.append(pred_close_ser)
#             truth_rows.append(truth_close_ser)

#         else:
#             # Horizon exceeds file – stop the back-test here
#             break

#         pred_ser = pd.Series(y_hat, name=t)  # pd.Series of predictions, index = tickers
#         preds.append(pred_ser)  # pd.Series of predictions, index = tickers)


#         # ground truth: next-bar return that the model tries to predict
#         if t + pd.tseries.frequencies.to_offset(model.freq) in now_dict[next(iter(now_dict))].index:
#             tgt_dict = cut_off(data_dict, t + pd.tseries.frequencies.to_offset(model.freq), lookback=1)
#             close_now  = {k: df.loc[t, "Close"]   for k, df in now_dict.items()}
#             close_next = {k: df.iloc[-1]["Close"] for k, df in tgt_dict.items()}
#             rtn = {k: (close_next[k] - close_now[k]) / close_now[k] for k in close_now}
#             targets.append(pd.Series(rtn, name=t))

#         # === optional retrain === (maintains correct chronology)
#         model.update(now_dict)

#     # concatenate & evaluate
#     pred_df  = pd.concat(pred_rows,  axis=1).T   # predicted closes
#     truth_df = pd.concat(truth_rows, axis=1).T   # realised closes

#     out_dir = Path(args.model_dir)
#     pred_df.to_csv(out_dir / "bt_pred_close.csv")
#     truth_df.to_csv(out_dir / "bt_truth_close.csv")

#     self.log.info("DONE")

