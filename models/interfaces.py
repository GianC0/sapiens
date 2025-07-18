"""
Shared typing protocol: any model which implements this interface can
be dropped into UMIStrategy without editing the strategy code.
"""
from __future__ import annotations
import pickle
from typing import Protocol, Dict, Any, Union, runtime_checkable
import pandas as pd
from nautilus_trader.cache.cache import Cache   # NT built-in cache ¹

DataDict = Dict[str, pd.DataFrame]
DataSource = Union[DataDict, Cache]             # accepted everywhere


@runtime_checkable
class MarketModel(Protocol):
    """
    The minimum contract – **no inheritance required**.
    A class *quacks* if it has the following public members.
    """

    # two public attributes readable by the caller
    L: int               # look-back window in bars (≥ 1)
    pred_len: int        # forecast horizon in bars
    is_initialized: bool # whether model has been initialized

    # -------- Mandatory lifecycle methods --------
    def fit(self, data: DataSource, *, n_epochs: int = 1, **kwargs) -> None:
        """(Re)train the model on fresher data."""
        ...

    def update(self, data: DataSource, **kwargs) -> None:
        """Light-weight maintenance (e.g. decide if re-training is needed)."""
        ...

    def predict(self, data: DataSource, **kwargs) -> Dict[str, float]:
        """
        Return next-bar return forecasts *per ticker*.
        Keys must be exactly the tickers present in `data`.
        """
        ...

    def initialize(self, data: DataSource, **kwargs) -> None:
        """
        Perform initial training on historical data.
        Called ONCE at strategy startup.
        
        Args:
            data: Historical training data
            **kwargs: Training parameters (epochs, batch_size, etc.)
        """
        ...

    # -------- State persistence (required) --------
    def save_state_dict(self) -> Dict[str, Any]:
        """Save model state for persistence."""
        ...
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load model state from dict."""
        ...

    # -------- Common helpers (provided by base) --------
    def _default_hparams(self) -> Dict[str, Any]:
        """
        Default hyperparameters for the model.
        Override in subclass to provide model-specific defaults.
        """
        return {}
    
    def _dump_model_metadata(self) -> None:
        """
        Save model metadata (params, size, etc.) for analysis.
        Can be overridden for model-specific metadata.
        """
        # Default implementation
        metadata = {
            "model_class": self.__class__.__name__,
            "L": self.L,
            "pred_len": self.pred_len,
            "model_dir": str(self.model_dir),
        }
        
        # Add state dict size estimation
        state = self.state_dict()
        size_bytes = len(pickle.dumps(state))
        metadata["size_mb"] = round(size_bytes / 1024 / 1024, 2)
        
        # Save metadata
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _train(self, train_data: Any, valid_data: Any, n_epochs: int) -> float:
        """
        Common training loop implementation.
        Models can override or use their own training logic.
        """
        raise NotImplementedError("Model must implement _train or override fit/initialize")
    
    def _build_panel(self, train_data: Any, valid_data: Any, n_epochs: int) -> float:
        """
        Convert input data type from Nautilus.Cache to
        whatever panel the model expects (e.g., {"StockTicker": Dataframe, ..., })
        """
        raise NotImplementedError("Model must implement _train or override fit/initialize")

    # -------- Optional advanced features --------
    def _build_submodules(self) -> None:
        """
        Optional: Build model architecture.
        Only needed for complex models with multiple components.
        """
        pass
    
