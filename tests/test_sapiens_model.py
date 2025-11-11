"""
Tests for SapiensModel base class and implementations.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from models.SapiensModel import SapiensModel


class MockModel(SapiensModel):
    """Minimal test implementation."""
    
    def _build_model(self):
        self.linear = torch.nn.Linear(self.F, 1).to(self._device)
    
    def _forward_train(self, batch):
        _, features, targets, mask = batch
        features = features.to(self._device)
        targets = targets.to(self._device)
        pred = self.linear(features[:, -1, :, :].mean(dim=1)).squeeze(-1)
        return torch.nn.functional.mse_loss(pred[mask], targets[mask])
    
    def _forward_predict(self, data, indexes, active_mask):
        return {ticker: 0.01 for ticker in data.keys()}


@pytest.fixture
def test_config():
    return {
        'freq': '1D',
        'train_start': '2024-01-01',
        'train_end': '2024-03-01',
        'train_offset': pd.DateOffset(months=2),
        'valid_start': '2024-03-01',
        'valid_end': '2024-04-01',
        'valid_split': 0.2,
        'feature_dim': 5,
        'window_len': 10,
        'pred_len': 1,
        'n_epochs': 2,
        'batch_size': 4,
        'patience': 2,
        'warm_start': True,
        'model_dir': 'tests/temp_model',
    }


@pytest.fixture
def mock_data():
    dates = pd.date_range('2024-01-01', '2024-04-01', freq='D')
    return {
        'AAPL': pd.DataFrame(torch.randn(len(dates), 5).numpy(), index=dates),
        'GOOGL': pd.DataFrame(torch.randn(len(dates), 5).numpy(), index=dates),
    }


def test_sapiens_model_init(test_config):
    """Test model initialization."""
    model = MockModel(**test_config)
    assert not model.is_initialized
    assert model.L == 10
    assert model.pred_len == 1


def test_sapiens_model_initialize(test_config, mock_data):
    """Test model training initialization."""
    model = MockModel(**test_config)
    loss = model.initialize(mock_data, total_bars=60)
    assert model.is_initialized
    assert loss < float('inf')
    assert (Path(test_config['model_dir']) / 'init.pt').exists()


def test_sapiens_model_predict(test_config, mock_data):
    """Test prediction interface."""
    model = MockModel(**test_config)
    model.initialize(mock_data, total_bars=60)
    
    mask = torch.ones(2, dtype=torch.bool)
    preds = model.predict(mock_data, indexes=10, active_mask=mask)
    
    assert isinstance(preds, dict)
    assert set(preds.keys()) == set(mock_data.keys())


def test_sapiens_model_update(test_config, mock_data):
    """Test warm-start update."""
    model = MockModel(**test_config)
    model.initialize(mock_data, total_bars=60)
    
    mask = torch.ones(2, dtype=torch.bool)
    model.update(mock_data, pd.Timestamp('2024-04-01'), 
                pd.Timestamp('2024-03-01'), mask, total_bars=60)
    
    assert (Path(test_config['model_dir']) / 'latest.pt').exists()


def test_sapiens_model_state_dict(test_config):
    """Test state persistence."""
    model = MockModel(**test_config)
    model.I = 2
    model._build_model()
    
    state = model.state_dict()
    assert 'linear.weight' in state
    
    model2 = MockModel(**test_config)
    model2.I = 2
    model2._build_model()
    model2.load_state_dict(state)