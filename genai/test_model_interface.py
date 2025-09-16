"""
Production-ready comprehensive test suite for MarketModel interface compliance.
Validates model implementations for compatibility with trading infrastructure.
"""

import gc
import json
import logging
import pickle
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import psutil
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from models.interfaces import MarketModel
from models.utils import freq2pdoffset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
@dataclass
class TestConfig:
    """Test suite configuration"""
    max_memory_mb: float = 1000.0
    prediction_timeout: float = 30.0
    training_timeout: float = 300.0
    tensor_tolerance: float = 1e-6
    prediction_bounds: Tuple[float, float] = (-10.0, 10.0)
    min_sharpe_ratio: float = -5.0
    max_sharpe_ratio: float = 5.0
    enable_gpu_tests: bool = torch.cuda.is_available()
    enable_stress_tests: bool = False
    save_failed_models: bool = True
    test_report_path: Path = Path("test_reports")


class TestMetrics:
    """Collect and report test metrics"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def add_result(self, test_name: str, passed: bool, duration: float, 
                   memory_mb: float = 0, details: Dict = None):
        self.results.append({
            "test": test_name,
            "passed": passed,
            "duration_ms": duration * 1000,
            "memory_mb": memory_mb,
            "details": details or {},
            "timestamp": time.time()
        })
    
    def generate_report(self, output_path: Path):
        """Generate test report"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "total_duration_s": time.time() - self.start_time,
            "avg_memory_mb": np.mean([r["memory_mb"] for r in self.results if r["memory_mb"] > 0]),
            "results": self.results
        }
        
        report_file = output_path / f"test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report


@contextmanager
def monitor_resources():
    """Context manager to monitor resource usage"""
    process = psutil.Process()
    gc.collect()
    
    mem_before = process.memory_info().rss / 1024 / 1024
    time_before = time.time()
    
    yield
    
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    time_after = time.time()
    
    return {
        "memory_delta_mb": mem_after - mem_before,
        "duration_s": time_after - time_before
    }


def validate_tensor_shapes(tensor: torch.Tensor, expected_shape: Tuple, 
                          name: str = "tensor") -> None:
    """Validate tensor shape with detailed error messages"""
    if tensor.shape != expected_shape:
        raise AssertionError(
            f"Shape mismatch for {name}: "
            f"expected {expected_shape}, got {tensor.shape}. "
            f"Tensor info: dtype={tensor.dtype}, device={tensor.device}, "
            f"has_nan={torch.isnan(tensor).any()}, has_inf={torch.isinf(tensor).any()}"
        )


class TestModelInterfaceProduction:
    """Production-ready test suite for MarketModel implementations"""
    
    @pytest.fixture(scope="class")
    def config(self):
        """Test configuration"""
        return TestConfig()
    
    @pytest.fixture(scope="class")
    def metrics(self):
        """Test metrics collector"""
        return TestMetrics()
    
    @pytest.fixture
    def sample_data_factory(self):
        """Factory for creating test data with different characteristics"""
        def _create_data(
            n_stocks: int = 3, 
            n_bars: int = 100,
            missing_pct: float = 0.0,
            volatility: float = 0.02,
            trend: float = 0.0001,
            start_date: str = '2021-01-01'
        ) -> Dict[str, pd.DataFrame]:
            
            dates = pd.date_range(start_date, periods=n_bars, freq='D', tz='UTC')
            tickers = [f"STOCK{i:03d}" for i in range(n_stocks)]
            data = {}
            
            for i, ticker in enumerate(tickers):
                # Generate realistic OHLCV data
                np.random.seed(42 + i)  # Reproducible
                returns = np.random.normal(trend, volatility, n_bars)
                close = 100 * np.exp(np.cumsum(returns))
                
                df = pd.DataFrame({
                    'Open': close * np.random.uniform(0.98, 1.02, n_bars),
                    'High': close * np.random.uniform(1.01, 1.05, n_bars),
                    'Low': close * np.random.uniform(0.95, 0.99, n_bars),
                    'Close': close,
                    'Volume': np.random.randint(1e6, 1e7, n_bars)
                }, index=dates)
                
                # Ensure OHLC consistency
                df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
                df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
                
                # Add missing data
                if missing_pct > 0:
                    mask = np.random.random(n_bars) < missing_pct
                    df.loc[mask, :] = np.nan
                    df.ffill(inplace=True)  # Forward fill
                
                data[ticker] = df
                
            return data
        
        return _create_data
    
    @pytest.fixture
    def model_params_factory(self, tmp_path):
        """Factory for model parameters with different configurations"""
        def _create_params(
            window_len: int = 60,
            pred_len: int = 1,
            n_epochs: int = 2,
            batch_size: int = 32
        ) -> Dict[str, Any]:
            return {
                'freq': '1D',
                'feature_dim': 5,  # OHLCV
                'window_len': window_len,
                'pred_len': pred_len,
                'train_offset': freq2pdoffset('3M'),
                'pred_offset': freq2pdoffset(f'{pred_len}D'),
                'train_end': pd.Timestamp('2021-03-01', tz='UTC'),
                'valid_end': pd.Timestamp('2021-04-01', tz='UTC'),
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'model_dir': tmp_path / f'test_model_{time.time()}',
                'calendar': 'NYSE',
                'patience': 3,
                'warm_start': True,
                'save_backups': False
            }
        return _create_params
    
    def test_model_initialization_comprehensive(
        self, model_class, model_params_factory, sample_data_factory, config, metrics
    ):
        """Comprehensive initialization test with multiple scenarios"""
        
        test_scenarios = [
            {"n_stocks": 3, "n_bars": 100, "name": "small"},
            {"n_stocks": 10, "n_bars": 500, "name": "medium"},
            {"n_stocks": 30, "n_bars": 1000, "name": "large", "skip_if_slow": True},
        ]
        
        for scenario in test_scenarios:
            if scenario.get("skip_if_slow") and not config.enable_stress_tests:
                continue
                
            with monitor_resources() as resources:
                # Create test data
                data = sample_data_factory(
                    n_stocks=scenario["n_stocks"],
                    n_bars=scenario["n_bars"]
                )
                params = model_params_factory()
                
                # Initialize model
                model = model_class(**params)
                
                # Validate pre-initialization state
                assert not model.is_initialized
                assert hasattr(model, 'L') and model.L == params['window_len']
                assert hasattr(model, 'pred_len') and model.pred_len == params['pred_len']
                
                # Initialize
                start_time = time.time()
                val_loss = model.initialize(data)
                init_time = time.time() - start_time
                
                # Validate post-initialization
                assert model.is_initialized
                assert isinstance(val_loss, (float, np.floating))
                assert 0 <= val_loss < 100, f"Unrealistic validation loss: {val_loss}"
                assert (model.model_dir / "init.pt").exists()
                
                # Log metrics
                metrics.add_result(
                    f"initialization_{scenario['name']}",
                    passed=True,
                    duration=init_time,
                    memory_mb=resources["memory_delta_mb"],
                    details={"n_stocks": scenario["n_stocks"], "val_loss": val_loss}
                )
                
                logger.info(f"✓ Initialization test passed for {scenario['name']} scenario")
    
    @pytest.mark.parametrize("missing_pct", [0.0, 0.1, 0.3])
    def test_missing_data_handling(
        self, model_class, model_params_factory, sample_data_factory, missing_pct
    ):
        """Test model robustness to missing data"""
        
        data = sample_data_factory(n_stocks=5, missing_pct=missing_pct)
        params = model_params_factory()
        model = model_class(**params)
        
        # Should handle missing data gracefully
        val_loss = model.initialize(data)
        assert not np.isnan(val_loss), f"Model failed with {missing_pct*100}% missing data"
        
        # Predictions should still work
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        predictions = model.predict(data, current_time, active_mask)
        
        assert len(predictions) == len(data)
        assert all(not np.isnan(v) for v in predictions.values())
    
    def test_prediction_consistency(
        self, model_class, model_params_factory, sample_data_factory, config
    ):
        """Test prediction consistency and determinism"""
        
        data = sample_data_factory(n_stocks=5)
        params = model_params_factory()
        
        # Train two identical models
        model1 = model_class(**params)
        torch.manual_seed(42)
        np.random.seed(42)
        val_loss1 = model1.initialize(data)
        
        params['model_dir'] = params['model_dir'].parent / 'test_model_2'
        model2 = model_class(**params)
        torch.manual_seed(42)
        np.random.seed(42)
        val_loss2 = model2.initialize(data)
        
        # Losses should be very close (allowing for numerical differences)
        assert abs(val_loss1 - val_loss2) < 0.01, "Training not deterministic with same seed"
        
        # Predictions should be identical
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        
        preds1 = model1.predict(data, current_time, active_mask)
        preds2 = model2.predict(data, current_time, active_mask)
        
        for ticker in preds1:
            assert abs(preds1[ticker] - preds2[ticker]) < config.tensor_tolerance, \
                f"Predictions not consistent for {ticker}"
    
    def test_prediction_performance(
        self, model_class, model_params_factory, sample_data_factory, 
        config, benchmark: BenchmarkFixture
    ):
        """Benchmark prediction performance"""
        
        data = sample_data_factory(n_stocks=50, n_bars=500)
        params = model_params_factory()
        model = model_class(**params)
        model.initialize(data)
        
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        
        # Benchmark prediction speed
        def predict():
            return model.predict(data, current_time, active_mask)
        
        result = benchmark.pedantic(predict, rounds=10, warmup_rounds=2)
        
        # Assert reasonable performance (< 1 second for 50 stocks)
        assert result.stats.mean < 1.0, f"Prediction too slow: {result.stats.mean:.2f}s"
    
    def test_active_mask_edge_cases(
        self, model_class, model_params_factory, sample_data_factory
    ):
        """Test edge cases for active mask handling"""
        
        data = sample_data_factory(n_stocks=5)
        params = model_params_factory()
        model = model_class(**params)
        model.initialize(data)
        
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        
        # Test all inactive
        all_inactive = torch.zeros(len(data), dtype=torch.bool)
        preds = model.predict(data, current_time, all_inactive)
        assert all(abs(v) < 1.0 for v in preds.values()), "Inactive stocks should have near-zero predictions"
        
        # Test single active
        single_active = torch.zeros(len(data), dtype=torch.bool)
        single_active[0] = True
        preds = model.predict(data, current_time, single_active)
        active_ticker = list(data.keys())[0]
        assert abs(preds[active_ticker]) > 0, "Active stock should have non-zero prediction"
        
        # Test alternating pattern
        alternating = torch.tensor([i % 2 == 0 for i in range(len(data))], dtype=torch.bool)
        preds = model.predict(data, current_time, alternating)
        for i, ticker in enumerate(data.keys()):
            if i % 2 == 0:
                assert abs(preds[ticker]) > 0, f"Active stock {ticker} should have prediction"
    
    def test_state_persistence_advanced(
        self, model_class, model_params_factory, sample_data_factory, tmp_path, config
    ):
        """Advanced state persistence testing including partial states"""
        
        data = sample_data_factory(n_stocks=5)
        params = model_params_factory()
        
        # Train original model
        model1 = model_class(**params)
        model1.initialize(data)
        
        # Get predictions
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        preds1 = model1.predict(data, current_time, active_mask)
        
        # Save state
        state = model1.state_dict()
        state_path = tmp_path / "model_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        # Verify state size is reasonable
        state_size_mb = state_path.stat().st_size / 1024 / 1024
        assert state_size_mb < 100, f"State too large: {state_size_mb:.1f}MB"
        
        # Load into new model
        params['model_dir'] = tmp_path / 'restored_model'
        model2 = model_class(**params)
        
        with open(state_path, 'rb') as f:
            loaded_state = pickle.load(f)
        
        model2.load_state_dict(loaded_state)
        
        # Verify identical predictions
        preds2 = model2.predict(data, current_time, active_mask)
        for ticker in preds1:
            assert abs(preds1[ticker] - preds2[ticker]) < config.tensor_tolerance
        
        # Test partial state loading (corrupted/incomplete state)
        if config.save_failed_models:
            partial_state = {k: v for k, v in state.items() if not k.startswith('_')}
            try:
                model3 = model_class(**params)
                model3.load_state_dict(partial_state)
                # Should handle gracefully or raise clear error
            except Exception as e:
                assert "state" in str(e).lower() or "missing" in str(e).lower()
    
    def test_concurrent_predictions(
        self, model_class, model_params_factory, sample_data_factory, config
    ):
        """Test thread safety for concurrent predictions"""
        
        data = sample_data_factory(n_stocks=10)
        params = model_params_factory()
        model = model_class(**params)
        model.initialize(data)
        
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        
        # Run predictions concurrently
        def predict_task(i):
            # Add slight time variation
            time_offset = pd.Timedelta(hours=i)
            return model.predict(data, current_time + time_offset, active_mask)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_task, i) for i in range(10)]
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=config.prediction_timeout)
                    results.append(result)
                except TimeoutError:
                    pytest.fail("Prediction timeout in concurrent execution")
        
        # All results should be valid
        assert len(results) == 10
        for result in results:
            assert len(result) == len(data)
            assert all(isinstance(v, (float, np.floating)) for v in result.values())
    
    def test_memory_leak_detection(
        self, model_class, model_params_factory, sample_data_factory, config
    ):
        """Detect memory leaks during repeated operations"""
        
        if not config.enable_stress_tests:
            pytest.skip("Stress tests disabled")
        
        data = sample_data_factory(n_stocks=20)
        params = model_params_factory()
        model = model_class(**params)
        model.initialize(data)
        
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        
        # Baseline memory
        gc.collect()
        process = psutil.Process()
        mem_baseline = process.memory_info().rss / 1024 / 1024
        
        # Run many predictions
        for i in range(100):
            _ = model.predict(data, current_time, active_mask)
            
            if i % 20 == 0:
                gc.collect()
                mem_current = process.memory_info().rss / 1024 / 1024
                mem_increase = mem_current - mem_baseline
                
                # Allow some increase but flag potential leak
                if mem_increase > config.max_memory_mb:
                    warnings.warn(
                        f"Potential memory leak: {mem_increase:.1f}MB increase after {i} predictions"
                    )
        
        # Final check
        gc.collect()
        mem_final = process.memory_info().rss / 1024 / 1024
        total_increase = mem_final - mem_baseline
        assert total_increase < config.max_memory_mb * 1.5, \
            f"Memory leak detected: {total_increase:.1f}MB increase"
    
    def test_mlflow_integration(
        self, model_class, model_params_factory, sample_data_factory, tmp_path
    ):
        """Test MLflow tracking integration"""
        
        # Setup MLflow
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        mlflow.set_experiment("test_experiment")
        
        with mlflow.start_run() as run:
            data = sample_data_factory(n_stocks=5)
            params = model_params_factory()
            model = model_class(**params)
            
            # Initialize and check MLflow logging
            val_loss = model.initialize(data)
            
            # Verify metrics were logged
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(run.info.run_id)
            metrics = run_data.data.metrics
            
            # Check for expected metrics
            expected_metrics = ['best_validation_loss', 'val_loss', 'train_loss']
            logged_metrics = list(metrics.keys())
            
            assert any(m in logged_metrics for m in expected_metrics), \
                f"No expected metrics logged. Found: {logged_metrics}"
            
            # Check for parameters
            params = run_data.data.params
            assert len(params) > 0, "No parameters logged to MLflow"
    
    def test_error_recovery(
        self, model_class, model_params_factory, sample_data_factory
    ):
        """Test model's ability to recover from errors"""
        
        params = model_params_factory()
        model = model_class(**params)
        
        # Test empty data
        with pytest.raises((ValueError, AssertionError)):
            model.initialize({})
        
        # Model should still be usable
        assert not model.is_initialized
        
        # Now initialize properly
        data = sample_data_factory(n_stocks=3)
        val_loss = model.initialize(data)
        assert model.is_initialized
        
        # Test prediction with wrong mask size
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        wrong_mask = torch.ones(10, dtype=torch.bool)  # Wrong size
        
        with pytest.raises((AssertionError, ValueError)):
            model.predict(data, current_time, wrong_mask)
        
        # Model should still work with correct mask
        correct_mask = torch.ones(len(data), dtype=torch.bool)
        preds = model.predict(data, current_time, correct_mask)
        assert len(preds) == len(data)
    
    def test_data_validation(
        self, model_class, model_params_factory, sample_data_factory
    ):
        """Test input data validation"""
        
        params = model_params_factory()
        model = model_class(**params)
        
        # Test with invalid data types
        invalid_data = {
            'INVALID': pd.DataFrame({
                'Open': ['not', 'a', 'number'],
                'Close': [1, 2, 3]
            })
        }
        
        with pytest.raises((ValueError, TypeError)):
            model.initialize(invalid_data)
        
        # Test with missing required columns
        missing_cols_data = {
            'TEST': pd.DataFrame({
                'Open': [1, 2, 3],
                'High': [2, 3, 4]
                # Missing Close, Low, Volume
            })
        }
        
        with pytest.raises((KeyError, ValueError)):
            model.initialize(missing_cols_data)
    
    @pytest.mark.parametrize("pred_len", [1, 5, 10, 20])
    def test_multi_step_prediction(
        self, model_class, model_params_factory, sample_data_factory, pred_len
    ):
        """Test multi-step prediction capabilities"""
        
        data = sample_data_factory(n_stocks=3)
        params = model_params_factory(pred_len=pred_len)
        model = model_class(**params)
        
        val_loss = model.initialize(data)
        assert model.pred_len == pred_len
        
        current_time = pd.Timestamp('2021-04-01', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        predictions = model.predict(data, current_time, active_mask)
        
        # For now, interface returns single value per ticker
        # Future: could return array of pred_len predictions
        assert all(isinstance(v, (float, np.floating)) for v in predictions.values())
    
    def test_update_workflow_advanced(
        self, model_class, model_params_factory, sample_data_factory
    ):
        """Advanced testing of model update/warm-start workflow"""
        
        data = sample_data_factory(n_stocks=5, n_bars=200)
        params = model_params_factory()
        model = model_class(**params)
        
        # Initial training
        val_loss_init = model.initialize(data)
        
        # Multiple update cycles
        update_times = pd.date_range('2021-04-01', periods=5, freq='W', tz='UTC')
        val_losses = [val_loss_init]
        
        for update_time in update_times:
            # Extend data slightly (simulate new data)
            for ticker in data:
                new_row = data[ticker].iloc[-1:].copy()
                new_row.index = [update_time]
                data[ticker] = pd.concat([data[ticker], new_row])
            
            active_mask = torch.ones(len(data), dtype=torch.bool)
            
            # Update model
            model.update(data, update_time, active_mask)
            
            # Get predictions
            preds = model.predict(data, update_time, active_mask)
            
            # Validate predictions
            assert len(preds) == len(data)
            assert all(-10 < v < 10 for v in preds.values())
        
        # Check that latest.pt exists after updates
        assert (model.model_dir / "latest.pt").exists()
    
    def test_hyperparameter_validation(
        self, model_class, model_params_factory, sample_data_factory
    ):
        """Test hyperparameter handling and validation"""
        
        data = sample_data_factory(n_stocks=3)
        params = model_params_factory()
        
        # Test with various hyperparameters
        test_hparams = [
            {'learning_rate': 0.001, 'dropout': 0.2},
            {'learning_rate': 0.01, 'hidden_dim': 256},
            {'batch_size': 16, 'num_layers': 3}
        ]
        
        for hparams in test_hparams:
            model = model_class(**params, **hparams)
            
            # Check if model stores hyperparameters
            if hasattr(model, 'hp') or hasattr(model, 'hparams'):
                hp_dict = getattr(model, 'hp', getattr(model, 'hparams', {}))
                for key, value in hparams.items():
                    if key in hp_dict:
                        assert hp_dict[key] == value
            
            # Model should still train
            val_loss = model.initialize(data)
            assert 0 <= val_loss < 100
    
    def test_prediction_distribution(
        self, model_class, model_params_factory, sample_data_factory, config
    ):
        """Test statistical properties of predictions"""
        
        data = sample_data_factory(n_stocks=50, n_bars=500)
        params = model_params_factory()
        model = model_class(**params)
        model.initialize(data)
        
        # Collect predictions over multiple time points
        predictions_list = []
        test_dates = pd.date_range('2021-04-01', periods=20, freq='D', tz='UTC')
        active_mask = torch.ones(len(data), dtype=torch.bool)
        
        for test_date in test_dates:
            preds = model.predict(data, test_date, active_mask)
            predictions_list.append(list(preds.values()))
        
        # Analyze prediction distribution
        all_preds = np.array(predictions_list).flatten()
        
        # Check bounds
        assert all(config.prediction_bounds[0] < p < config.prediction_bounds[1] for p in all_preds), \
            "Predictions outside reasonable bounds"
        
        # Check distribution properties
        mean_pred = np.mean(all_preds)
        std_pred = np.std(all_preds)
        
        # Predictions should be somewhat centered around 0 (for returns)
        assert abs(mean_pred) < 1.0, f"Prediction mean too high: {mean_pred:.4f}"
        
        # Should have reasonable variance (not all same value)
        assert 0.0001 < std_pred < 5.0, f"Prediction std unrealistic: {std_pred:.4f}"
        
        # Check for NaN or Inf
        assert not np.isnan(all_preds).any(), "NaN in predictions"
        assert not np.isinf(all_preds).any(), "Inf in predictions"
    
    def teardown_class(self, metrics, config):
        """Generate test report after all tests"""
        if hasattr(metrics, 'results') and metrics.results:
            report = metrics.generate_report(config.test_report_path)
            logger.info(f"Test report saved to {config.test_report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            print(f"Total tests: {report['total_tests']}")
            print(f"Passed: {report['passed']}")
            print(f"Failed: {report['failed']}")
            print(f"Total duration: {report['total_duration_s']:.2f}s")
            print(f"Avg memory: {report['avg_memory_mb']:.1f}MB")