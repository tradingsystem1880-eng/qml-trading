# Python Testing Skill

Pytest fixtures, mocking, and TDD patterns for trading systems.

## When to Use
- Writing unit tests for trading logic
- Mocking external APIs (Bybit, Binance)
- Testing trade simulation
- Test-driven development

## Pytest Fundamentals

### Project Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/
│   ├── test_detection.py
│   ├── test_simulation.py
│   └── test_risk.py
├── integration/
│   ├── test_backtest_pipeline.py
│   └── test_data_pipeline.py
└── fixtures/
    ├── sample_ohlcv.parquet
    └── sample_trades.json
```

### Basic Test Structure

```python
import pytest
import pandas as pd
from src.detection import get_detector

class TestPatternDetection:
    """Test pattern detection logic."""

    def test_detects_bullish_qml(self, sample_ohlcv):
        """Should detect bullish QML pattern."""
        detector = get_detector("atr")
        signals = detector.detect(sample_ohlcv)

        assert len(signals) >= 1
        assert signals[0].direction == "LONG"
        assert signals[0].pattern_type == "BULLISH"

    def test_no_detection_on_flat_market(self, flat_ohlcv):
        """Should not detect patterns in sideways market."""
        detector = get_detector("atr")
        signals = detector.detect(flat_ohlcv)

        assert len(signals) == 0

    def test_validity_score_range(self, sample_ohlcv):
        """Validity score should be 0-1."""
        detector = get_detector("atr")
        signals = detector.detect(sample_ohlcv)

        for signal in signals:
            assert 0 <= signal.validity_score <= 1
```

## Fixtures

### conftest.py

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data with known pattern."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="4h")

    # Create price series with embedded QML pattern
    base = 100
    prices = [base]
    for i in range(199):
        change = np.random.randn() * 0.5
        prices.append(prices[-1] * (1 + change/100))

    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "close": [p * (1 + np.random.randn()*0.001) for p in prices],
        "volume": np.random.uniform(1000, 5000, 200)
    }, index=dates)

    return df

@pytest.fixture
def flat_ohlcv():
    """Flat/sideways market data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="4h")
    base = 100

    return pd.DataFrame({
        "open": [base] * 100,
        "high": [base * 1.001] * 100,
        "low": [base * 0.999] * 100,
        "close": [base] * 100,
        "volume": [1000] * 100
    }, index=dates)

@pytest.fixture
def sample_trades():
    """Sample trade list for testing."""
    from dataclasses import dataclass

    @dataclass
    class Trade:
        entry_time: str
        exit_time: str
        direction: str
        entry_price: float
        exit_price: float
        pnl: float
        pnl_r: float

    return [
        Trade("2024-01-01", "2024-01-05", "LONG", 100, 110, 10, 2.0),
        Trade("2024-01-06", "2024-01-08", "SHORT", 110, 105, 5, 1.0),
        Trade("2024-01-09", "2024-01-12", "LONG", 105, 100, -5, -1.0),
    ]

@pytest.fixture
def mock_bybit_client(mocker):
    """Mock Bybit API client."""
    mock = mocker.Mock()
    mock.fetch_balance.return_value = {"USDT": {"free": 10000}}
    mock.create_order.return_value = {
        "id": "12345",
        "status": "filled",
        "filled": 0.01,
        "price": 42000
    }
    return mock
```

## Mocking External APIs

### Mock Exchange Client

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestBybitTrader:
    """Test Bybit paper trader."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock CCXT exchange."""
        exchange = Mock()
        exchange.fetch_ticker.return_value = {
            "symbol": "BTC/USDT",
            "last": 42000,
            "bid": 41990,
            "ask": 42010
        }
        exchange.fetch_balance.return_value = {
            "USDT": {"free": 10000, "used": 0, "total": 10000}
        }
        exchange.create_order.return_value = {
            "id": "order_123",
            "status": "open",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.01
        }
        return exchange

    def test_place_order(self, mock_exchange):
        """Test order placement."""
        from src.execution import BybitClient

        with patch.object(BybitClient, '_create_exchange', return_value=mock_exchange):
            client = BybitClient(testnet=True)
            order = client.place_order("BTC/USDT", "buy", 0.01)

            assert order["id"] == "order_123"
            mock_exchange.create_order.assert_called_once()

    def test_handles_network_error(self, mock_exchange):
        """Test retry on network error."""
        import ccxt

        mock_exchange.create_order.side_effect = [
            ccxt.NetworkError("Connection failed"),
            ccxt.NetworkError("Connection failed"),
            {"id": "order_123", "status": "filled"}
        ]

        from src.execution import BybitClient

        with patch.object(BybitClient, '_create_exchange', return_value=mock_exchange):
            client = BybitClient(testnet=True)
            order = client.place_order("BTC/USDT", "buy", 0.01)

            assert order["status"] == "filled"
            assert mock_exchange.create_order.call_count == 3
```

### Mock Data Fetcher

```python
@pytest.fixture
def mock_data_fetcher(mocker, sample_ohlcv):
    """Mock data fetching."""
    mocker.patch(
        "src.data_engine.fetch_ohlcv",
        return_value=sample_ohlcv
    )
    return sample_ohlcv

def test_backtest_with_mocked_data(mock_data_fetcher):
    """Run backtest with mocked data."""
    from cli.run_backtest import run_backtest

    results = run_backtest(symbol="BTCUSDT", timeframe="4h")

    assert results is not None
    assert "trades" in results
```

## Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("symbol,expected_min_trades", [
    ("BTCUSDT", 10),
    ("ETHUSDT", 8),
    ("SOLUSDT", 5),
])
def test_detection_across_symbols(symbol, expected_min_trades, sample_ohlcv):
    """Test detection works for multiple symbols."""
    detector = get_detector("atr")
    signals = detector.detect(sample_ohlcv)

    assert len(signals) >= expected_min_trades

@pytest.mark.parametrize("sl_mult,tp_mult,expected_rr", [
    (1.0, 3.0, 3.0),
    (1.5, 4.5, 3.0),
    (2.0, 4.0, 2.0),
])
def test_risk_reward_calculation(sl_mult, tp_mult, expected_rr):
    """Test R:R is calculated correctly."""
    from src.risk import calculate_rr

    rr = calculate_rr(sl_atr_mult=sl_mult, tp_atr_mult=tp_mult)
    assert rr == expected_rr
```

## Test Markers

```python
# pytest.ini or pyproject.toml
# [pytest]
# markers =
#     slow: marks tests as slow (deselect with '-m "not slow"')
#     integration: marks tests as integration tests
#     api: marks tests that hit real APIs

@pytest.mark.slow
def test_full_backtest():
    """Full backtest - slow test."""
    pass

@pytest.mark.integration
def test_data_pipeline():
    """Integration test for data pipeline."""
    pass

@pytest.mark.api
@pytest.mark.skipif(not os.getenv("BYBIT_API_KEY"), reason="No API key")
def test_real_api():
    """Test with real Bybit API."""
    pass
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_detection.py

# Run tests matching pattern
pytest -k "test_bullish"

# Run excluding slow tests
pytest -m "not slow"

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run in parallel
pytest -n auto
```

## Trade Simulation Testing

```python
class TestTradeSimulation:
    """Test trade simulation accuracy."""

    def test_sl_hit_correctly(self, sample_ohlcv):
        """SL should trigger when price hits stop."""
        from src.optimization.trade_simulator import simulate_trade

        signal = Mock(
            entry_price=100,
            stop_loss=95,
            take_profit=110,
            direction="LONG"
        )

        # Create data where SL is hit
        df = sample_ohlcv.copy()
        df.loc[df.index[10], "low"] = 94  # Below SL

        result = simulate_trade(signal, df)

        assert result.exit_reason == "SL"
        assert result.pnl < 0

    def test_tp_hit_correctly(self, sample_ohlcv):
        """TP should trigger when price hits target."""
        from src.optimization.trade_simulator import simulate_trade

        signal = Mock(
            entry_price=100,
            stop_loss=95,
            take_profit=110,
            direction="LONG"
        )

        # Create data where TP is hit
        df = sample_ohlcv.copy()
        df.loc[df.index[15], "high"] = 111  # Above TP

        result = simulate_trade(signal, df)

        assert result.exit_reason == "TP"
        assert result.pnl > 0
```
