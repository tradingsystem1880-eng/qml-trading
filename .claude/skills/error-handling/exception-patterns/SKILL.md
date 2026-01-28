# Exception Patterns Skill

Python exception handling best practices for trading systems.

## When to Use
- Creating custom exception classes
- Designing exception hierarchies
- Writing try/except blocks
- Handling trading-specific errors

## Trading Exception Hierarchy

```python
"""Custom exceptions for QML trading system."""

class QMLError(Exception):
    """Base exception for all QML trading errors."""
    pass

# Detection Errors
class DetectionError(QMLError):
    """Errors in pattern detection."""
    pass

class InsufficientDataError(DetectionError):
    """Not enough data for detection."""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(f"Need {required} bars, got {available}")

class InvalidPatternError(DetectionError):
    """Pattern fails validation criteria."""
    def __init__(self, reason: str, pattern_data: dict = None):
        self.reason = reason
        self.pattern_data = pattern_data
        super().__init__(f"Invalid pattern: {reason}")

# Execution Errors
class ExecutionError(QMLError):
    """Errors during trade execution."""
    pass

class OrderRejectedError(ExecutionError):
    """Order rejected by exchange."""
    def __init__(self, reason: str, order_id: str = None, symbol: str = None):
        self.reason = reason
        self.order_id = order_id
        self.symbol = symbol
        super().__init__(f"Order rejected ({symbol}): {reason}")

class InsufficientBalanceError(ExecutionError):
    """Not enough balance for trade."""
    def __init__(self, required: float, available: float, asset: str):
        self.required = required
        self.available = available
        self.asset = asset
        super().__init__(f"Need {required} {asset}, have {available}")

class PositionLimitError(ExecutionError):
    """Position limit exceeded."""
    pass

# Risk Errors
class RiskError(QMLError):
    """Risk management violations."""
    pass

class DailyLossLimitError(RiskError):
    """Daily loss limit exceeded."""
    def __init__(self, current_loss: float, limit: float):
        self.current_loss = current_loss
        self.limit = limit
        super().__init__(f"Daily loss {current_loss:.2%} exceeds limit {limit:.2%}")

class MaxDrawdownError(RiskError):
    """Maximum drawdown exceeded."""
    pass

class ConsecutiveLossError(RiskError):
    """Too many consecutive losses."""
    def __init__(self, count: int, threshold: int):
        self.count = count
        self.threshold = threshold
        super().__init__(f"{count} consecutive losses (threshold: {threshold})")

# Data Errors
class DataError(QMLError):
    """Data pipeline errors."""
    pass

class StaleDataError(DataError):
    """Data is too old."""
    def __init__(self, symbol: str, age_seconds: int, max_age: int):
        self.symbol = symbol
        self.age_seconds = age_seconds
        self.max_age = max_age
        super().__init__(f"{symbol} data is {age_seconds}s old (max: {max_age}s)")

class DataGapError(DataError):
    """Missing data in series."""
    pass
```

## Exception Handling Patterns

### Pattern 1: Specific to General

```python
# GOOD: Handle specific exceptions first
try:
    result = execute_trade(signal)
except InsufficientBalanceError as e:
    logger.warning(f"Insufficient balance: {e.asset}")
    reduce_position_size(e.available)
except OrderRejectedError as e:
    logger.error(f"Order rejected: {e.reason}")
    notify_operator(e)
except ExecutionError as e:
    logger.error(f"Execution failed: {e}")
    retry_later(signal)
except QMLError as e:
    logger.error(f"QML error: {e}")
```

### Pattern 2: Context Preservation

```python
# GOOD: Chain exceptions to preserve context
try:
    data = fetch_market_data(symbol)
except requests.RequestException as e:
    raise DataError(f"Failed to fetch {symbol}") from e

# Access original: error.__cause__
```

### Pattern 3: Cleanup with Finally

```python
def execute_with_cleanup(signal):
    position = None
    try:
        position = open_position(signal)
        monitor_position(position)
    except ExecutionError:
        if position:
            emergency_close(position)
        raise
    finally:
        release_margin_lock(signal.symbol)
```

### Pattern 4: Error Aggregation

```python
def validate_signals(signals: list) -> list:
    """Validate multiple signals, collecting all errors."""
    errors = []
    valid_signals = []

    for signal in signals:
        try:
            validated = validate_signal(signal)
            valid_signals.append(validated)
        except InvalidPatternError as e:
            errors.append((signal, e))

    if errors and not valid_signals:
        raise DetectionError(f"All {len(errors)} signals invalid")

    return valid_signals, errors
```

## Anti-Patterns to Avoid

```python
# BAD: Bare except
try:
    trade()
except:  # Catches SystemExit, KeyboardInterrupt
    pass

# BAD: Swallowing exceptions silently
try:
    trade()
except Exception:
    pass  # Lost error info

# BAD: Too broad exception
try:
    trade()
except Exception as e:
    print(e)  # No recovery, no context

# BAD: Using exceptions for control flow
try:
    value = dict[key]
except KeyError:
    value = default
# GOOD: Use dict.get(key, default)
```

## Error Recovery Strategies

```python
class ErrorRecovery:
    """Strategies for recovering from trading errors."""

    @staticmethod
    def retry_with_backoff(func, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                return func()
            except (DataError, ExecutionError) as e:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)

    @staticmethod
    def fallback_chain(primary, *fallbacks):
        """Try primary, then fallbacks in order."""
        for func in [primary] + list(fallbacks):
            try:
                return func()
            except QMLError:
                continue
        raise ExecutionError("All fallbacks failed")

    @staticmethod
    def graceful_degradation(signal, error):
        """Degrade to safer operation on error."""
        if isinstance(error, InsufficientBalanceError):
            signal.size *= 0.5
            return execute_trade(signal)
        elif isinstance(error, RiskError):
            return queue_for_manual_review(signal)
        raise error
```

## Usage in QML System

```python
# In src/execution/paper_trader_bybit.py
from src.exceptions import (
    ExecutionError,
    InsufficientBalanceError,
    DailyLossLimitError
)

class BybitPaperTrader:
    def execute_signal(self, signal):
        try:
            self._check_risk_limits()
            return self._place_order(signal)
        except DailyLossLimitError:
            self.state.paused = True
            self.logger.warning("Trading paused: daily loss limit")
            raise
        except InsufficientBalanceError as e:
            self._adjust_position_size(e.available)
            return self._place_order(signal)
```
