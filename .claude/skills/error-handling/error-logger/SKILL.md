# Error Logger Skill

Implement structured JSON logging with correlation IDs for trading system debugging.

## When to Use
- Setting up logging for new modules
- Adding trade-specific context to logs
- Debugging production issues
- Correlating events across async operations

## Logging Structure

```python
import logging
import json
import uuid
from datetime import datetime
from typing import Any, Optional
from contextlib import contextmanager

class TradingLogger:
    """Structured JSON logger with trade context."""

    def __init__(self, name: str, log_file: str = "logs/trading.jsonl"):
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self.correlation_id: Optional[str] = None
        self._setup_handler()

    def _setup_handler(self):
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def _log(self, level: str, message: str, **context):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "correlation_id": self.correlation_id,
            **context
        }
        self.logger.log(getattr(logging, level), json.dumps(entry))

    @contextmanager
    def trade_context(self, symbol: str, order_id: str = None, phase: int = None):
        """Context manager for trade-specific logging."""
        self.correlation_id = str(uuid.uuid4())[:8]
        self._log("INFO", "Trade context started",
                  symbol=symbol, order_id=order_id, phase=phase)
        try:
            yield self.correlation_id
        except Exception as e:
            self._log("ERROR", f"Trade failed: {e}",
                      symbol=symbol, order_id=order_id, error_type=type(e).__name__)
            raise
        finally:
            self._log("INFO", "Trade context ended", symbol=symbol)
            self.correlation_id = None

    def info(self, msg: str, **ctx): self._log("INFO", msg, **ctx)
    def error(self, msg: str, **ctx): self._log("ERROR", msg, **ctx)
    def debug(self, msg: str, **ctx): self._log("DEBUG", msg, **ctx)
    def warning(self, msg: str, **ctx): self._log("WARNING", msg, **ctx)
```

## Usage Example

```python
logger = TradingLogger("bybit_trader")

with logger.trade_context(symbol="BTCUSDT", order_id="12345", phase=1):
    logger.info("Placing order", side="BUY", size=0.01)
    # ... trading logic ...
    logger.info("Order filled", fill_price=42000.0, slippage_bps=2.5)
```

## Log Output Format

```json
{"timestamp": "2026-01-28T10:30:00", "level": "INFO", "message": "Placing order", "correlation_id": "a1b2c3d4", "symbol": "BTCUSDT", "side": "BUY", "size": 0.01}
```

## Key Fields for Trading

| Field | Purpose |
|-------|---------|
| `correlation_id` | Links all events in a trade lifecycle |
| `symbol` | Trading pair (BTCUSDT, ETHUSDT) |
| `order_id` | Exchange order identifier |
| `phase` | Forward test phase (1, 2, 3) |
| `error_type` | Exception class name |
| `fill_price` | Actual execution price |
| `slippage_bps` | Slippage in basis points |

## Integration with QML System

Add to `src/execution/paper_trader_bybit.py`:
```python
from src.logging.trading_logger import TradingLogger

class BybitPaperTrader:
    def __init__(self):
        self.logger = TradingLogger("bybit_paper")
```
