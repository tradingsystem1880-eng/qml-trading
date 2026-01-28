# API Error Handling Skill

Implement retry logic and graceful degradation for exchange API calls.

## When to Use
- Handling Bybit/CCXT API errors
- Implementing retry with exponential backoff
- Managing rate limits
- Building resilient API clients

## Retry Pattern

```python
import time
from functools import wraps
from typing import Type, Tuple, Callable
import ccxt

class APIError(Exception):
    """Base API error with retry metadata."""
    def __init__(self, message: str, retryable: bool = True, wait_seconds: int = 0):
        super().__init__(message)
        self.retryable = retryable
        self.wait_seconds = wait_seconds

def retry_api(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
        ccxt.RateLimitExceeded,
    )
):
    """Decorator for API calls with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt if exponential else 1)
                        # Rate limit: use exchange's retry-after if available
                        if isinstance(e, ccxt.RateLimitExceeded):
                            delay = max(delay, 60)  # Min 60s for rate limits
                        time.sleep(delay)
                except ccxt.AuthenticationError:
                    raise APIError("Invalid API credentials", retryable=False)
                except ccxt.InsufficientFunds:
                    raise APIError("Insufficient funds", retryable=False)
                except ccxt.InvalidOrder as e:
                    raise APIError(f"Invalid order: {e}", retryable=False)
            raise last_exception
        return wrapper
    return decorator
```

## Bybit-Specific Error Codes

```python
BYBIT_ERROR_CODES = {
    # Retryable
    10002: ("Request timeout", True, 5),
    10016: ("Server busy", True, 10),
    10018: ("Rate limit", True, 60),

    # Not retryable
    10001: ("Invalid request", False, 0),
    10003: ("Invalid API key", False, 0),
    10004: ("Invalid signature", False, 0),
    10005: ("Permission denied", False, 0),
    110001: ("Order not found", False, 0),
    110007: ("Insufficient balance", False, 0),
    110012: ("Order cancelled", False, 0),
    110017: ("Position not found", False, 0),
}

def handle_bybit_error(error_code: int, message: str) -> APIError:
    """Convert Bybit error code to APIError."""
    if error_code in BYBIT_ERROR_CODES:
        desc, retryable, wait = BYBIT_ERROR_CODES[error_code]
        return APIError(f"{desc}: {message}", retryable=retryable, wait_seconds=wait)
    return APIError(f"Unknown error {error_code}: {message}", retryable=False)
```

## Usage in BybitClient

```python
class BybitClient:
    @retry_api(max_retries=3, base_delay=2.0)
    def place_order(self, symbol: str, side: str, amount: float, price: float = None):
        try:
            return self.exchange.create_order(
                symbol=symbol,
                type="limit" if price else "market",
                side=side,
                amount=amount,
                price=price
            )
        except ccxt.ExchangeError as e:
            # Parse Bybit error code from response
            if hasattr(e, 'args') and len(e.args) > 0:
                # Example: "bybit {\"retCode\":110007,\"retMsg\":\"Insufficient\"}"
                import json, re
                match = re.search(r'\{.*\}', str(e.args[0]))
                if match:
                    data = json.loads(match.group())
                    raise handle_bybit_error(data.get('retCode', 0), data.get('retMsg', ''))
            raise
```

## Graceful Degradation

```python
class ResilientTrader:
    def __init__(self, primary: BybitClient, fallback_mode: str = "paper"):
        self.primary = primary
        self.fallback_mode = fallback_mode
        self.consecutive_failures = 0
        self.max_failures = 5

    def execute(self, signal):
        if self.consecutive_failures >= self.max_failures:
            self._log_degraded_mode(signal)
            return self._fallback_execute(signal)

        try:
            result = self.primary.place_order(**signal)
            self.consecutive_failures = 0
            return result
        except APIError as e:
            if not e.retryable:
                raise
            self.consecutive_failures += 1
            return self._fallback_execute(signal)

    def _fallback_execute(self, signal):
        """Log signal for manual execution or paper trading."""
        if self.fallback_mode == "paper":
            return {"status": "paper", "signal": signal}
        elif self.fallback_mode == "alert":
            self._send_alert(f"Manual execution needed: {signal}")
            return {"status": "alert_sent", "signal": signal}
```

## Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure: datetime = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if datetime.now() - self.last_failure > timedelta(seconds=self.reset_timeout):
                self.state = "half-open"
            else:
                raise APIError("Circuit breaker open", retryable=False)

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = datetime.now()
            if self.failures >= self.failure_threshold:
                self.state = "open"
            raise
```
