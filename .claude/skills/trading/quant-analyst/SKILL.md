# Quant Analyst Skill

Risk metrics, portfolio analysis, and position sizing for trading systems.

## When to Use
- Calculating risk-adjusted returns
- Implementing Kelly criterion sizing
- Analyzing drawdown and risk of ruin
- Evaluating strategy performance

## Core Risk Metrics

### Sharpe Ratio

```python
import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252):
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Daily or periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily, 365*24 for hourly)
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

# For crypto (24/7 trading)
def crypto_sharpe(returns: pd.Series, timeframe: str = "4h"):
    periods = {"1h": 365*24, "4h": 365*6, "1d": 365}
    return sharpe_ratio(returns, periods_per_year=periods[timeframe])
```

### Sortino Ratio

```python
def sortino_ratio(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252):
    """Sharpe variant using only downside volatility."""
    excess = returns - target_return
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess.mean() > 0 else 0.0

    return np.sqrt(periods_per_year) * excess.mean() / downside.std()
```

### Maximum Drawdown

```python
def max_drawdown(equity_curve: pd.Series):
    """Calculate maximum drawdown and duration."""
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax

    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find peak before max DD
    peak_idx = equity_curve[:max_dd_idx].idxmax()

    # Find recovery (if any)
    recovery_idx = None
    post_dd = equity_curve[max_dd_idx:]
    recovered = post_dd[post_dd >= equity_curve[peak_idx]]
    if len(recovered) > 0:
        recovery_idx = recovered.index[0]

    return {
        "max_dd": max_dd,
        "max_dd_pct": abs(max_dd) * 100,
        "peak_date": peak_idx,
        "trough_date": max_dd_idx,
        "recovery_date": recovery_idx,
        "underwater_days": (recovery_idx - peak_idx).days if recovery_idx else None
    }
```

### Calmar Ratio

```python
def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252):
    """Annual return / Max drawdown."""
    returns = equity_curve.pct_change().dropna()
    annual_return = (1 + returns.mean()) ** periods_per_year - 1
    mdd = abs(max_drawdown(equity_curve)["max_dd"])

    if mdd == 0:
        return np.inf if annual_return > 0 else 0.0
    return annual_return / mdd
```

## Kelly Criterion

```python
class KellyCalculator:
    """Kelly criterion for position sizing."""

    def __init__(self, kelly_fraction: float = 0.5):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.5 = half Kelly)
        """
        self.kelly_fraction = kelly_fraction

    def optimal_f(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal fraction of capital to risk.

        Kelly f* = (p * b - q) / b
        where:
            p = win probability
            q = loss probability (1 - p)
            b = win/loss ratio
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # Odds ratio
        p = win_rate
        q = 1 - p

        kelly_f = (p * b - q) / b

        # Never risk more than 100%, never negative
        kelly_f = max(0, min(1, kelly_f))

        return kelly_f * self.kelly_fraction

    def position_size(
        self,
        equity: float,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        atr: float,
        sl_atr_mult: float = 1.5
    ) -> dict:
        """
        Calculate position size in units.

        Returns:
            risk_pct: Percentage of equity to risk
            position_value: Dollar value of position
            units: Number of units to trade
        """
        optimal_risk = self.optimal_f(win_rate, avg_win_r, avg_loss_r)

        # Cap at reasonable maximum (3% of equity)
        risk_pct = min(optimal_risk, 0.03)

        risk_amount = equity * risk_pct
        sl_distance = atr * sl_atr_mult

        # Position size = risk amount / SL distance (per unit)
        position_value = risk_amount / (sl_distance / atr)  # Simplified

        return {
            "risk_pct": risk_pct,
            "risk_amount": risk_amount,
            "position_value": position_value,
            "kelly_raw": optimal_risk / self.kelly_fraction,
            "kelly_used": optimal_risk
        }
```

## Risk of Ruin

```python
def risk_of_ruin(win_rate: float, avg_win_r: float, avg_loss_r: float, risk_per_trade: float, ruin_threshold: float = 0.5):
    """
    Estimate probability of losing X% of capital.

    Uses simplified formula: RoR = ((1-edge)/(1+edge))^units
    where edge = (win_rate * avg_win_r) - ((1-win_rate) * avg_loss_r)
    """
    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    if expectancy <= 0:
        return 1.0  # Negative edge = certain ruin

    # Units to ruin
    units_to_ruin = ruin_threshold / risk_per_trade

    # Simplified RoR calculation
    edge = expectancy / avg_loss_r
    if edge >= 1:
        return 0.0  # Very positive edge

    ror = ((1 - edge) / (1 + edge)) ** units_to_ruin
    return min(1.0, max(0.0, ror))

# Example
ror = risk_of_ruin(
    win_rate=0.55,
    avg_win_r=3.8,
    avg_loss_r=1.0,
    risk_per_trade=0.01,  # 1% risk
    ruin_threshold=0.50   # 50% drawdown = ruin
)
print(f"Risk of Ruin: {ror:.4%}")
```

## Monte Carlo Analysis

```python
import numpy as np

def monte_carlo_equity(trades: list, n_simulations: int = 10000, starting_equity: float = 100000):
    """
    Monte Carlo simulation of equity curves.

    Args:
        trades: List of trade P&L values
        n_simulations: Number of paths to simulate
        starting_equity: Starting capital
    """
    trade_returns = np.array([t.pnl_pct for t in trades])
    n_trades = len(trade_returns)

    final_equities = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled = np.random.choice(trade_returns, size=n_trades, replace=True)

        # Build equity curve
        equity = starting_equity
        equity_curve = [equity]
        peak = equity

        for ret in shuffled:
            equity *= (1 + ret)
            equity_curve.append(equity)
            peak = max(peak, equity)

        final_equities.append(equity)
        max_dd = min((e - peak) / peak for e, peak in
                     zip(equity_curve, np.maximum.accumulate(equity_curve)))
        max_drawdowns.append(abs(max_dd))

    return {
        "median_final": np.median(final_equities),
        "p5_final": np.percentile(final_equities, 5),
        "p95_final": np.percentile(final_equities, 95),
        "median_dd": np.median(max_drawdowns),
        "p95_dd": np.percentile(max_drawdowns, 95),  # 95% worst case DD
        "prob_profit": np.mean(np.array(final_equities) > starting_equity)
    }
```

## Portfolio Metrics

```python
def analyze_strategy(trades: list, equity_curve: pd.Series):
    """Comprehensive strategy analysis."""
    returns = equity_curve.pct_change().dropna()

    # Basic stats
    total_trades = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl < 0]

    win_rate = len(winners) / total_trades if total_trades > 0 else 0
    avg_win = np.mean([t.pnl_r for t in winners]) if winners else 0
    avg_loss = np.mean([abs(t.pnl_r) for t in losers]) if losers else 0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Risk metrics
    dd_info = max_drawdown(equity_curve)

    return {
        # Trade stats
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win_r": avg_win,
        "avg_loss_r": avg_loss,
        "profit_factor": profit_factor,
        "expectancy_r": (win_rate * avg_win) - ((1 - win_rate) * avg_loss),

        # Risk metrics
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(equity_curve),
        "max_drawdown_pct": dd_info["max_dd_pct"],

        # Advanced
        "recovery_factor": (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) / dd_info["max_dd_pct"],
        "profit_per_dd": gross_profit / dd_info["max_dd_pct"] if dd_info["max_dd_pct"] > 0 else np.inf
    }
```

## Integration with QML System

```python
# In src/ml/kelly_sizer.py (existing)
from src.quant.metrics import KellyCalculator

# Usage in paper trader
kelly = KellyCalculator(kelly_fraction=0.5)  # Half-Kelly for safety

def size_position(signal, account_equity, historical_stats):
    return kelly.position_size(
        equity=account_equity,
        win_rate=historical_stats["win_rate"],
        avg_win_r=historical_stats["avg_win_r"],
        avg_loss_r=historical_stats["avg_loss_r"],
        atr=signal.atr,
        sl_atr_mult=signal.sl_atr_mult
    )
```
