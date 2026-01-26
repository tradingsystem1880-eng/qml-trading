# Backtest

Run a backtest on the QML trading system.

## Usage
```
/backtest [symbol] [timeframe]
```

## Examples
- `/backtest` - Run default backtest (BTCUSDT 4h)
- `/backtest ETHUSDT` - Backtest ETH on 4h
- `/backtest SOLUSDT 1h` - Backtest SOL on 1h timeframe

## Instructions

When the user invokes this skill:

1. Parse arguments (default: BTCUSDT, 4h)
2. Run the backtest command:
   ```bash
   python -m cli.run_backtest --symbol {symbol} --timeframe {timeframe}
   ```
3. Summarize key results:
   - Total trades
   - Win rate
   - Profit factor
   - Sharpe ratio
   - Max drawdown
4. If results look unusual (PF > 3.0 or negative Sharpe), warn about potential issues
