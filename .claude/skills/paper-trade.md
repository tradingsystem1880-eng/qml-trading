# Paper Trade

Start or manage paper trading session.

## Usage
```
/paper-trade [action]
```

## Actions
- `start` - Begin new paper trading session
- `status` - Check current positions and P&L
- `log [symbol] [direction] [entry] [sl] [tp]` - Log a trade
- `close [trade_id] [exit_price]` - Close a trade

## Examples
- `/paper-trade start` - Start new session
- `/paper-trade log BTCUSDT LONG 42000 41500 43000` - Log a long trade
- `/paper-trade status` - Show current session stats

## Instructions

When the user invokes this skill:

1. Paper trading uses Phase 7.9 baseline (fixed 1% position sizing)

2. For `start`:
   - Initialize session tracking
   - Remind user of the rules:
     - 1% risk per trade
     - Use detected SL/TP levels
     - Track all trades honestly

3. For `log`:
   - Record trade in session
   - Calculate position size based on 1% risk
   - Show R:R ratio

4. For `status`:
   - Show open trades
   - Calculate unrealized P&L
   - Show session statistics (wins, losses, total R)

5. For `close`:
   - Mark trade as closed
   - Calculate realized P&L in R
   - Update session stats

6. Reference the dashboard for visual tracking:
   ```
   streamlit run qml/dashboard/app_v2.py
   ```
