# Chart

Generate a chart visualization for a pattern or trade.

## Usage
```
/chart [symbol] [timeframe] [pattern_id]
```

## Examples
- `/chart BTCUSDT 4h` - Show latest detected pattern
- `/chart ETHUSDT 1h 123` - Show specific pattern by ID

## Instructions

When the user invokes this skill:

1. Parse arguments:
   - symbol: Required
   - timeframe: Default "4h"
   - pattern_id: Optional, uses latest if not provided

2. Load pattern data from detection results

3. Generate chart HTML using the pattern visualization spec:
   - Blue numbered swing points (P1-P5)
   - Orange prior trend line
   - Green/Red position box (TP/SL zones)
   - Entry, SL, TP price lines

4. Save chart to `results/charts/{symbol}_{timeframe}_{pattern_id}.html`

5. Open in browser:
   ```bash
   open results/charts/{filename}.html
   ```

6. Alternatively, direct user to dashboard Pattern Lab for interactive viewing
