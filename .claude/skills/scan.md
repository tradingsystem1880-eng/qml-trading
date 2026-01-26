# Scan

Scan for QML patterns across symbols.

## Usage
```
/scan [symbols] [timeframe]
```

## Examples
- `/scan` - Scan top 10 symbols on 4h
- `/scan BTCUSDT,ETHUSDT,SOLUSDT 1h` - Scan specific symbols on 1h

## Instructions

When the user invokes this skill:

1. Parse arguments:
   - symbols: Default top 10 by volume (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, DOTUSDT)
   - timeframe: Default "4h"

2. Run multi-symbol detection:
   ```bash
   python scripts/multi_symbol_detection.py --symbols {symbols} --timeframe {timeframe}
   ```

3. Report findings:
   - Symbols with active patterns
   - Pattern quality scores
   - Direction (BULLISH/BEARISH)
   - Entry, SL, TP levels
   - Risk:Reward ratio

4. Highlight any high-quality patterns (quality > 70%)
