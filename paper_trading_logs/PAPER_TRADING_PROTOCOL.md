# QML Paper Trading Protocol

**Start Date:** December 29, 2025  
**End Date:** January 26, 2026  
**Phase:** 4-Week Validation  
**Status:** ACTIVE

---

## Operational Parameters (LOCKED - DO NOT MODIFY)

| Parameter | Value |
|-----------|-------|
| Symbols | BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT |
| Timeframe | 1H |
| Filter | HIGH VOLATILITY (vol_pctl > 0.7) |
| Scan Frequency | Daily minimum |
| Risk per Trade | 2% (theoretical) |

---

## Success Criteria

| Metric | Threshold | Measurement Method |
|--------|-----------|-------------------|
| Operational Reliability | >95% | Scans without errors / Total scans |
| Signal Logging | 100% | All fields populated |
| Filter Accuracy | Correct | No high-vol periods missed |
| Price Realism | Verified | Entry prices vs actual market |

---

## Weekly Report Schedule

| Week | Start | End | Report Due |
|------|-------|-----|------------|
| Week 1 | Dec 29, 2025 | Jan 4, 2026 | Jan 5, 2026 |
| Week 2 | Jan 5, 2026 | Jan 11, 2026 | Jan 12, 2026 |
| Week 3 | Jan 12, 2026 | Jan 18, 2026 | Jan 19, 2026 |
| Week 4 | Jan 19, 2026 | Jan 25, 2026 | Jan 26, 2026 |

---

## Daily Scan Command

```bash
cd /Users/hunternovotny/Desktop/QML_SYSTEM
python -m src.trading.paper_trader --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT --filter on --scan
```

---

## Signal Log Format

Each signal must include:
- Timestamp
- Signal ID
- Symbol
- Pattern Type (BULLISH/BEARISH)
- Detection Context (head, shoulders)
- Trading Levels (entry, stop, target)
- Market Context (vol_pctl, ADX, RSI)
- Filter Decision (PASS/FAIL + reason)

---

## Escalation Protocol

- **Software Error:** Document and investigate immediately
- **Missed Scan:** Note in daily log, run catch-up scan
- **Anomalous Signal:** Flag for manual review, do not modify system

---

*Protocol established: December 29, 2025*

