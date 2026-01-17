# Pattern Visualization Engine - Documentation Overview

**For DeepSeek AI Knowledge Transfer**

## What Was Built

A standalone Python module at `src/dashboard/components/pattern_viz.py` that generates professional interactive Plotly charts for QML (Quasimodo) trading patterns.

## Architecture (CRITICAL)

**Geometry Extraction**: The visualization **EXTRACTS** P1-P5 coordinates from `features_json` stored in the database. It does **NOT** infer pattern points from OHLCV data. This ensures visualization uses the **same validated geometry** that QMLStrategy detected.

The feature extractor (`src/ml/feature_extractor.py`) stores raw geometry via `_extract_raw_geometry()`:
- P1-P5 timestamps, prices, and indices
- Entry, Stop-Loss, Take-Profit levels
- Pattern type

## Core Function

```python
def add_pattern_to_figure(fig: go.Figure, pattern_record: dict, ohlcv_df: pd.DataFrame) -> go.Figure
```

Takes an existing Plotly figure with candlestick data and adds 5 visualization layers:

| Layer | Description |
|-------|-------------|
| **ML Confidence Shading** | Vertical rectangle behind pattern, color intensity maps to confidence (red→green) |
| **Pattern Outline** | Line connecting P1→P2→P3→P4→P5 points |
| **Critical Points** | Markers for P1 (circle), P3 (triangle), P5 (square) with labels |
| **Trade Levels** | Dash-dot horizontal lines for Entry (blue), Stop-Loss (red), Take-Profit (green) |
| **Professional Styling** | Dark theme, consistent fonts, interactive legend |

## Data Flow

```
QMLStrategy → PatternFeatureExtractor._extract_raw_geometry() → ml_pattern_registry.features_json
                                                                           ↓
pattern_viz._extract_geometry() ← EXTRACTS from features_json (NOT inferred from OHLCV)
```

## Point Mapping (P1-P5)

| Point | Role | Bullish Pattern | Bearish Pattern |
|-------|------|-----------------|-----------------|
| P1 | Left Shoulder | Swing High | Swing Low |
| P2 | CHoCH Point | Interpolated | Interpolated |
| P3 | Head (Extreme) | Lowest Low | Highest High |
| P4 | BOS Point | Interpolated | Interpolated |
| P5 | Right Shoulder/Entry | Higher Low | Lower High |

## Features JSON Structure

The `features_json` column must contain:
```json
{
  "p1_idx": 8500,
  "p1_timestamp": "2025-12-15T00:00:00",
  "p1_price": 105000.0,
  "p2_idx": 8510,
  "p2_timestamp": "2025-12-16T16:00:00",
  "p2_price": 101500.0,
  "p3_idx": 8520,
  "p3_timestamp": "2025-12-18T08:00:00",
  "p3_price": 98000.0,
  "p4_idx": 8530,
  "p4_timestamp": "2025-12-19T04:00:00",
  "p4_price": 100500.0,
  "p5_idx": 8540,
  "p5_timestamp": "2025-12-20T00:00:00",
  "p5_price": 103000.0,
  "entry_price": 103000.0,
  "stop_loss_price": 97000.0,
  "take_profit_price": 115000.0
}
```

## Usage

```bash
# Run standalone test (generates HTML chart)
python src/dashboard/components/pattern_viz.py

# Output: results/pattern_viz_test.html
```

## Files Modified

| File | Change |
|------|--------|
| `src/ml/feature_extractor.py` | Added `_extract_raw_geometry()` method to store P1-P5 coordinates |
| `src/dashboard/components/pattern_viz.py` | Fixed `_extract_geometry()` to EXTRACT (not infer) from features_json |

## Key Design Decisions

1. **No Inference**: `_extract_geometry()` does NOT use OHLCV to infer points - it extracts validated data
2. **Flat Structure**: P1-P5 stored as flat keys (`p1_timestamp`, `p1_price`, etc.) in features_json
3. **Graceful Failure**: If geometry not found, logs warning and skips visualization (doesn't infer)
4. **Backward Compatibility**: Test query prefers patterns WITH geometry but falls back to any pattern
