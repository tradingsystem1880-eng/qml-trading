# COMPLETE TRADING PLATFORM OVERHAUL SPECIFICATION
## Version 1.0 | Priority: CRITICAL | Timeline: Full Rebuild

---

# TABLE OF CONTENTS
1. [Executive Summary](#1-executive-summary)
2. [Reference Standards](#2-reference-standards)
3. [Research Requirements](#3-research-requirements)
4. [Chart Visualization System](#4-chart-visualization-system)
5. [Position Box Specifications](#5-position-box-specifications)
6. [Pattern Annotation System](#6-pattern-annotation-system)
7. [Design System](#7-design-system)
8. [Component Library](#8-component-library)
9. [Dashboard Layout](#9-dashboard-layout)
10. [Implementation Phases](#10-implementation-phases)
11. [Quality Standards](#11-quality-standards)
12. [Execution Protocol](#12-execution-protocol)

---

# 1. EXECUTIVE SUMMARY

## Current State
- Dashboard is broken, ugly, and unusable
- Chart visualization does not work
- Pattern annotations are non-functional
- Position boxes do not render
- No visual cohesion or professional appearance
- Configuration and layout are horrible to use

## Target State
- Professional-grade trading interface rivaling TradingView, Binance Pro, and institutional platforms
- Fully functional chart visualization with pattern annotations
- Clean, precise position boxes showing entry/TP/SL zones
- Cohesive design system with consistent styling
- Intuitive, efficient user experience

## Success Criteria
The rebuild is successful when:
1. Charts display candlesticks with proper density (100-150 bars visible)
2. Pattern annotations render with numbered swing points and dashed connection lines
3. Position boxes show clear TP/SL zones matching reference images
4. All UI components follow the design system
5. The interface looks and feels like a professional trading platform

---

# 2. REFERENCE STANDARDS

## 2.1 Primary Visual References

The user has provided THREE reference images that define EXACTLY how charts should look:

### Reference Image 1: QML Pattern with Position Box
**What it shows:**
- Numbered swing points: 1, 2, 3, 4, 5
- Blue dashed lines connecting the swing points
- Green semi-transparent box for TP1 zone
- Extended lighter green box for TP2 zone
- Red/pink semi-transparent box for SL zone
- Labels: "TP 2", "TP 1", "SL" positioned on the boxes
- Clean gray chart background
- Approximately 100+ candles visible (condensed view)
- Professional, clean appearance

### Reference Image 2: QM-ZONE Pattern
**What it shows:**
- Structure labels: LH (Lower High), LL (Lower Low), HH (Higher High)
- "QM-ZONE" text label at the demand zone
- Green position box extending to chart edge
- Red stop loss box below entry
- Minimal, clean annotations
- Professional TradingView-style appearance

---

# 3. RESEARCH REQUIREMENTS

## 3.1 Technology Stack Decision

**Primary charting library:** lightweight-charts (TradingView's open-source solution)
- Provides exact visual style needed
- Well-documented API
- Active maintenance

## 3.2 Research Output

Create `/docs/research/` with:
- design-analysis.md - Visual patterns found
- code-patterns.md - Architecture patterns to adopt
- implementation-notes.md - Technical approaches to use

---

# 4. CHART VISUALIZATION SYSTEM

## 4.1 Technology Stack

**Required dependencies:**
- lightweight-charts
- React (existing)
- Streamlit (existing Python backend)

## 4.2 Chart Display Specifications

**Candle Density:**
- Load 500+ bars of historical data
- Display 100-150 bars in default view (condensed like references)
- Allow zoom range: 50 bars (detailed) to 300 bars (overview)

**Visual Style:**
- Background: #0d0d12
- Grid: rgba(255, 255, 255, 0.03) - Very subtle
- Bullish candles: #26a69a
- Bearish candles: #ef5350

---

# 5. POSITION BOX SPECIFICATIONS

**THIS IS THE MOST CRITICAL VISUAL ELEMENT**

## 5.1 Long Position Box

```
    Price
      |
 TP2 -+----+---------------------------------------------+
      |    |           TP2 ZONE                          | <- "TP 2" label
      |    |     rgba(38, 166, 91, 0.15)                 |
 TP1 -+----+---------------------------------------------+
      |    |           TP1 ZONE                          | <- "TP 1" label
      |    |     rgba(38, 166, 91, 0.25)                 |
Entry-+----+=============================================+ <- Entry line (blue)
      |    |         STOP LOSS ZONE                      | <- "SL" label
      |    |     rgba(231, 76, 60, 0.25)                 |
  SL -+----+---------------------------------------------+
      |
      +---------------------------------------------------> Time
           |                                            |
        Entry                                    Chart Edge
        Time                                (extend to edge)
```

## 5.2 Colors (EXACT - DO NOT CHANGE)

- TP_PRIMARY: rgba(38, 166, 91, 0.25) - TP1
- TP_SECONDARY: rgba(38, 166, 91, 0.15) - TP2, TP3
- SL_FILL: rgba(231, 76, 60, 0.25)
- ENTRY_LINE: #3498db (blue)

---

# 6. PATTERN ANNOTATION SYSTEM

## 6.1 Swing Point Markers

- Numbers positioned ABOVE swing highs, BELOW swing lows
- Small offset from the actual high/low (5-8 pixels)
- Font: 11px, semi-bold, monospace
- Color: #3498db (blue) or white

## 6.2 Connection Lines

- Stroke: #3498db
- Stroke width: 1.5
- Stroke dasharray: 6, 4 (6px dash, 4px gap)
- Opacity: 0.8

---

# 7. DESIGN SYSTEM

## 7.1 Color Palette

### Background Colors
- bg.primary: #0a0a0f (Main app background)
- bg.secondary: #12121a (Card/panel backgrounds)
- bg.tertiary: #1a1a24 (Hover states)
- bg.elevated: #22222e (Modals, dropdowns)

### Chart Colors
- chart.background: #0d0d12
- chart.bullish: #26a69a
- chart.bearish: #ef5350

### Text Colors
- text.primary: #ffffff
- text.secondary: #a0a0a0
- text.profit: #26a69a
- text.loss: #ef5350

## 7.2 Typography

- Primary font: Inter, sans-serif
- Monospace (for numbers): JetBrains Mono, monospace
- ALL numerical data must use monospace font

---

# 8. IMPLEMENTATION PHASES

## Phase 1: Foundation
- Set up design system
- Create base styles
- Configure dark theme

## Phase 2: Chart Foundation
- Integrate lightweight-charts
- Implement candlestick rendering
- Achieve proper visual density

## Phase 3: Annotation System
- Swing point markers
- Connection lines
- Zone overlays

## Phase 4: Position Boxes (CRITICAL)
- TP/SL zone rendering
- Entry line
- Labels

## Phase 5: Dashboard Layout
- Navigation
- Panel layout
- Responsive design

---

# 9. QUALITY STANDARDS

## Visual Quality Checklist

### Chart
- [ ] Candlesticks render with correct colors
- [ ] Chart shows 100-150 bars in default view
- [ ] Grid is subtle (barely visible)

### Position Boxes
- [ ] TP zones are green with correct transparency
- [ ] SL zone is red with correct transparency
- [ ] Entry line is blue
- [ ] Boxes extend to chart right edge
- [ ] Labels positioned correctly

### Typography
- [ ] All prices use monospace font
- [ ] All numbers are right-aligned in tables

---

# 10. EXECUTION PROTOCOL

1. Analyze current codebase
2. Research best approaches
3. Build foundation (design system)
4. Implement chart visualization
5. Add annotations and position boxes
6. Build UI components
7. Integrate and polish

The goal is a professional-grade trading platform. Accept nothing less.
