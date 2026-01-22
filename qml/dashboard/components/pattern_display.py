"""Pattern display components for Pattern Lab.

Reusable display functions for QML patterns with Arctic Pro styling.
"""

import streamlit as st
import pandas as pd
from typing import List, Optional

from theme import ARCTIC_PRO, TYPOGRAPHY
from src.detection import QMLPattern, PatternDirection, RegimeResult


def display_pattern_table(patterns: List[QMLPattern]) -> None:
    """Display patterns as styled DataFrame table."""
    if not patterns:
        st.info("No patterns to display")
        return

    data = []
    for p in patterns:
        direction = "SHORT" if p.direction == PatternDirection.BULLISH else "LONG"
        risk = abs(p.stop_loss - p.entry_price)
        reward = abs(p.take_profit_1 - p.entry_price)
        rr = reward / risk if risk > 0 else 0

        data.append({
            'ID': p.id,
            'Type': direction,
            'Strength': f"{p.pattern_strength:.3f}",
            'Head Ext': f"{p.head_extension_atr:.2f}",
            'BOS': p.bos_count,
            'Entry': f"${p.entry_price:,.2f}",
            'SL': f"${p.stop_loss:,.2f}",
            'TP1': f"${p.take_profit_1:,.2f}",
            'R:R': f"1:{rr:.1f}"
        })

    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def pattern_to_chart_dict(pattern: QMLPattern) -> dict:
    """Convert QMLPattern to dict format expected by chart component.

    Returns dict with keys matching tradingview_chart.py expectations:
    - p1_timestamp, p1_price through p5_timestamp, p5_price
    - entry_price, stop_loss_price, take_profit_price
    """
    return {
        'type': 'bullish_qml' if pattern.direction == PatternDirection.BULLISH else 'bearish_qml',
        'head_time': pattern.p3.timestamp,
        'detection_time': pattern.p5.timestamp,
        'entry_price': pattern.entry_price,
        'stop_loss_price': pattern.stop_loss,
        'take_profit_price': pattern.take_profit_1,
        'validity_score': pattern.pattern_strength,
        # Swing point data - use p*_timestamp keys to match chart component
        'p1_timestamp': pattern.p1.timestamp,
        'p1_price': pattern.p1.price,
        'p2_timestamp': pattern.p2.timestamp,
        'p2_price': pattern.p2.price,
        'p3_timestamp': pattern.p3.timestamp,
        'p3_price': pattern.p3.price,
        'p4_timestamp': pattern.p4.timestamp,
        'p4_price': pattern.p4.price,
        'p5_timestamp': pattern.p5.timestamp,
        'p5_price': pattern.p5.price,
    }
