#!/usr/bin/env python3
"""
QML Pattern Visualization Script - FIXED VERSION
=================================================
Shows every component of QML detection with correct chronological labeling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Optional

from src.data.fetcher import DataFetcher
from src.detection.detector import QMLDetector
from src.detection.swing import SwingDetector
from src.data.models import QMLPattern, PatternType, SwingPoint, SwingType


def normalize_time(t):
    """Remove timezone info from datetime."""
    if t is None:
        return None
    if hasattr(t, 'tzinfo') and t.tzinfo is not None:
        return t.replace(tzinfo=None)
    return t


class PatternVisualizer:
    """Creates detailed QML pattern visualizations."""
    
    COLORS = {
        'bullish': '#3FB950',
        'bearish': '#F85149',
        'swing_high': '#FF6B6B',
        'swing_low': '#4ECDC4',
        'head': '#FF6B9D',          # Pink - the extreme
        'choch': '#FFD93D',         # Yellow - change of character
        'bos': '#A855F7',           # Purple - break of structure
        'entry': '#00D9FF',         # Cyan - entry level
        'stop_loss': '#FF4757',     # Red - stop loss
        'take_profit': '#2ED573',   # Green - take profit
        'pattern_line': '#818CF8',  # Light purple - pattern structure
        'grid': '#1E293B',
        'bg': '#0F172A',
        'card_bg': '#1E293B',
        'text': '#E2E8F0',
        'text_muted': '#94A3B8',
    }
    
    def __init__(self, output_dir: str = "pattern_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = DataFetcher()
        self.swing_detector = SwingDetector()
        self.qml_detector = QMLDetector()
        print(f"📊 Pattern Visualizer initialized - Output: {self.output_dir}")
    
    def detect_and_visualize(self, symbol: str, timeframe: str, 
                            lookback_bars: int = 800, save_html: bool = True) -> List[go.Figure]:
        """Detect patterns and create detailed visualizations."""
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing {symbol} {timeframe}")
        print(f"{'='*60}")
        
        df = self.fetcher.get_data(symbol, timeframe, limit=lookback_bars)
        if df is None or len(df) < 100:
            print(f"❌ Insufficient data")
            return []
        
        print(f"✅ Loaded {len(df)} candles")
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        
        swings = self.swing_detector.detect(df, symbol)
        patterns = self.qml_detector.detect(symbol, timeframe, df)
        print(f"📍 Found {len(swings)} swing points, {len(patterns)} QML patterns")
        
        figures = []
        
        if not patterns:
            fig = self._create_overview_chart(df, symbol, timeframe, swings)
            figures.append(fig)
            if save_html:
                filepath = self.output_dir / f"{symbol.replace('/', '_')}_{timeframe}_overview.html"
                fig.write_html(str(filepath))
                print(f"   💾 Saved: {filepath}")
        else:
            for i, pattern in enumerate(patterns):
                print(f"\n📈 Creating chart {i+1}/{len(patterns)}")
                fig = self._create_pattern_chart(df, pattern, symbol, timeframe, swings)
                figures.append(fig)
                if save_html:
                    pt = normalize_time(pattern.detection_time)
                    ts = pt.strftime("%Y%m%d_%H%M") if pt else "unknown"
                    filepath = self.output_dir / f"{symbol.replace('/', '_')}_{timeframe}_{pattern.pattern_type.value}_{ts}.html"
                    fig.write_html(str(filepath))
                    print(f"   💾 Saved: {filepath}")
        
        return figures
    
    def _create_overview_chart(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                               swings: List[SwingPoint]) -> go.Figure:
        """Create overview chart with swing points."""
        plot_df = df.copy()
        if 'time' in plot_df.columns:
            plot_df['time'] = pd.to_datetime(plot_df['time']).dt.tz_localize(None)
            plot_df = plot_df.set_index('time')
        plot_df = plot_df.tail(300)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           row_heights=[0.75, 0.25])
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
            low=plot_df['low'], close=plot_df['close'], name='Price',
            increasing_line_color=self.COLORS['bullish'],
            decreasing_line_color=self.COLORS['bearish']
        ), row=1, col=1)
        
        # Volume
        colors = [self.COLORS['bullish'] if c >= o else self.COLORS['bearish'] 
                  for c, o in zip(plot_df['close'], plot_df['open'])]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], 
                            marker_color=colors, opacity=0.5, name='Volume'), row=2, col=1)
        
        self._add_swing_points(fig, swings, plot_df)
        
        fig.update_layout(
            title=f"<b>{symbol} {timeframe}</b> - Market Structure Overview",
            template='plotly_dark',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['bg'],
            font=dict(color=self.COLORS['text'], size=12),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=800,
            legend=dict(bgcolor='rgba(15,23,42,0.9)', bordercolor=self.COLORS['grid'])
        )
        fig.update_xaxes(gridcolor=self.COLORS['grid'], showgrid=True)
        fig.update_yaxes(gridcolor=self.COLORS['grid'], showgrid=True)
        
        return fig
    
    def _create_pattern_chart(self, df: pd.DataFrame, pattern: QMLPattern,
                             symbol: str, timeframe: str, swings: List[SwingPoint]) -> go.Figure:
        """Create detailed chart for a QML pattern with proper labeling."""
        plot_df = df.copy()
        if 'time' in plot_df.columns:
            plot_df['time'] = pd.to_datetime(plot_df['time']).dt.tz_localize(None)
            plot_df = plot_df.set_index('time')
        
        # Get pattern time and show MORE context (120 bars before, 40 after)
        pattern_time = normalize_time(pattern.detection_time)
        if pattern_time and pattern_time in plot_df.index:
            pattern_idx = plot_df.index.get_loc(pattern_time)
        else:
            pattern_idx = len(plot_df) - 1
        
        start_idx = max(0, pattern_idx - 120)  # More bars before
        end_idx = min(len(plot_df), pattern_idx + 40)   # Some bars after
        plot_df = plot_df.iloc[start_idx:end_idx]
        
        # Create figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           row_heights=[0.75, 0.25])
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
            low=plot_df['low'], close=plot_df['close'], name='Price',
            increasing_line_color=self.COLORS['bullish'],
            decreasing_line_color=self.COLORS['bearish']
        ), row=1, col=1)
        
        # Volume
        colors = [self.COLORS['bullish'] if c >= o else self.COLORS['bearish'] 
                  for c, o in zip(plot_df['close'], plot_df['open'])]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], 
                            marker_color=colors, opacity=0.5, name='Volume'), row=2, col=1)
        
        # Swing points
        self._add_swing_points(fig, swings, plot_df)
        
        # Pattern components with CORRECT labels
        self._add_pattern_markers(fig, pattern, plot_df)
        
        # Trading levels
        self._add_trading_levels(fig, pattern, plot_df)
        
        # Pattern structure lines
        self._add_structure_lines(fig, pattern)
        
        # Title and layout
        direction = "🟢 BULLISH" if pattern.pattern_type == PatternType.BULLISH else "🔴 BEARISH"
        pt_str = pattern_time.strftime('%Y-%m-%d %H:%M') if pattern_time else "N/A"
        
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol} {timeframe} - QML Pattern {direction}</b><br>" +
                     f"<span style='font-size:13px;color:{self.COLORS['text_muted']}'>Detection: {pt_str} | Validity: {pattern.validity_score:.1%}</span>",
                font=dict(size=18, color=self.COLORS['text'])
            ),
            template='plotly_dark',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['bg'],
            font=dict(color=self.COLORS['text'], size=12),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=900,
            legend=dict(
                bgcolor='rgba(15,23,42,0.95)', 
                bordercolor=self.COLORS['grid'],
                borderwidth=1,
                font=dict(size=11)
            )
        )
        fig.update_xaxes(gridcolor=self.COLORS['grid'], showgrid=True)
        fig.update_yaxes(gridcolor=self.COLORS['grid'], showgrid=True)
        
        return fig
    
    def _add_swing_points(self, fig: go.Figure, swings: List[SwingPoint], plot_df: pd.DataFrame):
        """Add swing markers."""
        highs_x, highs_y, lows_x, lows_y = [], [], [], []
        
        for swing in swings:
            st = normalize_time(swing.time)
            if st and plot_df.index.min() <= st <= plot_df.index.max():
                if swing.swing_type == SwingType.HIGH:
                    highs_x.append(st)
                    highs_y.append(swing.price)
                else:
                    lows_x.append(st)
                    lows_y.append(swing.price)
        
        if highs_x:
            fig.add_trace(go.Scatter(
                x=highs_x, y=highs_y, mode='markers+text',
                marker=dict(size=10, color=self.COLORS['swing_high'], symbol='triangle-down',
                           line=dict(width=1, color='white')),
                text=['SH']*len(highs_x), textposition='top center',
                textfont=dict(size=9, color=self.COLORS['swing_high']),
                name='Swing High', hovertemplate='<b>Swing High</b><br>$%{y:,.2f}<extra></extra>'
            ), row=1, col=1)
        
        if lows_x:
            fig.add_trace(go.Scatter(
                x=lows_x, y=lows_y, mode='markers+text',
                marker=dict(size=10, color=self.COLORS['swing_low'], symbol='triangle-up',
                           line=dict(width=1, color='white')),
                text=['SL']*len(lows_x), textposition='bottom center',
                textfont=dict(size=9, color=self.COLORS['swing_low']),
                name='Swing Low', hovertemplate='<b>Swing Low</b><br>$%{y:,.2f}<extra></extra>'
            ), row=1, col=1)
    
    def _add_pattern_markers(self, fig: go.Figure, pattern: QMLPattern, plot_df: pd.DataFrame):
        """Add pattern component markers with CORRECT chronological labels."""
        is_bullish = pattern.pattern_type == PatternType.BULLISH
        
        # HEAD - The extreme point (comes FIRST chronologically in the pattern)
        head_time = normalize_time(pattern.head_time)
        if head_time and pattern.head_price:
            fig.add_trace(go.Scatter(
                x=[head_time], y=[pattern.head_price],
                mode='markers+text',
                marker=dict(size=24, color=self.COLORS['head'], symbol='star',
                           line=dict(width=2, color='white')),
                text=['<b>1. HEAD</b><br>(Extreme)'],
                textposition='bottom center' if is_bullish else 'top center',
                textfont=dict(size=12, color=self.COLORS['head'], family='Arial'),
                name='① HEAD (Extreme)',
                hovertemplate='<b>HEAD - Pattern Extreme</b><br>Price: $%{y:,.2f}<br>This is the highest/lowest point<extra></extra>'
            ), row=1, col=1)
        
        # CHoCH LEVEL - Change of Character (comes AFTER the head)
        choch_time = normalize_time(pattern.left_shoulder_time)
        if choch_time and pattern.left_shoulder_price:
            fig.add_trace(go.Scatter(
                x=[choch_time], y=[pattern.left_shoulder_price],
                mode='markers+text',
                marker=dict(size=20, color=self.COLORS['choch'], symbol='diamond',
                           line=dict(width=2, color='white')),
                text=['<b>2. CHoCH</b><br>(Trend Break)'],
                textposition='top center' if is_bullish else 'bottom center',
                textfont=dict(size=11, color=self.COLORS['choch'], family='Arial'),
                name='② CHoCH (Change of Character)',
                hovertemplate='<b>CHoCH - Change of Character</b><br>Price: $%{y:,.2f}<br>First sign of trend reversal<extra></extra>'
            ), row=1, col=1)
            
            # CHoCH horizontal level
            fig.add_hline(y=pattern.left_shoulder_price, line_dash="dash",
                         line_color=self.COLORS['choch'], line_width=2, opacity=0.6,
                         annotation_text="CHoCH Level", annotation_position="left", row=1, col=1)
        
        # BoS LEVEL - Break of Structure / Entry Zone (comes LAST)
        # This is the detection_time / entry zone
        bos_time = normalize_time(pattern.detection_time)
        if bos_time and pattern.trading_levels:
            fig.add_trace(go.Scatter(
                x=[bos_time], y=[pattern.trading_levels.entry],
                mode='markers+text',
                marker=dict(size=20, color=self.COLORS['bos'], symbol='circle',
                           line=dict(width=2, color='white')),
                text=['<b>3. BoS</b><br>(Entry Zone)'],
                textposition='top center' if is_bullish else 'bottom center',
                textfont=dict(size=11, color=self.COLORS['bos'], family='Arial'),
                name='③ BoS (Break of Structure)',
                hovertemplate='<b>BoS - Break of Structure</b><br>Price: $%{y:,.2f}<br>Confirmation & Entry Zone<extra></extra>'
            ), row=1, col=1)
    
    def _add_trading_levels(self, fig: go.Figure, pattern: QMLPattern, plot_df: pd.DataFrame):
        """Add entry, stop loss, and take profit levels."""
        if not pattern.trading_levels:
            return
        
        tl = pattern.trading_levels
        x_range = [plot_df.index.min(), plot_df.index.max()]
        
        # Entry
        fig.add_trace(go.Scatter(
            x=x_range, y=[tl.entry, tl.entry], mode='lines',
            line=dict(color=self.COLORS['entry'], width=3),
            name=f'ENTRY: ${tl.entry:,.2f}'
        ), row=1, col=1)
        
        fig.add_annotation(
            x=plot_df.index.max(), y=tl.entry,
            text=f"<b>▶ ENTRY ${tl.entry:,.2f}</b>",
            showarrow=False, font=dict(size=13, color='white'),
            bgcolor=self.COLORS['entry'], borderpad=5, xanchor='right'
        )
        
        # Stop Loss
        fig.add_trace(go.Scatter(
            x=x_range, y=[tl.stop_loss, tl.stop_loss], mode='lines',
            line=dict(color=self.COLORS['stop_loss'], width=3, dash='dash'),
            name=f'STOP: ${tl.stop_loss:,.2f}'
        ), row=1, col=1)
        
        fig.add_annotation(
            x=plot_df.index.max(), y=tl.stop_loss,
            text=f"<b>✕ STOP ${tl.stop_loss:,.2f}</b>",
            showarrow=False, font=dict(size=13, color='white'),
            bgcolor=self.COLORS['stop_loss'], borderpad=5, xanchor='right'
        )
        
        # Take Profits
        for tp, label, alpha in [(tl.take_profit_1, 'TP1', 1.0), 
                                  (tl.take_profit_2, 'TP2', 0.7),
                                  (tl.take_profit_3, 'TP3', 0.5)]:
            fig.add_trace(go.Scatter(
                x=x_range, y=[tp, tp], mode='lines',
                line=dict(color=self.COLORS['take_profit'], width=2, dash='dot'),
                opacity=alpha, name=f'{label}: ${tp:,.2f}'
            ), row=1, col=1)
        
        # Main target annotation
        fig.add_annotation(
            x=plot_df.index.max(), y=tl.take_profit_1,
            text=f"<b>✓ TARGET ${tl.take_profit_1:,.2f}</b>",
            showarrow=False, font=dict(size=13, color='white'),
            bgcolor=self.COLORS['take_profit'], borderpad=5, xanchor='right'
        )
        
        # Trade info box
        risk = abs(tl.entry - tl.stop_loss)
        reward = abs(tl.take_profit_1 - tl.entry)
        rr = reward / risk if risk > 0 else 0
        
        fig.add_annotation(
            x=plot_df.index.min(), y=plot_df['high'].max(),
            text=f"<b>📊 TRADE SETUP</b><br>" +
                 f"Entry: ${tl.entry:,.2f}<br>" +
                 f"Risk: ${risk:,.2f}<br>" +
                 f"Reward: ${reward:,.2f}<br>" +
                 f"<b>R:R = {rr:.1f}:1</b>",
            showarrow=False, font=dict(size=11, color=self.COLORS['text']),
            bgcolor=self.COLORS['card_bg'], bordercolor=self.COLORS['entry'],
            borderwidth=2, borderpad=10, xanchor='left', yanchor='top'
        )
    
    def _add_structure_lines(self, fig: go.Figure, pattern: QMLPattern):
        """Add lines connecting pattern points."""
        points = []
        
        # HEAD first (chronologically first in the pattern)
        if pattern.head_time and pattern.head_price:
            points.append((normalize_time(pattern.head_time), pattern.head_price))
        
        # Then CHoCH level
        if pattern.left_shoulder_time and pattern.left_shoulder_price:
            points.append((normalize_time(pattern.left_shoulder_time), pattern.left_shoulder_price))
        
        # Then BoS/Entry
        if pattern.detection_time and pattern.trading_levels:
            points.append((normalize_time(pattern.detection_time), pattern.trading_levels.entry))
        
        if len(points) >= 2:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in points], y=[p[1] for p in points],
                mode='lines', line=dict(color=self.COLORS['pattern_line'], width=2, dash='dash'),
                name='Pattern Structure', hoverinfo='skip'
            ), row=1, col=1)


def main():
    print("\n" + "="*60)
    print("  🎯 QML PATTERN VISUALIZATION - FIXED")
    print("="*60)
    print("\nShows: HEAD → CHoCH → BoS (Entry) in correct order")
    print("-"*60)
    
    visualizer = PatternVisualizer()
    
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]:
        for tf in ["1h", "4h"]:
            try:
                visualizer.detect_and_visualize(symbol, tf, lookback_bars=800)
            except Exception as e:
                print(f"❌ Error: {symbol} {tf}: {e}")
    
    print("\n" + "="*60)
    print("✅ Complete! Open HTML files in browser.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
