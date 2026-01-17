"""
Trade Visualizer - Visual Forensics Lab
========================================
Interactive Plotly charts for manual trade verification.

Renders individual trades with:
- OHLC candlestick chart
- Entry/Exit markers
- Stop Loss / Take Profit levels
- Swing points (when available)
- Pattern zones

Designed for Jupyter notebook usage.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class TradeVisualizer:
    """
    Interactive trade visualization for forensic analysis.
    
    Plots individual trades with full context including:
    - Candlestick chart
    - Entry/exit markers
    - SL/TP levels
    - Volume
    - Pattern annotations
    
    Usage (in Jupyter):
        from src.analysis.visualizer import TradeVisualizer
        
        viz = TradeVisualizer()
        viz.plot_trade(trade, df, context_bars=50)
    """
    
    def __init__(self):
        """Initialize visualizer."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required: pip install plotly")
    
    def plot_trade(
        self,
        trade: Dict[str, Any],
        df: pd.DataFrame,
        context_bars: int = 50,
        show_volume: bool = True,
        show_atr: bool = False
    ) -> go.Figure:
        """
        Create interactive chart for a single trade.
        
        Args:
            trade: Trade dictionary with entry_time, exit_time, etc.
            df: OHLCV DataFrame
            context_bars: Number of bars before/after trade to show
            show_volume: Include volume subplot
            show_atr: Include ATR subplot
        
        Returns:
            Plotly Figure object
        """
        # Parse trade times
        entry_time = pd.to_datetime(trade.get('entry_time'))
        exit_time = pd.to_datetime(trade.get('exit_time'))
        
        # Ensure df has datetime index
        if 'time' in df.columns:
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        # Get trade timeframe
        entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = df.index.get_indexer([exit_time], method='nearest')[0]
        
        # Calculate slice bounds
        start_idx = max(0, entry_idx - context_bars)
        end_idx = min(len(df), exit_idx + context_bars)
        
        # Slice data
        plot_df = df.iloc[start_idx:end_idx].copy()
        
        # Create subplots
        rows = 2 if show_volume else 1
        row_heights = [0.7, 0.3] if show_volume else [1.0]
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=('Price Action', 'Volume') if show_volume else None
        )
        
        # Add candlesticks
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add volume if requested
        if show_volume and 'volume' in plot_df.columns:
            colors = ['#26a69a' if c >= o else '#ef5350' 
                     for o, c in zip(plot_df['open'], plot_df['close'])]
            
            fig.add_trace(
                go.Bar(
                    x=plot_df.index,
                    y=plot_df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
        
        # Trade details
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        side = trade.get('side', 'LONG')
        result = trade.get('result', '')
        pnl_pct = trade.get('pnl_pct', 0)
        
        # Entry marker
        entry_color = '#2196F3'  # Blue
        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode='markers+text',
                name='Entry',
                marker=dict(
                    symbol='triangle-up' if side == 'LONG' else 'triangle-down',
                    size=15,
                    color=entry_color
                ),
                text=['ENTRY'],
                textposition='top center',
                textfont=dict(size=10, color=entry_color)
            ),
            row=1, col=1
        )
        
        # Exit marker
        exit_color = '#27ae60' if result == 'WIN' else '#e74c3c'
        fig.add_trace(
            go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode='markers+text',
                name='Exit',
                marker=dict(
                    symbol='x',
                    size=12,
                    color=exit_color
                ),
                text=[f'EXIT ({pnl_pct:+.1f}%)'],
                textposition='bottom center',
                textfont=dict(size=10, color=exit_color)
            ),
            row=1, col=1
        )
        
        # Stop Loss line
        if stop_loss:
            fig.add_hline(
                y=stop_loss,
                line_dash='dash',
                line_color='#e74c3c',
                annotation_text=f'SL: ${stop_loss:,.0f}',
                annotation_position='right',
                row=1, col=1
            )
        
        # Take Profit line
        if take_profit:
            fig.add_hline(
                y=take_profit,
                line_dash='dash',
                line_color='#27ae60',
                annotation_text=f'TP: ${take_profit:,.0f}',
                annotation_position='right',
                row=1, col=1
            )
        
        # Entry price line
        fig.add_hline(
            y=entry_price,
            line_dash='dot',
            line_color='#2196F3',
            opacity=0.5,
            row=1, col=1
        )
        
        # Trade zone shading
        fig.add_vrect(
            x0=entry_time,
            x1=exit_time,
            fillcolor='rgba(33, 150, 243, 0.1)',
            layer='below',
            line_width=0,
            row=1, col=1
        )
        
        # Layout
        title = (f"Trade: {side} | "
                f"Entry: ${entry_price:,.0f} | "
                f"Exit: ${exit_price:,.0f} | "
                f"Result: {result} ({pnl_pct:+.2f}%)")
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            showlegend=False,
            hovermode='x unified',
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            font=dict(family='system-ui, -apple-system, sans-serif')
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        
        return fig
    
    def plot_trade_from_db(
        self,
        run_id: str,
        trade_index: int = 0,
        df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Plot a trade by loading from database.
        
        Args:
            run_id: Experiment run ID
            trade_index: Which trade to plot (0 = first)
            df: Optional pre-loaded DataFrame
        
        Returns:
            Plotly Figure
        """
        # Load trades from CSV
        from src.reporting.storage import ExperimentLogger
        import json
        
        logger = ExperimentLogger()
        run = logger.get_run(run_id)
        
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        
        # Get strategy and load trades
        config = json.loads(run['config_json'])
        strategy = config.get('detector_method', 'atr')
        
        trades_path = Path(__file__).parent.parent.parent / "results" / strategy / f"{run_id}_trades.csv"
        
        if not trades_path.exists():
            raise FileNotFoundError(f"Trades file not found: {trades_path}")
        
        trades_df = pd.read_csv(trades_path)
        
        if trade_index >= len(trades_df):
            raise IndexError(f"Trade index {trade_index} out of range (max: {len(trades_df)-1})")
        
        trade = trades_df.iloc[trade_index].to_dict()
        
        # Load candle data if not provided
        if df is None:
            symbol = config.get('symbol', 'BTCUSDT')
            timeframe = config.get('timeframe', '4h')
            
            data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "BTC" / f"{timeframe}_master.parquet"
            df = pd.read_parquet(data_path)
        
        return self.plot_trade(trade, df)
    
    def plot_signal(
        self,
        signal: Dict[str, Any],
        df: pd.DataFrame,
        context_bars: int = 30
    ) -> go.Figure:
        """
        Plot a detection signal (before trade execution).
        
        Args:
            signal: Signal dictionary
            df: OHLCV DataFrame
            context_bars: Context window size
        
        Returns:
            Plotly Figure
        """
        signal_time = pd.to_datetime(signal.get('timestamp'))
        
        # Ensure datetime index
        if 'time' in df.columns:
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        # Get signal index
        signal_idx = df.index.get_indexer([signal_time], method='nearest')[0]
        
        # Slice
        start_idx = max(0, signal_idx - context_bars)
        end_idx = min(len(df), signal_idx + context_bars)
        plot_df = df.iloc[start_idx:end_idx]
        
        # Create figure
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close'],
            name='OHLC'
        ))
        
        # Signal marker
        signal_type = signal.get('signal_type', 'UNKNOWN')
        price = signal.get('price', 0)
        
        fig.add_trace(go.Scatter(
            x=[signal_time],
            y=[price],
            mode='markers',
            name='Signal',
            marker=dict(
                symbol='diamond',
                size=20,
                color='#2196F3'
            )
        ))
        
        # SL/TP lines
        if signal.get('stop_loss'):
            fig.add_hline(y=signal['stop_loss'], line_dash='dash', line_color='red')
        if signal.get('take_profit'):
            fig.add_hline(y=signal['take_profit'], line_dash='dash', line_color='green')
        
        fig.update_layout(
            title=f"Signal: {signal_type} at ${price:,.0f}",
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig
