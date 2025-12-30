#!/usr/bin/env python3
"""
QML PATTERN VISUALIZATION GENERATOR - FIXED VERSION
====================================================
Generates high-quality, standardized visualizations for every QML pattern
detected in the v1.1.0 rolling-window backtest.

CRITICAL FIX: Uses mplfinance's `alines` parameter for drawing lines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import mplfinance as mpf

# Try to import ccxt for data fetching
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("Warning: ccxt not available, will use cached data only")


class QMLPatternVisualizer:
    """
    Generates professional QML pattern visualizations using mplfinance.
    
    CRITICAL: Uses alines parameter for drawing structure lines.
    """
    
    def __init__(self, output_dir: str = "pattern_charts"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define dark theme style
        self.mc = mpf.make_marketcolors(
            up='#26a69a',      # Green for up candles
            down='#ef5350',    # Red for down candles
            edge='inherit',
            wick='inherit',
            volume='in',
        )
        
        self.style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=self.mc,
            figcolor='#1e222d',
            facecolor='#1e222d',
            edgecolor='#363a45',
            gridcolor='#363a45',
            gridstyle='-',
            gridaxis='both',
            y_on_right=True,
        )
        
        # OHLCV data cache
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
    
    def load_patterns(self, csv_path: str) -> pd.DataFrame:
        """Load pattern data from CSV."""
        df = pd.read_csv(csv_path, parse_dates=['time'])
        print(f"Loaded {len(df)} patterns from {csv_path}")
        return df
    
    def fetch_ohlcv(self, symbol: str, start_time: datetime, 
                     end_time: datetime, timeframe: str = '1h') -> pd.DataFrame:
        """Fetch OHLCV data for visualization window."""
        cache_key = f"{symbol}_{start_time.date()}_{end_time.date()}"
        
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]
        
        if not HAS_CCXT:
            return pd.DataFrame()
        
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            all_candles = []
            current_ts = start_ts
            
            while current_ts < end_ts:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
                if not ohlcv:
                    break
                all_candles.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts <= current_ts:
                    break
                current_ts = last_ts + 1
            
            if not all_candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_candles, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.drop_duplicates(subset=['time']).sort_values('time')
            df = df.set_index('time')
            
            # Filter to window
            df = df[(df.index >= start_time) & (df.index <= end_time)]
            
            self._ohlcv_cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def find_swing_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """Find swing highs and lows in price data."""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Swing high
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                highs.append((df.index[i], df['High'].iloc[i]))
            
            # Swing low
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                lows.append((df.index[i], df['Low'].iloc[i]))
        
        return highs, lows
    
    def reconstruct_pattern_points(self, df: pd.DataFrame, pattern: pd.Series) -> Optional[Dict]:
        """
        Reconstruct the 5 QML points from pattern data.
        
        P1: Start of trend/Base
        P2: Left Shoulder (LS) - CHoCH level
        P3: Head (H) - Extreme point
        P4: Lower Low (LL) - Structure break / CHoCH
        P5: Right Shoulder (RS) - Entry retracement
        """
        pattern_time = pd.Timestamp(pattern['time'])
        pattern_type = pattern['pattern_type']
        
        head_price = pattern['head_price']
        ls_price = pattern['left_shoulder_price']
        choch_level = pattern['choch_level']
        entry_price = pattern['entry_price']
        
        is_bullish = 'bullish' in pattern_type
        
        # Find the pattern window in OHLCV data
        if pattern_time not in df.index:
            idx = df.index.get_indexer([pattern_time], method='nearest')[0]
            if idx >= len(df):
                return None
        else:
            idx = df.index.get_loc(pattern_time)
        
        # Look back to find pattern points
        lookback = min(100, idx)
        window_df = df.iloc[max(0, idx - lookback):idx + 10]
        
        if len(window_df) < 20:
            return None
        
        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(window_df, window=3)
        
        if is_bullish:
            # Bullish QML: P1(low) → P2(high/LS) → P3(low/Head) → P4(high/CHoCH) → P5(RS)
            
            # P3 (Head): Find swing low closest to head_price
            head_candidates = [(t, p) for t, p in swing_lows 
                               if t < pattern_time and abs(p - head_price) / head_price < 0.02]
            if head_candidates:
                p3_time, p3_price = min(head_candidates, key=lambda x: abs(x[1] - head_price))
            else:
                head_area = window_df[window_df['Low'] <= head_price * 1.01]
                if len(head_area) > 0:
                    p3_time = head_area['Low'].idxmin()
                    p3_price = head_area.loc[p3_time, 'Low']
                else:
                    p3_time = window_df['Low'].idxmin()
                    p3_price = window_df['Low'].min()
            
            # P2 (Left Shoulder): Swing high before head
            ls_candidates = [(t, p) for t, p in swing_highs 
                             if t < p3_time and abs(p - ls_price) / ls_price < 0.03]
            if ls_candidates:
                p2_time, p2_price = max(ls_candidates, key=lambda x: x[0])
            else:
                pre_head = window_df[window_df.index < p3_time]
                if len(pre_head) > 0:
                    p2_time = pre_head['High'].idxmax()
                    p2_price = pre_head.loc[p2_time, 'High']
                else:
                    return None
            
            # P1 (Base): Swing low before left shoulder
            base_candidates = [(t, p) for t, p in swing_lows if t < p2_time]
            if base_candidates:
                p1_time, p1_price = max(base_candidates, key=lambda x: x[0])
            else:
                pre_ls = window_df[window_df.index < p2_time]
                if len(pre_ls) > 0:
                    p1_time = pre_ls['Low'].idxmin()
                    p1_price = pre_ls.loc[p1_time, 'Low']
                else:
                    return None
            
            # P4 (CHoCH break): Swing high after head
            choch_candidates = [(t, p) for t, p in swing_highs 
                                if t > p3_time and t < pattern_time]
            if choch_candidates:
                p4_time, p4_price = min(choch_candidates, key=lambda x: x[0])
            else:
                post_head = window_df[(window_df.index > p3_time) & (window_df.index <= pattern_time)]
                if len(post_head) > 0:
                    p4_time = post_head['High'].idxmax()
                    p4_price = post_head.loc[p4_time, 'High']
                else:
                    return None
            
            # P5 (RS Entry)
            p5_time = pattern_time
            p5_price = entry_price
            
        else:
            # Bearish QML: P1(high) → P2(low/LS) → P3(high/Head) → P4(low/CHoCH) → P5(RS)
            
            # P3 (Head): Swing high closest to head_price
            head_candidates = [(t, p) for t, p in swing_highs 
                               if t < pattern_time and abs(p - head_price) / head_price < 0.02]
            if head_candidates:
                p3_time, p3_price = min(head_candidates, key=lambda x: abs(x[1] - head_price))
            else:
                head_area = window_df[window_df['High'] >= head_price * 0.99]
                if len(head_area) > 0:
                    p3_time = head_area['High'].idxmax()
                    p3_price = head_area.loc[p3_time, 'High']
                else:
                    p3_time = window_df['High'].idxmax()
                    p3_price = window_df['High'].max()
            
            # P2 (Left Shoulder): Swing low before head
            ls_candidates = [(t, p) for t, p in swing_lows 
                             if t < p3_time and abs(p - ls_price) / ls_price < 0.03]
            if ls_candidates:
                p2_time, p2_price = max(ls_candidates, key=lambda x: x[0])
            else:
                pre_head = window_df[window_df.index < p3_time]
                if len(pre_head) > 0:
                    p2_time = pre_head['Low'].idxmin()
                    p2_price = pre_head.loc[p2_time, 'Low']
                else:
                    return None
            
            # P1 (Base): Swing high before left shoulder
            base_candidates = [(t, p) for t, p in swing_highs if t < p2_time]
            if base_candidates:
                p1_time, p1_price = max(base_candidates, key=lambda x: x[0])
            else:
                pre_ls = window_df[window_df.index < p2_time]
                if len(pre_ls) > 0:
                    p1_time = pre_ls['High'].idxmax()
                    p1_price = pre_ls.loc[p1_time, 'High']
                else:
                    return None
            
            # P4 (CHoCH): Swing low after head
            choch_candidates = [(t, p) for t, p in swing_lows 
                                if t > p3_time and t < pattern_time]
            if choch_candidates:
                p4_time, p4_price = min(choch_candidates, key=lambda x: x[0])
            else:
                post_head = window_df[(window_df.index > p3_time) & (window_df.index <= pattern_time)]
                if len(post_head) > 0:
                    p4_time = post_head['Low'].idxmin()
                    p4_price = post_head.loc[p4_time, 'Low']
                else:
                    return None
            
            # P5 (RS Entry)
            p5_time = pattern_time
            p5_price = entry_price
        
        return {
            'P1': (p1_time, p1_price),
            'P2': (p2_time, p2_price),  # LS
            'P3': (p3_time, p3_price),  # Head
            'P4': (p4_time, p4_price),  # LL/CHoCH
            'P5': (p5_time, p5_price),  # RS
            'is_bullish': is_bullish,
        }
    
    def create_chart(self, df: pd.DataFrame, points: Dict, 
                     pattern: pd.Series, output_path: Path) -> bool:
        """
        Create mplfinance chart with QML structure overlay using alines.
        """
        try:
            # Get chart window with padding
            p1_time = points['P1'][0]
            p5_time = points['P5'][0]
            
            p1_idx = df.index.get_indexer([p1_time], method='nearest')[0]
            p5_idx = df.index.get_indexer([p5_time], method='nearest')[0]
            
            start_idx = max(0, p1_idx - 10)
            end_idx = min(len(df) - 1, p5_idx + 10)
            
            chart_df = df.iloc[start_idx:end_idx + 1].copy()
            
            if len(chart_df) < 10:
                print(f"  ⚠️ Insufficient data for chart")
                return False
            
            # ================================================================
            # CRITICAL: Build alines (arbitrary lines) for mplfinance
            # alines expects: [[(date1, price1), (date2, price2), ...], ...]
            # ================================================================
            
            point_order = ['P1', 'P2', 'P3', 'P4', 'P5']
            line_sequence = []
            
            print(f"  📍 Building QML structure line:")
            for i, point_name in enumerate(point_order):
                pt_time, pt_price = points[point_name]
                
                # Verify the point is valid
                if pd.isna(pt_price) or pt_price is None:
                    print(f"    ❌ {point_name}: INVALID (NaN price)")
                    return False
                
                # Ensure time is in chart range
                if pt_time < chart_df.index[0] or pt_time > chart_df.index[-1]:
                    # Find nearest valid time
                    nearest_idx = chart_df.index.get_indexer([pt_time], method='nearest')[0]
                    pt_time = chart_df.index[nearest_idx]
                
                line_sequence.append((pt_time, pt_price))
                
                labels = {'P1': 'Base', 'P2': 'LS', 'P3': 'H', 'P4': 'LL', 'P5': 'RS'}
                print(f"    {point_name} ({labels[point_name]}): {pt_time.strftime('%Y-%m-%d %H:%M')} @ ${pt_price:,.2f}")
            
            # Verify we have all 5 points
            if len(line_sequence) != 5:
                print(f"  ❌ Missing points: got {len(line_sequence)}, need 5")
                return False
            
            print(f"  ✓ Drawing line from {line_sequence[0][0].strftime('%m-%d %H:%M')} to {line_sequence[-1][0].strftime('%m-%d %H:%M')}")
            
            # Create the alines specification
            # alines = [sequence of (datetime, price) tuples]
            alines_data = [line_sequence]
            
            alines_kwargs = dict(
                colors=['white'],
                linewidths=2.0,
                alpha=1.0,
            )
            
            # ================================================================
            # Create the chart title
            # ================================================================
            title = f"{pattern['pattern_type'].upper()} | {pattern['time'].strftime('%Y-%m-%d %H:%M')} | Validity: {pattern['validity_score']:.2f}"
            
            # ================================================================
            # PLOT with alines
            # ================================================================
            fig, axes = mpf.plot(
                chart_df,
                type='candle',
                style=self.style,
                title=title,
                ylabel='Price (USDT)',
                volume=False,
                figsize=(14, 8),
                alines=dict(alines=alines_data, **alines_kwargs),
                returnfig=True,
                tight_layout=True,
            )
            
            ax = axes[0]
            
            # ================================================================
            # Add text labels for each point
            # ================================================================
            labels_map = {
                'P2': 'LS',
                'P3': 'H',
                'P4': 'LL',
                'P5': 'RS',
            }
            
            for point_name, label_text in labels_map.items():
                pt_time, pt_price = points[point_name]
                
                # Get x position (need to convert datetime to plot x-coordinate)
                # In mplfinance, x is the integer index in the dataframe
                if pt_time in chart_df.index:
                    x_pos = chart_df.index.get_loc(pt_time)
                else:
                    x_pos = chart_df.index.get_indexer([pt_time], method='nearest')[0]
                
                # Offset y position based on point type
                is_bullish = points['is_bullish']
                if point_name in ['P2', 'P4']:  # These are highs in bullish, lows in bearish
                    if is_bullish:
                        y_offset = pt_price * 1.01  # Above
                        va = 'bottom'
                    else:
                        y_offset = pt_price * 0.99  # Below
                        va = 'top'
                else:  # P3, P5
                    if is_bullish:
                        y_offset = pt_price * 0.99  # Below for lows
                        va = 'top'
                    else:
                        y_offset = pt_price * 1.01  # Above for highs
                        va = 'bottom'
                
                ax.annotate(
                    label_text,
                    xy=(x_pos, pt_price),
                    xytext=(x_pos, y_offset),
                    fontsize=11,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va=va,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e222d', edgecolor='white', alpha=0.8),
                )
            
            # Add entry/SL info
            info_text = f"Entry: ${pattern['entry_price']:,.2f} | SL: ${pattern['stop_loss']:,.2f}"
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                   fontsize=9, color='#d1d4dc', verticalalignment='bottom',
                   family='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e222d', edgecolor='#363a45'))
            
            # Save
            fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='#1e222d', edgecolor='none')
            plt.close(fig)
            
            print(f"  ✅ Chart saved: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"  ❌ Chart creation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_all_charts(self, patterns_csv: str, symbol: str = 'BTC/USDT'):
        """Generate charts for all patterns in the CSV."""
        patterns = self.load_patterns(patterns_csv)
        
        # Fetch OHLCV data
        print("\n📡 Fetching OHLCV data...")
        min_time = patterns['time'].min() - timedelta(days=5)
        max_time = patterns['time'].max() + timedelta(days=2)
        
        ohlcv_df = self.fetch_ohlcv(symbol, min_time, max_time, '1h')
        
        if ohlcv_df.empty:
            print("❌ Failed to fetch OHLCV data")
            return
        
        print(f"   Loaded {len(ohlcv_df)} candles: {ohlcv_df.index.min()} to {ohlcv_df.index.max()}")
        
        # Generate charts
        print(f"\n🎨 Generating {len(patterns)} pattern charts with alines overlay...")
        print("=" * 70)
        success_count = 0
        
        for idx, pattern in patterns.iterrows():
            pattern_num = idx + 1
            pattern_type = pattern['pattern_type']
            pattern_time = pattern['time']
            
            print(f"\n[{pattern_num}/{len(patterns)}] {pattern_type} @ {pattern_time}")
            
            # Reconstruct points
            points = self.reconstruct_pattern_points(ohlcv_df, pattern)
            
            if not points:
                print(f"  ⚠️ Could not reconstruct pattern points")
                continue
            
            # Generate chart filename
            filename = f"qml_{pattern_num:02d}_{pattern_type}_{pattern_time.strftime('%Y%m%d_%H%M')}.png"
            output_path = self.output_dir / filename
            
            if self.create_chart(ohlcv_df, points, pattern, output_path):
                success_count += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Generated {success_count}/{len(patterns)} pattern charts")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("  QML PATTERN VISUALIZATION GENERATOR - FIXED VERSION")
    print("  Using mplfinance alines for structure overlay")
    print("=" * 70)
    
    patterns_csv = Path(__file__).parent.parent / "btc_backtest_labels.csv"
    output_dir = Path(__file__).parent / "pattern_charts"
    
    if not patterns_csv.exists():
        print(f"❌ Patterns CSV not found: {patterns_csv}")
        return
    
    visualizer = QMLPatternVisualizer(output_dir=str(output_dir))
    visualizer.generate_all_charts(str(patterns_csv), symbol='BTC/USDT')


if __name__ == "__main__":
    main()
