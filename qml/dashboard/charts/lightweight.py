"""
TradingView Lightweight Charts Integration
==========================================
Premium chart rendering for QML patterns.

Uses the battle-tested TradingView Lightweight Charts library (7.5k+ stars).
Renders professional charts with patterns, zones, and trade boxes.

Usage:
    from qml.dashboard.charts import LightweightChart, render_pattern_chart
    
    # Simple rendering
    html = render_pattern_chart(df, pattern)
    st.components.v1.html(html, height=600)
    
    # Or with more control
    chart = LightweightChart(theme="dark")
    html = chart.render(df, pattern, title="BTC/USDT QML Pattern")
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger


class LightweightChart:
    """
    TradingView Lightweight Charts wrapper.
    
    Renders professional charts with:
    - Candlesticks (green/red)
    - Volume bars
    - Pattern annotations (BOS, L, S, R labels)
    - Trend lines connecting swing points
    - Support/resistance zones
    - Trade entry/exit boxes
    """
    
    def __init__(self, theme: str = "dark"):
        """
        Initialize chart renderer.
        
        Args:
            theme: Color theme ("dark" or "light")
        """
        self.theme = theme
        self._template = None
        self._theme_config = None
        
        # Load resources
        self._load_template()
        self._load_theme()
        
        logger.debug(f"LightweightChart initialized with {theme} theme")
    
    def _load_template(self):
        """Load HTML template."""
        template_path = Path(__file__).parent / "templates" / "chart.html"
        
        if template_path.exists():
            self._template = template_path.read_text()
            logger.debug(f"Loaded template from {template_path}")
        else:
            logger.warning(f"Template not found at {template_path}, using inline")
            self._template = self._get_inline_template()
    
    def _load_theme(self):
        """Load theme configuration."""
        theme_path = Path(__file__).parent / "styles" / f"{self.theme}_theme.json"
        
        if theme_path.exists():
            self._theme_config = json.loads(theme_path.read_text())
            logger.debug(f"Loaded theme from {theme_path}")
        else:
            logger.warning(f"Theme not found at {theme_path}, using default")
            self._theme_config = self._get_default_theme()
    
    def render(
        self,
        df: pd.DataFrame,
        pattern: Optional[Dict] = None,
        title: str = "QML Pattern"
    ) -> str:
        """
        Render complete pattern chart.
        
        Args:
            df: OHLCV DataFrame (columns: time, open, high, low, close, volume)
            pattern: Pattern data dict with swing points, entry, etc.
            title: Chart title
            
        Returns:
            HTML string ready for st.components.html()
        """
        logger.info(f"Rendering chart: {title}")
        logger.debug(f"Data: {len(df)} candles")
        
        # Prepare data
        candlesticks = self._format_candlesticks(df)
        volume = self._format_volume(df)
        
        # Extract pattern elements
        markers = []
        lines = []
        zones = []
        pattern_info = None
        
        if pattern:
            markers = self._create_markers(pattern, df)
            lines = self._create_trend_lines(pattern, df)
            zones = self._create_zones(pattern)
            pattern_info = self._create_pattern_info(pattern)
            
            logger.debug(f"Pattern elements: {len(markers)} markers, {len(lines)} lines, {len(zones)} zones")
        
        # Build config
        config = {
            "candlesticks": candlesticks,
            "volume": volume,
            "markers": markers,
            "lines": lines,
            "zones": zones,
            "patternInfo": pattern_info,
            "theme": self._theme_config
        }
        
        # Render HTML
        html = self._render_html(config, title)
        
        logger.success(f"Chart rendered successfully")
        return html
    
    def _format_candlesticks(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to TradingView candlestick format."""
        candlesticks = []
        
        for _, row in df.iterrows():
            # Handle timestamp
            if 'time' in df.columns:
                time_col = row['time']
            elif 'timestamp' in df.columns:
                time_col = row['timestamp']
            else:
                time_col = row.name  # Use index
            
            # Convert to unix timestamp
            if isinstance(time_col, (pd.Timestamp, datetime)):
                timestamp = int(time_col.timestamp())
            else:
                timestamp = int(time_col)
            
            candlesticks.append({
                "time": timestamp,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })
        
        return candlesticks
    
    def _format_volume(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to TradingView volume format."""
        if 'volume' not in df.columns:
            return []
        
        volume = []
        
        for _, row in df.iterrows():
            # Handle timestamp
            if 'time' in df.columns:
                time_col = row['time']
            elif 'timestamp' in df.columns:
                time_col = row['timestamp']
            else:
                time_col = row.name
            
            if isinstance(time_col, (pd.Timestamp, datetime)):
                timestamp = int(time_col.timestamp())
            else:
                timestamp = int(time_col)
            
            # Color based on close vs open
            color = '#22c55e' if row['close'] >= row['open'] else '#ef4444'
            
            volume.append({
                "time": timestamp,
                "value": float(row["volume"]),
                "color": color
            })
        
        return volume
    
    def _create_markers(self, pattern: Dict, df: pd.DataFrame) -> List[Dict]:
        """Create markers for swing points and pattern labels."""
        markers = []
        
        # Swing point labels: P1, P2, P3, P4, P5
        swing_points = pattern.get("swing_points", [])
        
        for i, point in enumerate(swing_points, 1):
            label = f"P{i}"
            time = point.get("time")
            price = point.get("price")
            
            if time and price:
                if isinstance(time, (pd.Timestamp, datetime)):
                    timestamp = int(time.timestamp())
                else:
                    timestamp = int(time)
                
                # Determine position and color
                is_high = point.get("type", "high") == "high"
                
                markers.append({
                    "time": timestamp,
                    "position": "aboveBar" if is_high else "belowBar",
                    "color": "#38BDF8",  # Cyan
                    "shape": "circle",
                    "text": label,
                    "size": 1
                })
        
        # BOS (Break of Structure) label
        if pattern.get("bos_time") and pattern.get("bos_price"):
            bos_time = pattern["bos_time"]
            if isinstance(bos_time, (pd.Timestamp, datetime)):
                bos_timestamp = int(bos_time.timestamp())
            else:
                bos_timestamp = int(bos_time)
            
            markers.append({
                "time": bos_timestamp,
                "position": "aboveBar",
                "color": "#ef4444",  # Red
                "shape": "text",
                "text": "BOS",
                "size": 2
            })
        
        # Entry marker
        if pattern.get("entry_time") and pattern.get("entry_price"):
            entry_time = pattern["entry_time"]
            if isinstance(entry_time, (pd.Timestamp, datetime)):
                entry_timestamp = int(entry_time.timestamp())
            else:
                entry_timestamp = int(entry_time)
            
            is_bullish = pattern.get("type", "bullish") == "bullish"
            
            markers.append({
                "time": entry_timestamp,
                "position": "belowBar" if is_bullish else "aboveBar",
                "color": "#22c55e" if is_bullish else "#ef4444",
                "shape": "arrowUp" if is_bullish else "arrowDown",
                "text": "ENTRY",
                "size": 2
            })
        
        return markers
    
    def _create_trend_lines(self, pattern: Dict, df: pd.DataFrame) -> List[Dict]:
        """Create trend lines connecting swing points."""
        lines = []
        
        swing_points = pattern.get("swing_points", [])
        
        if len(swing_points) >= 2:
            # Connect swing points with lines
            line_data = []
            
            for point in swing_points:
                time = point.get("time")
                price = point.get("price")
                
                if time and price:
                    if isinstance(time, (pd.Timestamp, datetime)):
                        timestamp = int(time.timestamp())
                    else:
                        timestamp = int(time)
                    
                    line_data.append({
                        "time": timestamp,
                        "value": float(price)
                    })
            
            if line_data:
                lines.append({
                    "data": line_data,
                    "color": "#38BDF8",  # Cyan
                    "width": 2,
                    "style": 0  # Solid
                })
        
        return lines
    
    def _create_zones(self, pattern: Dict) -> List[Dict]:
        """Create support/resistance zones."""
        zones = []
        
        # Get zone data from pattern
        pattern_zones = pattern.get("zones", [])
        
        for zone in pattern_zones:
            zones.append({
                "high": float(zone.get("high", 0)),
                "low": float(zone.get("low", 0)),
                "color": zone.get("color", "#38BDF8"),
                "title": zone.get("label", "")
            })
        
        # Create zone for entry/SL/TP if available
        if pattern.get("entry_price") and pattern.get("stop_loss"):
            entry = float(pattern["entry_price"])
            sl = float(pattern["stop_loss"])
            
            zones.append({
                "high": max(entry, sl),
                "low": min(entry, sl),
                "color": "#ef4444",  # Red for risk zone
                "title": "SL"
            })
        
        if pattern.get("entry_price") and pattern.get("take_profit"):
            entry = float(pattern["entry_price"])
            tp = float(pattern["take_profit"])
            
            zones.append({
                "high": max(entry, tp),
                "low": min(entry, tp),
                "color": "#22c55e",  # Green for profit zone
                "title": "TP"
            })
        
        return zones
    
    def _create_pattern_info(self, pattern: Dict) -> Dict:
        """Create pattern info for display panel."""
        return {
            "type": pattern.get("type", "unknown"),
            "validity": f"{pattern.get('validity', 0) * 100:.0f}",
            "entry": f"{pattern.get('entry_price', 0):.2f}",
            "riskReward": f"{pattern.get('risk_reward', 0):.2f}"
        }
    
    def _render_html(self, config: Dict, title: str) -> str:
        """Render final HTML with config injected."""
        # Convert config to JSON
        config_json = json.dumps(config, default=str)
        
        # Replace placeholders in template
        html = self._template.replace("{{ CONFIG_JSON }}", config_json)
        html = html.replace("{{ CHART_TITLE }}", title)
        
        return html
    
    def _get_inline_template(self) -> str:
        """Fallback inline template if file not found."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { margin: 0; background: #0F1419; font-family: sans-serif; }
        #chart-container { width: 100%; height: 600px; }
        .chart-title { position: absolute; top: 10px; left: 10px; color: #D9D9D9; z-index: 10; }
    </style>
</head>
<body>
    <div id="chart-container">
        <div class="chart-title">{{ CHART_TITLE }}</div>
    </div>
    <script>
        const config = {{ CONFIG_JSON }};
        const chart = LightweightCharts.createChart(document.getElementById('chart-container'), {
            ...config.theme,
            width: window.innerWidth,
            height: 600
        });
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#22c55e', downColor: '#ef4444',
            borderUpColor: '#22c55e', borderDownColor: '#ef4444',
            wickUpColor: '#22c55e', wickDownColor: '#ef4444'
        });
        candlestickSeries.setData(config.candlesticks);
        if (config.markers) candlestickSeries.setMarkers(config.markers);
        chart.timeScale().fitContent();
    </script>
</body>
</html>
'''
    
    def _get_default_theme(self) -> Dict:
        """Default dark theme config."""
        return {
            "layout": {"background": {"color": "#0F1419"}, "textColor": "#D9D9D9"},
            "grid": {"vertLines": {"color": "#1E252E"}, "horzLines": {"color": "#1E252E"}},
            "crosshair": {"mode": 1},
            "priceScale": {"borderColor": "#2B3139"},
            "timeScale": {"borderColor": "#2B3139", "timeVisible": True}
        }


def render_pattern_chart(
    df: pd.DataFrame,
    pattern: Optional[Dict] = None,
    title: str = "QML Pattern",
    theme: str = "dark"
) -> str:
    """
    Convenience function to render a pattern chart.
    
    Args:
        df: OHLCV DataFrame
        pattern: Pattern data dict
        title: Chart title
        theme: Color theme
        
    Returns:
        HTML string for st.components.html()
    
    Example:
        html = render_pattern_chart(df, pattern, "BTC/USDT QML")
        st.components.v1.html(html, height=600)
    """
    chart = LightweightChart(theme=theme)
    return chart.render(df, pattern, title)
