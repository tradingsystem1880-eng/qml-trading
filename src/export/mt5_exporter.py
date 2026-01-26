"""
MT5 Pattern Exporter
====================
Exports QML patterns to MetaTrader 5 for auto-drawing on charts.

Usage:
    from src.export.mt5_exporter import export_pattern_to_mt5
    export_pattern_to_mt5(pattern, symbol, timeframe)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import platform


def get_mt5_files_path() -> Optional[Path]:
    """Find the MT5 MQL5/Files folder based on OS."""
    system = platform.system()

    if system == "Darwin":  # macOS
        # MT5 on Mac uses Wine
        base_paths = [
            Path.home() / "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5/MQL5/Files",
            Path.home() / "Library/Application Support/MetaTrader 5/MQL5/Files",
        ]
    elif system == "Windows":
        # Standard Windows paths
        base_paths = [
            Path.home() / "AppData/Roaming/MetaQuotes/Terminal" / "*" / "MQL5/Files",
            Path("C:/Program Files/MetaTrader 5/MQL5/Files"),
            Path("C:/Program Files (x86)/MetaTrader 5/MQL5/Files"),
        ]
    else:  # Linux
        base_paths = [
            Path.home() / ".wine/drive_c/Program Files/MetaTrader 5/MQL5/Files",
        ]

    for path in base_paths:
        if "*" in str(path):
            # Handle wildcard paths
            parent = path.parent.parent
            if parent.exists():
                for terminal_dir in parent.iterdir():
                    files_path = terminal_dir / "MQL5" / "Files"
                    if files_path.exists():
                        return files_path
        elif path.exists():
            return path

    return None


def format_timestamp(ts: Any) -> int:
    """Convert various timestamp formats to Unix timestamp."""
    if ts is None:
        return 0

    if isinstance(ts, (int, float)):
        return int(ts)

    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp())

    if isinstance(ts, str):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except:
            return 0

    return 0


def export_pattern_to_mt5(
    pattern: Any,
    symbol: str,
    timeframe: str,
    trend_swings: Optional[list] = None
) -> Dict[str, Any]:
    """
    Export a QML pattern to MT5 for auto-drawing.

    Args:
        pattern: QMLPattern object or dict with pattern data
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "4h")
        trend_swings: Optional list of trend swing points before the pattern

    Returns:
        Dict with export status and file path
    """
    # Find MT5 files folder
    mt5_path = get_mt5_files_path()
    if mt5_path is None:
        return {
            "success": False,
            "error": "MT5 Files folder not found. Is MetaTrader 5 installed?",
            "path": None
        }

    # Ensure folder exists
    mt5_path.mkdir(parents=True, exist_ok=True)

    # Extract pattern data
    if hasattr(pattern, 'entry_price'):
        # QMLPattern object
        pattern_data = {
            "pattern_id": getattr(pattern, 'id', str(datetime.now().timestamp())),
            "symbol": symbol.replace("/", ""),
            "timeframe": timeframe,
            "direction": "LONG" if str(getattr(pattern, 'direction', '')).upper() == "BEARISH" else "SHORT",
            "quality": f"{getattr(pattern, 'pattern_strength', 0):.0%}",

            # Swing points
            "p1_price": getattr(pattern.p1, 'price', 0) if hasattr(pattern, 'p1') else 0,
            "p1_time": format_timestamp(getattr(pattern.p1, 'timestamp', None)) if hasattr(pattern, 'p1') else 0,
            "p2_price": getattr(pattern.p2, 'price', 0) if hasattr(pattern, 'p2') else 0,
            "p2_time": format_timestamp(getattr(pattern.p2, 'timestamp', None)) if hasattr(pattern, 'p2') else 0,
            "p3_price": getattr(pattern.p3, 'price', 0) if hasattr(pattern, 'p3') else 0,
            "p3_time": format_timestamp(getattr(pattern.p3, 'timestamp', None)) if hasattr(pattern, 'p3') else 0,
            "p4_price": getattr(pattern.p4, 'price', 0) if hasattr(pattern, 'p4') else 0,
            "p4_time": format_timestamp(getattr(pattern.p4, 'timestamp', None)) if hasattr(pattern, 'p4') else 0,
            "p5_price": getattr(pattern.p5, 'price', 0) if hasattr(pattern, 'p5') else 0,
            "p5_time": format_timestamp(getattr(pattern.p5, 'timestamp', None)) if hasattr(pattern, 'p5') else 0,

            # Trading levels
            "entry": getattr(pattern, 'entry_price', 0),
            "stop_loss": getattr(pattern, 'stop_loss', 0),
            "take_profit": getattr(pattern, 'take_profit_1', 0),

            # Metadata
            "exported_at": datetime.now().isoformat(),
        }
    else:
        # Dict format
        pattern_data = {
            "pattern_id": pattern.get('pattern_id', str(datetime.now().timestamp())),
            "symbol": symbol.replace("/", ""),
            "timeframe": timeframe,
            "direction": pattern.get('direction', 'UNKNOWN'),
            "quality": pattern.get('quality', '0%') if isinstance(pattern.get('quality'), str) else f"{pattern.get('quality', 0):.0%}",

            # Swing points
            "p1_price": pattern.get('p1_price', 0),
            "p1_time": format_timestamp(pattern.get('p1_time')),
            "p2_price": pattern.get('p2_price', 0),
            "p2_time": format_timestamp(pattern.get('p2_time')),
            "p3_price": pattern.get('p3_price', 0),
            "p3_time": format_timestamp(pattern.get('p3_time')),
            "p4_price": pattern.get('p4_price', 0),
            "p4_time": format_timestamp(pattern.get('p4_time')),
            "p5_price": pattern.get('p5_price', 0),
            "p5_time": format_timestamp(pattern.get('p5_time')),

            # Trading levels
            "entry": pattern.get('entry', 0),
            "stop_loss": pattern.get('stop_loss', 0),
            "take_profit": pattern.get('take_profit', 0),

            # Metadata
            "exported_at": datetime.now().isoformat(),
        }

    # Add trend swings if provided
    if trend_swings and len(trend_swings) >= 2:
        for i, swing in enumerate(trend_swings[:4]):
            pattern_data[f"trend{i+1}_price"] = swing.get('price', 0)
            pattern_data[f"trend{i+1}_time"] = format_timestamp(swing.get('time'))

    # Write to file
    output_file = mt5_path / "qml_pattern.json"
    with open(output_file, 'w') as f:
        json.dump(pattern_data, f, indent=2)

    return {
        "success": True,
        "path": str(output_file),
        "pattern_id": pattern_data["pattern_id"],
        "message": f"Pattern exported to MT5. Open {symbol} {timeframe} chart with QML_Pattern_Drawer EA."
    }


def check_mt5_installed() -> Dict[str, Any]:
    """Check if MT5 is installed and return installation info."""
    mt5_path = get_mt5_files_path()

    if mt5_path:
        return {
            "installed": True,
            "files_path": str(mt5_path),
            "message": "MT5 is installed and ready for pattern export."
        }
    else:
        return {
            "installed": False,
            "files_path": None,
            "message": "MT5 not found. Please install MetaTrader 5."
        }
