"""
QML Pattern Data Schemas
========================
Defines strict output formats for pattern files to ensure compatibility
between detection scripts (experiments) and the visualization system.

Version: 1.0.0
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# =============================================================================
# REQUIRED COLUMNS FOR PATTERN FILES
# =============================================================================

REQUIRED_PATTERN_COLUMNS = [
    'pattern_id',
    'pattern_type',
    'validity_score',
    'entry_price',
    'stop_loss',
    'take_profit',
    'TS_Date',
    'TS_Price',
    'P1_Date',
    'P1_Price',
    'P2_Date',
    'P2_Price',
    'P3_Date',
    'P3_Price',
    'P4_Date',
    'P4_Price',
    'P5_Date',
    'P5_Price',
]

# Date columns that need datetime parsing
DATE_COLUMNS = ['TS_Date', 'P1_Date', 'P2_Date', 'P3_Date', 'P4_Date', 'P5_Date']

# Valid pattern types
VALID_PATTERN_TYPES = ['bullish_qml', 'bearish_qml']

# Supported timeframes
SUPPORTED_TIMEFRAMES = ['1h', '4h']

# Timeframe detection thresholds (in seconds)
TIMEFRAME_THRESHOLDS = {
    '1h': (3000, 4200),    # 50-70 minutes
    '4h': (13800, 16200),  # 230-270 minutes
}


# =============================================================================
# OHLCV MASTER DATA SCHEMA
# =============================================================================

REQUIRED_OHLCV_COLUMNS = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR']

# Alternative column name mappings (for flexibility)
OHLCV_COLUMN_ALIASES = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'atr': 'ATR',
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_pattern_csv(path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a pattern CSV file against the required schema.
    
    Args:
        path: Path to the pattern CSV file
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: List[str] = []
    
    if not path.exists():
        return False, [f"File not found: {path}"]
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, [f"Failed to read CSV: {e}"]
    
    # Check required columns
    missing_cols = [col for col in REQUIRED_PATTERN_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataframe
    if len(df) == 0:
        errors.append("Pattern file is empty")
        return False, errors
    
    # Validate pattern types
    if 'pattern_type' in df.columns:
        invalid_types = df[~df['pattern_type'].isin(VALID_PATTERN_TYPES)]['pattern_type'].unique()
        if len(invalid_types) > 0:
            errors.append(f"Invalid pattern types: {list(invalid_types)}")
    
    # Validate numeric columns
    numeric_cols = ['validity_score', 'entry_price', 'stop_loss', 'take_profit',
                    'TS_Price', 'P1_Price', 'P2_Price', 'P3_Price', 'P4_Price', 'P5_Price']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} should be numeric")
    
    # Validate validity score range
    if 'validity_score' in df.columns:
        out_of_range = df[(df['validity_score'] < 0) | (df['validity_score'] > 1)]
        if len(out_of_range) > 0:
            errors.append(f"validity_score should be between 0 and 1 ({len(out_of_range)} violations)")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def infer_timeframe(df: pd.DataFrame) -> str:
    """
    Infer timeframe (1h or 4h) from pattern data by analyzing point spacing.
    
    Uses the time delta between P1 and P2 (or P2 and P3) to determine
    the underlying candle timeframe.
    
    Args:
        df: Pattern DataFrame with parsed date columns
        
    Returns:
        Inferred timeframe string ('1h' or '4h')
        
    Raises:
        ValueError: If timeframe cannot be inferred
    """
    if len(df) == 0:
        raise ValueError("Cannot infer timeframe from empty DataFrame")
    
    # Parse date columns if needed
    for col in ['P1_Date', 'P2_Date', 'P3_Date']:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    
    # Calculate median time delta between P1 and P2
    deltas = []
    for _, row in df.iterrows():
        if pd.notna(row.get('P1_Date')) and pd.notna(row.get('P2_Date')):
            delta = (row['P2_Date'] - row['P1_Date']).total_seconds()
            if delta > 0:
                deltas.append(delta)
    
    if not deltas:
        raise ValueError("No valid date pairs found for timeframe inference")
    
    # Use median delta to be robust against outliers
    median_delta = sorted(deltas)[len(deltas) // 2]
    
    # Normalize to per-bar delta (typically 1-5 bars between points)
    # We check which timeframe's range best fits
    for timeframe, (min_sec, max_sec) in TIMEFRAME_THRESHOLDS.items():
        # Check if delta fits within 1-10 bars of this timeframe
        bars = median_delta / ((min_sec + max_sec) / 2)
        if 1 <= bars <= 10:
            return timeframe
    
    # Default heuristic: if median delta > 2.5 hours, assume 4h
    if median_delta > 9000:  # 2.5 hours
        return '4h'
    return '1h'


def get_date_range(df: pd.DataFrame, padding_bars: int = 60) -> Tuple[datetime, datetime]:
    """
    Extract the date range from pattern data with optional padding.
    
    Args:
        df: Pattern DataFrame with parsed date columns
        padding_bars: Number of bars to pad on each side (for visualization context)
        
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    # Parse date columns if needed
    for col in DATE_COLUMNS:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    
    # Find min/max across all date columns
    min_date = None
    max_date = None
    
    for col in DATE_COLUMNS:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if pd.notna(col_min):
                min_date = col_min if min_date is None else min(min_date, col_min)
            if pd.notna(col_max):
                max_date = col_max if max_date is None else max(max_date, col_max)
    
    if min_date is None or max_date is None:
        raise ValueError("Could not determine date range from pattern data")
    
    # Infer timeframe for padding calculation
    try:
        timeframe = infer_timeframe(df)
        hours_per_bar = 4 if timeframe == '4h' else 1
    except ValueError:
        hours_per_bar = 1  # Default to 1h
    
    # Apply padding
    padding_hours = padding_bars * hours_per_bar
    start_date = min_date - pd.Timedelta(hours=padding_hours)
    end_date = max_date + pd.Timedelta(hours=padding_hours)
    
    return start_date.to_pydatetime(), end_date.to_pydatetime()


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV column names to the standard format.
    
    Args:
        df: DataFrame with potentially varied column names
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    # Apply aliases
    rename_map = {}
    for old_name, new_name in OHLCV_COLUMN_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            rename_map[old_name] = new_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


# =============================================================================
# DATACLASS DEFINITIONS
# =============================================================================

@dataclass
class PatternMetadata:
    """Metadata extracted from a pattern file."""
    path: Path
    pattern_count: int
    timeframe: str
    date_range: Tuple[datetime, datetime]
    bullish_count: int
    bearish_count: int
    validity_range: Tuple[float, float]
    
    @classmethod
    def from_csv(cls, path: Path) -> 'PatternMetadata':
        """Create metadata from a pattern CSV file."""
        df = pd.read_csv(path, parse_dates=DATE_COLUMNS)
        
        timeframe = infer_timeframe(df)
        date_range = get_date_range(df, padding_bars=0)
        
        bullish_count = len(df[df['pattern_type'] == 'bullish_qml'])
        bearish_count = len(df[df['pattern_type'] == 'bearish_qml'])
        
        validity_range = (
            df['validity_score'].min(),
            df['validity_score'].max()
        )
        
        return cls(
            path=path,
            pattern_count=len(df),
            timeframe=timeframe,
            date_range=date_range,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            validity_range=validity_range
        )
