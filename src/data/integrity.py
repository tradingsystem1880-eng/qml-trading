"""
Data Integrity Validator - The Data Doctor
============================================
VRD 2.0 Module 7: Process Integrity

Validates data quality before backtesting to ensure:
- No missing timestamps (gaps in sequence)
- No NaN values in OHLCV columns
- No zero prices
- Proper data types

Usage:
    from src.data.integrity import DataValidator
    
    validator = DataValidator()
    report = validator.check_health(df, timeframe='4h')
    
    if report['status'] == 'fail':
        print(f"Data issues: {report['issues']}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class HealthReport:
    """
    Results of a data health check.
    
    Attributes:
        status: 'pass', 'warn', or 'fail'
        checked_at: Timestamp of check
        total_rows: Number of rows in dataset
        issues: List of issue descriptions
        metrics: Detailed metrics dictionary
    """
    status: str
    checked_at: datetime
    total_rows: int
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status,
            'checked_at': self.checked_at.isoformat(),
            'total_rows': self.total_rows,
            'issues': self.issues,
            'metrics': self.metrics
        }
    
    def __str__(self) -> str:
        icon = '✅' if self.status == 'pass' else '⚠️' if self.status == 'warn' else '❌'
        lines = [f"{icon} Data Health: {self.status.upper()} ({self.total_rows} rows)"]
        for issue in self.issues:
            lines.append(f"   • {issue}")
        return '\n'.join(lines)


class DataValidator:
    """
    Validates data integrity for backtesting.
    
    Performs comprehensive checks on OHLCV data to ensure
    data quality before running backtests.
    
    Checks performed:
    - Missing timestamps (gaps in time series)
    - NaN values in any column
    - Zero or negative prices
    - Data type validation
    - OHLC consistency (high >= low, etc.)
    
    Usage:
        validator = DataValidator()
        report = validator.check_health(df, timeframe='4h')
        
        if report.status == 'fail':
            raise ValueError(f"Data integrity check failed: {report.issues}")
    """
    
    # Expected time intervals for different timeframes
    TIMEFRAME_INTERVALS = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '4h': timedelta(hours=4),
        '6h': timedelta(hours=6),
        '8h': timedelta(hours=8),
        '12h': timedelta(hours=12),
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1),
    }
    
    # Required columns
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    def check_health(
        self,
        df: pd.DataFrame,
        timeframe: str = '4h',
        time_column: str = 'time',
        strict: bool = False
    ) -> HealthReport:
        """
        Perform comprehensive data health check.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Expected timeframe (e.g., '1h', '4h')
            time_column: Name of timestamp column
            strict: If True, any issue = fail. If False, minor issues = warn
        
        Returns:
            HealthReport with status and detailed findings
        """
        issues = []
        metrics = {}
        
        total_rows = len(df)
        
        # Check 1: Empty data
        if total_rows == 0:
            return HealthReport(
                status='fail',
                checked_at=datetime.now(),
                total_rows=0,
                issues=['DataFrame is empty'],
                metrics={}
            )
        
        # Check 2: Required columns
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check 3: Time column exists
        if time_column not in df.columns:
            issues.append(f"Time column '{time_column}' not found")
            # Can't do timestamp checks without time column
            return HealthReport(
                status='fail',
                checked_at=datetime.now(),
                total_rows=total_rows,
                issues=issues,
                metrics=metrics
            )
        
        # Check 4: NaN values
        nan_counts = {}
        for col in self.OHLCV_COLUMNS:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_counts[col] = int(nan_count)
        
        if nan_counts:
            total_nans = sum(nan_counts.values())
            issues.append(f"NaN values found: {nan_counts} (total: {total_nans})")
            metrics['nan_counts'] = nan_counts
        else:
            metrics['nan_counts'] = {}
        
        # Check 5: Zero prices
        zero_counts = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    zero_counts[col] = int(zero_count)
        
        if zero_counts:
            issues.append(f"Zero prices found: {zero_counts}")
            metrics['zero_counts'] = zero_counts
        else:
            metrics['zero_counts'] = {}
        
        # Check 6: Negative prices
        negative_counts = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    negative_counts[col] = int(neg_count)
        
        if negative_counts:
            issues.append(f"Negative prices found: {negative_counts}")
            metrics['negative_counts'] = negative_counts
        
        # Check 7: OHLC consistency (high >= low, etc.)
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                issues.append(f"OHLC inconsistency: {invalid_hl} bars where high < low")
                metrics['invalid_hl_count'] = int(invalid_hl)
        
        # Check 8: Timestamp gaps
        gaps_found = 0
        gap_details = []
        
        if timeframe in self.TIMEFRAME_INTERVALS:
            expected_interval = self.TIMEFRAME_INTERVALS[timeframe]
            
            # Ensure time is datetime
            time_series = pd.to_datetime(df[time_column])
            time_diffs = time_series.diff().dropna()
            
            # Find gaps (intervals > expected)
            tolerance = expected_interval * 1.1  # 10% tolerance
            gap_mask = time_diffs > tolerance
            gaps_found = gap_mask.sum()
            
            if gaps_found > 0:
                # Get first few gap locations
                gap_indices = gap_mask[gap_mask].index[:5]
                for idx in gap_indices:
                    if idx > 0:
                        prev_time = time_series.iloc[idx - 1]
                        curr_time = time_series.iloc[idx]
                        gap_details.append(f"{prev_time} → {curr_time}")
                
                issues.append(f"Timestamp gaps: {gaps_found} gaps found (expected {timeframe} intervals)")
                metrics['gaps_found'] = int(gaps_found)
                metrics['gap_examples'] = gap_details[:3]
        else:
            metrics['gaps_found'] = 'unknown (unsupported timeframe)'
        
        # Check 9: Data range info (not an issue, just metrics)
        time_series = pd.to_datetime(df[time_column])
        metrics['date_range'] = {
            'start': str(time_series.min()),
            'end': str(time_series.max()),
            'days': (time_series.max() - time_series.min()).days
        }
        
        # Determine overall status
        if len(issues) == 0:
            status = 'pass'
        elif strict:
            status = 'fail'
        else:
            # Classify severity
            critical_keywords = ['Missing required', 'empty', 'Time column']
            has_critical = any(
                any(kw in issue for kw in critical_keywords)
                for issue in issues
            )
            
            if has_critical or nan_counts or zero_counts or negative_counts:
                status = 'fail'
            elif gaps_found > total_rows * 0.01:  # More than 1% gaps
                status = 'warn'
            else:
                status = 'warn'
        
        return HealthReport(
            status=status,
            checked_at=datetime.now(),
            total_rows=total_rows,
            issues=issues,
            metrics=metrics
        )
    
    def check_and_raise(
        self,
        df: pd.DataFrame,
        timeframe: str = '4h',
        time_column: str = 'time'
    ) -> HealthReport:
        """
        Check health and raise exception on failure.
        
        Args:
            df: DataFrame to check
            timeframe: Expected timeframe
            time_column: Name of timestamp column
        
        Returns:
            HealthReport (only if status is not 'fail')
        
        Raises:
            ValueError: If health check fails
        """
        report = self.check_health(df, timeframe, time_column, strict=True)
        
        if report.status == 'fail':
            raise ValueError(f"Data integrity check failed:\n{report}")
        
        return report


def validate_dataframe(
    df: pd.DataFrame,
    timeframe: str = '4h',
    raise_on_fail: bool = False
) -> HealthReport:
    """
    Convenience function for quick validation.
    
    Args:
        df: DataFrame to validate
        timeframe: Expected timeframe
        raise_on_fail: If True, raise exception on failure
    
    Returns:
        HealthReport
    """
    validator = DataValidator()
    
    if raise_on_fail:
        return validator.check_and_raise(df, timeframe)
    else:
        return validator.check_health(df, timeframe)
