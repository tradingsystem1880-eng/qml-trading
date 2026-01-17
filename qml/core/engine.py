"""
QML Engine
==========
Central hub for the QML Trading System.

The "brain" that orchestrates:
- Pattern detection
- Backtesting
- Validation
- Reporting

This is the main interface that the dashboard uses.

Usage:
    from qml import QMLEngine
    
    engine = QMLEngine()
    
    # Detect patterns
    patterns = engine.detect_patterns("BTC/USDT", "4h", days=365)
    
    # Run backtest
    results = engine.backtest(patterns)
    
    # Validate strategy
    validation = engine.validate(results)
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from qml.core.config import QMLConfig, default_config


@dataclass
class PatternResult:
    """Result from pattern detection."""
    patterns: List[Dict]
    symbol: str
    timeframe: str
    total_found: int
    bullish_count: int
    bearish_count: int


@dataclass
class BacktestResult:
    """Result from backtesting."""
    trades: pd.DataFrame
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


@dataclass 
class ValidationResult:
    """Result from validation."""
    verdict: str  # "DEPLOY", "CAUTION", "REJECT"
    confidence_score: float
    p_value: float
    is_significant: bool
    reasons: List[str]


class QMLEngine:
    """
    Central QML Trading Engine.
    
    Orchestrates all system operations in a clean, unified interface.
    """
    
    def __init__(self, config: Optional[QMLConfig] = None):
        """
        Initialize QML Engine.
        
        Args:
            config: Configuration (uses default if not provided)
        """
        self.config = config or default_config
        logger.info("QML Engine initialized")
    
    def detect_patterns(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365,
        min_validity: Optional[float] = None
    ) -> PatternResult:
        """
        Detect QML patterns in historical data.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "4h")
            days: Days of history to analyze
            min_validity: Minimum pattern validity score
            
        Returns:
            PatternResult with detected patterns
        """
        from qml.core.data import DataLoader
        from qml.strategy.detector import PatternDetector
        
        # Load data
        loader = DataLoader(config=self.config)
        df = loader.load_ohlcv(symbol, timeframe, days=days)
        
        # Detect patterns
        detector = PatternDetector(config=self.config)
        patterns = detector.detect(df, symbol, timeframe)
        
        # Filter by validity
        validity_threshold = min_validity or self.config.detection.min_validity
        patterns = [p for p in patterns if p.get("validity", 0) >= validity_threshold]
        
        # Count types
        bullish = sum(1 for p in patterns if p.get("type") == "bullish")
        bearish = sum(1 for p in patterns if p.get("type") == "bearish")
        
        logger.info(f"Detected {len(patterns)} patterns: {bullish} bullish, {bearish} bearish")
        
        return PatternResult(
            patterns=patterns,
            symbol=symbol,
            timeframe=timeframe,
            total_found=len(patterns),
            bullish_count=bullish,
            bearish_count=bearish
        )
    
    def backtest(
        self,
        patterns: PatternResult,
        initial_capital: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on detected patterns.
        
        Args:
            patterns: Result from detect_patterns()
            initial_capital: Starting capital
            
        Returns:
            BacktestResult with performance metrics
        """
        from qml.backtest.engine import BacktestEngine
        
        capital = initial_capital or self.config.backtest.initial_capital
        
        engine = BacktestEngine(config=self.config)
        result = engine.run(patterns.patterns, initial_capital=capital)
        
        logger.info(f"Backtest complete: Sharpe={result.sharpe_ratio:.2f}, Win={result.win_rate:.1%}")
        
        return result
    
    def validate(self, backtest: BacktestResult) -> ValidationResult:
        """
        Run statistical validation on backtest results.
        
        Args:
            backtest: Result from backtest()
            
        Returns:
            ValidationResult with verdict and statistics
        """
        from qml.validation.validator import Validator
        
        validator = Validator(config=self.config)
        result = validator.run(backtest)
        
        logger.info(f"Validation: {result.verdict} (confidence={result.confidence_score:.1f})")
        
        return result
    
    def run_full_pipeline(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Pattern Detection → Backtest → Validation
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Days of history
            
        Returns:
            Complete results dictionary
        """
        logger.info(f"Running full pipeline: {symbol} {timeframe}")
        
        # Step 1: Detect patterns
        patterns = self.detect_patterns(symbol, timeframe, days)
        
        # Step 2: Backtest
        backtest = self.backtest(patterns)
        
        # Step 3: Validate
        validation = self.validate(backtest)
        
        return {
            "patterns": patterns,
            "backtest": backtest,
            "validation": validation,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status for dashboard."""
        return {
            "version": "2.0.0",
            "config": self.config.model_dump(),
            "status": "ready"
        }
