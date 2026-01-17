"""
QML Configuration
=================
Unified configuration system using Pydantic.
Single source of truth for all settings.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data loading configuration."""
    base_dir: Path = Path("user_data/data")
    default_exchange: str = "binance"
    default_symbols: List[str] = ["BTC/USDT", "ETH/USDT"]
    default_timeframes: List[str] = ["4h", "1d"]
    cache_ttl_minutes: int = 15


class DetectionConfig(BaseModel):
    """Pattern detection configuration."""
    atr_period: int = 14
    swing_lookback: int = 10
    min_validity: float = 0.5
    pattern_types: List[str] = ["bullish_qml", "bearish_qml"]


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    initial_capital: float = 10000.0
    position_size_pct: float = 2.0
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    max_trades: int = 1000


class ValidationConfig(BaseModel):
    """Validation configuration."""
    walk_forward_folds: int = 5
    purge_bars: int = 5
    embargo_bars: int = 5
    n_permutations: int = 1000
    n_monte_carlo: int = 10000
    confidence_threshold: float = 0.05


class DashboardConfig(BaseModel):
    """Dashboard UI configuration."""
    theme: str = "dark"
    refresh_interval: int = 60
    default_symbol: str = "BTC/USDT"
    default_timeframe: str = "4h"


class QMLConfig(BaseModel):
    """
    Master configuration for QML Trading System.
    
    All settings in one place - follows Freqtrade pattern.
    
    Usage:
        config = QMLConfig()
        config.data.default_exchange = "binance"
        config.detection.atr_period = 14
    """
    data: DataConfig = Field(default_factory=DataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    # Paths
    user_data_dir: Path = Path("user_data")
    results_dir: Path = Path("user_data/results")
    models_dir: Path = Path("user_data/models")
    configs_dir: Path = Path("user_data/configs")
    
    class Config:
        extra = "allow"
    
    @classmethod
    def from_yaml(cls, path: str) -> "QMLConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Global default config
default_config = QMLConfig()
