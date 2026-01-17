"""
QML Trading System Configuration
================================
Centralized configuration management using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")
    
    user: str = Field(default="qml_user", description="Database user")
    password: str = Field(default="qml_secure_password", description="Database password")
    db: str = Field(default="qml_trading", description="Database name")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    
    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class ExchangeSettings(BaseSettings):
    """Exchange API settings."""
    
    model_config = SettingsConfigDict(env_prefix="BINANCE_")
    
    api_key: str = Field(default="", description="Binance API key")
    secret: str = Field(default="", description="Binance API secret")
    testnet: bool = Field(default=False, description="Use testnet")
    rate_limit: int = Field(default=1200, description="API rate limit per minute")


class TelegramSettings(BaseSettings):
    """Telegram bot settings."""
    
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_")
    
    bot_token: str = Field(default="", description="Telegram bot token")
    chat_id: str = Field(default="", description="Telegram chat ID for alerts")
    enabled: bool = Field(default=False, description="Enable Telegram alerts")
    
    @field_validator("enabled", mode="before")
    @classmethod
    def check_enabled(cls, v: bool, info) -> bool:
        """Auto-enable if token and chat_id are provided."""
        # This will be validated after all fields are set
        return v


class DetectionSettings(BaseSettings):
    """QML Pattern detection parameters."""
    
    model_config = SettingsConfigDict(env_prefix="DETECTION_")
    
    # Timeframes for detection (swing trading focused)
    timeframes: List[str] = Field(
        default=["1h", "4h", "1d"],
        description="Timeframes for pattern detection"
    )
    
    # Symbols to monitor
    symbols: List[str] = Field(
        default=[
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
            "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT",
            "MATIC/USDT", "LINK/USDT", "UNI/USDT", "ATOM/USDT",
            "LTC/USDT", "ETC/USDT", "FIL/USDT", "APT/USDT",
            "ARB/USDT", "OP/USDT", "INJ/USDT", "SUI/USDT"
        ],
        description="Symbols to monitor for patterns"
    )
    
    # ATR Configuration
    atr_period: int = Field(default=14, description="ATR calculation period")
    
    # Swing Point Detection (Timeframe-specific ATR multipliers)
    # Lower values = more swing points detected (more sensitive)
    # Higher values = fewer, more significant swings only
    swing_atr_multiplier_1h: float = Field(
        default=0.3, 
        description="ATR multiplier for swing significance on 1H"
    )
    swing_atr_multiplier_4h: float = Field(
        default=0.5,
        description="ATR multiplier for swing significance on 4H"
    )
    swing_atr_multiplier_1d: float = Field(
        default=0.7,
        description="ATR multiplier for swing significance on 1D"
    )
    
    # CHoCH Detection
    choch_break_atr_multiplier: float = Field(
        default=0.3,
        description="Minimum ATR multiple for CHoCH break confirmation"
    )
    choch_confirmation_bars: int = Field(
        default=2,
        description="Bars needed to confirm CHoCH (close-based)"
    )
    
    # BoS Detection
    bos_break_atr_multiplier: float = Field(
        default=0.5,
        description="Minimum ATR multiple for BoS break confirmation"
    )
    bos_volume_spike_threshold: float = Field(
        default=1.5,
        description="Volume spike threshold (multiplier of average)"
    )
    
    # Pattern Validation
    min_pattern_validity_score: float = Field(
        default=0.5,
        description="Minimum validity score for pattern acceptance"
    )
    min_head_depth_atr: float = Field(
        default=0.2,
        description="Minimum head depth in ATR units"
    )
    max_head_depth_atr: float = Field(
        default=10.0,
        description="Maximum head depth in ATR units"
    )
    
    # Lookback Configuration
    swing_lookback_bars: int = Field(
        default=100,
        description="Number of bars to look back for swing detection"
    )
    structure_lookback_swings: int = Field(
        default=6,
        description="Number of swings to consider for structure analysis"
    )
    
    def get_swing_atr_multiplier(self, timeframe: str) -> float:
        """Get timeframe-specific ATR multiplier."""
        multipliers = {
            "1h": self.swing_atr_multiplier_1h,
            "4h": self.swing_atr_multiplier_4h,
            "1d": self.swing_atr_multiplier_1d,
        }
        return multipliers.get(timeframe, 1.0)


class TradingSettings(BaseSettings):
    """Trading and risk management parameters."""
    
    model_config = SettingsConfigDict(env_prefix="TRADING_")
    
    # Risk Management
    default_risk_reward: float = Field(
        default=3.0,
        description="Default risk-to-reward ratio"
    )
    stop_loss_atr_multiplier: float = Field(
        default=1.5,
        description="Stop loss distance in ATR units below/above pattern"
    )
    
    # Position Sizing (for semi-automated guidance)
    max_risk_per_trade_pct: float = Field(
        default=1.0,
        description="Maximum risk per trade as percentage of portfolio"
    )
    max_concurrent_positions: int = Field(
        default=5,
        description="Maximum number of concurrent positions"
    )
    max_correlation_exposure: float = Field(
        default=0.7,
        description="Maximum correlation between concurrent positions"
    )


class MLSettings(BaseSettings):
    """Machine learning model settings."""
    
    model_config = SettingsConfigDict(env_prefix="ML_")
    
    model_path: str = Field(
        default="./models",
        description="Path to store/load models"
    )
    production_model_version: Optional[str] = Field(
        default=None,
        description="Currently deployed model version"
    )
    
    # Training Configuration
    walk_forward_splits: int = Field(
        default=5,
        description="Number of walk-forward splits"
    )
    purge_gap_bars: int = Field(
        default=10,
        description="Gap between train/test to prevent leakage"
    )
    
    # XGBoost Defaults
    xgb_n_estimators: int = Field(default=100, description="Number of trees")
    xgb_max_depth: int = Field(default=5, description="Maximum tree depth")
    xgb_learning_rate: float = Field(default=0.1, description="Learning rate")
    xgb_min_child_weight: int = Field(default=5, description="Minimum child weight")
    
    # Confidence Thresholds
    min_prediction_confidence: float = Field(
        default=0.6,
        description="Minimum ML confidence for alerts"
    )


class BacktestSettings(BaseSettings):
    """Backtesting configuration."""
    
    model_config = SettingsConfigDict(env_prefix="BACKTEST_")
    
    start_date: str = Field(
        default="2020-01-01",
        description="Backtest start date"
    )
    end_date: str = Field(
        default="2024-12-31",
        description="Backtest end date"
    )
    
    # Transaction Costs
    commission_pct: float = Field(
        default=0.1,
        description="Commission percentage per trade"
    )
    slippage_pct: float = Field(
        default=0.05,
        description="Expected slippage percentage"
    )
    
    # Initial Capital
    initial_capital: float = Field(
        default=100000.0,
        description="Initial capital for backtesting"
    )


class Settings(BaseSettings):
    """Main settings class aggregating all configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = Field(default="QML Trading System", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to avoid re-reading environment on every call.
    Call get_settings.cache_clear() to reload settings.
    """
    return Settings()


# Convenience accessor
settings = get_settings()

