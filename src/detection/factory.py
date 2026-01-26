"""
Detector Factory
================
Simple factory function to instantiate the correct detector based on method name.
This provides a clean interface for the rest of the system to get detectors
without needing to know the specific class names.
"""

from typing import Any, Dict, Optional, Union

from src.detection.base import BaseDetector, DetectorConfig


def get_detector(
    method_name: str,
    config: Optional[Union[Dict[str, Any], DetectorConfig]] = None
) -> BaseDetector:
    """
    Factory function to get the appropriate detector instance.

    This is the primary interface for obtaining detector instances.
    It handles configuration and returns the correct detector based
    on the method name.

    Supported methods:
    - "rolling_window" or "v1": RollingWindowDetector (v1.1.0)
    - "atr" or "atr_directional_change" or "v2": ATRDetector (v2.0.0)
    - "historical" or "batch" or "backtest": HistoricalSwingDetector (Phase 7.5)

    Args:
        method_name: Name of the detection method. Case-insensitive.
                    Accepts: "rolling_window", "v1", "atr", "atr_directional_change", "v2",
                             "historical", "batch", "backtest"
        config: Configuration dictionary or DetectorConfig instance.
               If dict, will be converted to appropriate config class.

    Returns:
        Configured detector instance (subclass of BaseDetector)

    Raises:
        ValueError: If method_name is not recognized

    Examples:
        # Get ATR detector with default config
        detector = get_detector("atr")

        # Get rolling window detector with custom config
        detector = get_detector("rolling_window", {"window_size": 150, "step_size": 8})

        # Get detector using config object
        from src.detection.v2_atr import ATRDetectorConfig
        config = ATRDetectorConfig(atr_lookback=20)
        detector = get_detector("v2", config)

        # Get historical detector for backtesting
        detector = get_detector("historical")
    """
    # Normalize method name
    method = method_name.lower().strip()
    
    # Map method names to detector classes
    if method in ("rolling_window", "v1", "rolling", "v1_rolling"):
        from src.detection.v1_rolling import RollingWindowDetector, RollingWindowConfig
        
        if config is None:
            detector_config = RollingWindowConfig()
        elif isinstance(config, dict):
            detector_config = RollingWindowConfig(**config)
        elif isinstance(config, RollingWindowConfig):
            detector_config = config
        elif isinstance(config, DetectorConfig):
            # Convert base config to specific config
            detector_config = RollingWindowConfig(
                min_validity_score=config.min_validity_score,
                atr_period=config.atr_period,
                stop_loss_atr_mult=config.stop_loss_atr_mult,
                take_profit_atr_mult=config.take_profit_atr_mult,
            )
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
        
        return RollingWindowDetector(detector_config)
    
    elif method in ("atr", "atr_directional_change", "v2", "v2_atr"):
        from src.detection.v2_atr import ATRDetector, ATRDetectorConfig
        
        if config is None:
            detector_config = ATRDetectorConfig()
        elif isinstance(config, dict):
            detector_config = ATRDetectorConfig(**config)
        elif isinstance(config, ATRDetectorConfig):
            detector_config = config
        elif isinstance(config, DetectorConfig):
            # Convert base config to specific config
            detector_config = ATRDetectorConfig(
                min_validity_score=config.min_validity_score,
                atr_period=config.atr_period,
                stop_loss_atr_mult=config.stop_loss_atr_mult,
                take_profit_atr_mult=config.take_profit_atr_mult,
            )
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
        
        return ATRDetector(detector_config)

    elif method in ("historical", "batch", "backtest"):
        # Phase 7.5: Historical/batch detector for backtesting
        from src.detection.historical_detector import HistoricalSwingDetector
        from src.detection.config import SwingDetectionConfig

        if config is None:
            swing_config = SwingDetectionConfig()
        elif isinstance(config, dict):
            swing_config = SwingDetectionConfig(**config.get("swing", config))
        elif isinstance(config, DetectorConfig):
            swing_config = SwingDetectionConfig(
                atr_period=config.atr_period,
            )
        else:
            swing_config = SwingDetectionConfig()

        return HistoricalSwingDetector(swing_config)

    else:
        available = [
            "rolling_window (v1)",
            "atr_directional_change (v2)",
            "historical (Phase 7.5 batch detector)"
        ]
        raise ValueError(
            f"Unknown detection method: '{method_name}'. "
            f"Available methods: {', '.join(available)}"
        )


def list_available_detectors() -> Dict[str, Dict[str, str]]:
    """
    List all available detector methods with their descriptions.

    Returns:
        Dictionary mapping method names to info dicts

    Example:
        >>> list_available_detectors()
        {
            'rolling_window': {
                'version': '1.1.0',
                'description': 'Fixed-size sliding window...',
                'aliases': ['v1', 'rolling']
            },
            'atr_directional_change': {
                'version': '2.0.0',
                'description': 'ATR-driven detection...',
                'aliases': ['v2', 'atr']
            },
            'historical': {
                'version': '7.5.0',
                'description': 'Idempotent batch detector...',
                'aliases': ['batch', 'backtest']
            }
        }
    """
    return {
        'rolling_window': {
            'version': '1.1.0',
            'description': (
                'Fixed-size sliding window approach. Runs detection every N bars. '
                'Simple but may miss patterns between step intervals.'
            ),
            'aliases': ['v1', 'rolling', 'v1_rolling'],
            'recommended_for': 'Backward compatibility, testing against old results',
        },
        'atr_directional_change': {
            'version': '2.0.0',
            'description': (
                'ATR-driven detection triggered at confirmed swing points. '
                'More accurate and aligned with market structure.'
            ),
            'aliases': ['v2', 'atr', 'v2_atr'],
            'recommended_for': 'Production use, new development',
        },
        'historical': {
            'version': '7.5.0',
            'description': (
                'Idempotent batch detector using scipy argrelextrema. '
                'Uses lookforward data (not causal). Z-score based filtering.'
            ),
            'aliases': ['batch', 'backtest'],
            'recommended_for': 'Backtesting, historical analysis, strategy research',
        }
    }


def get_default_detector() -> BaseDetector:
    """
    Get the default (recommended) detector with default configuration.
    
    Currently returns ATR Detector (v2.0.0) as it's the primary algorithm.
    
    Returns:
        Configured default detector instance
    """
    return get_detector("atr")
