"""
Base Stage Interface
====================
Base class and shared types for pipeline stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import time
from loguru import logger


@dataclass
class StageContext:
    """Shared context passed between pipeline stages."""
    
    # Input data
    df: Any = None  # OHLCV DataFrame
    strategy_name: str = ""
    backtest_fn: Any = None
    param_grid: Dict = field(default_factory=dict)
    optimization_metric: str = "sharpe_ratio"
    
    # Stage outputs (accumulated)
    features_df: Any = None
    regime_labels: Any = None
    walk_forward_result: Any = None
    permutation_result: Any = None
    monte_carlo_result: Any = None
    bootstrap_result: Any = None
    diagnostics: Dict = field(default_factory=dict)
    
    # Metadata
    experiment_id: str = ""
    timestamp: str = ""
    config: Any = None


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    success: bool
    duration_seconds: float
    result: Any = None
    error: Optional[str] = None
    
    def __str__(self):
        status = "✅" if self.success else "❌"
        return f"{status} {self.stage_name} ({self.duration_seconds:.2f}s)"


class BaseStage(ABC):
    """
    Base class for all pipeline stages.
    
    Each stage:
    - Receives context with accumulated results
    - Performs its specific task
    - Returns updated context
    - Logs progress
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def run(self, context: StageContext) -> StageResult:
        """
        Execute stage with error handling and timing.
        
        Args:
            context: Shared pipeline context
            
        Returns:
            StageResult with success status and outputs
        """
        logger.info(f"{'='*60}")
        logger.info(f"Stage: {self.name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Execute stage logic
            result = self.execute(context)
            
            duration = time.time() - start_time
            stage_result = StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=duration,
                result=result
            )
            
            logger.info(f"✅ {self.name} completed in {duration:.2f}s")
            return stage_result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {self.name} failed: {e}")
            
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=duration,
                error=str(e)
            )
    
    @abstractmethod
    def execute(self, context: StageContext) -> Any:
        """
        Stage-specific execution logic.
        
        Implement this in subclasses.
        
        Args:
            context: Pipeline context
            
        Returns:
            Stage-specific result (stored in StageResult.result)
        """
        pass
