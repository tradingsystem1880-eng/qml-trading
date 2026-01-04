"""
Parameter Sensitivity Analysis
==============================
Tests parameters across ±40% range to identify stability islands vs cliffs.
Outputs data structures ready for 3D surface plotting.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SensitivityConfig:
    """Configuration for parameter sensitivity analysis."""
    
    # Variation range (e.g., 0.4 = ±40%)
    variation_range: float = 0.4
    
    # Number of steps per parameter
    n_steps: int = 20
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "sharpe_ratio",
        "profit_factor",
        "max_drawdown_pct",
        "win_rate",
        "total_trades",
    ])
    
    # Primary optimization metric
    primary_metric: str = "sharpe_ratio"


@dataclass
class ParameterSweep:
    """Result of a single parameter sweep."""
    
    param_name: str
    param_values: np.ndarray
    metric_values: Dict[str, np.ndarray]
    base_value: float
    
    @property
    def best_value(self) -> float:
        """Best parameter value for primary metric."""
        idx = np.argmax(self.metric_values.get("sharpe_ratio", self.param_values))
        return float(self.param_values[idx])
    
    @property
    def stability_score(self) -> float:
        """
        Stability score (0-1).
        Higher = more stable (performance doesn't vary much).
        """
        sharpe_vals = self.metric_values.get("sharpe_ratio", np.array([0]))
        if len(sharpe_vals) < 2 or np.std(sharpe_vals) == 0:
            return 1.0
        
        # Coefficient of variation (inverted for stability)
        mean_val = np.mean(sharpe_vals)
        if mean_val == 0:
            return 0.0
        
        cv = np.std(sharpe_vals) / abs(mean_val)
        return float(max(0, 1 - cv))


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis result."""
    
    # Single parameter sweeps
    single_sweeps: Dict[str, ParameterSweep]
    
    # 2D grid results (for 3D surface plots)
    grid_results: Optional[Dict[str, np.ndarray]] = None
    grid_param1_name: Optional[str] = None
    grid_param1_values: Optional[np.ndarray] = None
    grid_param2_name: Optional[str] = None
    grid_param2_values: Optional[np.ndarray] = None
    
    # Configuration used
    config: SensitivityConfig = field(default_factory=SensitivityConfig)
    
    def get_stability_ranking(self) -> List[Tuple[str, float]]:
        """Rank parameters by stability (most stable first)."""
        rankings = [
            (name, sweep.stability_score)
            for name, sweep in self.single_sweeps.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_sensitivity_ranking(self) -> List[Tuple[str, float]]:
        """Rank parameters by sensitivity (most sensitive first)."""
        rankings = [
            (name, 1 - sweep.stability_score)
            for name, sweep in self.single_sweeps.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)


class ParameterScanner:
    """
    Parameter Sensitivity Scanner.
    
    Tests parameters across ±40% range in configurable steps.
    Identifies:
    - Stability Islands: Regions where performance is consistently good
    - Cliffs: Regions where small changes cause large performance drops
    
    Outputs data structures ready for 3D surface plotting.
    """
    
    def __init__(self, config: Optional[SensitivityConfig] = None):
        """
        Initialize parameter scanner.
        
        Args:
            config: Scanner configuration
        """
        self.config = config or SensitivityConfig()
        
        logger.info(
            f"ParameterScanner initialized: ±{self.config.variation_range*100:.0f}% range, "
            f"{self.config.n_steps} steps"
        )
    
    def scan_single_parameter(
        self,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        param_name: str,
        base_value: float,
        base_params: Dict[str, Any]
    ) -> ParameterSweep:
        """
        Scan a single parameter across its range.
        
        Args:
            objective_fn: Function that takes params dict and returns metrics dict
            param_name: Name of parameter to vary
            base_value: Base/default value of parameter
            base_params: Base parameters dictionary
            
        Returns:
            ParameterSweep result
        """
        # Generate parameter values
        min_val = base_value * (1 - self.config.variation_range)
        max_val = base_value * (1 + self.config.variation_range)
        param_values = np.linspace(min_val, max_val, self.config.n_steps)
        
        # Initialize metric storage
        metric_values = {m: np.zeros(self.config.n_steps) for m in self.config.metrics}
        
        logger.info(f"Scanning {param_name}: {min_val:.4f} to {max_val:.4f}")
        
        # Run sweep
        for i, val in enumerate(param_values):
            params = base_params.copy()
            params[param_name] = val
            
            try:
                metrics = objective_fn(params)
                for metric in self.config.metrics:
                    metric_values[metric][i] = metrics.get(metric, np.nan)
            except Exception as e:
                logger.warning(f"Failed at {param_name}={val}: {e}")
                for metric in self.config.metrics:
                    metric_values[metric][i] = np.nan
        
        return ParameterSweep(
            param_name=param_name,
            param_values=param_values,
            metric_values=metric_values,
            base_value=base_value,
        )
    
    def scan_all_parameters(
        self,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        base_params: Dict[str, Any],
        params_to_scan: Optional[List[str]] = None
    ) -> SensitivityResult:
        """
        Scan all specified parameters independently.
        
        Args:
            objective_fn: Objective function
            base_params: Base parameters
            params_to_scan: List of parameter names to scan (default: all numeric)
            
        Returns:
            SensitivityResult with all sweeps
        """
        if params_to_scan is None:
            # Scan all numeric parameters
            params_to_scan = [
                k for k, v in base_params.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
        
        logger.info(f"Scanning {len(params_to_scan)} parameters")
        
        single_sweeps = {}
        
        for param_name in params_to_scan:
            base_value = base_params[param_name]
            sweep = self.scan_single_parameter(
                objective_fn, param_name, base_value, base_params
            )
            single_sweeps[param_name] = sweep
        
        return SensitivityResult(
            single_sweeps=single_sweeps,
            config=self.config,
        )
    
    def scan_parameter_grid(
        self,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        param1_name: str,
        param1_base: float,
        param2_name: str,
        param2_base: float,
        base_params: Dict[str, Any],
        n_steps: Optional[int] = None
    ) -> SensitivityResult:
        """
        Scan a 2D parameter grid for 3D surface plotting.
        
        Args:
            objective_fn: Objective function
            param1_name: First parameter name
            param1_base: First parameter base value
            param2_name: Second parameter name
            param2_base: Second parameter base value
            base_params: Base parameters
            n_steps: Optional override for number of steps (default from config)
            
        Returns:
            SensitivityResult with grid data
        """
        n = n_steps or self.config.n_steps
        
        # Generate value ranges
        param1_min = param1_base * (1 - self.config.variation_range)
        param1_max = param1_base * (1 + self.config.variation_range)
        param1_values = np.linspace(param1_min, param1_max, n)
        
        param2_min = param2_base * (1 - self.config.variation_range)
        param2_max = param2_base * (1 + self.config.variation_range)
        param2_values = np.linspace(param2_min, param2_max, n)
        
        logger.info(
            f"Scanning 2D grid: {param1_name} x {param2_name} ({n}x{n} = {n*n} points)"
        )
        
        # Initialize result grids
        grid_results = {m: np.zeros((n, n)) for m in self.config.metrics}
        
        # Run grid scan
        total = n * n
        for i, v1 in enumerate(param1_values):
            for j, v2 in enumerate(param2_values):
                params = base_params.copy()
                params[param1_name] = v1
                params[param2_name] = v2
                
                try:
                    metrics = objective_fn(params)
                    for metric in self.config.metrics:
                        grid_results[metric][i, j] = metrics.get(metric, np.nan)
                except Exception as e:
                    for metric in self.config.metrics:
                        grid_results[metric][i, j] = np.nan
                
                if (i * n + j + 1) % (total // 10) == 0:
                    logger.info(f"  Progress: {(i * n + j + 1) / total * 100:.0f}%")
        
        # Also run single sweeps for context
        sweep1 = self.scan_single_parameter(
            objective_fn, param1_name, param1_base, base_params
        )
        sweep2 = self.scan_single_parameter(
            objective_fn, param2_name, param2_base, base_params
        )
        
        return SensitivityResult(
            single_sweeps={param1_name: sweep1, param2_name: sweep2},
            grid_results=grid_results,
            grid_param1_name=param1_name,
            grid_param1_values=param1_values,
            grid_param2_name=param2_name,
            grid_param2_values=param2_values,
            config=self.config,
        )
    
    def identify_stability_islands(
        self,
        result: SensitivityResult,
        metric: str = "sharpe_ratio",
        threshold_pct: float = 0.8
    ) -> Dict[str, Any]:
        """
        Identify stability islands in parameter space.
        
        A stability island is a region where performance stays above
        threshold_pct of the maximum.
        
        Args:
            result: SensitivityResult from scan
            metric: Metric to analyze
            threshold_pct: Threshold as percentage of max (0.8 = 80% of max)
            
        Returns:
            Dictionary with stability analysis
        """
        analysis = {}
        
        # Single parameter stability
        for param_name, sweep in result.single_sweeps.items():
            values = sweep.metric_values.get(metric, np.array([]))
            if len(values) == 0:
                continue
            
            max_val = np.nanmax(values)
            threshold = max_val * threshold_pct
            
            # Find stable region
            stable_mask = values >= threshold
            stable_indices = np.where(stable_mask)[0]
            
            if len(stable_indices) > 0:
                stable_start = sweep.param_values[stable_indices[0]]
                stable_end = sweep.param_values[stable_indices[-1]]
                stable_width = (stable_end - stable_start) / sweep.base_value * 100
                
                analysis[param_name] = {
                    "stable_range": (float(stable_start), float(stable_end)),
                    "stable_width_pct": float(stable_width),
                    "optimal_value": float(sweep.param_values[np.nanargmax(values)]),
                    "stability_score": sweep.stability_score,
                    "is_stable": stable_width > 20,  # >20% range is stable
                }
        
        # 2D grid stability (for surface)
        if result.grid_results is not None and metric in result.grid_results:
            grid = result.grid_results[metric]
            max_val = np.nanmax(grid)
            threshold = max_val * threshold_pct
            stable_region = grid >= threshold
            
            analysis["grid_stability"] = {
                "stable_area_pct": float(np.sum(stable_region) / grid.size * 100),
                "max_value": float(max_val),
                "threshold": float(threshold),
            }
        
        return analysis
    
    def to_surface_plot_data(
        self,
        result: SensitivityResult,
        metric: str = "sharpe_ratio"
    ) -> Dict[str, np.ndarray]:
        """
        Convert grid result to surface plot data.
        
        Args:
            result: SensitivityResult with grid data
            metric: Metric for Z axis
            
        Returns:
            Dictionary with X, Y, Z arrays for 3D plotting
        """
        if result.grid_results is None:
            raise ValueError("No grid results available. Run scan_parameter_grid first.")
        
        if metric not in result.grid_results:
            raise ValueError(f"Metric {metric} not in grid results")
        
        # Create meshgrid
        X, Y = np.meshgrid(result.grid_param1_values, result.grid_param2_values)
        Z = result.grid_results[metric].T  # Transpose for correct orientation
        
        return {
            "X": X,
            "Y": Y,
            "Z": Z,
            "param1_name": result.grid_param1_name,
            "param2_name": result.grid_param2_name,
            "metric": metric,
        }
    
    def generate_report(self, result: SensitivityResult) -> str:
        """
        Generate text report of sensitivity analysis.
        
        Args:
            result: SensitivityResult
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "PARAMETER SENSITIVITY ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Variation range: ±{self.config.variation_range*100:.0f}%",
            f"Steps per parameter: {self.config.n_steps}",
            "",
            "PARAMETER STABILITY RANKING (most stable first):",
            "-" * 60,
        ]
        
        for param_name, score in result.get_stability_ranking():
            sweep = result.single_sweeps[param_name]
            sharpe_vals = sweep.metric_values.get("sharpe_ratio", np.array([0]))
            best_val = sweep.param_values[np.nanargmax(sharpe_vals)]
            
            lines.append(
                f"  {param_name:25s}: stability={score:.3f}, "
                f"base={sweep.base_value:.4f}, best={best_val:.4f}"
            )
        
        lines.extend([
            "",
            "SENSITIVITY RANKING (most sensitive first):",
            "-" * 60,
        ])
        
        for param_name, sensitivity in result.get_sensitivity_ranking():
            lines.append(f"  {param_name:25s}: sensitivity={sensitivity:.3f}")
        
        # Stability islands
        stability_analysis = self.identify_stability_islands(result)
        
        lines.extend([
            "",
            "STABILITY ISLANDS:",
            "-" * 60,
        ])
        
        for param_name, info in stability_analysis.items():
            if param_name == "grid_stability":
                continue
            lines.append(
                f"  {param_name}: stable range {info['stable_range'][0]:.4f} to "
                f"{info['stable_range'][1]:.4f} ({info['stable_width_pct']:.1f}% of base)"
            )
        
        if "grid_stability" in stability_analysis:
            grid_info = stability_analysis["grid_stability"]
            lines.append(
                f"  2D Grid: {grid_info['stable_area_pct']:.1f}% of space is stable"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def scan_parameter_sensitivity(
    objective_fn: Callable[[Dict], Dict],
    base_params: Dict[str, Any],
    params_to_scan: Optional[List[str]] = None,
    variation_range: float = 0.4,
    n_steps: int = 20
) -> SensitivityResult:
    """
    Convenience function for parameter sensitivity scan.
    
    Args:
        objective_fn: Function taking params and returning metrics
        base_params: Base parameter values
        params_to_scan: Parameters to scan (default: all numeric)
        variation_range: Range to vary (0.4 = ±40%)
        n_steps: Number of steps
        
    Returns:
        SensitivityResult
    """
    config = SensitivityConfig(
        variation_range=variation_range,
        n_steps=n_steps,
    )
    scanner = ParameterScanner(config=config)
    return scanner.scan_all_parameters(objective_fn, base_params, params_to_scan)
