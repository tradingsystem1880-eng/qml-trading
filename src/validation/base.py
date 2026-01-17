"""
Validation Base Interface
=========================
VRD 2.0 Validation Module

Abstract base class for all validation methods (Monte Carlo, Permutation, Bootstrap).
Ensures consistent interface across all validators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationStatus(Enum):
    """Result status of validation test."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class ValidationResult:
    """
    Standardized output from any validator.
    
    Attributes:
        validator_name: Name of the validator (e.g., 'permutation_test')
        status: Overall pass/warn/fail status
        metrics: Dictionary of calculated metrics
        p_value: Statistical significance (if applicable)
        confidence: Confidence level (0.0 to 1.0)
        interpretation: Human-readable interpretation
        details: Additional details for debugging
        timestamp: When validation was run
    """
    validator_name: str
    status: ValidationStatus
    metrics: Dict[str, Any] = field(default_factory=dict)
    p_value: Optional[float] = None
    confidence: Optional[float] = None
    interpretation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'validator_name': self.validator_name,
            'status': self.status.value,
            'metrics': self.metrics,
            'p_value': self.p_value,
            'confidence': self.confidence,
            'interpretation': self.interpretation,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        icon = {
            ValidationStatus.PASS: '✅',
            ValidationStatus.WARN: '⚠️',
            ValidationStatus.FAIL: '❌',
            ValidationStatus.ERROR: '💥'
        }.get(self.status, '❓')
        
        return f"{icon} {self.validator_name}: {self.status.value.upper()} - {self.interpretation}"


class BaseValidator(ABC):
    """
    Abstract base class for all validation methods.
    
    All validators must implement:
    - validate(): Run the validation and return ValidationResult
    - name property: Return the validator name
    
    Validators may optionally implement:
    - configure(): Update validator settings
    - get_default_config(): Return default configuration
    
    Usage:
        class MyValidator(BaseValidator):
            @property
            def name(self) -> str:
                return "my_validator"
            
            def validate(self, backtest_result: Dict[str, Any]) -> ValidationResult:
                # Run validation logic
                return ValidationResult(
                    validator_name=self.name,
                    status=ValidationStatus.PASS,
                    metrics={'some_metric': 0.95},
                    interpretation="Validation passed"
                )
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with optional configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self.get_default_config()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return validator name."""
        pass
    
    @abstractmethod
    def validate(
        self,
        backtest_result: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        Run validation on backtest results.
        
        Args:
            backtest_result: Results dictionary from BacktestEngine
            trades: Optional list of trade dictionaries
        
        Returns:
            ValidationResult with metrics and status
        """
        pass
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for this validator.
        
        Override in subclasses to provide validator-specific defaults.
        """
        return {}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update validator configuration.
        
        Args:
            config: New configuration values
        """
        self.config.update(config)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass
class ValidationSuite:
    """
    Collection of validation results from multiple validators.
    
    Provides aggregate status and easy access to individual results.
    """
    results: List[ValidationResult] = field(default_factory=list)
    run_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def overall_status(self) -> ValidationStatus:
        """Get aggregate status (worst case)."""
        if not self.results:
            return ValidationStatus.ERROR
        
        statuses = [r.status for r in self.results]
        
        if ValidationStatus.ERROR in statuses:
            return ValidationStatus.ERROR
        if ValidationStatus.FAIL in statuses:
            return ValidationStatus.FAIL
        if ValidationStatus.WARN in statuses:
            return ValidationStatus.WARN
        return ValidationStatus.PASS
    
    @property
    def passed(self) -> bool:
        """Check if all validators passed."""
        return self.overall_status == ValidationStatus.PASS
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
    
    def get_result(self, validator_name: str) -> Optional[ValidationResult]:
        """Get result by validator name."""
        for r in self.results:
            if r.validator_name == validator_name:
                return r
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_status': self.overall_status.value,
            'run_timestamp': self.run_timestamp.isoformat(),
            'validators': [r.to_dict() for r in self.results]
        }
    
    def __str__(self) -> str:
        icon = {
            ValidationStatus.PASS: '✅',
            ValidationStatus.WARN: '⚠️',
            ValidationStatus.FAIL: '❌',
            ValidationStatus.ERROR: '💥'
        }.get(self.overall_status, '❓')
        
        lines = [f"{icon} VALIDATION SUITE: {self.overall_status.value.upper()}"]
        lines.append("-" * 50)
        for r in self.results:
            lines.append(f"  {r}")
        
        return '\n'.join(lines)
