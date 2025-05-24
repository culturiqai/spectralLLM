"""
SpectralLLM Validation Framework
==============================

Comprehensive validation and testing tools for SpectralLLM training and evaluation.

Features:
- Training metrics validation and consistency checking
- Perplexity calculation verification
- Data leakage detection between train/val/test sets
- Non-overlapping sequence validation
- Baseline model evaluation
- Comprehensive validation suite
"""

from .perplexity_validator import PerplexityValidator
from .baseline_evaluator import BaselineEvaluator
from .dataset_validator import DatasetValidator

try:
    from .metrics_validator import (
        MetricsValidator, 
        validate_training_run, 
        run_comprehensive_validation_suite
    )
    METRICS_VALIDATOR_AVAILABLE = True
except ImportError:
    METRICS_VALIDATOR_AVAILABLE = False

__all__ = [
    'PerplexityValidator',
    'BaselineEvaluator', 
    'DatasetValidator'
]

if METRICS_VALIDATOR_AVAILABLE:
    __all__.extend([
        'MetricsValidator', 
        'validate_training_run',
        'run_comprehensive_validation_suite'
    ])

# Convenience function for quick validation
def quick_validation(training_history=None, **kwargs):
    """
    Quick validation function that runs appropriate tests based on provided data.
    
    Args:
        training_history: Training metrics history
        **kwargs: Additional arguments passed to run_comprehensive_validation_suite
        
    Returns:
        Validation results
    """
    if not METRICS_VALIDATOR_AVAILABLE:
        raise ImportError("Enhanced metrics validator not available")
    
    return run_comprehensive_validation_suite(training_history=training_history, **kwargs) 