# src/__init__.py
"""
Predictive Maintenance for Vehicles - ML Package

This package provides functionality for predicting vehicle maintenance needs
using machine learning models.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .utils import (
    load_config,
    save_model,
    load_model,
    setup_logging,
    calculate_business_metrics
)

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'load_config',
    'save_model',
    'load_model',
    'setup_logging',
    'calculate_business_metrics'
]