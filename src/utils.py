# src/utils.py
"""
Utility functions for the predictive maintenance project.

Provides helper functions for configuration, logging, file operations, and calculations.
"""

import yaml
import json
import pickle
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
from datetime import datetime, timedelta
import hashlib
import warnings
from pathlib import Path

def setup_logging(log_file: Optional[str] = None, 
                  log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('predictive_maintenance')
    
    if not logger.handlers:
        # Set log level
        level = getattr(logging, log_level.upper())
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str = 'config/settings.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger = logging.getLogger('predictive_maintenance.utils')
        logger.info(f"Configuration loaded from {config_path}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {str(e)}")

def save_config(config: Dict, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger = logging.getLogger('predictive_maintenance.utils')
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        raise ValueError(f"Error saving configuration: {str(e)}")

def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None):
    """
    Save trained model to file.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        metadata: Optional metadata to save with the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create model package
        model_package = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        # Save based on file extension
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(model_package, f)
        elif filepath.endswith('.joblib'):
            joblib.dump(model_package, filepath)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .joblib")
        
        logger = logging.getLogger('predictive_maintenance.utils')
        logger.info(f"Model saved to {filepath}")
        
    except Exception as e:
        raise ValueError(f"Error saving model: {str(e)}")

def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Load trained model from file.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Tuple of (model, metadata)
    """
    try:
        # Load based on file extension
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                model_package = pickle.load(f)
        elif filepath.endswith('.joblib'):
            model_package = joblib.load(filepath)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .joblib")
        
        model = model_package.get('model')
        metadata = model_package.get('metadata', {})
        
        logger = logging.getLogger('predictive_maintenance.utils')
        logger.info(f"Model loaded from {filepath}")
        
        return model, metadata
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def calculate_data_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive statistics for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'basic': {
            'n_records': len(df),
            'n_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'data_types': df.dtypes.astype(str).to_dict()
        },
        'missing_values': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist()
        },
        'numerical_stats': {},
        'categorical_stats': {}
    }
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        stats['numerical_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            '25%': float(df[col].quantile(0.25)),
            '50%': float(df[col].quantile(0.50)),
            '75%': float(df[col].quantile(0.75)),
            'max': float(df[col].max()),
            'skew': float(df[col].skew()),
            'kurtosis': float(df[col].kurtosis())
        }
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        stats['categorical_stats'][col] = {
            'n_unique': int(df[col].nunique()),
            'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'top_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'unique_values': df[col].unique().tolist()[:10]  # Limit to first 10
        }
    
    return stats

def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              cost_params: Dict) -> Dict:
    """
    Calculate business impact metrics for predictive maintenance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_params: Dictionary with cost parameters
        
    Returns:
        Dictionary of business metrics
    """
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Extract cost parameters with defaults
    preventive_cost = cost_params.get('preventive_maintenance_cost', 500)
    reactive_cost = cost_params.get('reactive_maintenance_cost', 2000)
    false_alarm_cost = cost_params.get('false_alarm_cost', 100)
    downtime_cost_per_hour = cost_params.get('downtime_cost_per_hour', 200)
    avg_downtime_hours = cost_params.get('avg_downtime_hours', 24)
    
    # Calculate various costs
    total_preventive_cost = (tp + fp) * preventive_cost
    total_reactive_cost = fn * reactive_cost
    total_false_alarm_cost = fp * false_alarm_cost
    
    # Calculate avoided costs
    avoided_reactive_cost = tp * reactive_cost
    avoided_downtime_cost = tp * downtime_cost_per_hour * avg_downtime_hours
    
    # Total costs with predictive maintenance
    total_cost_with_pm = total_preventive_cost + total_reactive_cost + total_false_alarm_cost
    
    # Total costs without predictive maintenance (all reactive)
    total_cost_without_pm = len(y_true) * reactive_cost
    
    # Calculate savings
    total_savings = total_cost_without_pm - total_cost_with_pm
    savings_percentage = (total_savings / total_cost_without_pm) * 100 if total_cost_without_pm > 0 else 0
    
    # Calculate ROI
    investment_cost = cost_params.get('system_implementation_cost', 50000)
    roi_percentage = (total_savings / investment_cost) * 100 if investment_cost > 0 else 0
    
    # Calculate payback period (in years)
    annual_savings = total_savings * 12  # Assuming monthly predictions
    payback_period_years = investment_cost / annual_savings if annual_savings > 0 else float('inf')
    
    business_metrics = {
        'cost_analysis': {
            'total_cost_with_pm': float(total_cost_with_pm),
            'total_cost_without_pm': float(total_cost_without_pm),
            'total_savings': float(total_savings),
            'savings_percentage': float(savings_percentage),
            'investment_cost': float(investment_cost),
            'roi_percentage': float(roi_percentage),
            'payback_period_years': float(payback_period_years)
        },
        'operations': {
            'preventive_maintenances': int(tp + fp),
            'reactive_maintenances_avoided': int(tp),
            'missed_failures': int(fn),
            'false_alarms': int(fp),
            'detection_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'false_alarm_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        },
        'downtime_analysis': {
            'downtime_hours_avoided': int(tp * avg_downtime_hours),
            'downtime_cost_avoided': float(avoided_downtime_cost),
            'avg_downtime_hours': float(avg_downtime_hours)
        }
    }
    
    return business_metrics

def generate_data_hash(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """
    Generate MD5 hash of DataFrame for versioning.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to include in hash (None = all columns)
        
    Returns:
        MD5 hash string
    """
    if columns:
        df_to_hash = df[columns].copy()
    else:
        df_to_hash = df.copy()
    
    # Convert to string representation
    data_str = df_to_hash.to_csv(index=False)
    
    # Generate hash
    return hashlib.md5(data_str.encode()).hexdigest()

def check_data_quality(df: pd.DataFrame, rules: Dict) -> Dict:
    """
    Check data quality against predefined rules.
    
    Args:
        df: Input DataFrame
        rules: Dictionary of data quality rules
        
    Returns:
        Dictionary of quality check results
    """
    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    # Check for missing values
    missing_threshold = rules.get('missing_threshold', 0.1)
    missing_percentage = df.isnull().sum().max() / len(df)
    
    if missing_percentage > missing_threshold:
        results['failed'].append({
            'check': 'missing_values',
            'message': f'Missing values exceed threshold: {missing_percentage:.2%} > {missing_threshold:.2%}'
        })
    else:
        results['passed'].append({
            'check': 'missing_values',
            'message': f'Missing values within threshold: {missing_percentage:.2%}'
        })
    
    # Check for duplicate records
    duplicate_threshold = rules.get('duplicate_threshold', 0.05)
    duplicate_percentage = df.duplicated().sum() / len(df)
    
    if duplicate_percentage > duplicate_threshold:
        results['failed'].append({
            'check': 'duplicates',
            'message': f'Duplicates exceed threshold: {duplicate_percentage:.2%} > {duplicate_threshold:.2%}'
        })
    else:
        results['passed'].append({
            'check': 'duplicates',
            'message': f'Duplicates within threshold: {duplicate_percentage:.2%}'
        })
    
    # Check for outliers
    outlier_threshold = rules.get('outlier_threshold', 0.01)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols[:5]:  # Check first 5 numerical columns
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_percentage = outliers / len(df)
        
        if outlier_percentage > outlier_threshold:
            results['warnings'].append({
                'check': f'outliers_{col}',
                'message': f'Outliers in {col}: {outlier_percentage:.2%} > {outlier_threshold:.2%}'
            })
    
    # Check data types
    expected_types = rules.get('expected_types', {})
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if expected_type not in actual_type:
                results['failed'].append({
                    'check': f'data_type_{col}',
                    'message': f'Data type mismatch for {col}: expected {expected_type}, got {actual_type}'
                })
    
    # Calculate overall score
    total_checks = len(results['passed']) + len(results['failed']) + len(results['warnings'])
    if total_checks > 0:
        results['quality_score'] = len(results['passed']) / total_checks
    else:
        results['quality_score'] = 1.0
    
    return results

def create_experiment_tracking(experiment_name: str, 
                              params: Dict,
                              metrics: Dict,
                              artifacts: Optional[Dict] = None) -> str:
    """
    Create experiment tracking record.
    
    Args:
        experiment_name: Name of the experiment
        params: Experiment parameters
        metrics: Experiment metrics
        artifacts: Optional artifacts (paths to files)
        
    Returns:
        Experiment ID
    """
    import uuid
    from datetime import datetime
    
    experiment_id = str(uuid.uuid4())[:8]
    
    experiment_record = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'metrics': metrics,
        'artifacts': artifacts or {},
        'version': '1.0'
    }
    
    # Create experiments directory
    experiments_dir = 'experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Save experiment record
    record_path = os.path.join(experiments_dir, f'{experiment_id}_{experiment_name}.json')
    with open(record_path, 'w') as f:
        json.dump(experiment_record, f, indent=2)
    
    logger = logging.getLogger('predictive_maintenance.utils')
    logger.info(f"Experiment {experiment_name} tracked with ID: {experiment_id}")
    
    return experiment_id

def validate_prediction_input(input_data: Dict, expected_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate input data for prediction.
    
    Args:
        input_data: Dictionary of input features
        expected_features: List of expected feature names
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for missing features
    for feature in expected_features:
        if feature not in input_data:
            errors.append(f"Missing required feature: {feature}")
    
    # Check data types (basic validation)
    for feature, value in input_data.items():
        if isinstance(value, (int, float)):
            # Check for NaN or infinite values
            if np.isnan(value) or np.isinf(value):
                errors.append(f"Invalid value for {feature}: {value}")
        elif isinstance(value, str):
            # Check for empty strings
            if not value.strip():
                errors.append(f"Empty string for {feature}")
    
    return len(errors) == 0, errors

def get_version() -> str:
    """Get the current version of the package."""
    return "1.0.0"

def get_system_info() -> Dict:
    """Get system information for debugging and logging."""
    import platform
    import sys
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'system': platform.system(),
        'machine': platform.machine(),
        'timestamp': datetime.now().isoformat()
    }