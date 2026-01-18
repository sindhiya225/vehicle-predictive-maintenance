# src/data_preprocessing.py
"""
Data preprocessing module for vehicle predictive maintenance.

Handles data cleaning, validation, and preparation for ML modeling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime, timedelta
import yaml
import json

class DataPreprocessor:
    """Preprocess vehicle sensor and maintenance data."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw vehicle data from CSV/JSON file.
        
        Args:
            filepath: Path to raw data file
            
        Returns:
            DataFrame containing raw vehicle data
        """
        self.logger.info(f"Loading raw data from {filepath}")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        self.logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate data quality and report issues.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (validated DataFrame, validation report)
        """
        self.logger.info("Starting data validation")
        
        validation_report = {
            'total_records': len(df),
            'columns_validated': list(df.columns),
            'issues_found': {}
        }
        
        # Check for missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        validation_report['missing_values'] = missing_values[missing_values > 0].to_dict()
        validation_report['missing_percentage'] = missing_percentage[missing_percentage > 0].to_dict()
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        validation_report['duplicate_records'] = int(duplicates)
        
        # Validate data types
        expected_types = self.config.get('data_types', {})
        type_issues = {}
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    type_issues[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        if type_issues:
            validation_report['data_type_issues'] = type_issues
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numerical_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_report[col] = {
                        'outliers_count': int(outliers),
                        'outliers_percentage': (outliers / len(df)) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
        
        if outlier_report:
            validation_report['outliers'] = outlier_report
        
        self.logger.info(f"Validation complete. Issues found: {bool(validation_report['issues_found'])}")
        return df, validation_report
    
    def clean_data(self, df: pd.DataFrame, validation_report: Dict) -> pd.DataFrame:
        """
        Clean data based on validation findings.
        
        Args:
            df: Input DataFrame
            validation_report: Report from validate_data method
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning")
        
        df_clean = df.copy()
        
        # Handle missing values
        missing_cols = validation_report.get('missing_values', {})
        for col, missing_count in missing_cols.items():
            if missing_count > 0:
                if col in self.config.get('categorical_columns', []):
                    # For categorical, fill with mode
                    mode_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_value, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with mode: {mode_value}")
                elif col in self.config.get('numerical_columns', []):
                    # For numerical, fill with median
                    median_value = df_clean[col].median()
                    df_clean[col].fillna(median_value, inplace=True)
                    self.logger.info(f"Filled missing values in {col} with median: {median_value}")
                else:
                    # For other columns, fill with forward fill
                    df_clean[col].fillna(method='ffill', inplace=True)
        
        # Remove duplicates
        duplicates = validation_report.get('duplicate_records', 0)
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            self.logger.info(f"Removed {duplicates} duplicate records")
        
        # Handle outliers (cap them)
        outliers = validation_report.get('outliers', {})
        for col, outlier_info in outliers.items():
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']
            
            df_clean[col] = np.where(
                df_clean[col] < lower_bound, lower_bound,
                np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            )
            self.logger.info(f"Capped outliers in {col} to bounds: [{lower_bound}, {upper_bound}]")
        
        # Convert date columns
        date_columns = self.config.get('date_columns', [])
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col])
                self.logger.info(f"Converted {col} to datetime")
        
        self.logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean
    
    def create_target_variable(self, df: pd.DataFrame, 
                              fault_date_col: str = 'fault_date',
                              prediction_horizon: int = 30) -> pd.DataFrame:
        """
        Create binary target variable for maintenance prediction.
        
        Args:
            df: Input DataFrame
            fault_date_col: Column name containing fault dates
            prediction_horizon: Days in future to predict
            
        Returns:
            DataFrame with target variable
        """
        self.logger.info(f"Creating target variable with {prediction_horizon}-day horizon")
        
        df_target = df.copy()
        
        # Ensure we have datetime
        if fault_date_col in df_target.columns:
            df_target[fault_date_col] = pd.to_datetime(df_target[fault_date_col])
        
        # Create target: 1 if maintenance needed in next X days, else 0
        if 'timestamp' in df_target.columns:
            df_target['timestamp'] = pd.to_datetime(df_target['timestamp'])
            
            # For each record, check if fault occurs within prediction horizon
            df_target['days_to_fault'] = (
                df_target[fault_date_col] - df_target['timestamp']
            ).dt.days
            
            df_target['target'] = np.where(
                (df_target['days_to_fault'] >= 0) & 
                (df_target['days_to_fault'] <= prediction_horizon),
                1,  # Maintenance needed
                0   # No maintenance needed
            )
            
            self.logger.info(f"Target distribution: {df_target['target'].value_counts().to_dict()}")
        
        return df_target
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final training dataset.
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        self.logger.info("Preparing training data")
        
        # Separate features and target
        target_col = 'target'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Drop non-feature columns
        columns_to_drop = self.config.get('columns_to_drop', []) + [
            'timestamp', 'vehicle_id', 'fault_date', 'days_to_fault'
        ]
        
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        X = df.drop(columns=columns_to_drop + [target_col], errors='ignore')
        y = df[target_col]
        
        # Ensure no missing values
        if X.isnull().sum().sum() > 0:
            self.logger.warning("Missing values found in features. Filling with median.")
            X = X.fillna(X.median())
        
        self.logger.info(f"Training data prepared. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """
        Save processed data to file.
        
        Args:
            df: Processed DataFrame
            filepath: Output file path
        """
        self.logger.info(f"Saving processed data to {filepath}")
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        elif filepath.endswith('.pkl'):
            df.to_pickle(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV, Parquet, or PKL.")
        
        self.logger.info(f"Data saved successfully. Shape: {df.shape}")