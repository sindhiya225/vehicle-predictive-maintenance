"""
Feature engineering for vehicle predictive maintenance
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class VehicleFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create advanced features from vehicle sensor data"""
    
    def __init__(self, target_window: str = '7d'):
        self.target_window = target_window
        self.feature_columns = None
        
    def fit(self, X: pd.DataFrame, y=None):
        # Store feature names for later use
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into feature-rich dataset"""
        X = X.copy()
        
        # 1. Rolling statistics (past 7, 30 days)
        X = self._add_rolling_features(X)
        
        # 2. Rate of change features
        X = self._add_rate_of_change_features(X)
        
        # 3. Engineering features
        X = self._add_engineering_features(X)
        
        # 4. Threshold crossing features
        X = self._add_threshold_features(X)
        
        # 5. Interaction features
        X = self._add_interaction_features(X)
        
        # 6. Time-based features
        X = self._add_time_features(X)
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        return X
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        if 'vehicle_id' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ['failure_in_next_7d', 'failure_in_next_30d']]
            
            for col in numeric_cols:
                # Rolling means
                df[f'{col}_rolling_mean_7d'] = df.groupby('vehicle_id')[col].transform(
                    lambda x: x.rolling(window=7, min_periods=3).mean()
                )
                df[f'{col}_rolling_std_7d'] = df.groupby('vehicle_id')[col].transform(
                    lambda x: x.rolling(window=7, min_periods=3).std()
                )
                
                # Percent change
                df[f'{col}_pct_change_1d'] = df.groupby('vehicle_id')[col].pct_change()
                df[f'{col}_pct_change_7d'] = df.groupby('vehicle_id')[col].pct_change(periods=7)
        
        return df
    
    def _add_rate_of_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change and derivative features"""
        if 'vehicle_id' in df.columns:
            critical_params = ['engine_temperature', 'oil_pressure', 'battery_voltage']
            
            for param in critical_params:
                if param in df.columns:
                    # First derivative (instantaneous rate of change)
                    df[f'{param}_roc'] = df.groupby('vehicle_id')[param].diff()
                    
                    # Acceleration (second derivative)
                    df[f'{param}_acceleration'] = df.groupby('vehicle_id')[f'{param}_roc'].diff()
                    
                    # Volatility
                    df[f'{param}_volatility'] = df.groupby('vehicle_id')[param].transform(
                        lambda x: x.rolling(window=5, min_periods=3).std()
                    )
        
        return df
    
    def _add_engineering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific engineering features"""
        # Engine stress indicator
        if all(col in df.columns for col in ['engine_temperature', 'rpm']):
            df['engine_stress'] = (df['engine_temperature'] / 100) * (df['rpm'] / 3000)
        
        # Battery health indicator
        if 'battery_voltage' in df.columns:
            df['battery_health'] = np.where(
                df['battery_voltage'] > 12.4, 'Good',
                np.where(df['battery_voltage'] > 12.0, 'Fair', 'Poor')
            )
        
        # Oil pressure to temperature ratio
        if all(col in df.columns for col in ['oil_pressure', 'engine_temperature']):
            df['oil_temp_ratio'] = df['oil_pressure'] / (df['engine_temperature'] + 1e-5)
        
        # Fuel efficiency
        if all(col in df.columns for col in ['distance_km', 'fuel_consumption']):
            df['fuel_efficiency'] = df['distance_km'] / (df['fuel_consumption'] + 1e-5)
        
        return df
    
    def _add_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on threshold crossings"""
        # Count of threshold violations in past 7 days
        thresholds = {
            'engine_temperature': {'high': 110, 'low': 80},
            'oil_pressure': {'high': 55, 'low': 25},
            'battery_voltage': {'low': 11.5}
        }
        
        for param, bounds in thresholds.items():
            if param in df.columns:
                # High threshold violations
                if 'high' in bounds:
                    df[f'{param}_high_violations_7d'] = df.groupby('vehicle_id')[param].transform(
                        lambda x: x.rolling(window=7, min_periods=3).apply(
                            lambda y: (y > bounds['high']).sum()
                        )
                    )
                
                # Low threshold violations
                if 'low' in bounds:
                    df[f'{param}_low_violations_7d'] = df.groupby('vehicle_id')[param].transform(
                        lambda x: x.rolling(window=7, min_periods=3).apply(
                            lambda y: (y < bounds['low']).sum()
                        )
                    )
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between parameters"""
        # Temperature-pressure interaction
        if all(col in df.columns for col in ['engine_temperature', 'oil_pressure']):
            df['temp_pressure_interaction'] = df['engine_temperature'] * df['oil_pressure']
        
        # Speed-distance interaction
        if all(col in df.columns for col in ['max_speed_kmh', 'distance_km']):
            df['speed_distance_interaction'] = df['max_speed_kmh'] * df['distance_km']
        
        # Age-usage interaction
        if all(col in df.columns for col in ['vehicle_age_days', 'total_distance']):
            df['age_usage_interaction'] = df['vehicle_age_days'] * df['total_distance'] / 1000
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            
            # Sine/Cosine encoding for cyclical features
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
            
            # Days since last maintenance (simulated)
            df['days_since_last_maintenance'] = df.groupby('vehicle_id').cumcount() % 90
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Fill with forward fill then backward fill within vehicle group
        if 'vehicle_id' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['failure_in_next_7d', 'failure_in_next_30d']:
                    df[col] = df.groupby('vehicle_id')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
        
        # For any remaining NaN, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['failure_in_next_7d', 'failure_in_next_30d']:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def get_feature_importance_template(self) -> pd.DataFrame:
        """Return template for feature importance documentation"""
        feature_categories = {
            'Engine Parameters': [
                'engine_temperature', 'oil_pressure', 'rpm',
                'engine_temperature_rolling_mean_7d',
                'oil_pressure_rolling_std_7d',
                'engine_stress', 'oil_temp_ratio'
            ],
            'Electrical System': [
                'battery_voltage', 'battery_voltage_roc',
                'battery_voltage_low_violations_7d'
            ],
            'Usage Patterns': [
                'distance_km', 'idle_time_minutes', 'max_speed_kmh',
                'distance_km_rolling_mean_7d',
                'fuel_efficiency', 'speed_distance_interaction'
            ],
            'Historical Trends': [
                'vehicle_age_days', 'total_distance',
                'days_since_last_maintenance',
                'age_usage_interaction'
            ],
            'Threshold Violations': [
                'engine_temperature_high_violations_7d',
                'oil_pressure_low_violations_7d'
            ],
            'Temporal Features': [
                'month', 'day_of_week', 'is_weekend',
                'day_sin', 'day_cos'
            ]
        }
        
        return pd.DataFrame([
            {'Feature': feature, 'Category': category, 'Business Impact': 'High', 'Description': ''}
            for category, features in feature_categories.items()
            for feature in features
        ])