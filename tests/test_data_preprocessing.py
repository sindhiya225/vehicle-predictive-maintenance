# tests/test_data_preprocessing.py
"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import yaml

# Import the module to test
import sys
sys.path.append('src')
from data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test class for DataPreprocessor."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        config = {
            'data_types': {
                'vehicle_id': 'object',
                'engine_rpm': 'float64',
                'oil_temperature': 'float64',
                'fuel_consumption': 'float64'
            },
            'categorical_columns': ['vehicle_type', 'fuel_type'],
            'numerical_columns': ['engine_rpm', 'oil_temperature', 'fuel_consumption', 'mileage'],
            'date_columns': ['timestamp', 'fault_date'],
            'columns_to_drop': ['unused_column']
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample vehicle data for testing."""
        np.random.seed(42)
        
        n_samples = 100
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        data = {
            'vehicle_id': [f'V{i:03d}' for i in range(n_samples)],
            'timestamp': dates,
            'engine_rpm': np.random.normal(2000, 500, n_samples),
            'oil_temperature': np.random.normal(90, 10, n_samples),
            'fuel_consumption': np.random.normal(8, 2, n_samples),
            'mileage': np.random.uniform(0, 200000, n_samples),
            'vehicle_type': np.random.choice(['SUV', 'Sedan', 'Truck'], n_samples),
            'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric'], n_samples),
            'unused_column': np.zeros(n_samples)
        }
        
        # Add some faults
        fault_dates = dates + pd.to_timedelta(np.random.randint(1, 60, n_samples), unit='D')
        data['fault_date'] = fault_dates
        
        # Add some missing values
        data['engine_rpm'][:5] = np.nan
        data['fuel_type'][10:15] = None
        
        # Add some outliers
        data['oil_temperature'][20] = 200  # Outlier
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self, sample_config):
        """Create DataPreprocessor instance for testing."""
        return DataPreprocessor(sample_config)
    
    def test_load_raw_data_csv(self, preprocessor):
        """Test loading raw data from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            result = preprocessor.load_raw_data(csv_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert list(result.columns) == ['A', 'B']
        finally:
            os.unlink(csv_path)
    
    def test_load_raw_data_json(self, preprocessor):
        """Test loading raw data from JSON file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            df.to_json(f.name, orient='records')
            json_path = f.name
        
        try:
            result = preprocessor.load_raw_data(json_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert list(result.columns) == ['A', 'B']
        finally:
            os.unlink(json_path)
    
    def test_load_raw_data_unsupported_format(self, preprocessor):
        """Test loading raw data with unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("unsupported format")
            txt_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                preprocessor.load_raw_data(txt_path)
        finally:
            os.unlink(txt_path)
    
    def test_validate_data(self, preprocessor, sample_data):
        """Test data validation functionality."""
        df, report = preprocessor.validate_data(sample_data)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(report, dict)
        
        # Check report structure
        assert 'total_records' in report
        assert 'columns_validated' in report
        assert 'issues_found' in report
        
        # Check specific validations
        assert report['total_records'] == 100
        assert 'engine_rpm' in report.get('missing_values', {})
        assert 'fuel_type' in report.get('missing_values', {})
        assert 'oil_temperature' in report.get('outliers', {})
    
    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning functionality."""
        # First validate to get report
        df, report = preprocessor.validate_data(sample_data)
        
        # Then clean
        df_clean = preprocessor.clean_data(df, report)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert df_clean.shape[0] <= df.shape[0]  # May remove duplicates
        
        # Check missing values are handled
        assert df_clean['engine_rpm'].isnull().sum() == 0
        assert df_clean['fuel_type'].isnull().sum() == 0
        
        # Check outliers are capped
        oil_temp_max = df_clean['oil_temperature'].max()
        assert oil_temp_max < 200  # Outlier should be capped
    
    def test_create_target_variable(self, preprocessor, sample_data):
        """Test target variable creation."""
        df_with_target = preprocessor.create_target_variable(
            sample_data,
            fault_date_col='fault_date',
            prediction_horizon=30
        )
        
        assert 'target' in df_with_target.columns
        assert 'days_to_fault' in df_with_target.columns
        
        # Check target values are binary
        assert set(df_with_target['target'].unique()).issubset({0, 1})
        
        # Check days_to_fault calculation
        assert df_with_target['days_to_fault'].dtype in [np.int64, np.float64]
    
    def test_prepare_training_data(self, preprocessor, sample_data):
        """Test training data preparation."""
        # First create target variable
        df_with_target = preprocessor.create_target_variable(
            sample_data,
            fault_date_col='fault_date',
            prediction_horizon=30
        )
        
        # Then prepare training data
        X, y = preprocessor.prepare_training_data(df_with_target)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check shapes
        assert len(X) == len(y)
        assert X.shape[1] > 0
        
        # Check target column is not in features
        assert 'target' not in X.columns
        
        # Check non-feature columns are dropped
        assert 'unused_column' not in X.columns
        assert 'vehicle_id' not in X.columns
    
    def test_save_processed_data_csv(self, preprocessor, sample_data, tmp_path):
        """Test saving processed data to CSV."""
        output_path = tmp_path / "test_output.csv"
        
        preprocessor.save_processed_data(sample_data, str(output_path))
        
        # Check file exists
        assert output_path.exists()
        
        # Check can be loaded back
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == len(sample_data)
    
    def test_save_processed_data_parquet(self, preprocessor, sample_data, tmp_path):
        """Test saving processed data to Parquet."""
        output_path = tmp_path / "test_output.parquet"
        
        preprocessor.save_processed_data(sample_data, str(output_path))
        
        # Check file exists
        assert output_path.exists()
        
        # Check can be loaded back
        df_loaded = pd.read_parquet(output_path)
        assert len(df_loaded) == len(sample_data)
    
    def test_save_processed_data_pkl(self, preprocessor, sample_data, tmp_path):
        """Test saving processed data to Pickle."""
        output_path = tmp_path / "test_output.pkl"
        
        preprocessor.save_processed_data(sample_data, str(output_path))
        
        # Check file exists
        assert output_path.exists()
        
        # Check can be loaded back
        df_loaded = pd.read_pickle(output_path)
        assert len(df_loaded) == len(sample_data)
    
    def test_save_processed_data_unsupported_format(self, preprocessor, sample_data, tmp_path):
        """Test saving processed data with unsupported format."""
        output_path = tmp_path / "test_output.txt"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            preprocessor.save_processed_data(sample_data, str(output_path))
    
    def test_data_preprocessing_pipeline(self, preprocessor, sample_data):
        """Test complete data preprocessing pipeline."""
        # Step 1: Validate
        df_validated, report = preprocessor.validate_data(sample_data)
        assert isinstance(report, dict)
        
        # Step 2: Clean
        df_clean = preprocessor.clean_data(df_validated, report)
        assert df_clean.isnull().sum().sum() == 0
        
        # Step 3: Create target
        df_with_target = preprocessor.create_target_variable(
            df_clean,
            fault_date_col='fault_date',
            prediction_horizon=30
        )
        assert 'target' in df_with_target.columns
        
        # Step 4: Prepare training data
        X, y = preprocessor.prepare_training_data(df_with_target)
        assert len(X) == len(y)
        assert X.shape[1] > 0
        
        # Step 5: Check no data leakage
        assert 'fault_date' not in X.columns
        assert 'timestamp' not in X.columns
        assert 'target' not in X.columns
    
    def test_edge_cases(self, preprocessor):
        """Test edge cases in data preprocessing."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        df, report = preprocessor.validate_data(empty_df)
        assert len(df) == 0
        assert report['total_records'] == 0
        
        # Test with single record
        single_df = pd.DataFrame({
            'vehicle_id': ['V001'],
            'timestamp': [datetime.now()],
            'engine_rpm': [2000],
            'fault_date': [datetime.now() + timedelta(days=10)]
        })
        
        df_with_target = preprocessor.create_target_variable(single_df)
        assert len(df_with_target) == 1
        assert 'target' in df_with_target.columns
    
    def test_missing_columns_handling(self, preprocessor, sample_data):
        """Test handling of missing columns in configuration."""
        # Remove a column that's in config
        df_missing_col = sample_data.drop(columns=['engine_rpm'])
        
        # Should handle gracefully
        df, report = preprocessor.validate_data(df_missing_col)
        assert 'engine_rpm' not in df.columns
        
        # Cleaning should still work
        df_clean = preprocessor.clean_data(df, report)
        assert df_clean.isnull().sum().sum() == 0
    
    def test_duplicate_handling(self, preprocessor):
        """Test duplicate record handling."""
        # Create DataFrame with duplicates
        duplicate_data = pd.DataFrame({
            'vehicle_id': ['V001', 'V001', 'V002', 'V002'],
            'timestamp': pd.date_range('2023-01-01', periods=4, freq='H'),
            'engine_rpm': [2000, 2000, 2100, 2100],
            'fault_date': pd.date_range('2023-02-01', periods=4, freq='D')
        })
        
        df, report = preprocessor.validate_data(duplicate_data)
        assert report['duplicate_records'] > 0
        
        df_clean = preprocessor.clean_data(df, report)
        assert df_clean.duplicated().sum() == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])