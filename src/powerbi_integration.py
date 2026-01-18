# src/powerbi_integration.py
"""
Power BI integration module for predictive maintenance dashboard.

Provides functionality to export model results and predictions for Power BI visualization.
"""

import pandas as pd
import numpy as np
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import csv

class PowerBIIntegration:
    """Handle integration with Power BI for dashboard creation."""
    
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
    
    def create_powerbi_dataset(self, predictions_df: pd.DataFrame,
                              model_metrics: Dict,
                              business_metrics: Dict) -> Dict[str, pd.DataFrame]:
        """
        Create datasets optimized for Power BI consumption.
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            model_metrics: Model performance metrics
            business_metrics: Business impact metrics
            
        Returns:
            Dictionary of DataFrames for different Power BI tables
        """
        self.logger.info("Creating Power BI datasets")
        
        datasets = {}
        
        # 1. Main Predictions Table
        datasets['predictions'] = self._create_predictions_table(predictions_df)
        
        # 2. Model Performance Metrics Table
        datasets['model_metrics'] = self._create_metrics_table(model_metrics)
        
        # 3. Business Metrics Table
        datasets['business_metrics'] = self._create_business_metrics_table(business_metrics)
        
        # 4. Feature Importance Table
        if 'feature_importance' in model_metrics:
            datasets['feature_importance'] = self._create_feature_importance_table(
                model_metrics['feature_importance']
            )
        
        # 5. Time Series Predictions (if timestamp available)
        if 'timestamp' in predictions_df.columns:
            datasets['time_series_predictions'] = self._create_time_series_table(predictions_df)
        
        # 6. Vehicle Fleet Summary
        if 'vehicle_id' in predictions_df.columns:
            datasets['fleet_summary'] = self._create_fleet_summary_table(predictions_df)
        
        self.logger.info(f"Created {len(datasets)} Power BI datasets")
        return datasets
    
    def _create_predictions_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create main predictions table for Power BI."""
        table_data = df.copy()
        
        # Ensure required columns
        required_cols = ['prediction', 'probability']
        for col in required_cols:
            if col not in table_data.columns:
                self.logger.warning(f"Required column '{col}' not found in predictions")
        
        # Add derived columns for Power BI
        table_data['prediction_label'] = table_data['prediction'].map({
            0: 'No Maintenance Needed',
            1: 'Maintenance Needed'
        })
        
        table_data['risk_category'] = pd.cut(
            table_data['probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Add prediction date
        table_data['prediction_date'] = datetime.now().date()
        
        # Select and order columns for Power BI
        powerbi_cols = []
        if 'vehicle_id' in table_data.columns:
            powerbi_cols.append('vehicle_id')
        if 'timestamp' in table_data.columns:
            powerbi_cols.append('timestamp')
        
        powerbi_cols.extend([
            'prediction', 'prediction_label', 'probability', 'risk_category',
            'prediction_date'
        ])
        
        # Add any additional features
        feature_cols = [col for col in table_data.columns 
                       if col not in powerbi_cols + ['target', 'true_label']]
        powerbi_cols.extend(feature_cols[:10])  # Limit to top 10 features
        
        return table_data[powerbi_cols]
    
    def _create_metrics_table(self, metrics: Dict) -> pd.DataFrame:
        """Create model metrics table for Power BI."""
        metrics_data = []
        
        for metric_name, metric_value in metrics.items():
            if metric_name != 'confusion_matrix' and metric_name != 'feature_importance':
                metrics_data.append({
                    'metric_name': metric_name,
                    'metric_value': float(metric_value) if isinstance(metric_value, (int, float)) else 0,
                    'metric_category': self._categorize_metric(metric_name),
                    'update_date': datetime.now().date()
                })
        
        # Add confusion matrix metrics
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            for key, value in cm.items():
                metrics_data.append({
                    'metric_name': f'confusion_matrix_{key}',
                    'metric_value': float(value),
                    'metric_category': 'Confusion Matrix',
                    'update_date': datetime.now().date()
                })
        
        return pd.DataFrame(metrics_data)
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric for Power BI grouping."""
        accuracy_metrics = {'accuracy', 'balanced_accuracy'}
        precision_metrics = {'precision', 'average_precision'}
        recall_metrics = {'recall', 'sensitivity', 'detection_rate'}
        fpr_metrics = {'false_positive_rate', 'false_alarm_rate', 'specificity'}
        auc_metrics = {'roc_auc'}
        
        if metric_name in accuracy_metrics:
            return 'Accuracy'
        elif metric_name in precision_metrics:
            return 'Precision'
        elif metric_name in recall_metrics:
            return 'Recall'
        elif metric_name in fpr_metrics:
            return 'False Positive Rate'
        elif metric_name in auc_metrics:
            return 'AUC'
        else:
            return 'Other'
    
    def _create_business_metrics_table(self, business_metrics: Dict) -> pd.DataFrame:
        """Create business metrics table for Power BI."""
        metrics_data = []
        
        for metric_name, metric_value in business_metrics.items():
            metrics_data.append({
                'business_metric': metric_name.replace('_', ' ').title(),
                'value': float(metric_value) if isinstance(metric_value, (int, float)) else str(metric_value),
                'unit': self._get_business_metric_unit(metric_name),
                'category': self._categorize_business_metric(metric_name),
                'update_date': datetime.now().date()
            })
        
        return pd.DataFrame(metrics_data)
    
    def _get_business_metric_unit(self, metric_name: str) -> str:
        """Get unit for business metric."""
        if 'cost' in metric_name or 'savings' in metric_name:
            return 'USD'
        elif 'percentage' in metric_name:
            return '%'
        elif 'per_vehicle' in metric_name:
            return 'USD/vehicle'
        else:
            return 'count'
    
    def _categorize_business_metric(self, metric_name: str) -> str:
        """Categorize business metric."""
        if 'cost' in metric_name:
            return 'Cost Analysis'
        elif 'savings' in metric_name:
            return 'Savings'
        elif 'maintenance' in metric_name:
            return 'Maintenance Operations'
        elif 'alarm' in metric_name or 'failure' in metric_name:
            return 'Risk Analysis'
        else:
            return 'Other'
    
    def _create_feature_importance_table(self, feature_importance: Dict) -> pd.DataFrame:
        """Create feature importance table for Power BI."""
        if isinstance(feature_importance, dict):
            data = [{'feature': k, 'importance': float(v)} 
                   for k, v in feature_importance.items()]
        elif isinstance(feature_importance, list):
            data = [{'feature': f[0], 'importance': float(f[1])} 
                   for f in feature_importance]
        else:
            self.logger.warning("Unsupported feature importance format")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('importance', ascending=False).head(20)  # Top 20 features
        df['category'] = df['feature'].apply(self._categorize_feature)
        df['update_date'] = datetime.now().date()
        
        return df
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize features for Power BI grouping."""
        engine_keywords = ['rpm', 'temp', 'pressure', 'fuel', 'oil', 'engine', 'torque']
        electrical_keywords = ['voltage', 'current', 'battery', 'sensor', 'electrical']
        usage_keywords = ['mileage', 'distance', 'hours', 'speed', 'usage', 'trip']
        time_keywords = ['age', 'days', 'months', 'years']
        
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in engine_keywords):
            return 'Engine Parameters'
        elif any(keyword in feature_lower for keyword in electrical_keywords):
            return 'Electrical System'
        elif any(keyword in feature_lower for keyword in usage_keywords):
            return 'Usage Metrics'
        elif any(keyword in feature_lower for keyword in time_keywords):
            return 'Time-based Features'
        else:
            return 'Other Features'
    
    def _create_time_series_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series predictions table for Power BI."""
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        time_series_data = df.copy()
        time_series_data['timestamp'] = pd.to_datetime(time_series_data['timestamp'])
        time_series_data['date'] = time_series_data['timestamp'].dt.date
        time_series_data['hour'] = time_series_data['timestamp'].dt.hour
        time_series_data['day_of_week'] = time_series_data['timestamp'].dt.day_name()
        
        # Aggregate by date
        daily_stats = time_series_data.groupby('date').agg({
            'prediction': ['sum', 'count', 'mean'],
            'probability': ['mean', 'std', 'max']
        }).reset_index()
        
        # Flatten column names
        daily_stats.columns = ['_'.join(col).strip('_') for col in daily_stats.columns]
        
        # Rename columns
        daily_stats = daily_stats.rename(columns={
            'prediction_sum': 'maintenance_predictions',
            'prediction_count': 'total_predictions',
            'prediction_mean': 'prediction_rate',
            'probability_mean': 'avg_risk_score',
            'probability_std': 'risk_score_std',
            'probability_max': 'max_risk_score'
        })
        
        daily_stats['update_date'] = datetime.now().date()
        
        return daily_stats
    
    def _create_fleet_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fleet summary table for Power BI."""
        if 'vehicle_id' not in df.columns:
            return pd.DataFrame()
        
        fleet_summary = df.groupby('vehicle_id').agg({
            'prediction': ['sum', 'count', 'mean'],
            'probability': ['mean', 'max', 'min', 'std']
        }).reset_index()
        
        # Flatten column names
        fleet_summary.columns = ['_'.join(col).strip('_') for col in fleet_summary.columns]
        
        # Rename columns
        fleet_summary = fleet_summary.rename(columns={
            'prediction_sum': 'maintenance_alerts',
            'prediction_count': 'total_readings',
            'prediction_mean': 'alert_rate',
            'probability_mean': 'avg_risk_score',
            'probability_max': 'max_risk_score',
            'probability_min': 'min_risk_score',
            'probability_std': 'risk_score_std'
        })
        
        # Add risk categorization
        fleet_summary['risk_category'] = pd.cut(
            fleet_summary['avg_risk_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        fleet_summary['update_date'] = datetime.now().date()
        
        return fleet_summary
    
    def export_to_csv(self, datasets: Dict[str, pd.DataFrame], 
                     output_dir: str = 'powerbi/data'):
        """
        Export datasets to CSV files for Power BI.
        
        Args:
            datasets: Dictionary of DataFrames
            output_dir: Output directory for CSV files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, dataset_df in datasets.items():
            output_path = os.path.join(output_dir, f'{dataset_name}.csv')
            dataset_df.to_csv(output_path, index=False)
            self.logger.info(f"Exported {dataset_name} to {output_path}")
    
    def export_to_sqlite(self, datasets: Dict[str, pd.DataFrame],
                        db_path: str = 'powerbi/vehicle_predictions.db'):
        """
        Export datasets to SQLite database for Power BI.
        
        Args:
            datasets: Dictionary of DataFrames
            db_path: Path to SQLite database
        """
        import os
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            for dataset_name, dataset_df in datasets.items():
                dataset_df.to_sql(dataset_name, conn, if_exists='replace', index=False)
                self.logger.info(f"Exported {dataset_name} to SQLite database")
    
    def create_powerbi_connection_script(self, data_source: str = 'csv',
                                        connection_string: Optional[str] = None,
                                        output_path: str = 'powerbi/data_connection_script.py'):
        """
        Create Power BI data connection script.
        
        Args:
            data_source: Type of data source ('csv', 'sqlite', 'api')
            connection_string: Connection string for the data source
            output_path: Path to save the connection script
        """
        script_template = """# Power BI Data Connection Script
# Generated: {timestamp}
# Data Source: {data_source}

import pandas as pd
import numpy as np
from datetime import datetime

class PowerBIDataConnector:
    \"\"\"
    Power BI Data Connector for Vehicle Predictive Maintenance Dashboard.
    
    This script provides data loading functionality for Power BI dashboards.
    \"\"\"
    
    def __init__(self, data_source='{data_source}'):
        self.data_source = data_source
        self._setup_logging()
        
    def _setup_logging(self):
        \"\"\"Setup basic logging for data operations.\"\"\"
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_predictions_data(self):
        \"\"\"Load main predictions dataset.\"\"\"
        self.logger.info("Loading predictions data")
        return self._load_dataset('predictions')
    
    def load_model_metrics(self):
        \"\"\"Load model performance metrics.\"\"\"
        self.logger.info("Loading model metrics")
        return self._load_dataset('model_metrics')
    
    def load_business_metrics(self):
        \"\"\"Load business impact metrics.\"\"\"
        self.logger.info("Loading business metrics")
        return self._load_dataset('business_metrics')
    
    def load_feature_importance(self):
        \"\"\"Load feature importance data.\"\"\"
        self.logger.info("Loading feature importance")
        return self._load_dataset('feature_importance')
    
    def load_time_series_data(self):
        \"\"\"Load time series predictions data.\"\"\"
        self.logger.info("Loading time series data")
        return self._load_dataset('time_series_predictions')
    
    def load_fleet_summary(self):
        \"\"\"Load fleet summary data.\"\"\"
        self.logger.info("Loading fleet summary")
        return self._load_dataset('fleet_summary')
    
    def _load_dataset(self, dataset_name):
        \"\"\"Generic dataset loader based on data source.\"\"\"
        if self.data_source == 'csv':
            return self._load_csv_dataset(dataset_name)
        elif self.data_source == 'sqlite':
            return self._load_sqlite_dataset(dataset_name)
        elif self.data_source == 'api':
            return self._load_api_dataset(dataset_name)
        else:
            raise ValueError(f"Unsupported data source: {{self.data_source}}")
    
    def _load_csv_dataset(self, dataset_name):
        \"\"\"Load dataset from CSV file.\"\"\"
        filepath = f'data/{{dataset_name}}.csv'
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {{dataset_name}} from {{filepath}}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {{dataset_name}}: {{str(e)}}")
            return pd.DataFrame()
    
    def _load_sqlite_dataset(self, dataset_name):
        \"\"\"Load dataset from SQLite database.\"\"\"
        import sqlite3
        db_path = '{connection_string}' or 'vehicle_predictions.db'
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(f'SELECT * FROM {{dataset_name}}', conn)
                self.logger.info(f"Loaded {{dataset_name}} from SQLite database")
                return df
        except Exception as e:
            self.logger.error(f"Error loading {{dataset_name}}: {{str(e)}}")
            return pd.DataFrame()
    
    def _load_api_dataset(self, dataset_name):
        \"\"\"Load dataset from API endpoint.\"\"\"
        import requests
        api_url = '{connection_string}' or 'http://localhost:8000/api'
        try:
            response = requests.get(f'{{api_url}}/{{dataset_name}}')
            response.raise_for_status()
            df = pd.DataFrame(response.json())
            self.logger.info(f"Loaded {{dataset_name}} from API")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {{dataset_name}}: {{str(e)}}")
            return pd.DataFrame()
    
    def refresh_all_data(self):
        \"\"\"Refresh all datasets for Power BI dashboard.\"\"\"
        self.logger.info("Refreshing all Power BI datasets")
        
        data = {{
            'predictions': self.load_predictions_data(),
            'model_metrics': self.load_model_metrics(),
            'business_metrics': self.load_business_metrics(),
            'feature_importance': self.load_feature_importance(),
            'time_series_predictions': self.load_time_series_data(),
            'fleet_summary': self.load_fleet_summary()
        }}
        
        self.logger.info("Data refresh complete")
        return data

# Usage example
if __name__ == "__main__":
    connector = PowerBIDataConnector(data_source='{data_source}')
    all_data = connector.refresh_all_data()
    print(f"Loaded {{len(all_data)}} datasets for Power BI dashboard")
"""
        
        script_content = script_template.format(
            timestamp=datetime.now().isoformat(),
            data_source=data_source,
            connection_string=connection_string or ''
        )
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        self.logger.info(f"Power BI connection script created at {output_path}")