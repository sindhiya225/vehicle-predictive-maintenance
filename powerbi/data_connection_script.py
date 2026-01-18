"""
PowerBI data connection and visualization preparation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class PowerBIDataExporter:
    """Prepare data for PowerBI visualization dashboard"""
    
    def __init__(self, model_path='models/trained_models'):
        self.model_path = model_path
        self.predictions = None
        self.business_metrics = None
        
    def prepare_dashboard_data(self, data_path, predictions_path=None):
        """Prepare comprehensive dataset for PowerBI dashboard"""
        
        print("Loading and processing data for PowerBI...")
        
        # Load original data
        df = pd.read_csv(data_path, parse_dates=['date'])
        
        # Load predictions if available
        if predictions_path:
            pred_df = pd.read_csv(predictions_path)
            df = pd.merge(df, pred_df, on=['vehicle_id', 'date'], how='left')
        
        # Calculate key metrics for dashboard
        dashboard_data = self._calculate_kpis(df)
        
        # Prepare vehicle-level summary
        vehicle_summary = self._create_vehicle_summary(df)
        
        # Prepare time series data
        time_series = self._create_time_series_data(df)
        
        # Prepare maintenance recommendations
        recommendations = self._create_recommendations(df)
        
        # Prepare feature importance data
        feature_importance = self._prepare_feature_importance()
        
        # Export to PowerBI compatible format
        self._export_to_files({
            'dashboard_kpis': dashboard_data,
            'vehicle_summary': vehicle_summary,
            'time_series': time_series,
            'recommendations': recommendations,
            'feature_importance': feature_importance
        })
        
        print("PowerBI data preparation complete!")
        return dashboard_data
    
    def _calculate_kpis(self, df):
        """Calculate key performance indicators"""
        
        kpis = {
            'date_generated': datetime.now().strftime('%Y-%m-%d'),
            'total_vehicles': df['vehicle_id'].nunique(),
            'total_observations': len(df),
            'failure_rate': df['failure_in_next_7d'].mean(),
            
            # Prediction metrics (if predictions available)
            'predicted_failures': df.get('predicted_failure', 0).sum() if 'predicted_failure' in df.columns else 0,
            'true_positives': df.get('true_positive', 0).sum() if 'true_positive' in df.columns else 0,
            'false_positives': df.get('false_positive', 0).sum() if 'false_positive' in df.columns else 0,
            
            # Business metrics
            'avg_engine_temp': df['engine_temperature'].mean(),
            'avg_oil_pressure': df['oil_pressure'].mean(),
            'avg_fuel_efficiency': (df['distance_km'] / df['fuel_consumption']).mean() 
                                  if 'fuel_consumption' in df.columns else 0,
            
            # Maintenance metrics
            'vehicles_requiring_maintenance': df.get('maintenance_required', 0).sum() 
                                            if 'maintenance_required' in df.columns else 0,
            'high_risk_vehicles': df.get('risk_score', lambda x: (x > 0.7).sum() if 'risk_score' in df.columns else 0),
        }
        
        # Convert to DataFrame for PowerBI
        kpi_df = pd.DataFrame([kpis])
        
        # Calculate weekly trends
        weekly_trends = df.groupby(pd.Grouper(key='date', freq='W')).agg({
            'failure_in_next_7d': 'mean',
            'engine_temperature': 'mean',
            'distance_km': 'sum'
        }).reset_index()
        
        return {
            'kpis': kpi_df,
            'weekly_trends': weekly_trends
        }
    
    def _create_vehicle_summary(self, df):
        """Create vehicle-level summary for drill-down analysis"""
        
        vehicle_summary = df.groupby('vehicle_id').agg({
            'date': ['min', 'max', 'count'],
            'engine_temperature': ['mean', 'max', 'std'],
            'oil_pressure': ['mean', 'min', 'std'],
            'battery_voltage': ['mean', 'min'],
            'distance_km': 'sum',
            'failure_in_next_7d': 'sum',
            'total_distance': 'last'
        }).round(2)
        
        # Flatten multi-level columns
        vehicle_summary.columns = ['_'.join(col).strip() for col in vehicle_summary.columns.values]
        vehicle_summary = vehicle_summary.reset_index()
        
        # Calculate risk scores
        vehicle_summary['risk_score'] = self._calculate_risk_score(vehicle_summary)
        vehicle_summary['risk_category'] = pd.cut(
            vehicle_summary['risk_score'],
            bins=[0, 0.3, 0.7, 1],
            labels=['Low', 'Medium', 'High']
        )
        
        # Add maintenance priority
        vehicle_summary['maintenance_priority'] = vehicle_summary.apply(
            lambda x: 'Immediate' if x['risk_score'] > 0.8 else 
                     'Within 7 days' if x['risk_score'] > 0.5 else
                     'Within 30 days' if x['risk_score'] > 0.3 else 'Monitor',
            axis=1
        )
        
        return vehicle_summary
    
    def _calculate_risk_score(self, vehicle_data):
        """Calculate risk score for each vehicle"""
        
        risk_factors = []
        
        # Engine temperature risk
        if 'engine_temperature_mean' in vehicle_data.columns:
            temp_risk = np.where(
                vehicle_data['engine_temperature_mean'] > 105, 0.8,
                np.where(vehicle_data['engine_temperature_mean'] > 100, 0.5,
                        np.where(vehicle_data['engine_temperature_mean'] < 85, 0.3, 0.1))
            )
            risk_factors.append(temp_risk)
        
        # Oil pressure risk
        if 'oil_pressure_mean' in vehicle_data.columns:
            oil_risk = np.where(
                vehicle_data['oil_pressure_mean'] < 30, 0.7,
                np.where(vehicle_data['oil_pressure_mean'] < 35, 0.4,
                        np.where(vehicle_data['oil_pressure_mean'] > 50, 0.5, 0.1))
            )
            risk_factors.append(oil_risk)
        
        # Battery risk
        if 'battery_voltage_min' in vehicle_data.columns:
            battery_risk = np.where(
                vehicle_data['battery_voltage_min'] < 11.8, 0.6,
                np.where(vehicle_data['battery_voltage_min'] < 12.2, 0.3, 0.1)
            )
            risk_factors.append(battery_risk)
        
        # Usage risk
        if 'distance_km_sum' in vehicle_data.columns:
            usage_risk = np.where(
                vehicle_data['distance_km_sum'] > 5000, 0.4,
                np.where(vehicle_data['distance_km_sum'] > 3000, 0.2, 0.1)
            )
            risk_factors.append(usage_risk)
        
        # Calculate weighted average risk
        if risk_factors:
            risk_score = np.mean(risk_factors, axis=0)
            return np.clip(risk_score, 0, 1)
        else:
            return np.zeros(len(vehicle_data))
    
    def _create_time_series_data(self, df):
        """Create time series data for trend analysis"""
        
        # Daily aggregates
        daily_data = df.groupby('date').agg({
            'vehicle_id': 'nunique',
            'engine_temperature': 'mean',
            'oil_pressure': 'mean',
            'failure_in_next_7d': 'sum',
            'distance_km': 'sum'
        }).reset_index()
        
        daily_data.columns = ['date', 'active_vehicles', 'avg_engine_temp', 
                             'avg_oil_pressure', 'failures', 'total_distance']
        
        # Rolling averages
        daily_data['avg_engine_temp_7d_ma'] = daily_data['avg_engine_temp'].rolling(7).mean()
        daily_data['failure_rate_7d_ma'] = daily_data['failures'].rolling(7).mean()
        
        return daily_data
    
    def _create_recommendations(self, df):
        """Create maintenance recommendations"""
        
        recommendations = []
        
        # Sample recommendation logic
        high_temp_vehicles = df[df['engine_temperature'] > 105]['vehicle_id'].unique()[:10]
        low_oil_vehicles = df[df['oil_pressure'] < 30]['vehicle_id'].unique()[:10]
        high_mileage = df.sort_values('total_distance', ascending=False)['vehicle_id'].unique()[:5]
        
        for vid in high_temp_vehicles:
            recommendations.append({
                'vehicle_id': vid,
                'issue': 'High Engine Temperature',
                'severity': 'High',
                'recommendation': 'Check cooling system, thermostat, and coolant levels',
                'estimated_cost': 250,
                'estimated_downtime_hours': 4
            })
        
        for vid in low_oil_vehicles:
            recommendations.append({
                'vehicle_id': vid,
                'issue': 'Low Oil Pressure',
                'severity': 'Critical',
                'recommendation': 'Check oil level, oil pump, and for leaks',
                'estimated_cost': 500,
                'estimated_downtime_hours': 8
            })
        
        for vid in high_mileage:
            recommendations.append({
                'vehicle_id': vid,
                'issue': 'High Mileage Maintenance',
                'severity': 'Medium',
                'recommendation': 'Schedule comprehensive maintenance check',
                'estimated_cost': 800,
                'estimated_downtime_hours': 24
            })
        
        return pd.DataFrame(recommendations)
    
    def _prepare_feature_importance(self):
        """Prepare feature importance data for visualization"""
        
        try:
            feature_imp = pd.read_csv(f'{self.model_path}/feature_importance.csv')
            
            # Categorize features
            def categorize_feature(feature_name):
                if 'temperature' in feature_name.lower():
                    return 'Engine Temperature'
                elif 'oil' in feature_name.lower():
                    return 'Oil System'
                elif 'battery' in feature_name.lower():
                    return 'Electrical System'
                elif 'distance' in feature_name.lower() or 'mileage' in feature_name.lower():
                    return 'Usage Patterns'
                elif 'rolling' in feature_name.lower() or 'trend' in feature_name.lower():
                    return 'Historical Trends'
                elif 'violation' in feature_name.lower():
                    return 'Threshold Violations'
                else:
                    return 'Other Features'
            
            feature_imp['category'] = feature_imp['feature'].apply(categorize_feature)
            
            return feature_imp.head(20)  # Top 20 features
            
        except FileNotFoundError:
            # Return sample data if file not found
            return pd.DataFrame({
                'feature': ['engine_temperature_trend', 'oil_pressure_violations', 
                           'battery_voltage_min', 'distance_7d_avg', 'vehicle_age'],
                'importance': [0.15, 0.12, 0.10, 0.08, 0.07],
                'category': ['Engine Temperature', 'Oil System', 'Electrical System', 
                            'Usage Patterns', 'Historical Trends']
            })
    
    def _export_to_files(self, data_dict):
        """Export all data to CSV files for PowerBI"""
        
        export_dir = 'powerbi/data'
        os.makedirs(export_dir, exist_ok=True)
        
        for name, data in data_dict.items():
            if isinstance(data, dict):
                for subname, subdata in data.items():
                    filepath = f'{export_dir}/{name}_{subname}.csv'
                    subdata.to_csv(filepath, index=False)
                    print(f"Exported: {filepath}")
            else:
                filepath = f'{export_dir}/{name}.csv'
                data.to_csv(filepath, index=False)
                print(f"Exported: {filepath}")
        
        # Create PowerBI connection string file
        self._create_powerbi_connection_file(export_dir)
    
    def _create_powerbi_connection_file(self, data_dir):
        """Create PowerBI data connection script"""
        
        connection_script = f"""
        // PowerBI Data Connection Script
        // Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        let
            Source = Folder.Files("{os.path.abspath(data_dir)}"),
            #"Filtered CSV" = Table.SelectRows(Source, each Text.EndsWith([Name], ".csv")),
            #"Combined Data" = Table.Combine(
            #    List.Transform(
            #        #"Filtered CSV"[Content],
            #        each Csv.Document(_, [Delimiter=",", Encoding=1252, QuoteStyle=QuoteStyle.None])
            #    )
            // Import each CSV file as separate table
            kpis = Csv.Document(File.Contents("{os.path.abspath(data_dir)}/dashboard_kpis_kpis.csv")),
            vehicle_summary = Csv.Document(File.Contents("{os.path.abspath(data_dir)}/vehicle_summary.csv")),
            time_series = Csv.Document(File.Contents("{os.path.abspath(data_dir)}/time_series.csv")),
            recommendations = Csv.Document(File.Contents("{os.path.abspath(data_dir)}/recommendations.csv")),
            feature_importance = Csv.Document(File.Contents("{os.path.abspath(data_dir)}/feature_importance.csv"))
        in
            #"Create named tables for each dataset"
        """
        
        with open(f'{data_dir}/powerbi_connection.txt', 'w') as f:
            f.write(connection_script)
        
        print(f"PowerBI connection script created: {data_dir}/powerbi_connection.txt")

# Usage example
if __name__ == "__main__":
    exporter = PowerBIDataExporter()
    dashboard_data = exporter.prepare_dashboard_data('data/raw/vehicle_sensor_data.csv')