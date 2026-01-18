"""
Model training for predictive maintenance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc)
import joblib
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from .feature_engineering import VehicleFeatureEngineer

class PredictiveMaintenanceModel:
    """Train and optimize predictive maintenance models"""
    
    def __init__(self, target_column: str = 'failure_in_next_7d'):
        self.target_column = target_column
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.scaler = None
        self.feature_engineer = VehicleFeatureEngineer()
        
    def prepare_data(self, data_path: str, test_size: float = 0.2):
        """Load, preprocess, and split data"""
        print("Loading data...")
        df = pd.read_csv(data_path, parse_dates=['date'])
        
        print("Engineering features...")
        X = self.feature_engineer.fit_transform(df)
        
        # Separate features and target
        y = df[self.target_column].astype(int)
        
        # Drop target columns from features
        drop_cols = ['failure_in_next_7d', 'failure_in_next_30d', 'date']
        drop_cols = [c for c in drop_cols if c in X.columns]
        X = X.drop(columns=drop_cols)
        
        # Remove non-numeric columns for modeling
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Dropping non-numeric columns: {list(non_numeric)}")
            X = X.drop(columns=non_numeric)
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        print(f"Data shape: {X.shape}")
        print(f"Failure rate: {y.mean():.2%}")
        
        # Time-based split (maintain temporal order)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Handle class imbalance with SMOTE
        print("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Training set after SMOTE: {X_train_resampled.shape}")
        print(f"New failure rate: {y_train_resampled.mean():.2%}")
        
        return X_train_resampled, X_test, y_train_resampled, y_test, X.columns
    
    def train_models(self, X_train, y_train, X_test, y_test, feature_names):
        """Train multiple models and select the best one"""
        
        # Define models with initial hyperparameters
        models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_score = 0
        self.best_model = None
        
        for model_name, model_config in models.items():
            print(f"\nTraining {model_name}...")
            print("-" * 50)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=tscv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store model
            self.models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Test AUC: {test_auc:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")
            
            # Update best model
            if test_auc > best_score:
                best_score = test_auc
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name
        
        print(f"\n{'='*60}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Test AUC: {best_score:.4f}")
        
        # Get feature importance from best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.best_model
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                           index=['Actual Negative', 'Actual Positive'],
                           columns=['Predicted Negative', 'Predicted Positive'])
        
        # Classification report
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        metrics['auc_pr'] = pr_auc
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        print("\nConfusion Matrix:")
        print(cm_df)
        
        print("\nClassification Report:")
        print(pd.DataFrame(cr).transpose())
        
        print("\nKey Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Calculate business metrics
        self._calculate_business_metrics(y_test, y_pred, cm)
        
        return metrics, cm_df, cr
    
    def _calculate_business_metrics(self, y_test, y_pred, cm):
        """Calculate business impact metrics"""
        # Costs and savings (example values - adjust based on business case)
        cost_unplanned_downtime = 5000  # $ per occurrence
        cost_planned_maintenance = 1000  # $ per occurrence
        savings_per_early_detection = 4000  # $ savings
        
        # From confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        # Business calculations
        total_failures = tp + fn
        detected_failures = tp
        false_alarms = fp
        
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Cost savings
        savings = (detected_failures * savings_per_early_detection) - (false_alarms * cost_planned_maintenance)
        
        print("\n" + "="*60)
        print("BUSINESS IMPACT ANALYSIS")
        print("="*60)
        print(f"\nTotal Actual Failures: {total_failures}")
        print(f"Failures Detected Early: {detected_failures} ({detection_rate:.1%})")
        print(f"False Alarms: {false_alarms} ({false_alarm_rate:.1%})")
        print(f"\nEstimated Cost Savings: ${savings:,.0f}")
        print(f"ROI (first year): {savings/(cost_planned_maintenance*100):.1f}x")
        
        return {
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'estimated_savings': savings
        }
    
    def save_model(self, model_path: str):
        """Save trained model and metadata"""
        if self.best_model:
            # Save model
            joblib.dump(self.best_model, f"{model_path}/best_model.pkl")
            
            # Save feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_csv(f"{model_path}/feature_importance.csv", index=False)
            
            # Save model metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'features_used': len(self.feature_importance) if self.feature_importance is not None else 0,
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(f"{model_path}/model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        self.best_model = joblib.load(f"{model_path}/best_model.pkl")
        return self.best_model

def main():
    """Main training pipeline"""
    print("="*60)
    print("VEHICLE PREDICTIVE MAINTENANCE - MODEL TRAINING")
    print("="*60)
    
    # Initialize model trainer
    trainer = PredictiveMaintenanceModel(target_column='failure_in_next_7d')
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
        'data/raw/vehicle_sensor_data.csv',
        test_size=0.2
    )
    
    # Train models
    best_model = trainer.train_models(X_train, y_train, X_test, y_test, feature_names)
    
    # Evaluate best model
    metrics, cm, cr = trainer.evaluate_model(best_model, X_test, y_test)
    
    # Save model
    trainer.save_model('models/trained_models')
    
    # Print feature importance
    if trainer.feature_importance is not None:
        print("\n" + "="*60)
        print("TOP 20 FEATURES BY IMPORTANCE")
        print("="*60)
        print(trainer.feature_importance.head(20).to_string())
    
    return trainer

if __name__ == "__main__":
    trainer = main()