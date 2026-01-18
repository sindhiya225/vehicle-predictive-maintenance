# src/model_evaluation.py
"""
Model evaluation module for predictive maintenance.

Provides comprehensive evaluation metrics, visualization, and business impact analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
import logging
import json
import yaml
from datetime import datetime
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve

class ModelEvaluator:
    """Evaluate predictive maintenance models."""
    
    def __init__(self, config_path: str = 'config/settings.yaml'):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        plt.style.use('seaborn-v0_8-darkgrid')
        
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
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for positive class)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Calculating evaluation metrics")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = self._calculate_average_precision(y_true, y_prob)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
        
        # Calculate derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Business-oriented metrics
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        self.logger.info(f"Metrics calculated: {json.dumps({k: round(v, 4) if isinstance(v, float) else v 
                                                           for k, v in metrics.items() if k != 'confusion_matrix'})}")
        
        return metrics
    
    def _calculate_average_precision(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate average precision score."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # Use trapezoidal integration
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    def generate_classification_report(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Formatted classification report
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        return json.dumps(report, indent=2)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix with annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Maintenance', 'Maintenance'],
                   yticklabels=['No Maintenance', 'Maintenance'],
                   ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix - Predictive Maintenance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve with AUC score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc_score:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Predictive Maintenance')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = self._calculate_average_precision(y_true, y_prob)
        
        ax.plot(recall, precision, 'r-', 
                label=f'Precision-Recall (AP = {avg_precision:.3f})', 
                linewidth=2)
        
        # Plot no-skill line
        baseline = len(y_true[y_true==1]) / len(y_true)
        ax.plot([0, 1], [baseline, baseline], 'k--', 
                label='No-Skill Classifier', linewidth=1)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - Predictive Maintenance')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve to assess probability calibration.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        ax.plot(prob_pred, prob_true, 's-', label='Model Calibration', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve - Predictive Maintenance')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Calibration curve saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               save_path: Optional[str] = None,
                               top_n: int = 15) -> plt.Figure:
        """
        Plot feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            save_path: Path to save the figure
            top_n: Number of top features to display
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            self.logger.warning("Model doesn't have feature importances attribute")
            return fig
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), 
                      importance_df['importance'], 
                      color=colors)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances - Predictive Maintenance')
        ax.invert_yaxis()  # Highest importance at top
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  cost_matrix: Dict[str, float]) -> Dict:
        """
        Calculate business impact metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Dictionary with cost parameters
            
        Returns:
            Dictionary of business metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Extract costs from configuration
        preventive_maintenance_cost = cost_matrix.get('preventive_maintenance_cost', 500)
        reactive_maintenance_cost = cost_matrix.get('reactive_maintenance_cost', 2000)
        false_alarm_cost = cost_matrix.get('false_alarm_cost', 100)
        missed_failure_cost = cost_matrix.get('missed_failure_cost', 5000)
        
        # Calculate costs
        total_cost = (
            tp * preventive_maintenance_cost +
            fp * (preventive_maintenance_cost + false_alarm_cost) +
            fn * missed_failure_cost +
            tn * 0  # No cost for correct negative predictions
        )
        
        # Cost without predictive maintenance (all reactive)
        cost_without_pm = len(y_true) * reactive_maintenance_cost
        
        # Calculate savings
        cost_savings = cost_without_pm - total_cost
        savings_percentage = (cost_savings / cost_without_pm) * 100
        
        business_metrics = {
            'total_cost_with_pm': float(total_cost),
            'total_cost_without_pm': float(cost_without_pm),
            'cost_savings': float(cost_savings),
            'savings_percentage': float(savings_percentage),
            'preventive_maintenances': int(tp + fp),
            'reactive_maintenances_avoided': int(tp),
            'missed_failures': int(fn),
            'false_alarms': int(fp),
            'cost_per_vehicle': float(total_cost / len(y_true))
        }
        
        self.logger.info(f"Business metrics calculated: Savings of ${cost_savings:,.2f} "
                        f"({savings_percentage:.1f}%)")
        
        return business_metrics
    
    def generate_evaluation_report(self, metrics: Dict, business_metrics: Dict,
                                  model_name: str, dataset_info: Dict) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Model performance metrics
            business_metrics: Business impact metrics
            model_name: Name of the model
            dataset_info: Information about the dataset
            
        Returns:
            Formatted evaluation report
        """
        report = {
            'model_evaluation_report': {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'dataset_info': dataset_info,
                'performance_metrics': metrics,
                'business_metrics': business_metrics,
                'model_interpretation': self._generate_interpretation(metrics, business_metrics)
            }
        }
        
        return json.dumps(report, indent=2)
    
    def _generate_interpretation(self, metrics: Dict, business_metrics: Dict) -> Dict:
        """Generate human-readable interpretation of results."""
        interpretation = {
            'performance_summary': '',
            'business_impact': '',
            'recommendations': []
        }
        
        # Performance summary
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        interpretation['performance_summary'] = (
            f"The model achieves {accuracy:.1%} accuracy with {precision:.1%} precision "
            f"and {recall:.1%} recall. This indicates good predictive capability for "
            f"vehicle maintenance needs."
        )
        
        # Business impact
        savings = business_metrics.get('cost_savings', 0)
        savings_pct = business_metrics.get('savings_percentage', 0)
        
        interpretation['business_impact'] = (
            f"Implementing this predictive maintenance system could save "
            f"${savings:,.2f} ({savings_pct:.1f}%) compared to reactive maintenance. "
            f"It would enable {business_metrics.get('reactive_maintenances_avoided', 0)} "
            f"preventive interventions while generating "
            f"{business_metrics.get('false_alarms', 0)} false alarms."
        )
        
        # Recommendations
        if metrics.get('recall', 0) < 0.7:
            interpretation['recommendations'].append(
                "Consider tuning the model to improve recall (reduce missed failures), "
                "as these are costly in maintenance scenarios."
            )
        
        if metrics.get('precision', 0) < 0.6:
            interpretation['recommendations'].append(
                "The model has relatively high false alarms. Consider adjusting the "
                "decision threshold or collecting more data on normal operation."
            )
        
        if business_metrics.get('false_alarms', 0) > business_metrics.get('reactive_maintenances_avoided', 0):
            interpretation['recommendations'].append(
                "The false alarm rate is high compared to true detections. "
                "Review feature engineering and consider adding more contextual data."
            )
        
        if not interpretation['recommendations']:
            interpretation['recommendations'].append(
                "The model performs well across both technical and business metrics. "
                "Proceed with deployment and monitor performance in production."
            )
        
        return interpretation