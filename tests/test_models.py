# tests/test_models.py
"""
Unit tests for model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tempfile
import os
import json
import yaml

# Import the modules to test
import sys
sys.path.append('src')
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from data_preprocessing import DataPreprocessor

class TestModelTraining:
    """Test class for ModelTrainer."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        config = {
            'model_training': {
                'test_size': 0.2,
                'random_state': 42,
                'models': {
                    'logistic_regression': {
                        'max_iter': 1000,
                        'random_state': 42
                    },
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    }
                },
                'cross_validation': {
                    'cv': 5,
                    'scoring': 'f1'
                }
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        
        n_samples = 1000
        n_features = 10
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic target with some relationship to features
        # Make the first 3 features predictive
        y_proba = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3)))
        y = (y_proba > 0.5).astype(int)
        
        # Add some noise
        noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        y[noise_indices] = 1 - y[noise_indices]
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    @pytest.fixture
    def model_trainer(self, sample_config):
        """Create ModelTrainer instance for testing."""
        return ModelTrainer(sample_config)
    
    def test_train_logistic_regression(self, model_trainer, sample_training_data):
        """Test training logistic regression model."""
        X, y = sample_training_data
        
        model, metrics = model_trainer.train_logistic_regression(X, y)
        
        assert model is not None
        assert isinstance(metrics, dict)
        
        # Check model has been trained
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_train_random_forest(self, model_trainer, sample_training_data):
        """Test training random forest model."""
        X, y = sample_training_data
        
        model, metrics = model_trainer.train_random_forest(X, y)
        
        assert model is not None
        assert isinstance(metrics, dict)
        
        # Check model has been trained
        assert hasattr(model, 'feature_importances_')
        assert hasattr(model, 'estimators_')
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check feature importances
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert all(0 <= imp <= 1 for imp in importances)
        assert np.sum(importances) > 0
    
    def test_train_xgboost(self, model_trainer, sample_training_data):
        """Test training XGBoost model."""
        X, y = sample_training_data
        
        model, metrics = model_trainer.train_xgboost(X, y)
        
        assert model is not None
        assert isinstance(metrics, dict)
        
        # Check model has been trained
        assert hasattr(model, 'feature_importances_')
        
        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Feature importances should be available
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
    
    def test_train_all_models(self, model_trainer, sample_training_data):
        """Test training all models."""
        X, y = sample_training_data
        
        models_dict, results_df = model_trainer.train_all_models(X, y)
        
        assert isinstance(models_dict, dict)
        assert isinstance(results_df, pd.DataFrame)
        
        # Check all models are trained
        assert 'logistic_regression' in models_dict
        assert 'random_forest' in models_dict
        
        # Check results DataFrame
        assert len(results_df) > 0
        assert 'model_name' in results_df.columns
        assert 'accuracy' in results_df.columns
        assert 'precision' in results_df.columns
        assert 'recall' in results_df.columns
        assert 'f1_score' in results_df.columns
    
    def test_hyperparameter_tuning(self, model_trainer, sample_training_data):
        """Test hyperparameter tuning."""
        X, y = sample_training_data
        
        # Use smaller dataset for faster testing
        X_small = X.iloc[:200]
        y_small = y.iloc[:200]
        
        best_model, best_params, best_score = model_trainer.tune_hyperparameters(
            X_small, y_small, model_type='random_forest'
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        
        # Check best_params contains expected parameters
        assert 'n_estimators' in best_params or 'max_depth' in best_params
        
        # Check score is reasonable
        assert 0 <= best_score <= 1
    
    def test_cross_validation(self, model_trainer, sample_training_data):
        """Test cross-validation."""
        X, y = sample_training_data
        
        # Use smaller dataset for faster testing
        X_small = X.iloc[:200]
        y_small = y.iloc[:200]
        
        cv_scores = model_trainer.cross_validate(
            X_small, y_small, model_type='logistic_regression'
        )
        
        assert isinstance(cv_scores, dict)
        
        # Check cv scores structure
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert 'all_scores' in cv_scores
        
        # Check scores are reasonable
        assert 0 <= cv_scores['mean_score'] <= 1
        assert 0 <= cv_scores['std_score'] <= 1
    
    def test_save_and_load_model(self, model_trainer, sample_training_data, tmp_path):
        """Test model saving and loading."""
        X, y = sample_training_data
        
        # Train a model
        model, metrics = model_trainer.train_logistic_regression(X.iloc[:100], y.iloc[:100])
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        metadata = {
            'training_date': '2023-01-01',
            'features': list(X.columns),
            'metrics': metrics
        }
        
        model_trainer.save_model(model, str(model_path), metadata)
        
        # Check file exists
        assert model_path.exists()
        
        # Load model
        loaded_model, loaded_metadata = model_trainer.load_model(str(model_path))
        
        # Check model loaded correctly
        assert loaded_model is not None
        assert isinstance(loaded_metadata, dict)
        
        # Check predictions are consistent
        X_test = X.iloc[100:110]
        original_preds = model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        
        assert np.array_equal(original_preds, loaded_preds)
        
        # Check metadata
        assert 'training_date' in loaded_metadata
        assert 'features' in loaded_metadata
        assert 'metrics' in loaded_metadata
    
    def test_feature_selection(self, model_trainer, sample_training_data):
        """Test feature selection functionality."""
        X, y = sample_training_data
        
        selected_features = model_trainer.select_features(X, y, method='correlation', top_k=5)
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 5
        assert all(feat in X.columns for feat in selected_features)
        
        # Test with different selection methods
        selected_features_rf = model_trainer.select_features(X, y, method='random_forest', top_k=3)
        assert len(selected_features_rf) <= 3
    
    def test_model_ensemble(self, model_trainer, sample_training_data):
        """Test model ensemble creation."""
        X, y = sample_training_data
        
        # Use smaller dataset for faster testing
        X_small = X.iloc[:200]
        y_small = y.iloc[:200]
        
        ensemble_model, metrics = model_trainer.create_ensemble(
            X_small, y_small, models=['logistic_regression', 'random_forest']
        )
        
        assert ensemble_model is not None
        assert isinstance(metrics, dict)
        
        # Check ensemble can make predictions
        predictions = ensemble_model.predict(X_small.iloc[:10])
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})
    
    def test_handle_imbalanced_data(self, model_trainer):
        """Test handling of imbalanced data."""
        # Create imbalanced dataset
        np.random.seed(42)
        X_imbalanced = np.random.randn(1000, 5)
        y_imbalanced = np.array([0] * 900 + [1] * 100)  # 90% negative, 10% positive
        
        X_df = pd.DataFrame(X_imbalanced, columns=[f'feature_{i}' for i in range(5)])
        
        # Train model with class weighting
        model, metrics = model_trainer.train_random_forest(
            X_df, y_imbalanced, handle_imbalance=True
        )
        
        assert model is not None
        assert 'recall' in metrics
        
        # Check recall is reasonable (should be better than random)
        assert metrics['recall'] > 0.1
    
    def test_edge_cases(self, model_trainer):
        """Test edge cases in model training."""
        # Test with empty data
        X_empty = pd.DataFrame()
        y_empty = pd.Series([], dtype=int)
        
        with pytest.raises(ValueError):
            model_trainer.train_logistic_regression(X_empty, y_empty)
        
        # Test with single class
        X_single = pd.DataFrame({'feature': [1, 2, 3]})
        y_single = pd.Series([0, 0, 0])  # Only one class
        
        with pytest.raises(ValueError):
            model_trainer.train_logistic_regression(X_single, y_single)
        
        # Test with NaN in features
        X_nan = pd.DataFrame({'feature': [1, np.nan, 3]})
        y_nan = pd.Series([0, 1, 0])
        
        model, metrics = model_trainer.train_logistic_regression(X_nan, y_nan)
        assert model is not None
        # Model should handle NaN internally or through preprocessing

class TestModelEvaluation:
    """Test class for ModelEvaluator."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        config = {
            'business_metrics': {
                'preventive_maintenance_cost': 500,
                'reactive_maintenance_cost': 2000,
                'false_alarm_cost': 100,
                'missed_failure_cost': 5000
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        
        n_samples = 1000
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_pred = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        y_prob = np.random.rand(n_samples)
        
        return y_true, y_pred, y_prob
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.choice([0, 1], size=100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    @pytest.fixture
    def model_evaluator(self, sample_config):
        """Create ModelEvaluator instance for testing."""
        return ModelEvaluator(sample_config)
    
    def test_calculate_metrics(self, model_evaluator, sample_predictions):
        """Test metrics calculation."""
        y_true, y_pred, y_prob = sample_predictions
        
        metrics = model_evaluator.calculate_metrics(y_true, y_pred, y_prob)
        
        assert isinstance(metrics, dict)
        
        # Check required metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check confusion matrix
        assert 'confusion_matrix' in metrics
        cm = metrics['confusion_matrix']
        assert 'true_negative' in cm
        assert 'false_positive' in cm
        assert 'false_negative' in cm
        assert 'true_positive' in cm
        
        # Check values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_generate_classification_report(self, model_evaluator, sample_predictions):
        """Test classification report generation."""
        y_true, y_pred, _ = sample_predictions
        
        report = model_evaluator.generate_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        
        # Should be valid JSON
        report_dict = json.loads(report)
        assert '0' in report_dict or '1' in report_dict
        assert 'accuracy' in report_dict
    
    def test_plot_confusion_matrix(self, model_evaluator, sample_predictions, tmp_path):
        """Test confusion matrix plotting."""
        y_true, y_pred, _ = sample_predictions
        
        save_path = tmp_path / "confusion_matrix.png"
        
        fig = model_evaluator.plot_confusion_matrix(y_true, y_pred, str(save_path))
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_roc_curve(self, model_evaluator, sample_predictions, tmp_path):
        """Test ROC curve plotting."""
        y_true, _, y_prob = sample_predictions
        
        save_path = tmp_path / "roc_curve.png"
        
        fig = model_evaluator.plot_roc_curve(y_true, y_prob, str(save_path))
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_precision_recall_curve(self, model_evaluator, sample_predictions, tmp_path):
        """Test precision-recall curve plotting."""
        y_true, _, y_prob = sample_predictions
        
        save_path = tmp_path / "pr_curve.png"
        
        fig = model_evaluator.plot_precision_recall_curve(y_true, y_prob, str(save_path))
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_feature_importance(self, model_evaluator, sample_model, tmp_path):
        """Test feature importance plotting."""
        feature_names = [f'feature_{i}' for i in range(10)]
        
        save_path = tmp_path / "feature_importance.png"
        
        fig = model_evaluator.plot_feature_importance(
            sample_model, feature_names, str(save_path), top_n=5
        )
        
        assert fig is not None
        assert save_path.exists()
    
    def test_calculate_business_metrics(self, model_evaluator, sample_predictions):
        """Test business metrics calculation."""
        y_true, y_pred, _ = sample_predictions
        
        cost_matrix = {
            'preventive_maintenance_cost': 500,
            'reactive_maintenance_cost': 2000,
            'false_alarm_cost': 100,
            'missed_failure_cost': 5000
        }
        
        business_metrics = model_evaluator.calculate_business_metrics(
            y_true, y_pred, cost_matrix
        )
        
        assert isinstance(business_metrics, dict)
        
        # Check required business metrics
        assert 'total_cost_with_pm' in business_metrics
        assert 'total_cost_without_pm' in business_metrics
        assert 'cost_savings' in business_metrics
        assert 'savings_percentage' in business_metrics
        
        # Check operational metrics
        assert 'preventive_maintenances' in business_metrics
        assert 'reactive_maintenances_avoided' in business_metrics
        assert 'missed_failures' in business_metrics
        assert 'false_alarms' in business_metrics
        
        # Check logical relationships
        assert business_metrics['total_cost_with_pm'] >= 0
        assert business_metrics['cost_savings'] >= -business_metrics['total_cost_with_pm']
    
    def test_generate_evaluation_report(self, model_evaluator, sample_predictions):
        """Test evaluation report generation."""
        y_true, y_pred, y_prob = sample_predictions
        
        metrics = model_evaluator.calculate_metrics(y_true, y_pred, y_prob)
        
        cost_matrix = {
            'preventive_maintenance_cost': 500,
            'reactive_maintenance_cost': 2000,
            'false_alarm_cost': 100,
            'missed_failure_cost': 5000
        }
        
        business_metrics = model_evaluator.calculate_business_metrics(
            y_true, y_pred, cost_matrix
        )
        
        dataset_info = {
            'n_samples': len(y_true),
            'positive_class_ratio': y_true.mean(),
            'features': ['feature_1', 'feature_2', 'feature_3']
        }
        
        report = model_evaluator.generate_evaluation_report(
            metrics, business_metrics, 'test_model', dataset_info
        )
        
        assert isinstance(report, str)
        
        # Should be valid JSON
        report_dict = json.loads(report)
        assert 'model_evaluation_report' in report_dict
        assert 'performance_metrics' in report_dict['model_evaluation_report']
        assert 'business_metrics' in report_dict['model_evaluation_report']
        assert 'model_interpretation' in report_dict['model_evaluation_report']
    
    def test_calibration_curve_plotting(self, model_evaluator, sample_predictions, tmp_path):
        """Test calibration curve plotting."""
        y_true, _, y_prob = sample_predictions
        
        save_path = tmp_path / "calibration_curve.png"
        
        fig = model_evaluator.plot_calibration_curve(y_true, y_prob, str(save_path))
        
        assert fig is not None
        assert save_path.exists()
    
    def test_edge_cases_evaluation(self, model_evaluator):
        """Test edge cases in model evaluation."""
        # Test with all correct predictions
        y_true_all_correct = np.array([0, 1, 0, 1])
        y_pred_all_correct = np.array([0, 1, 0, 1])
        y_prob_all_correct = np.array([0.1, 0.9, 0.2, 0.8])
        
        metrics = model_evaluator.calculate_metrics(
            y_true_all_correct, y_pred_all_correct, y_prob_all_correct
        )
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        
        # Test with all incorrect predictions
        y_true_all_wrong = np.array([0, 1, 0, 1])
        y_pred_all_wrong = np.array([1, 0, 1, 0])
        
        metrics = model_evaluator.calculate_metrics(
            y_true_all_wrong, y_pred_all_wrong, None
        )
        
        assert metrics['accuracy'] == 0.0
        
        # Test with single class in predictions
        y_true_single = np.array([0, 0, 0, 0])
        y_pred_single = np.array([0, 0, 0, 0])
        
        metrics = model_evaluator.calculate_metrics(y_true_single, y_pred_single, None)
        
        assert metrics['precision'] == 0  # Because no positive predictions
    
    def test_perfect_and_worst_case_business_metrics(self, model_evaluator):
        """Test business metrics in perfect and worst cases."""
        # Perfect predictions
        y_true_perfect = np.array([0, 1, 0, 1, 0, 1])
        y_pred_perfect = np.array([0, 1, 0, 1, 0, 1])
        
        cost_matrix = {
            'preventive_maintenance_cost': 500,
            'reactive_maintenance_cost': 2000,
            'false_alarm_cost': 100,
            'missed_failure_cost': 5000
        }
        
        perfect_metrics = model_evaluator.calculate_business_metrics(
            y_true_perfect, y_pred_perfect, cost_matrix
        )
        
        # Should have no missed failures and no false alarms
        assert perfect_metrics['missed_failures'] == 0
        assert perfect_metrics['false_alarms'] == 0
        
        # Worst predictions (all wrong)
        y_pred_worst = 1 - y_true_perfect
        
        worst_metrics = model_evaluator.calculate_business_metrics(
            y_true_perfect, y_pred_worst, cost_matrix
        )
        
        # Should have maximum missed failures and false alarms
        assert worst_metrics['missed_failures'] == 3  # Half the samples are positive
        assert worst_metrics['false_alarms'] == 3     # Half the samples are negative

if __name__ == "__main__":
    pytest.main([__file__, "-v"])