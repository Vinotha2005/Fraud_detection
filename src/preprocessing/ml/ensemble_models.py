"""
Ensemble Machine Learning Models for Fraud Detection
Combines multiple algorithms for maximum accuracy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FraudEnsembleModel:
    """
    Advanced ensemble of gradient boosting models
    
    Combines:
    - XGBoost (optimized for structured data)
    - LightGBM (fast, efficient)
    - CatBoost (handles categorical features well)
    """
    
    def __init__(
        self,
        use_smote: bool = True,
        smote_ratio: float = 0.3,
        n_folds: int = 5
    ):
        self.use_smote = use_smote
        self.smote_ratio = smote_ratio
        self.n_folds = n_folds
        
        # Model weights (learned during training)
        self.weights = {'xgb': 0.4, 'lgb': 0.35, 'cat': 0.25}
        
        # Initialize models
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def _get_xgboost_params(self) -> Dict:
        """Optimized XGBoost parameters for fraud detection"""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 10,  # Handle class imbalance
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _get_lightgbm_params(self) -> Dict:
        """Optimized LightGBM parameters"""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 10,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def _get_catboost_params(self) -> Dict:
        """Optimized CatBoost parameters"""
        return {
            'iterations': 500,
            'depth': 7,
            'learning_rate': 0.05,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'scale_pos_weight': 10,
            'l2_leaf_reg': 3
        }
    
    def handle_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE + Undersampling
        
        Args:
            X: Features
            y: Labels
        
        Returns:
            Resampled X, y
        """
        
        if not self.use_smote:
            return X, y
        
        # Calculate sampling strategy
        fraud_count = np.sum(y == 1)
        non_fraud_count = np.sum(y == 0)
        
        print(f"Original class distribution - Fraud: {fraud_count}, Non-fraud: {non_fraud_count}")
        
        # SMOTE to oversample minority class
        smote = SMOTE(
            sampling_strategy=self.smote_ratio,
            random_state=42,
            k_neighbors=5
        )
        
        # Undersample majority class
        under = RandomUnderSampler(
            sampling_strategy=0.5,
            random_state=42
        )
        
        # Pipeline
        pipeline = ImbPipeline([
            ('smote', smote),
            ('under', under)
        ])
        
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        
        fraud_count_new = np.sum(y_resampled == 1)
        non_fraud_count_new = np.sum(y_resampled == 0)
        
        print(f"Resampled distribution - Fraud: {fraud_count_new}, Non-fraud: {non_fraud_count_new}")
        
        return X_resampled, y_resampled
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None
    ):
        """
        Train ensemble of models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(
            X_train_scaled, y_train
        )
        
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        self.xgb_model = xgb.XGBClassifier(**self._get_xgboost_params())
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.xgb_model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train_balanced, y_train_balanced)
        
        print("\n" + "="*50)
        print("Training LightGBM...")
        print("="*50)
        
        self.lgb_model = lgb.LGBMClassifier(**self._get_lightgbm_params())
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.lgb_model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.lgb_model.fit(X_train_balanced, y_train_balanced)
        
        print("\n" + "="*50)
        print("Training CatBoost...")
        print("="*50)
        
        self.cat_model = CatBoostClassifier(**self._get_catboost_params())
        self.cat_model.fit(X_train_balanced, y_train_balanced)
        
        # Optimize ensemble weights using validation set
        if X_val is not None:
            self._optimize_weights(X_val, y_val)
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Ensemble Weights: {self.weights}")
        print("="*50)
    
    def _optimize_weights(self, X_val: pd.DataFrame, y_val: np.ndarray):
        """
        Optimize ensemble weights using validation performance
        """
        
        X_val_scaled = self.scaler.transform(X_val)
        
        # Get individual model predictions
        xgb_pred = self.xgb_model.predict_proba(X_val_scaled)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X_val_scaled)[:, 1]
        cat_pred = self.cat_model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate individual AUC scores
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        cat_auc = roc_auc_score(y_val, cat_pred)
        
        print(f"\nIndividual Model Performance:")
        print(f"  XGBoost AUC: {xgb_auc:.4f}")
        print(f"  LightGBM AUC: {lgb_auc:.4f}")
        print(f"  CatBoost AUC: {cat_auc:.4f}")
        
        # Weight by performance
        total_auc = xgb_auc + lgb_auc + cat_auc
        self.weights = {
            'xgb': xgb_auc / total_auc,
            'lgb': lgb_auc / total_auc,
            'cat': cat_auc / total_auc
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ensemble prediction
        
        Args:
            X: Features
        
        Returns:
            Fraud probabilities
        """
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X_scaled)[:, 1]
        cat_pred = self.cat_model.predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble
        ensemble_pred = (
            self.weights['xgb'] * xgb_pred +
            self.weights['lgb'] * lgb_pred +
            self.weights['cat'] * cat_pred
        )
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive evaluation
        
        Returns:
            Dictionary with metrics
        """
        
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Precision-Recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # False positive rate
        fpr = fp / (fp + tn)
        
        metrics = {
            'auc': auc,
            'best_f1': f1_scores[best_f1_idx],
            'best_threshold': best_threshold,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': f1_score(y_test, y_pred),
            'false_positive_rate': fpr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        # Estimated savings calculation
        avg_fraud_amount = 250  # Average fraud transaction amount
        false_positive_cost = 5  # Cost of investigating false positive
        
        fraud_prevented = tp * avg_fraud_amount
        investigation_cost = fp * false_positive_cost
        net_savings = fraud_prevented - investigation_cost
        
        metrics['estimated_fraud_prevented'] = fraud_prevented
        metrics['investigation_cost'] = investigation_cost
        metrics['net_savings'] = net_savings
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance across all models
        """
        
        # XGBoost importance
        xgb_imp = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': self.xgb_model.feature_importances_
        })
        
        # LightGBM importance
        lgb_imp = pd.DataFrame({
            'feature': self.feature_names,
            'lgb_importance': self.lgb_model.feature_importances_
        })
        
        # CatBoost importance
        cat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'cat_importance': self.cat_model.feature_importances_
        })
        
        # Merge
        importance_df = xgb_imp.merge(lgb_imp, on='feature').merge(cat_imp, on='feature')
        
        # Weighted average
        importance_df['ensemble_importance'] = (
            self.weights['xgb'] * importance_df['xgb_importance'] +
            self.weights['lgb'] * importance_df['lgb_importance'] +
            self.weights['cat'] * importance_df['cat_importance']
        )
        
        return importance_df.nlargest(top_n, 'ensemble_importance')
    
    def save(self, filepath: str):
        """Save trained ensemble"""
        joblib.dump({
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'cat_model': self.cat_model,
            'scaler': self.scaler,
            'weights': self.weights,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained ensemble"""
        data = joblib.load(filepath)
        self.xgb_model = data['xgb_model']
        self.lgb_model = data['lgb_model']
        self.cat_model = data['cat_model']
        self.scaler = data['scaler']
        self.weights = data['weights']
        self.feature_names = data['feature_names']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Ensemble ML Models - Ready")
    print("XGBoost + LightGBM + CatBoost with optimized weights")