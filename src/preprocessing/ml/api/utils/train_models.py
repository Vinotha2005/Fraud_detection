"""
Complete Training Pipeline for FraudShield
Orchestrates the entire ML workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime

# Import our modules
from preprocessing.feature_engineering import AdvancedFeatureEngineer
from ml.ensemble_models import FraudEnsembleModel
from ml.anomaly_detection import HybridAnomalyDetector
from ml.explainability import FraudExplainer
from utils.data_generator import load_or_generate_data


class FraudShieldTrainer:
    """
    Main training pipeline for FraudShield
    """
    
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_model = None
        self.anomaly_detector = None
        self.explainer = None
        
        self.feature_names = None
        self.metrics = {}
    
    def _default_config(self) -> dict:
        """Default training configuration"""
        return {
            'data': {
                'filepath': '../data/fraud_transactions.csv',
                'num_transactions': 100000,
                'fraud_ratio': 0.02
            },
            'training': {
                'test_size': 0.2,
                'val_size': 0.1,
                'random_state': 42,
                'use_smote': True,
                'smote_ratio': 0.3
            },
            'models': {
                'ensemble_weight': 0.7,  # Weight for supervised models
                'anomaly_weight': 0.3    # Weight for anomaly detection
            },
            'output': {
                'model_dir': '../models',
                'metrics_file': '../models/metrics.json'
            }
        }
    
    def run(self):
        """Execute complete training pipeline"""
        
        print("\n" + "="*70)
        print("FRAUDSHIELD TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load or Generate Data
        print("\n[STEP 1] Loading Data...")
        df = self._load_data()
        
        # Step 2: Feature Engineering
        print("\n[STEP 2] Feature Engineering...")
        df_features = self._engineer_features(df)
        
        # Step 3: Prepare Training Data
        print("\n[STEP 3] Preparing Training Data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(df_features)
        
        # Step 4: Train Supervised Models
        print("\n[STEP 4] Training Supervised Ensemble...")
        self._train_ensemble(X_train, y_train, X_val, y_val)
        
        # Step 5: Train Anomaly Detector
        print("\n[STEP 5] Training Anomaly Detector...")
        self._train_anomaly_detector(X_train[y_train == 0])  # Train on normal transactions
        
        # Step 6: Evaluate Models
        print("\n[STEP 6] Evaluating Models...")
        self._evaluate(X_test, y_test)
        
        # Step 7: Setup Explainability
        print("\n[STEP 7] Setting up Explainability...")
        self._setup_explainer(X_train[:1000])
        
        # Step 8: Save Models
        print("\n[STEP 8] Saving Models...")
        self._save_models()
        
        # Step 9: Generate Sample Predictions
        print("\n[STEP 9] Generating Sample Predictions...")
        self._generate_samples(X_test, y_test)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        
        return self.metrics
    
    def _load_data(self) -> pd.DataFrame:
        """Load or generate transaction data"""
        
        df = load_or_generate_data(
            filepath=self.config['data']['filepath'],
            num_transactions=self.config['data']['num_transactions'],
            fraud_ratio=self.config['data']['fraud_ratio']
        )
        
        print(f"Loaded {len(df)} transactions")
        print(f"Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        
        df_features = self.feature_engineer.engineer_features(df)
        
        # Get feature names (exclude non-feature columns)
        self.feature_names = self.feature_engineer.get_feature_names(df_features)
        
        print(f"Created {len(self.feature_names)} features")
        
        return df_features
    
    def _prepare_data(self, df: pd.DataFrame):
        """Split data into train/val/test sets"""
        
        # Separate features and labels
        X = df[self.feature_names].fillna(0)
        y = df['is_fraud'].values
        
        # First split: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        # Second split: train / val
        val_ratio = self.config['training']['val_size'] / (1 - self.config['training']['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config['training']['random_state'],
            stratify=y_temp
        )
        
        print(f"Train size: {len(X_train)}")
        print(f"Val size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble of gradient boosting models"""
        
        self.ensemble_model = FraudEnsembleModel(
            use_smote=self.config['training']['use_smote'],
            smote_ratio=self.config['training']['smote_ratio']
        )
        
        self.ensemble_model.train(X_train, y_train, X_val, y_val)
    
    def _train_anomaly_detector(self, X_normal):
        """Train anomaly detection models"""
        
        X_normal_np = X_normal.values if hasattr(X_normal, 'values') else X_normal
        
        self.anomaly_detector = HybridAnomalyDetector(
            input_dim=len(self.feature_names),
            contamination=self.config['data']['fraud_ratio']
        )
        
        self.anomaly_detector.train(X_normal_np, epochs=30)
    
    def _evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        
        # Supervised model metrics
        supervised_metrics = self.ensemble_model.evaluate(X_test, y_test)
        
        # Anomaly detector metrics
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        anomaly_scores = self.anomaly_detector.predict_anomaly_score(X_test_np)
        
        # Hybrid prediction (combine supervised + anomaly)
        supervised_pred = self.ensemble_model.predict_proba(X_test)
        
        # Normalize anomaly scores
        anomaly_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
        
        # Weighted ensemble
        hybrid_pred = (
            self.config['models']['ensemble_weight'] * supervised_pred +
            self.config['models']['anomaly_weight'] * anomaly_norm
        )
        
        # Calculate hybrid metrics
        from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
        
        hybrid_auc = roc_auc_score(y_test, hybrid_pred)
        
        precision, recall, thresholds = precision_recall_curve(y_test, hybrid_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        hybrid_pred_binary = (hybrid_pred >= best_threshold).astype(int)
        
        # Store metrics
        self.metrics = {
            'supervised': supervised_metrics,
            'hybrid': {
                'auc': float(hybrid_auc),
                'best_f1': float(best_f1),
                'best_threshold': float(best_threshold),
                'f1': float(f1_score(y_test, hybrid_pred_binary))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"\nSupervised Ensemble:")
        print(f"  AUC: {supervised_metrics['auc']:.4f}")
        print(f"  F1 Score: {supervised_metrics['f1']:.4f}")
        print(f"  Precision: {supervised_metrics['precision']:.4f}")
        print(f"  Recall: {supervised_metrics['recall']:.4f}")
        print(f"  FPR: {supervised_metrics['false_positive_rate']:.4f}")
        
        print(f"\nHybrid Model (Ensemble + Anomaly):")
        print(f"  AUC: {hybrid_auc:.4f}")
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Optimal Threshold: {best_threshold:.4f}")
        
        print(f"\nEstimated Impact:")
        print(f"  Fraud Prevented: ${supervised_metrics['estimated_fraud_prevented']:,.2f}")
        print(f"  Investigation Cost: ${supervised_metrics['investigation_cost']:,.2f}")
        print(f"  Net Savings: ${supervised_metrics['net_savings']:,.2f}")
        print("="*70)
    
    def _setup_explainer(self, X_sample):
        """Initialize explainability tools"""
        
        self.explainer = FraudExplainer(
            model=self.ensemble_model,
            feature_names=self.feature_names
        )
        
        # Initialize SHAP with sample data
        X_sample_np = X_sample.values if hasattr(X_sample, 'values') else X_sample
        self.explainer.initialize_shap(X_sample_np, method='tree')
        
        print("Explainer initialized with SHAP")
    
    def _save_models(self):
        """Save trained models"""
        
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        
        # Save ensemble model
        ensemble_path = os.path.join(
            self.config['output']['model_dir'],
            'ensemble_model.joblib'
        )
        self.ensemble_model.save(ensemble_path)
        
        # Save anomaly detector
        anomaly_path = os.path.join(
            self.config['output']['model_dir'],
            'anomaly_detector.pt'
        )
        import torch
        torch.save({
            'ae_model': self.anomaly_detector.ae_detector.model.state_dict(),
            'ae_scaler': self.anomaly_detector.ae_detector.scaler,
            'ae_threshold': self.anomaly_detector.ae_detector.threshold,
            'if_model': self.anomaly_detector.if_detector.model,
            'if_scaler': self.anomaly_detector.if_detector.scaler
        }, anomaly_path)
        
        # Save feature names
        feature_path = os.path.join(
            self.config['output']['model_dir'],
            'feature_names.json'
        )
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save metrics
        with open(self.config['output']['metrics_file'], 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Models saved to {self.config['output']['model_dir']}")
    
    def _generate_samples(self, X_test, y_test):
        """Generate sample predictions with explanations"""
        
        # Find some fraud examples
        fraud_indices = np.where(y_test == 1)[0][:5]
        
        print("\nSample Fraud Detections:")
        print("="*70)
        
        for idx in fraud_indices:
            X_sample = X_test.iloc[idx:idx+1] if hasattr(X_test, 'iloc') else X_test[idx:idx+1]
            
            # Get prediction
            fraud_prob = self.ensemble_model.predict_proba(X_sample)[0]
            
            print(f"\nTransaction Index: {idx}")
            print(f"True Label: FRAUD")
            print(f"Predicted Probability: {fraud_prob*100:.2f}%")
            print(f"Classification: {'FRAUD' if fraud_prob > 0.5 else 'LEGITIMATE'}")
            print("-" * 70)


def main():
    """Main entry point"""
    
    # Create trainer
    trainer = FraudShieldTrainer()
    
    # Run training pipeline
    metrics = trainer.run()
    
    return metrics


if __name__ == "__main__":
    metrics = main()