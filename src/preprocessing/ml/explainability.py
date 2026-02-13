"""
Explainable AI for Fraud Detection
Provides interpretable explanations for model predictions
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FraudExplainer:
    """
    Comprehensive explainability for fraud detection models
    
    Provides:
    - SHAP values (global and local)
    - LIME explanations
    - Feature importance
    - Decision rules
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: List[str] = ['Legitimate', 'Fraud']
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.shap_explainer = None
        self.lime_explainer = None
    
    def initialize_shap(
        self,
        X_background: np.ndarray,
        method: str = 'tree'
    ):
        """
        Initialize SHAP explainer
        
        Args:
            X_background: Background dataset for SHAP
            method: 'tree' for tree-based models, 'kernel' for others
        """
        
        print("Initializing SHAP explainer...")
        
        if method == 'tree':
            # For XGBoost, LightGBM, CatBoost
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif method == 'kernel':
            # For other models
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background
            )
        
        print("SHAP explainer ready")
    
    def initialize_lime(
        self,
        X_train: np.ndarray,
        mode: str = 'classification'
    ):
        """Initialize LIME explainer"""
        
        print("Initializing LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            random_state=42
        )
        
        print("LIME explainer ready")
    
    def explain_prediction_shap(
        self,
        X: np.ndarray,
        instance_idx: int = 0
    ) -> Dict:
        """
        Explain a single prediction using SHAP
        
        Args:
            X: Feature data
            instance_idx: Index of instance to explain
        
        Returns:
            Dictionary with explanation details
        """
        
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap() first.")
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X[instance_idx:instance_idx+1])
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get fraud class
        
        # Get base value (expected prediction)
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        # Create explanation
        explanation = {
            'base_value': float(base_value),
            'shap_values': shap_values[0].tolist() if hasattr(shap_values[0], 'tolist') else shap_values[0],
            'feature_values': X[instance_idx].tolist() if hasattr(X[instance_idx], 'tolist') else X[instance_idx],
            'feature_names': self.feature_names,
            'prediction': float(self.model.predict_proba(X[instance_idx:instance_idx+1])[0, 1])
        }
        
        # Top contributing features
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[-10:][::-1]
        
        explanation['top_features'] = [
            {
                'feature': self.feature_names[i],
                'value': float(X[instance_idx, i]),
                'shap_value': float(shap_values[0, i]),
                'contribution': 'increases fraud risk' if shap_values[0, i] > 0 else 'decreases fraud risk'
            }
            for i in top_indices
        ]
        
        return explanation
    
    def explain_prediction_lime(
        self,
        X: np.ndarray,
        instance_idx: int = 0,
        num_features: int = 10
    ) -> Dict:
        """
        Explain a single prediction using LIME
        
        Args:
            X: Feature data
            instance_idx: Index of instance to explain
            num_features: Number of top features to show
        
        Returns:
            Dictionary with explanation
        """
        
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")
        
        # Get LIME explanation
        lime_exp = self.lime_explainer.explain_instance(
            X[instance_idx],
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Extract explanation
        explanation = {
            'prediction': float(self.model.predict_proba(X[instance_idx:instance_idx+1])[0, 1]),
            'lime_score': lime_exp.score,
            'top_features': []
        }
        
        # Get feature contributions
        for feature, weight in lime_exp.as_list():
            explanation['top_features'].append({
                'feature': feature,
                'weight': float(weight),
                'contribution': 'increases fraud risk' if weight > 0 else 'decreases fraud risk'
            })
        
        return explanation
    
    def generate_fraud_report(
        self,
        X: np.ndarray,
        instance_idx: int,
        transaction_details: Dict = None
    ) -> str:
        """
        Generate human-readable fraud detection report
        
        Args:
            X: Feature data
            instance_idx: Transaction index
            transaction_details: Additional transaction info
        
        Returns:
            Formatted report string
        """
        
        # Get prediction
        fraud_prob = self.model.predict_proba(X[instance_idx:instance_idx+1])[0, 1]
        is_fraud = fraud_prob > 0.5
        
        # Get SHAP explanation
        shap_exp = self.explain_prediction_shap(X, instance_idx)
        
        # Build report
        report = []
        report.append("="*70)
        report.append("FRAUD DETECTION REPORT")
        report.append("="*70)
        report.append("")
        
        if transaction_details:
            report.append("TRANSACTION DETAILS:")
            for key, value in transaction_details.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        report.append(f"FRAUD PROBABILITY: {fraud_prob*100:.2f}%")
        report.append(f"CLASSIFICATION: {'⚠️  FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE'}")
        report.append("")
        
        report.append("TOP RISK FACTORS:")
        report.append("-" * 70)
        
        for i, feature_info in enumerate(shap_exp['top_features'][:5], 1):
            feature = feature_info['feature']
            value = feature_info['value']
            shap_val = feature_info['shap_value']
            contribution = feature_info['contribution']
            
            impact = "HIGH" if abs(shap_val) > 0.1 else "MEDIUM" if abs(shap_val) > 0.05 else "LOW"
            
            report.append(f"{i}. {feature}")
            report.append(f"   Value: {value:.4f}")
            report.append(f"   Impact: {impact} ({contribution})")
            report.append(f"   SHAP: {shap_val:+.4f}")
            report.append("")
        
        report.append("="*70)
        
        if is_fraud:
            report.append("RECOMMENDED ACTION: Flag for manual review")
            report.append("ESCALATION: High priority")
        else:
            report.append("RECOMMENDED ACTION: Approve transaction")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def plot_shap_waterfall(
        self,
        X: np.ndarray,
        instance_idx: int,
        save_path: str = None
    ):
        """
        Create SHAP waterfall plot showing feature contributions
        """
        
        shap_values = self.shap_explainer.shap_values(X[instance_idx:instance_idx+1])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.shap_explainer.expected_value[1] if isinstance(
                    self.shap_explainer.expected_value, (list, np.ndarray)
                ) else self.shap_explainer.expected_value,
                data=X[instance_idx],
                feature_names=self.feature_names
            )
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.close()
    
    def plot_shap_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save_path: str = None
    ):
        """
        Create SHAP summary plot showing overall feature importance
        """
        
        shap_values = self.shap_explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.close()
    
    def get_decision_rules(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_depth: int = 3
    ) -> str:
        """
        Extract simple decision rules from the model
        
        Returns:
            Human-readable decision rules
        """
        
        from sklearn.tree import DecisionTreeClassifier, export_text
        
        # Train a simple decision tree as a surrogate model
        surrogate_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )
        
        # Use model predictions as labels
        y_pred = self.model.predict(X)
        surrogate_tree.fit(X, y_pred)
        
        # Extract rules
        rules = export_text(
            surrogate_tree,
            feature_names=self.feature_names
        )
        
        return rules
    
    def explain_ensemble(
        self,
        X: np.ndarray,
        instance_idx: int,
        model_components: Dict
    ) -> Dict:
        """
        Explain ensemble model prediction
        Shows contribution from each component model
        
        Args:
            X: Feature data
            instance_idx: Instance index
            model_components: Dictionary of {name: model, weight}
        
        Returns:
            Explanation with component breakdowns
        """
        
        explanation = {
            'final_prediction': None,
            'components': {}
        }
        
        weighted_sum = 0
        
        for name, (model, weight) in model_components.items():
            pred = model.predict_proba(X[instance_idx:instance_idx+1])[0, 1]
            contribution = pred * weight
            weighted_sum += contribution
            
            explanation['components'][name] = {
                'prediction': float(pred),
                'weight': float(weight),
                'contribution': float(contribution)
            }
        
        explanation['final_prediction'] = float(weighted_sum)
        
        return explanation


if __name__ == "__main__":
    print("Explainability Module - Ready")
    print("SHAP + LIME for transparent fraud detection")