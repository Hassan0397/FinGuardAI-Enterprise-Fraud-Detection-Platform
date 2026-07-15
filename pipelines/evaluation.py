# pipelines/evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any

class ModelEvaluator:
    """Evaluate model performance with various metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # Calculate costs if applicable
        self.metrics['false_negative_rate'] = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        self.metrics['false_positive_rate'] = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        return self.metrics
    
    def get_summary(self) -> str:
        """Get human-readable evaluation summary"""
        summary = []
        summary.append("Model Performance Summary:")
        summary.append(f"  - Accuracy: {self.metrics.get('accuracy', 0):.4f}")
        summary.append(f"  - Precision: {self.metrics.get('precision', 0):.4f}")
        summary.append(f"  - Recall: {self.metrics.get('recall', 0):.4f}")
        summary.append(f"  - F1 Score: {self.metrics.get('f1_score', 0):.4f}")
        
        if 'roc_auc' in self.metrics:
            summary.append(f"  - ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        return "\n".join(summary)