# utils/cost_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class CostEngine:
    """Cost-aware decision engine for fraud detection"""
    
    def __init__(self, threshold_config: Dict[str, Any]):
        self.thresholds = threshold_config.get('xgboost', {})
        self.cost_params = threshold_config.get('cost_params', {
            'false_negative_cost': 1000,
            'false_positive_cost': 50
        })
    
    def make_decisions(self, predictions: pd.DataFrame) -> np.ndarray:
        """Make Approve/Review/Block decisions based on fraud probability"""
        
        approve_threshold = self.thresholds.get('approve', 0.44)
        review_threshold = self.thresholds.get('review_end', 0.84)
        
        decisions = []
        for prob in predictions['fraud_probability']:
            if prob <= approve_threshold:
                decisions.append('Approve')
            elif prob <= review_threshold:
                decisions.append('Review')
            else:
                decisions.append('Block')
        
        return np.array(decisions)
    
    def calculate_cost(self, predictions: pd.DataFrame, decisions: np.ndarray) -> Dict[str, float]:
        """Calculate total cost based on decisions"""
        total_cost = 0
        false_negatives = 0
        false_positives = 0
        
        # Note: In real scenario, compare with actual labels
        # Here we estimate based on fraud probability
        
        for prob, decision in zip(predictions['fraud_probability'], decisions):
            # Estimate actual fraud (simulated)
            is_actually_fraud = prob > 0.5
            
            if is_actually_fraud and decision in ['Approve', 'Review']:
                total_cost += self.cost_params['false_negative_cost']
                false_negatives += 1
            elif not is_actually_fraud and decision == 'Block':
                total_cost += self.cost_params['false_positive_cost']
                false_positives += 1
        
        return {
            'total_cost': total_cost,
            'false_negatives': false_negatives,
            'false_positives': false_positives,
            'avg_cost_per_transaction': total_cost / len(predictions) if len(predictions) > 0 else 0
        }
    
    def get_decision_summary(self, predictions: pd.DataFrame, decisions: np.ndarray) -> Dict[str, int]:
        """Get summary statistics of decisions"""
        decision_counts = pd.Series(decisions).value_counts()
        
        return {
            'approve': int(decision_counts.get('Approve', 0)),
            'review': int(decision_counts.get('Review', 0)),
            'block': int(decision_counts.get('Block', 0)),
            'total': len(decisions)
        }
    
    def calculate_risk_score(self, fraud_probability: float) -> float:
        """Calculate normalized risk score (0-100)"""
        return min(100, fraud_probability * 100)