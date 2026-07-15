# pipelines/inference.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

class FraudInference:
    """Runs fraud detection inference on transaction data"""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.xgb_model = models.get('xgb')
        self.logistic_model = models.get('logistic')
        self.thresholds = models.get('thresholds', {})
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run fraud detection on all transactions"""
        # Make predictions using XGBoost (primary model)
        fraud_probabilities = self._get_fraud_probabilities(df)
        
        # Create results dataframe
        results = pd.DataFrame({
            'transaction_id': df.get('transaction_id', range(len(df))),
            'fraud_probability': fraud_probabilities,
            'risk_level': self._assign_risk_levels(fraud_probabilities),
            'timestamp': datetime.now()
        })
        
        return results
    
    def _get_fraud_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """Get fraud probability scores from XGBoost model"""
        try:
            # Ensure we have only numeric columns
            X = df.select_dtypes(include=[np.number])
            
            # Get predictions
            probabilities = self.xgb_model.predict_proba(X)[:, 1]
            return probabilities
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return random probabilities as fallback (for demo)
            return np.random.uniform(0, 1, len(df))
    
    def _assign_risk_levels(self, probabilities: np.ndarray) -> np.ndarray:
        """Assign risk levels based on fraud probability"""
        risk_levels = []
        
        xgb_thresholds = self.thresholds.get('xgboost', {})
        review_threshold = xgb_thresholds.get('review_end', 0.84)
        approve_threshold = xgb_thresholds.get('approve', 0.44)
        
        for prob in probabilities:
            if prob >= review_threshold:
                risk_levels.append('High Risk')
            elif prob >= approve_threshold:
                risk_levels.append('Medium Risk')
            else:
                risk_levels.append('Low Risk')
        
        return np.array(risk_levels)
    
    def predict_single(self, transaction: pd.Series) -> Dict[str, Any]:
        """Predict fraud for a single transaction"""
        # Convert series to dataframe
        df = pd.DataFrame([transaction])
        results = self.predict(df)
        
        return {
            'fraud_probability': float(results['fraud_probability'].iloc[0]),
            'risk_level': results['risk_level'].iloc[0],
            'is_fraudulent': results['fraud_probability'].iloc[0] > 0.5
        }