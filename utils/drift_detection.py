# utils/drift_detection.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats

class DriftDetector:
    """Detects data drift between training and inference data"""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.training_stats = self._get_training_statistics()
    
    def _get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics from training data (simulated)"""
        # In production, load actual training statistics
        return {
            'mean': {
                'amount': 500,
                'merchant_risk_score': 0.3,
                'transaction_velocity_1h': 2,
                'transaction_velocity_24h': 10,
                'customer_age': 35
            },
            'std': {
                'amount': 300,
                'merchant_risk_score': 0.2,
                'transaction_velocity_1h': 3,
                'transaction_velocity_24h': 8,
                'customer_age': 12
            }
        }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift between training and current data"""
        
        drift_report = {
            'drift_score': 0.0,
            'drifted_features_count': 0,
            'feature_drift': {},
            'overall_status': 'Stable'
        }
        
        # Check drift for key numerical features
        key_features = ['amount', 'merchant_risk_score', 'transaction_velocity_1h']
        
        total_drift = 0
        features_checked = 0
        
        for feature in key_features:
            if feature in current_data.columns:
                drift_info = self._calculate_feature_drift(current_data[feature], feature)
                drift_report['feature_drift'][feature] = drift_info
                
                if drift_info['drifted']:
                    drift_report['drifted_features_count'] += 1
                
                total_drift += drift_info['score']
                features_checked += 1
        
        # Calculate overall drift score
        if features_checked > 0:
            drift_report['drift_score'] = total_drift / features_checked
        
        # Set overall status
        if drift_report['drift_score'] < 20:
            drift_report['overall_status'] = 'Stable'
        elif drift_report['drift_score'] < 50:
            drift_report['overall_status'] = 'Warning'
        else:
            drift_report['overall_status'] = 'Severe Drift'
        
        return drift_report
    
    def _calculate_feature_drift(self, current_series: pd.Series, feature_name: str) -> Dict[str, Any]:
        """Calculate drift for a single feature"""
        
        # Get training statistics
        train_mean = self.training_stats['mean'].get(feature_name, current_series.mean())
        train_std = self.training_stats['std'].get(feature_name, current_series.std())
        
        # Calculate current statistics
        current_mean = current_series.mean()
        current_std = current_series.std()
        
        # Calculate drift using PSI (Population Stability Index)
        if train_std > 0:
            mean_shift = abs(current_mean - train_mean) / train_std
        else:
            mean_shift = 0
        
        # Drift score (0-100)
        drift_score = min(100, mean_shift * 20)
        
        # Determine if drifted
        drifted = drift_score > 30
        status = 'Stable'
        if drift_score > 50:
            status = 'Severe'
        elif drift_score > 30:
            status = 'Warning'
        
        return {
            'score': drift_score,
            'drifted': drifted,
            'status': status,
            'train_mean': train_mean,
            'current_mean': current_mean,
            'percent_change': ((current_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
        }
    
    def get_drift_recommendations(self, drift_report: Dict[str, Any]) -> List[str]:
        """Get recommendations based on drift detection"""
        recommendations = []
        
        if drift_report['overall_status'] == 'Severe Drift':
            recommendations.append("⚠️ Severe data drift detected - Immediate model retraining recommended")
            recommendations.append("Review data pipeline for changes in data generation process")
        elif drift_report['overall_status'] == 'Warning':
            recommendations.append("⚠️ Moderate drift detected - Consider retraining within next month")
            recommendations.append("Monitor feature distributions for further changes")
        
        if drift_report['drifted_features_count'] > 0:
            drifted_features = [f for f, info in drift_report['feature_drift'].items() if info['drifted']]
            recommendations.append(f"Drifted features: {', '.join(drifted_features)}")
        
        if not recommendations:
            recommendations.append("✅ No significant drift detected - Model is stable")
        
        return recommendations