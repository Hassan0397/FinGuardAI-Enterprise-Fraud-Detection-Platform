# utils/ab_testing.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import hashlib
from scipy import stats
from scipy.stats import ttest_ind
import warnings
import os
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import plotly.graph_objects as go
import plotly.express as px
import json

warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ABTestEngine:
    """Enterprise-grade A/B Testing Framework for Model Comparison"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)
        self.test_results = {}
        self.test_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.test_start_time = datetime.now()
        
        self.metrics_config = {
            'accuracy': {'weight': 0.30, 'higher_is_better': True},
            'precision': {'weight': 0.20, 'higher_is_better': True},
            'recall': {'weight': 0.20, 'higher_is_better': True},
            'f1_score': {'weight': 0.20, 'higher_is_better': True},
            'inference_time_ms': {'weight': 0.10, 'higher_is_better': False}
        }
    
    def _get_z_score(self, confidence_level: float) -> float:
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        return z_scores.get(confidence_level, 1.96)
    
    def run_ab_test(self, 
                    model_a_predictions: np.ndarray,
                    model_b_predictions: np.ndarray,
                    y_true: np.ndarray,
                    model_a_name: str = "Model A",
                    model_b_name: str = "Model B",
                    model_a_proba: Optional[np.ndarray] = None,
                    model_b_proba: Optional[np.ndarray] = None,
                    inference_time_a: float = 0.0,
                    inference_time_b: float = 0.0) -> Dict[str, Any]:
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Calculate metrics
        metrics_a = {
            'accuracy': accuracy_score(y_true, model_a_predictions),
            'precision': precision_score(y_true, model_a_predictions, zero_division=0),
            'recall': recall_score(y_true, model_a_predictions, zero_division=0),
            'f1_score': f1_score(y_true, model_a_predictions, zero_division=0),
            'inference_time_ms': inference_time_a
        }
        
        metrics_b = {
            'accuracy': accuracy_score(y_true, model_b_predictions),
            'precision': precision_score(y_true, model_b_predictions, zero_division=0),
            'recall': recall_score(y_true, model_b_predictions, zero_division=0),
            'f1_score': f1_score(y_true, model_b_predictions, zero_division=0),
            'inference_time_ms': inference_time_b
        }
        
        # Add AUC if probabilities available
        if model_a_proba is not None and model_b_proba is not None:
            try:
                metrics_a['auc'] = roc_auc_score(y_true, model_a_proba)
                metrics_b['auc'] = roc_auc_score(y_true, model_b_proba)
                self.metrics_config['auc'] = {'weight': 0.10, 'higher_is_better': True}
            except:
                metrics_a['auc'] = 0.5
                metrics_b['auc'] = 0.5
        
        # Statistical tests
        statistical_tests = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            statistical_tests[metric] = self._compare_proportions(
                metrics_a[metric], metrics_b[metric], len(y_true)
            )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(metrics_a, metrics_b, len(y_true))
        
        # Determine winner
        scores = self._calculate_weighted_scores(metrics_a, metrics_b)
        
        # Handle NaN scores
        model_a_score = scores['model_a'] if not np.isnan(scores['model_a']) else 0
        model_b_score = scores['model_b'] if not np.isnan(scores['model_b']) else 0
        
        winner = model_a_name if model_a_score > model_b_score else model_b_name
        
        self.test_results = {
            'test_id': self.test_id,
            'test_time': self.test_start_time.isoformat(),
            'model_a': model_a_name,
            'model_b': model_b_name,
            'sample_size': len(y_true),
            'confidence_level': self.confidence_level,
            'metrics': {model_a_name: metrics_a, model_b_name: metrics_b},
            'statistical_tests': statistical_tests,
            'confidence_intervals': confidence_intervals,
            'scores': {'model_a': model_a_score, 'model_b': model_b_score},
            'winner': {
                'name': winner, 
                'score': max(model_a_score, model_b_score), 
                'margin': abs(model_a_score - model_b_score)
            },
            'recommendations': self._generate_recommendations(metrics_a, metrics_b, model_a_name, model_b_name),
            'lift_analysis': self._calculate_lift_analysis(metrics_a, metrics_b)
        }
        
        return self.test_results
    
    def _compare_proportions(self, prop_a: float, prop_b: float, sample_size: int) -> Dict[str, Any]:
        p_pooled = (prop_a + prop_b) / 2
        se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / sample_size)) if sample_size > 0 and p_pooled > 0 else 0.01
        z_stat = (prop_a - prop_b) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        is_significant = p_value < (1 - self.confidence_level)
        lift = ((prop_a - prop_b) / prop_b * 100) if prop_b > 0 else 0
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'lift_percentage': lift,
            'interpretation': f"{'Significant' if is_significant else 'Not significant'} difference detected"
        }
    
    def _calculate_confidence_intervals(self, metrics_a: Dict, metrics_b: Dict, sample_size: int) -> Dict:
        intervals = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in metrics_a and metric in metrics_b:
                se = np.sqrt((metrics_a[metric] * (1 - metrics_a[metric]) + metrics_b[metric] * (1 - metrics_b[metric])) / sample_size)
                margin = self.z_score * se
                intervals[metric] = {
                    'model_a': {'lower': max(0, metrics_a[metric] - margin), 'upper': min(1, metrics_a[metric] + margin)},
                    'model_b': {'lower': max(0, metrics_b[metric] - margin), 'upper': min(1, metrics_b[metric] + margin)}
                }
        return intervals
    
    def _calculate_lift_analysis(self, metrics_a: Dict, metrics_b: Dict) -> Dict:
        lift = {}
        for metric in metrics_a.keys():
            if metric in metrics_b and metrics_a[metric] > 0:
                lift[metric] = {
                    'absolute': metrics_a[metric] - metrics_b[metric],
                    'percentage': ((metrics_a[metric] - metrics_b[metric]) / metrics_b[metric] * 100) if metrics_b[metric] > 0 else 0
                }
        return lift
    
    def _calculate_weighted_scores(self, metrics_a: Dict, metrics_b: Dict) -> Dict[str, float]:
        scores = {'model_a': 0, 'model_b': 0}
        total_weight = 0
        
        for metric, config in self.metrics_config.items():
            if metric in metrics_a and metric in metrics_b:
                weight = config['weight']
                total_weight += weight
                
                val_a = metrics_a[metric]
                val_b = metrics_b[metric]
                
                if metric == 'inference_time_ms':
                    max_time = max(val_a, val_b, 100)
                    norm_a = max(0, 1 - (val_a / max_time))
                    norm_b = max(0, 1 - (val_b / max_time))
                else:
                    norm_a = val_a
                    norm_b = val_b
                
                scores['model_a'] += norm_a * weight
                scores['model_b'] += norm_b * weight
        
        if total_weight > 0:
            scores['model_a'] = (scores['model_a'] / total_weight) * 100
            scores['model_b'] = (scores['model_b'] / total_weight) * 100
        
        return scores
    
    def _generate_recommendations(self, metrics_a: Dict, metrics_b: Dict, name_a: str, name_b: str) -> List[str]:
        recommendations = []
        
        if metrics_a['accuracy'] > metrics_b['accuracy'] + 0.05:
            recommendations.append(f"🎯 For higher accuracy, deploy {name_a} ({(metrics_a['accuracy']-metrics_b['accuracy'])*100:.1f}% improvement)")
        elif metrics_b['accuracy'] > metrics_a['accuracy'] + 0.05:
            recommendations.append(f"🎯 For higher accuracy, deploy {name_b} ({(metrics_b['accuracy']-metrics_a['accuracy'])*100:.1f}% improvement)")
        
        if metrics_a['f1_score'] > metrics_b['f1_score'] + 0.05:
            recommendations.append(f"📊 For balanced precision-recall, {name_a} provides better F1 score")
        elif metrics_b['f1_score'] > metrics_a['f1_score'] + 0.05:
            recommendations.append(f"📊 For balanced precision-recall, {name_b} provides better F1 score")
        
        time_ratio = max(metrics_a['inference_time_ms'], metrics_b['inference_time_ms']) / max(min(metrics_a['inference_time_ms'], metrics_b['inference_time_ms']), 1)
        if time_ratio > 1.5:
            faster = name_a if metrics_a['inference_time_ms'] < metrics_b['inference_time_ms'] else name_b
            recommendations.append(f"⚡ For real-time applications, {faster} is {time_ratio:.0f}x faster")
        
        recommendations.append(f"📊 Test conducted at {self.confidence_level*100}% confidence level")
        
        return recommendations
    
    def get_test_summary(self) -> str:
        """Get a beautiful formatted summary of the A/B test results"""
        if not self.test_results:
            return "No test results available. Please run an A/B test first."
        
        results = self.test_results
        winner = results['winner']
        model_a = results['model_a']
        model_b = results['model_b']
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"🔬 A/B TEST RESULTS - {results['test_id']}".center(80))
        summary_lines.append("=" * 80)
        summary_lines.append(f"📅 Test Time: {results['test_time'][:19]}")
        summary_lines.append(f"📊 Sample Size: {results['sample_size']:,} transactions")
        summary_lines.append(f"🎯 Confidence Level: {results['confidence_level']*100}%")
        summary_lines.append("")
        
        # Winner announcement
        summary_lines.append("🏆 WINNER ANNOUNCEMENT".center(80))
        summary_lines.append("-" * 80)
        summary_lines.append(f"   Winner: {winner['name']}")
        summary_lines.append(f"   Score: {winner['score']:.1f}/100")
        summary_lines.append(f"   Margin: {winner['margin']:.1f} points")
        summary_lines.append("")
        
        # Performance comparison
        summary_lines.append("📊 PERFORMANCE COMPARISON".center(80))
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'Metric':<25} {model_a:<22} {model_b:<22} {'Better':<10}")
        summary_lines.append("-" * 80)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time_ms']:
            if metric in results['metrics'][model_a]:
                val_a = results['metrics'][model_a][metric]
                val_b = results['metrics'][model_b][metric]
                metric_name = metric.replace('_', ' ').title()
                
                if metric == 'inference_time_ms':
                    better = model_a if val_a < val_b else model_b
                    val_a_str = f"{val_a:.0f}ms"
                    val_b_str = f"{val_b:.0f}ms"
                else:
                    better = model_a if val_a > val_b else model_b
                    val_a_str = f"{val_a:.4f}"
                    val_b_str = f"{val_b:.4f}"
                
                summary_lines.append(f"{metric_name:<25} {val_a_str:<22} {val_b_str:<22} {better:<10}")
        
        # FIXED: Corrected the formatting for AUC line
        if 'auc' in results['metrics'][model_a]:
            val_a = results['metrics'][model_a]['auc']
            val_b = results['metrics'][model_b]['auc']
            better = model_a if val_a > val_b else model_b
            val_a_str = f"{val_a:.4f}"
            val_b_str = f"{val_b:.4f}"
            summary_lines.append(f"{'AUC':<25} {val_a_str:<22} {val_b_str:<22} {better:<10}")
        
        summary_lines.append("")
        
        # Statistical significance
        summary_lines.append("📈 STATISTICAL SIGNIFICANCE".center(80))
        summary_lines.append("-" * 80)
        
        for metric, test in results['statistical_tests'].items():
            metric_name = metric.replace('_', ' ').title()
            status = "✅ SIGNIFICANT" if test['is_significant'] else "❌ NOT SIGNIFICANT"
            summary_lines.append(f"   {metric_name:<18} p-value: {test['p_value']:.4f} - {status}")
            summary_lines.append(f"   {'':18} Lift: {test['lift_percentage']:+.1f}%")
            summary_lines.append("")
        
        # Recommendations
        if results['recommendations']:
            summary_lines.append("💡 RECOMMENDATIONS".center(80))
            summary_lines.append("-" * 80)
            for rec in results['recommendations']:
                summary_lines.append(f"   • {rec}")
            summary_lines.append("")
        
        summary_lines.append("=" * 80)
        summary_lines.append("🏦 FinGuard AI - Cost-Aware Fraud Detection Platform".center(80))
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)


class ABTestVisualizer:
    """Professional visualization utilities"""
    
    @staticmethod
    def create_metrics_chart(results: Dict[str, Any]):
        """Create metrics comparison bar chart"""
        model_a = results['model_a']
        model_b = results['model_b']
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'auc' in results['metrics'][model_a] and not np.isnan(results['metrics'][model_a]['auc']):
            metrics.append('auc')
        
        values_a = [results['metrics'][model_a][m] for m in metrics]
        values_b = [results['metrics'][model_b][m] for m in metrics]
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        
        fig = go.Figure(data=[
            go.Bar(name=model_a, x=metric_labels, y=values_a, 
                   marker_color='#3b82f6', text=[f"{v:.3f}" for v in values_a], 
                   textposition='auto', textfont=dict(size=12, color='white'),
                   width=0.4),
            go.Bar(name=model_b, x=metric_labels, y=values_b, 
                   marker_color='#f59e0b', text=[f"{v:.3f}" for v in values_b], 
                   textposition='auto', textfont=dict(size=12, color='white'),
                   width=0.4)
        ])
        
        fig.update_layout(
            title=dict(text="Model Performance Comparison", font=dict(size=18, family="Inter"), x=0.5),
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            height=500,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12)),
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0', tickangle=0)
        fig.update_yaxes(range=[0, 1.1], gridcolor='#e2e8f0', tickformat='.0%')
        
        return fig
    
    @staticmethod
    def create_speed_chart(results: Dict[str, Any]):
        """Create inference speed comparison chart"""
        model_a = results['model_a']
        model_b = results['model_b']
        
        time_a = results['metrics'][model_a]['inference_time_ms']
        time_b = results['metrics'][model_b]['inference_time_ms']
        
        colors_a = '#ef4444' if time_a > time_b else '#10b981'
        colors_b = '#ef4444' if time_b > time_a else '#10b981'
        
        fig = go.Figure(data=[
            go.Bar(name=model_a, x=['Inference Time'], y=[time_a], 
                   marker_color=colors_a, text=[f"{time_a:.0f}ms"], 
                   textposition='auto', textfont=dict(size=14, color='white')),
            go.Bar(name=model_b, x=['Inference Time'], y=[time_b], 
                   marker_color=colors_b, text=[f"{time_b:.0f}ms"], 
                   textposition='auto', textfont=dict(size=14, color='white'))
        ])
        
        fig.update_layout(
            title=dict(text="Inference Speed Comparison (Lower is Better)", font=dict(size=16), x=0.5),
            xaxis_title="Metric",
            yaxis_title="Time (milliseconds)",
            barmode='group',
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        return fig
    
    @staticmethod
    def create_radar_chart(results: Dict[str, Any]):
        """Create radar chart for performance comparison"""
        model_a = results['model_a']
        model_b = results['model_b']
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Speed']
        if 'auc' in results['metrics'][model_a] and not np.isnan(results['metrics'][model_a]['auc']):
            categories.insert(4, 'AUC')
        
        # Normalize speed (lower is better) - scale to 0-1 where lower time = higher score
        time_a = results['metrics'][model_a]['inference_time_ms']
        time_b = results['metrics'][model_b]['inference_time_ms']
        max_time = max(time_a, time_b, 1)
        speed_a = 1 - (time_a / max_time)
        speed_b = 1 - (time_b / max_time)
        
        values_a = [
            results['metrics'][model_a]['accuracy'],
            results['metrics'][model_a]['precision'],
            results['metrics'][model_a]['recall'],
            results['metrics'][model_a]['f1_score'],
            speed_a
        ]
        
        values_b = [
            results['metrics'][model_b]['accuracy'],
            results['metrics'][model_b]['precision'],
            results['metrics'][model_b]['recall'],
            results['metrics'][model_b]['f1_score'],
            speed_b
        ]
        
        if 'auc' in results['metrics'][model_a] and not np.isnan(results['metrics'][model_a]['auc']):
            values_a.insert(4, results['metrics'][model_a]['auc'])
            values_b.insert(4, results['metrics'][model_b]['auc'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_a,
            theta=categories,
            fill='toself',
            name=model_a,
            line_color='#3b82f6',
            line_width=2,
            fillcolor='rgba(59, 130, 246, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values_b,
            theta=categories,
            fill='toself',
            name=model_b,
            line_color='#f59e0b',
            line_width=2,
            fillcolor='rgba(245, 158, 11, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%', 
                              gridcolor='#e2e8f0', linecolor='#e2e8f0'),
                angularaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', 
                               tickfont=dict(size=11)),
                bgcolor='white'
            ),
            title=dict(text="Performance Radar Chart", font=dict(size=16), x=0.5),
            height=500,
            template='plotly_white',
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=12))
        )
        
        return fig
    
    @staticmethod
    def create_significance_chart(results: Dict[str, Any]):
        """Create statistical significance chart"""
        metrics = []
        p_values = []
        colors = []
        lift_values = []
        
        for metric, test in results['statistical_tests'].items():
            metrics.append(metric.replace('_', ' ').title())
            p_values.append(test['p_value'])
            colors.append('#10b981' if test['is_significant'] else '#ef4444')
            lift_values.append(test['lift_percentage'])
        
        fig = go.Figure()
        
        # Add bars for p-values
        fig.add_trace(go.Bar(
            x=metrics, 
            y=p_values, 
            marker_color=colors, 
            text=[f"{p:.4f}" for p in p_values], 
            textposition='auto',
            name='P-Value',
            yaxis='y'
        ))
        
        # Add line for lift
        fig.add_trace(go.Scatter(
            x=metrics,
            y=lift_values,
            mode='lines+markers',
            name='Lift %',
            line=dict(color='#8b5cf6', width=2),
            marker=dict(size=8, color='#8b5cf6'),
            yaxis='y2'
        ))
        
        fig.add_hline(y=0.05, line_dash="dash", line_color="#ef4444", line_width=2,
                     annotation_text="Significance Threshold (0.05)", 
                     annotation_position="top right",
                     annotation_font_size=10)
        
        fig.update_layout(
            title=dict(text="Statistical Significance Analysis", font=dict(size=16), x=0.5),
            xaxis_title="Metrics",
            height=450,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            hovermode='x unified',
            yaxis=dict(title="P-Value", range=[0, 1], gridcolor='#e2e8f0', tickformat='.2f'),
            yaxis2=dict(title="Lift (%)", overlaying='y', side='right', 
                       tickformat='+.0f%', gridcolor='rgba(0,0,0,0)')
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        
        return fig
    
    @staticmethod
    def create_confidence_interval_chart(results: Dict[str, Any]):
        """Create confidence interval visualization"""
        intervals = results.get('confidence_intervals', {})
        if not intervals:
            return None
        
        model_a = results['model_a']
        model_b = results['model_b']
        
        fig = go.Figure()
        
        metrics = list(intervals.keys())
        
        for metric in metrics:
            ci_a = intervals[metric]['model_a']
            ci_b = intervals[metric]['model_b']
            metric_name = metric.replace('_', ' ').title()
            
            fig.add_trace(go.Scatter(
                x=[ci_a['lower'], ci_a['upper']],
                y=[metric_name, metric_name],
                mode='lines',
                line=dict(color='#3b82f6', width=6),
                name=f"{model_a} - {metric_name}",
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[ci_b['lower'], ci_b['upper']],
                y=[metric_name, metric_name],
                mode='lines',
                line=dict(color='#f59e0b', width=6),
                name=f"{model_b} - {metric_name}",
                showlegend=False
            ))
            
            # Add markers for point estimates
            fig.add_trace(go.Scatter(
                x=[results['metrics'][model_a][metric]],
                y=[metric_name],
                mode='markers',
                marker=dict(color='#3b82f6', size=12, symbol='diamond'),
                name=f"{model_a} estimate",
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[results['metrics'][model_b][metric]],
                y=[metric_name],
                mode='markers',
                marker=dict(color='#f59e0b', size=12, symbol='diamond'),
                name=f"{model_b} estimate",
                showlegend=False
            ))
        
        # Add legend manually
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', line=dict(color='#3b82f6', width=6),
            name=model_a, showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', line=dict(color='#f59e0b', width=6),
            name=model_b, showlegend=True
        ))
        
        fig.update_layout(
            title=dict(text="95% Confidence Intervals", font=dict(size=16), x=0.5),
            xaxis_title="Metric Value",
            yaxis_title="Metrics",
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        fig.update_xaxes(range=[0, 1], gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        return fig
    
    @staticmethod
    def create_winner_gauge(results: Dict[str, Any]):
        """Create winner score gauge chart"""
        winner = results['winner']
        loser_score = 100 - winner['score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=winner['score'],
            title=dict(text=f"Winner: {winner['name']}", font=dict(size=18)),
            delta=dict(reference=loser_score, increasing=dict(color="green")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="darkblue"),
                bar=dict(color="#10b981"),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, 30], color="#ef4444"),
                    dict(range=[30, 70], color="#f59e0b"),
                    dict(range=[70, 100], color="#10b981")
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=winner['score']
                )
            )
        ))
        
        fig.update_layout(
            height=350,
            font=dict(family="Inter, sans-serif", size=14)
        )
        
        return fig
    
    @staticmethod
    def create_lift_chart(results: Dict[str, Any]):
        """Create lift analysis chart"""
        lift_data = results.get('lift_analysis', {})
        if not lift_data:
            return None
        
        metrics = []
        lift_percentages = []
        colors = []
        
        for metric, data in lift_data.items():
            if metric != 'inference_time_ms':
                metrics.append(metric.replace('_', ' ').title())
                lift_percentages.append(data['percentage'])
                colors.append('#10b981' if data['percentage'] > 0 else '#ef4444')
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=lift_percentages,
                marker_color=colors,
                text=[f"{l:+.1f}%" for l in lift_percentages],
                textposition='auto',
                textfont=dict(size=12)
            )
        ])
        
        fig.add_hline(y=0, line_dash="solid", line_color="#64748b", line_width=1)
        
        fig.update_layout(
            title=dict(text="Lift Analysis (Model A vs Model B)", font=dict(size=16), x=0.5),
            xaxis_title="Metrics",
            yaxis_title="Lift Percentage (%)",
            height=400,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12)
        )
        
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        return fig


class ABTestReportGenerator:
    """Generate professional PDF and HTML reports with graphs"""
    
    def __init__(self):
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        self.charts_dir = os.path.join(self.reports_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def _generate_charts_as_images(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all charts as image files for PDF embedding"""
        chart_paths = {}
        visualizer = ABTestVisualizer()
        
        # Generate and save each chart
        charts = {
            'metrics_chart': visualizer.create_metrics_chart(results),
            'radar_chart': visualizer.create_radar_chart(results),
            'speed_chart': visualizer.create_speed_chart(results),
            'significance_chart': visualizer.create_significance_chart(results),
            'winner_gauge': visualizer.create_winner_gauge(results),
            'lift_chart': visualizer.create_lift_chart(results),
            'confidence_chart': visualizer.create_confidence_interval_chart(results)
        }
        
        for name, fig in charts.items():
            if fig is not None:
                path = os.path.join(self.charts_dir, f"{name}_{results['test_id']}.png")
                fig.write_image(path, width=800, height=500, scale=2)
                chart_paths[name] = path
        
        return chart_paths
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate beautiful HTML report with embedded visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/ab_test_report_{timestamp}.html"
        
        winner = results['winner']
        model_a = results['model_a']
        model_b = results['model_b']
        
        # Generate charts as HTML
        visualizer = ABTestVisualizer()
        
        metrics_chart_html = visualizer.create_metrics_chart(results).to_html(full_html=False, include_plotlyjs='cdn')
        radar_chart_html = visualizer.create_radar_chart(results).to_html(full_html=False, include_plotlyjs=False)
        speed_chart_html = visualizer.create_speed_chart(results).to_html(full_html=False, include_plotlyjs=False)
        significance_chart_html = visualizer.create_significance_chart(results).to_html(full_html=False, include_plotlyjs=False)
        winner_gauge_html = visualizer.create_winner_gauge(results).to_html(full_html=False, include_plotlyjs=False)
        
        # Create metrics table rows
        metrics_rows = ""
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time_ms']:
            if metric in results['metrics'][model_a]:
                val_a = results['metrics'][model_a][metric]
                val_b = results['metrics'][model_b][metric]
                metric_name = metric.replace('_', ' ').title()
                
                if metric == 'inference_time_ms':
                    better = model_a if val_a < val_b else model_b
                    val_a_str = f"{val_a:.0f}ms"
                    val_b_str = f"{val_b:.0f}ms"
                else:
                    better = model_a if val_a > val_b else model_b
                    val_a_str = f"{val_a:.4f}"
                    val_b_str = f"{val_b:.4f}"
                
                metrics_rows += f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td>{val_a_str}</td>
                    <td>{val_b_str}</td>
                    <td><span class="badge badge-success">{better}</span></td>
                </tr>"""
        
        if 'auc' in results['metrics'][model_a] and not np.isnan(results['metrics'][model_a]['auc']):
            val_a = results['metrics'][model_a]['auc']
            val_b = results['metrics'][model_b]['auc']
            better = model_a if val_a > val_b else model_b
            metrics_rows += f"""
            <tr>
                <td><strong>AUC</strong></td>
                <td>{val_a:.4f}</td>
                <td>{val_b:.4f}</td>
                <td><span class="badge badge-success">{better}</span></td>
            </tr>"""
        
        # Create significance rows
        significance_rows = ""
        for metric, test in results['statistical_tests'].items():
            metric_name = metric.replace('_', ' ').title()
            sig_class = "badge-success" if test['is_significant'] else "badge-warning"
            sig_text = "Significant ✓" if test['is_significant'] else "Not Significant ✗"
            significance_rows += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{test['p_value']:.4f}</td>
                <td><span class='badge {sig_class}'>{sig_text}</span></td>
                <td class="{'positive' if test['lift_percentage'] > 0 else 'negative'}">{test['lift_percentage']:+.1f}%</td>
            </tr>"""
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>A/B Test Report - FinGuard AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
            padding: 40px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 32px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 48px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }}
        
        .header p {{
            color: #94a3b8;
            font-size: 1rem;
        }}
        
        .winner-card {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            margin: 30px;
            padding: 40px;
            border-radius: 28px;
            text-align: center;
            color: white;
            box-shadow: 0 20px 35px -8px rgba(16, 185, 129, 0.4);
            animation: slideIn 0.5s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .winner-card h2 {{
            font-size: 2rem;
            margin: 16px 0 8px;
        }}
        
        .winner-card .winner-icon {{
            font-size: 4rem;
        }}
        
        .content {{
            padding: 48px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 24px;
            margin: 32px 0;
        }}
        
        .metric-box {{
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            padding: 24px;
            border-radius: 20px;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-box:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -6px rgba(0,0,0,0.1);
        }}
        
        .metric-box h4 {{
            color: #64748b;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }}
        
        .metric-box .value {{
            font-size: 1.75rem;
            font-weight: 800;
            color: #0f172a;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }}
        
        .chart-card {{
            background: white;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s;
        }}
        
        .chart-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.1);
        }}
        
        .chart-card h3 {{
            margin-bottom: 16px;
            color: #0f172a;
            font-size: 1.2rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        th, td {{
            padding: 14px 20px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            background: #f1f5f9;
            font-weight: 700;
            color: #0f172a;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        tr:hover {{
            background: #f8fafc;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-success {{ background: #d1fae5; color: #065f46; }}
        .badge-warning {{ background: #fed7aa; color: #92400e; }}
        
        .winner-highlight {{
            font-weight: 700;
            color: #10b981;
        }}
        
        .positive {{
            color: #10b981;
            font-weight: 600;
        }}
        
        .negative {{
            color: #ef4444;
            font-weight: 600;
        }}
        
        .recommendations {{
            background: #fef3c7;
            padding: 28px;
            border-radius: 20px;
            margin: 32px 0;
            border-left: 4px solid #f59e0b;
        }}
        
        .recommendations h3 {{
            color: #92400e;
            margin-bottom: 16px;
            font-size: 1.25rem;
        }}
        
        .recommendations ul {{ margin-left: 24px; }}
        .recommendations li {{ margin: 12px 0; color: #92400e; }}
        
        .footer {{
            background: #f8fafc;
            padding: 32px;
            text-align: center;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            margin: 40px 0 20px;
            padding-bottom: 12px;
            border-bottom: 3px solid #3b82f6;
            display: inline-block;
        }}
        
        @media print {{
            body {{ padding: 0; background: white; }}
            .container {{ box-shadow: none; border-radius: 0; }}
            .chart-card {{ break-inside: avoid; }}
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 20px; }}
            .content {{ padding: 24px; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="font-size: 3rem;">🏦</div>
            <h1>FinGuard AI - A/B Test Report</h1>
            <p>Statistical Model Comparison & Performance Analysis</p>
            <small style="color: #64748b;">Test ID: {results['test_id']} | Date: {results['test_time'][:10]}</small>
        </div>
        
        <div class="winner-card">
            <div class="winner-icon">🏆</div>
            <h2>Winner: {winner['name']}</h2>
            <p style="font-size: 1.25rem; margin-top: 8px;">Overall Score: {winner['score']:.1f}/100</p>
            <p>Margin: {winner['margin']:.1f} points</p>
        </div>
        
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-box">
                    <h4>Sample Size</h4>
                    <div class="value">{results['sample_size']:,}</div>
                    <small>transactions</small>
                </div>
                <div class="metric-box">
                    <h4>Confidence Level</h4>
                    <div class="value">{results['confidence_level']*100:.0f}%</div>
                    <small>statistical threshold</small>
                </div>
                <div class="metric-box">
                    <h4>Model A</h4>
                    <div class="value">{model_a}</div>
                </div>
                <div class="metric-box">
                    <h4>Model B</h4>
                    <div class="value">{model_b}</div>
                </div>
            </div>
            
            <div class="chart-grid">
                <div class="chart-card">
                    <h3>📊 Performance Comparison</h3>
                    <div id="metrics-chart">{metrics_chart_html}</div>
                </div>
                <div class="chart-card">
                    <h3>🎯 Winner Score Gauge</h3>
                    <div id="winner-gauge">{winner_gauge_html}</div>
                </div>
                <div class="chart-card">
                    <h3>🕸️ Radar Chart Analysis</h3>
                    <div id="radar-chart">{radar_chart_html}</div>
                </div>
                <div class="chart-card">
                    <h3>⚡ Speed Comparison</h3>
                    <div id="speed-chart">{speed_chart_html}</div>
                </div>
                <div class="chart-card">
                    <h3>📈 Statistical Significance</h3>
                    <div id="significance-chart">{significance_chart_html}</div>
                </div>
            </div>
            
            <h3 class="section-title">📊 Performance Metrics</h3>
            <table>
                <thead>
                    <tr><th>Metric</th><th>{model_a}</th><th>{model_b}</th><th>Better Model</th></tr>
                </thead>
                <tbody>
                    {metrics_rows}
                </tbody>
            </table>
            
            <h3 class="section-title">📈 Statistical Significance</h3>
            <table>
                <thead>
                    <tr><th>Metric</th><th>P-Value</th><th>Significant?</th><th>Lift</th></tr>
                </thead>
                <tbody>
                    {significance_rows}
                </tbody>
            </table>
            
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <ul>"""
        
        for rec in results['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += f"""
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by <strong>FinGuard AI</strong> A/B Testing Framework</p>
            <p style="font-size: 0.75rem; margin-top: 12px;">© 2026 FinGuard AI. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def generate_pdf_report(self, results: Dict[str, Any]) -> BytesIO:
        """Generate professional PDF report with embedded graphs"""
        buffer = BytesIO()
        
        # Generate charts as images
        chart_paths = self._generate_charts_as_images(results)
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), 
                                rightMargin=36, leftMargin=36,
                                topMargin=36, bottomMargin=36)
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0f172a'),
            alignment=TA_CENTER,
            spaceAfter=15,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header
        story.append(Paragraph("🏦 FinGuard AI - A/B Test Report", title_style))
        story.append(Paragraph(f"Test ID: {results['test_id']} | Date: {results['test_time'][:10]}", 
                              ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.HexColor('#64748b'), fontSize=9)))
        story.append(Spacer(1, 20))
        
        # Winner Section
        winner = results['winner']
        winner_data = [
            [Paragraph(f"🏆 WINNER: {winner['name']}", 
                      ParagraphStyle('WinnerTitle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=16, textColor=colors.white))],
            [Paragraph(f"Overall Score: {winner['score']:.1f}/100", 
                      ParagraphStyle('WinnerScore', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.HexColor('#d1fae5'), fontSize=12))],
            [Paragraph(f"Margin: {winner['margin']:.1f} points", 
                      ParagraphStyle('WinnerMargin', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.HexColor('#d1fae5'), fontSize=10))]
        ]
        
        winner_table = Table(winner_data, colWidths=[600])
        winner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#10b981')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('ROUNDEDCORNERS', [15, 15, 15, 15]),
        ]))
        story.append(winner_table)
        story.append(Spacer(1, 20))
        
        # Metrics Summary
        story.append(Paragraph("📊 Test Configuration", heading_style))
        
        config_data = [
            ["Sample Size", f"{results['sample_size']:,} transactions"],
            ["Confidence Level", f"{results['confidence_level']*100:.0f}%"],
            ["Model A", results['model_a']],
            ["Model B", results['model_b']],
        ]
        
        config_table = Table(config_data, colWidths=[150, 200])
        config_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0f172a')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(config_table)
        story.append(Spacer(1, 20))
        
        # Add charts to PDF
        if 'metrics_chart' in chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("📈 Performance Comparison", heading_style))
            img = Image(chart_paths['metrics_chart'], width=500, height=350)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'radar_chart' in chart_paths:
            story.append(Paragraph("🕸️ Radar Chart Analysis", heading_style))
            img = Image(chart_paths['radar_chart'], width=500, height=350)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'speed_chart' in chart_paths:
            story.append(Paragraph("⚡ Speed Comparison", heading_style))
            img = Image(chart_paths['speed_chart'], width=500, height=300)
            story.append(img)
            story.append(Spacer(1, 10))
        
        if 'significance_chart' in chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("📊 Statistical Significance Analysis", heading_style))
            img = Image(chart_paths['significance_chart'], width=500, height=350)
            story.append(img)
            story.append(Spacer(1, 10))
        
        # Performance Metrics Table
        story.append(PageBreak())
        story.append(Paragraph("📊 Performance Metrics", heading_style))
        
        metrics_data = [["Metric", results['model_a'], results['model_b'], "Better"]]
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time_ms']:
            if metric in results['metrics'][results['model_a']]:
                val_a = results['metrics'][results['model_a']][metric]
                val_b = results['metrics'][results['model_b']][metric]
                metric_name = metric.replace('_', ' ').title()
                
                if metric == 'inference_time_ms':
                    better = results['model_a'] if val_a < val_b else results['model_b']
                    val_a_str = f"{val_a:.0f}ms"
                    val_b_str = f"{val_b:.0f}ms"
                else:
                    better = results['model_a'] if val_a > val_b else results['model_b']
                    val_a_str = f"{val_a:.4f}"
                    val_b_str = f"{val_b:.4f}"
                
                metrics_data.append([metric_name, val_a_str, val_b_str, better])
        
        if 'auc' in results['metrics'][results['model_a']] and not np.isnan(results['metrics'][results['model_a']]['auc']):
            val_a = results['metrics'][results['model_a']]['auc']
            val_b = results['metrics'][results['model_b']]['auc']
            better = results['model_a'] if val_a > val_b else results['model_b']
            metrics_data.append(['AUC', f"{val_a:.4f}", f"{val_b:.4f}", better])
        
        metrics_table = Table(metrics_data, colWidths=[120, 100, 100, 100])
        metrics_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Statistical Significance Table
        story.append(Paragraph("📊 Statistical Significance", heading_style))
        
        sig_data = [["Metric", "P-Value", "Significant?", "Lift"]]
        
        for metric, test in results['statistical_tests'].items():
            metric_name = metric.replace('_', ' ').title()
            sig_text = "Yes ✓" if test['is_significant'] else "No ✗"
            sig_data.append([metric_name, f"{test['p_value']:.4f}", sig_text, f"{test['lift_percentage']:+.1f}%"])
        
        sig_table = Table(sig_data, colWidths=[120, 100, 100, 100])
        sig_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(sig_table)
        story.append(Spacer(1, 20))
        
        # Lift Chart
        if 'lift_chart' in chart_paths:
            story.append(Paragraph("📈 Lift Analysis", heading_style))
            img = Image(chart_paths['lift_chart'], width=500, height=350)
            story.append(img)
            story.append(Spacer(1, 10))
        
        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("💡 Recommendations", heading_style))
        
        for rec in results['recommendations']:
            story.append(Paragraph(f"• {rec}", ParagraphStyle('RecStyle', parent=styles['Normal'], leftIndent=20, spaceAfter=6, fontSize=9)))
        
        story.append(Spacer(1, 30))
        
        # Footer
        footer_text = Paragraph("Built by <b>FinGuard AI Team</b> | Enterprise Fraud Detection Platform", 
                               ParagraphStyle('FooterStyle', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.HexColor('#64748b'), fontSize=8))
        story.append(footer_text)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Clean up chart images
        for path in chart_paths.values():
            try:
                os.remove(path)
            except:
                pass
        
        return buffer


def generate_sample_ab_test_data(n_samples: int = 1000, 
                                 fraud_rate: float = 0.05,
                                 model_a_accuracy: float = 0.95,
                                 model_b_accuracy: float = 0.90):
    """Generate sample data for A/B testing simulation"""
    np.random.seed(42)
    
    y_true = np.random.choice([0, 1], size=n_samples, p=[1-fraud_rate, fraud_rate])
    
    y_pred_a = []
    y_pred_b = []
    
    for true_label in y_true:
        if np.random.random() < model_a_accuracy:
            y_pred_a.append(true_label)
        else:
            y_pred_a.append(1 - true_label)
        
        if np.random.random() < model_b_accuracy:
            y_pred_b.append(true_label)
        else:
            y_pred_b.append(1 - true_label)
    
    return y_true, np.array(y_pred_a), np.array(y_pred_b)