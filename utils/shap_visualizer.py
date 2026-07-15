# utils/shap_visualizer.py
"""
Advanced SHAP Visualizations for Enterprise Fraud Detection
Premium-level interactive visualizations 
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class SHAPVisualizer:
    """Professional SHAP visualization engine - Fixed for compatibility"""
    
    @staticmethod
    def create_waterfall_chart(shap_data: Dict, title: str = "SHAP Waterfall Plot") -> go.Figure:
        """Create interactive waterfall chart for SHAP values"""
        
        waterfall_data = shap_data.get('waterfall_data', [])
        if not waterfall_data:
            # Create sample data if empty
            waterfall_data = [
                {'feature': 'Base Value', 'shap_value': 0.15, 'start_value': 0.15, 'end_value': 0.15, 'direction': 'positive'},
                {'feature': 'Transaction Amount', 'shap_value': 0.25, 'start_value': 0.15, 'end_value': 0.40, 'direction': 'positive'},
                {'feature': 'Merchant Risk', 'shap_value': 0.18, 'start_value': 0.40, 'end_value': 0.58, 'direction': 'positive'},
                {'feature': 'International', 'shap_value': 0.12, 'start_value': 0.58, 'end_value': 0.70, 'direction': 'positive'}
            ]
        
        # Prepare data
        features = [item.get('feature', f'Feature_{i}')[:30] for i, item in enumerate(waterfall_data)]
        shap_values = [item.get('shap_value', 0) for item in waterfall_data]
        directions = [item.get('direction', 'positive') for item in waterfall_data]
        
        colors = ['#dc2626' if d == 'positive' else '#10b981' for d in directions]
        
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=shap_values,
            y=features,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2),
                cornerradius=8
            ),
            text=[f"{v:+.3f}" for v in shap_values],
            textposition='outside',
            textfont=dict(size=11, color='#1e293b'),
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:+.3f}<extra></extra>'
        ))
        
        # Add baseline annotation
        base_value = shap_data.get('base_value', 0.5)
        final_value = base_value + sum(shap_values)
        
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="#64748b",
            line_width=1.5,
            annotation_text="Base",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Base: {base_value:.3f} → Final: {final_value:.3f}</sub>",
                font=dict(size=16, color='#0f172a')
            ),
            xaxis_title="<b>SHAP Value (Impact on Prediction)</b>",
            yaxis_title="<b>Features</b>",
            height=max(400, len(features) * 32),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='#e2e8f0',
                gridwidth=1,
                tickfont=dict(size=11),
                title_font=dict(size=12, color='#475569'),
                zeroline=True,
                zerolinecolor='#cbd5e1',
                zerolinewidth=1
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                gridwidth=1,
                tickfont=dict(size=11),
                title_font=dict(size=12, color='#475569')
            ),
            margin=dict(l=20, r=80, t=80, b=30),
            hoverlabel=dict(bgcolor='white', font_size=12)
        )
        
        return fig
    
    @staticmethod
    def create_force_plot_3d(shap_data: Dict, feature_names: List[str]) -> go.Figure:
        """Create 3D force plot for SHAP values"""
        
        shap_values = shap_data.get('shap_values', [])
        if not shap_values:
            return go.Figure()
        
        # Create 3D scatter plot
        n_features = min(10, len(feature_names), len(shap_values))
        if n_features == 0:
            return go.Figure()
            
        colors = ['#dc2626' if v > 0 else '#10b981' for v in shap_values[:n_features]]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=list(range(n_features)),
            y=[abs(v) * 100 for v in shap_values[:n_features]],
            z=[1] * n_features,
            mode='markers+text',
            marker=dict(
                size=[max(10, min(50, abs(v) * 50 + 10)) for v in shap_values[:n_features]],
                color=colors,
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            text=feature_names[:n_features],
            textposition="top center",
            textfont=dict(size=10, color='#0f172a'),
            hovertemplate='<b>%{text}</b><br>Impact: %{marker.size:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>3D SHAP Impact Visualization</b>",
                font=dict(size=16, color='#0f172a')
            ),
            scene=dict(
                xaxis_title="Feature Index",
                yaxis_title="Impact Magnitude (%)",
                zaxis_title="",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis=dict(gridcolor='#e2e8f0'),
                yaxis=dict(gridcolor='#e2e8f0')
            ),
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_shap_dashboard(shap_data: Dict, feature_names: List[str]) -> go.Figure:
        """Create comprehensive SHAP dashboard with separate plots (FIXED)"""
        
        shap_values = shap_data.get('shap_values', [])
        feature_importance = shap_data.get('feature_importance', {})
        
        # Create a container figure with multiple traces
        fig = go.Figure()
        
        # Chart 1: Feature Impact (Top 10 features) - Horizontal Bar
        if shap_values and feature_names:
            top_indices = np.argsort(np.abs(shap_values))[-10:][::-1]
            top_indices = [i for i in top_indices if i < len(feature_names) and i < len(shap_values)]
            
            if top_indices:
                top_features = [feature_names[i][:25] for i in top_indices]
                top_shap = [shap_values[i] for i in top_indices]
                colors_bar = ['#dc2626' if v > 0 else '#10b981' for v in top_shap]
                
                # Feature Impact Chart
                fig.add_trace(go.Bar(
                    x=top_shap,
                    y=top_features,
                    orientation='h',
                    marker=dict(color=colors_bar, line=dict(color='white', width=1)),
                    text=[f"{v:+.3f}" for v in top_shap],
                    textposition='outside',
                    name='Feature Impact',
                    hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:+.3f}<extra></extra>'
                ))
                
                # Add annotation
                fig.add_annotation(
                    x=0.5, y=1.08, xref="paper", yref="paper",
                    text="<b>📊 Feature Impact Analysis</b>",
                    showarrow=False,
                    font=dict(size=16, color='#0f172a')
                )
        
        # Chart 2: Cumulative Impact - Line chart
        sorted_shap = sorted([abs(v) for v in shap_values], reverse=True)
        if sorted_shap:
            cumulative = np.cumsum(sorted_shap)
            cumulative_normalized = cumulative / max(cumulative) if max(cumulative) > 0 else cumulative
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative) + 1)),
                y=cumulative_normalized,
                mode='lines+markers',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6, color='#3b82f6', symbol='circle'),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)',
                name='Cumulative Impact',
                yaxis='y2',
                hovertemplate='Features: %{x}<br>Cumulative Impact: %{y:.1%}<extra></extra>'
            ))
        
        # Add reference line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b", opacity=0.5)
        
        # Update layout with secondary y-axis for cumulative
        fig.update_layout(
            title=dict(
                text="<b>🏆 SHAP Analysis Dashboard</b>",
                font=dict(size=20, color='#0f172a'),
                x=0.5
            ),
            height=600,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            xaxis=dict(
                title="<b>Features / Cumulative Index</b>",
                gridcolor='#e2e8f0',
                gridwidth=1,
                zerolinecolor='#cbd5e1'
            ),
            yaxis=dict(
                title="<b>SHAP Value</b>",
                gridcolor='#e2e8f0',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='#cbd5e1',
                side='left'
            ),
            yaxis2=dict(
                title="<b>Cumulative Impact</b>",
                overlaying='y',
                side='right',
                range=[0, 1.1],
                tickformat='.0%',
                gridcolor='rgba(0,0,0,0)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=80, t=80, b=50)
        )
        
        # Add annotation for contribution summary
        positive_sum = sum(v for v in shap_values if v > 0)
        negative_sum = abs(sum(v for v in shap_values if v < 0))
        contribution_text = f"📊 Contribution Analysis: Risk Increasing (+{positive_sum:.3f}) | Risk Decreasing (-{negative_sum:.3f})"
        fig.add_annotation(
            x=0.5, y=-0.12, xref="paper", yref="paper",
            text=contribution_text,
            showarrow=False,
            font=dict(size=11, color='#64748b'),
            bgcolor='#f8fafc',
            bordercolor='#e2e8f0',
            borderwidth=1,
            borderpad=4
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_radar(feature_importance: Dict) -> go.Figure:
        """Create radar chart for feature importance comparison"""
        
        if not feature_importance:
            return go.Figure()
        
        # Take top 8 features
        if isinstance(list(feature_importance.values())[0], dict):
            top_items = sorted(feature_importance.items(), key=lambda x: x[1].get('importance', 0), reverse=True)[:8]
            top_features = [item[0][:20] for item in top_items]
            top_scores = [item[1].get('importance', 0) for item in top_items]
        else:
            top_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            top_features = [item[0][:20] for item in top_items]
            max_score = max(top_items[0][1], 1) if top_items else 1
            top_scores = [item[1] / max_score for item in top_items]
        
        if not top_features:
            return go.Figure()
        
        # Close the loop for radar chart
        top_features.append(top_features[0])
        top_scores.append(top_scores[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=top_scores,
            theta=top_features,
            fill='toself',
            name='Feature Importance',
            line=dict(color='#3b82f6', width=2),
            fillcolor='rgba(59, 130, 246, 0.3)',
            hovertemplate='%{theta}: %{r:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10),
                    gridcolor='#e2e8f0',
                    tickformat='.0%'
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, color='#475569'),
                    gridcolor='#e2e8f0',
                    rotation=90
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            title=dict(
                text="<b>Feature Importance Radar</b>",
                font=dict(size=16, color='#0f172a'),
                x=0.5
            ),
            height=500,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_confidence_gauge(confidence: float) -> go.Figure:
        """Create confidence gauge chart"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            title={'text': "SHAP Confidence Score", 'font': {'size': 14, 'color': '#0f172a'}},
            delta={'reference': 80, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#dc2626"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickfont': {'size': 10}},
                'bar': {'color': "#3b82f6", 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 80], 'color': "#fed7aa"},
                    {'range': [80, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "#0f172a", 'width': 2},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            },
            number={'font': {'size': 40, 'color': '#0f172a'}}
        ))
        
        fig.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add confidence interpretation
        if confidence >= 0.8:
            interpretation = "High Confidence - Reliable Prediction"
            color = "#10b981"
        elif confidence >= 0.6:
            interpretation = "Medium Confidence - Use with Caution"
            color = "#f59e0b"
        else:
            interpretation = "Low Confidence - Verify Manually"
            color = "#dc2626"
        
        fig.add_annotation(
            x=0.5, y=-0.2, xref="paper", yref="paper",
            text=interpretation,
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor='#f8fafc',
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            width=250,
            height=30
        )
        
        return fig
    
    @staticmethod
    def create_shap_summary_plot(shap_values: np.ndarray, feature_names: List[str], 
                                  X_sample: pd.DataFrame, max_display: int = 20) -> go.Figure:
        """Create a summary plot for SHAP values"""
        
        if shap_values is None or len(shap_values) == 0:
            return go.Figure()
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
        top_features = [feature_names[i][:25] for i in top_indices if i < len(feature_names)]
        top_shap_means = [mean_abs_shap[i] for i in top_indices]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_shap_means,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_shap_means,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Mean |SHAP|")
            ),
            text=[f"{v:.4f}" for v in top_shap_means],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="<b>Global SHAP Feature Importance</b>",
                font=dict(size=16, color='#0f172a'),
                x=0.5
            ),
            xaxis_title="<b>Mean |SHAP Value|</b>",
            yaxis_title="<b>Features</b>",
            height=max(400, len(top_features) * 30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#e2e8f0', gridwidth=1),
            yaxis=dict(gridcolor='#e2e8f0', gridwidth=1),
            margin=dict(l=20, r=50, t=60, b=30)
        )
        
        return fig
    
    @staticmethod
    def create_dependence_plot(shap_values: np.ndarray, X_sample: pd.DataFrame, 
                                feature_name: str, feature_idx: int) -> go.Figure:
        """Create a SHAP dependence plot for a specific feature"""
        
        if shap_values is None or feature_idx >= shap_values.shape[1]:
            return go.Figure()
        
        feature_values = X_sample.iloc[:, feature_idx] if feature_idx < len(X_sample.columns) else X_sample.iloc[:, 0]
        shap_vals_for_feature = shap_values[:, feature_idx]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=shap_vals_for_feature,
            mode='markers',
            marker=dict(
                size=8,
                color=shap_vals_for_feature,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="SHAP Value"),
                opacity=0.7
            ),
            hovertemplate=f'<b>{feature_name}</b><br>Value: %{{x:.3f}}<br>SHAP: %{{y:.3f}}<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(feature_values, shap_vals_for_feature, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=sorted(feature_values),
            y=p(sorted(feature_values)),
            mode='lines',
            line=dict(color='#dc2626', width=2, dash='dash'),
            name='Trend',
            hovertemplate='Trend: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>SHAP Dependence Plot: {feature_name}</b>",
                font=dict(size=16, color='#0f172a'),
                x=0.5
            ),
            xaxis_title=f"<b>{feature_name} Value</b>",
            yaxis_title="<b>SHAP Value (Impact on Prediction)</b>",
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#e2e8f0', gridwidth=1, zeroline=True),
            yaxis=dict(gridcolor='#e2e8f0', gridwidth=1, zeroline=True, zerolinecolor='#cbd5e1'),
            hovermode='closest'
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b", opacity=0.5)
        
        return fig
