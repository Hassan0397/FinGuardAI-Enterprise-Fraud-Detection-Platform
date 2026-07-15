# utils/pdf_report_generator.py
"""
PROFESSIONAL PDF REPORT GENERATOR - ENTERPRISE GRADE
Clean, modern design with perfect alignment and spacing
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Professional matplotlib configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#E2E8F0'


class UltraPremiumPDFReport:
    """Professional PDF Report Generator with Clean Design"""
    
    def __init__(self):
        # Professional color palette - Financial services grade
        self.colors = {
            # Primary brand colors
            'primary': '#0A2540',      # Dark navy
            'primary-dark': '#061A2E',
            'secondary': '#0052FF',    # Stripe blue
            'secondary-dark': '#0038CC',
            
            # Status colors
            'success': '#00C9A7',      # Teal green
            'warning': '#F5A623',      # Amber
            'danger': '#FF3B5C',       # Red
            'info': '#00B4D8',         # Light blue
            
            # Neutrals - Clean grayscale
            'gray-1': '#FFFFFF',       # White
            'gray-2': '#F8F9FC',       # Off white
            'gray-3': '#F1F3F8',       # Light gray
            'gray-4': '#E4E7EC',       # Border gray
            'gray-5': '#CDD1D9',       # Medium light
            'gray-6': '#9CA3AF',       # Medium
            'gray-7': '#6B7280',       # Dark medium
            'gray-8': '#374151',       # Dark
            'gray-9': '#1F2937',       # Darker
            
            # Text colors
            'text-primary': '#111827',
            'text-secondary': '#6B7280',
            'text-tertiary': '#9CA3AF',
            'text-inverse': '#FFFFFF',
        }
        
    def generate_complete_report(self, explanation: Dict, predictions_df: pd.DataFrame, 
                                   selected_txn: str) -> bytes:
        """Generate professional PDF report"""
        
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(letter),
            rightMargin=50,
            leftMargin=50,
            topMargin=45,
            bottomMargin=45,
            title=f"FinGuard_Report_{selected_txn}"
        )
        
        story = []
        styles = self._create_professional_styles()
        
        # Cover Page
        story.extend(self._create_cover_page(explanation, selected_txn, styles))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(explanation, styles))
        story.append(PageBreak())
        
        # SHAP Waterfall
        story.extend(self._create_shap_waterfall(explanation, styles))
        story.append(PageBreak())
        
        # Feature Impact
        story.extend(self._create_feature_impact(explanation, styles))
        story.append(PageBreak())
        
        # Risk Decomposition
        story.extend(self._create_risk_decomposition(explanation, styles))
        story.append(PageBreak())
        
        # Detailed Analysis
        story.extend(self._create_detailed_analysis(explanation, styles))
        story.append(PageBreak())
        
        # AI Explanations
        story.extend(self._create_ai_explanations(explanation, styles))
        story.append(PageBreak())
        
        # Recommendations
        story.extend(self._create_recommendations(explanation, styles))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_professional_styles(self):
        """Create clean, professional typography styles"""
        styles = getSampleStyleSheet()
        
        # Clear hierarchy with consistent spacing
        # Only add styles that don't exist yet
        
        if 'CoverTitle' not in styles:
            styles.add(ParagraphStyle(
                name='CoverTitle',
                fontSize=48,
                textColor=HexColor(self.colors['text-primary']),
                alignment=1,
                spaceAfter=12,
                fontName='Helvetica-Bold',
                leading=56
            ))
        
        if 'CoverSubtitle' not in styles:
            styles.add(ParagraphStyle(
                name='CoverSubtitle',
                fontSize=12,
                textColor=HexColor(self.colors['text-secondary']),
                alignment=1,
                spaceAfter=40,
                fontName='Helvetica',
                leading=18
            ))
        
        if 'SectionTitle' not in styles:
            styles.add(ParagraphStyle(
                name='SectionTitle',
                fontSize=20,
                textColor=HexColor(self.colors['text-primary']),
                spaceAfter=6,
                fontName='Helvetica-Bold',
                leading=28
            ))
        
        if 'SectionSubtitle' not in styles:
            styles.add(ParagraphStyle(
                name='SectionSubtitle',
                fontSize=10,
                textColor=HexColor(self.colors['text-secondary']),
                spaceAfter=20,
                fontName='Helvetica',
                leading=14
            ))
        
        if 'CardTitle' not in styles:
            styles.add(ParagraphStyle(
                name='CardTitle',
                fontSize=11,
                textColor=HexColor(self.colors['text-primary']),
                spaceAfter=8,
                fontName='Helvetica-Bold',
                leading=16
            ))
        
        # Use a different name to avoid conflict
        if 'DescriptionText' not in styles:
            styles.add(ParagraphStyle(
                name='DescriptionText',
                fontSize=9,
                textColor=HexColor(self.colors['text-secondary']),
                spaceAfter=4,
                fontName='Helvetica',
                leading=13
            ))
        
        if 'MetricValue' not in styles:
            styles.add(ParagraphStyle(
                name='MetricValue',
                fontSize=32,
                textColor=HexColor(self.colors['secondary']),
                alignment=1,
                fontName='Helvetica-Bold',
                leading=38
            ))
        
        if 'MetricLabel' not in styles:
            styles.add(ParagraphStyle(
                name='MetricLabel',
                fontSize=8,
                textColor=HexColor(self.colors['text-tertiary']),
                alignment=1,
                fontName='Helvetica',
                leading=10
            ))
        
        if 'FooterText' not in styles:
            styles.add(ParagraphStyle(
                name='FooterText',
                fontSize=7,
                textColor=HexColor(self.colors['text-tertiary']),
                alignment=1,
                fontName='Helvetica',
                leading=10
            ))
        
        return styles
    
    def _create_cover_page(self, explanation: Dict, txn_id: str, styles) -> List:
        """Create clean, professional cover page"""
        story = []
        
        # Top accent line
        accent_line = Table([[' ']], colWidths=[7.5*inch], rowHeights=[3])
        accent_line.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))
        ]))
        story.append(accent_line)
        story.append(Spacer(1, 1.2*inch))
        
        # Title
        story.append(Paragraph("FINGUARD AI", styles['CoverTitle']))
        story.append(Paragraph("Fraud Intelligence Report", styles['CoverSubtitle']))
        
        story.append(Spacer(1, 0.6*inch))
        
        # Risk gauge
        risk_score = explanation.get('risk_score', 0)
        gauge_img = self._create_clean_gauge(risk_score)
        story.append(Image(gauge_img, width=3*inch, height=2.8*inch))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Risk level
        if risk_score > 84:
            risk_text = "CRITICAL RISK"
            risk_color = self.colors['danger']
        elif risk_score > 44:
            risk_text = "ELEVATED RISK"
            risk_color = self.colors['warning']
        else:
            risk_text = "LOW RISK"
            risk_color = self.colors['success']
        
        risk_badge = Table([[risk_text]], colWidths=[3.5*inch], rowHeights=[0.45*inch])
        risk_badge.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(risk_color + '10')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor(risk_color)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), 20),
        ]))
        story.append(risk_badge)
        
        story.append(Spacer(1, 0.5*inch))
        
        # Info section
        info_data = [
            ["Report ID", f"FGR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"],
            ["Transaction ID", txn_id[:20] + "..." if len(txn_id) > 20 else txn_id],
            ["Date", datetime.now().strftime('%B %d, %Y')],
            ["Time", datetime.now().strftime('%I:%M %p')],
        ]
        
        info_table = Table(info_data, colWidths=[1.5*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor(self.colors['text-secondary'])),
            ('TEXTCOLOR', (1, 0), (1, -1), HexColor(self.colors['text-primary'])),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(info_table)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer accent
        story.append(Spacer(1, 0.5*inch))
        bottom_line = Table([[' ']], colWidths=[7.5*inch], rowHeights=[1])
        bottom_line.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['gray-4']))
        ]))
        story.append(bottom_line)
        
        return story
    
    def _create_clean_gauge(self, risk_score: float) -> io.BytesIO:
        """Create clean, minimal gauge chart"""
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='white')
        fig.patch.set_facecolor('white')
        
        theta = np.linspace(0, np.pi, 200)
        
        # Gradient zones
        zones = [
            (0, 0.44, '#00C9A7'),
            (0.44, 0.84, '#F5A623'),
            (0.84, 1.0, '#FF3B5C')
        ]
        
        for start, end, color in zones:
            theta_zone = np.linspace(start * np.pi, end * np.pi, 100)
            ax.fill_between(np.cos(theta_zone), 0, np.sin(theta_zone), 
                           color=color, alpha=0.2, edgecolor='white', linewidth=0.5)
        
        # Outer rings
        ax.plot(np.cos(theta), np.sin(theta), color='#E2E8F0', linewidth=2)
        ax.plot(np.cos(theta) * 0.82, np.sin(theta) * 0.82, color='#E2E8F0', linewidth=1)
        
        # Tick marks
        for val in [0, 44, 84, 100]:
            angle = val / 100 * np.pi
            ax.plot([np.cos(angle) * 0.82, np.cos(angle) * 0.92],
                   [np.sin(angle) * 0.82, np.sin(angle) * 0.92],
                   color='#6B7280', linewidth=1.5)
            ax.text(np.cos(angle) * 0.98, np.sin(angle) * 0.98 - 0.03, 
                   str(val), ha='center', fontsize=7, color='#6B7280')
        
        # Labels
        labels = [(0.22*np.pi, 'LOW'), (0.64*np.pi, 'MODERATE'), (0.92*np.pi, 'HIGH')]
        for angle, label in labels:
            ax.text(np.cos(angle) * 1.08, np.sin(angle) * 1.08, label,
                   ha='center', fontsize=7, color='#6B7280')
        
        # Needle
        angle = risk_score / 100 * np.pi
        ax.arrow(0, 0, np.cos(angle) * 0.7, np.sin(angle) * 0.7,
                head_width=0.06, head_length=0.08, fc='#0A2540', 
                ec='#0A2540', width=0.012, zorder=10)
        
        # Center
        ax.add_patch(Circle((0, 0), 0.08, color='#0A2540', zorder=11))
        ax.add_patch(Circle((0, 0), 0.04, color='white', zorder=12))
        
        # Score
        ax.text(0, -0.25, f'{risk_score:.0f}', ha='center', va='center',
               fontsize=22, fontweight='bold', color='#0052FF')
        ax.text(0, -0.33, 'RISK SCORE', ha='center', va='center',
               fontsize=6, color='#9CA3AF')
        
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.45, 1.15)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        return buf
    
    def _create_executive_summary(self, explanation: Dict, styles) -> List:
        """Create clean executive summary"""
        story = []
        
        story.append(Paragraph("Executive Summary", styles['SectionTitle']))
        story.append(Paragraph("Key findings and risk assessment overview", styles['SectionSubtitle']))
        
        # Horizontal line
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        # Metrics row
        risk_score = explanation.get('risk_score', 0)
        fraud_prob = explanation.get('fraud_probability', 0)
        confidence = explanation.get('confidence', 0.8)
        
        metrics = [
            ("Risk Score", f"{risk_score:.0f}", "/100"),
            ("Fraud Probability", f"{fraud_prob:.1%}", ""),
            ("Model Confidence", f"{confidence:.1%}", "")
        ]
        
        metric_cards = []
        for label, value, suffix in metrics:
            card = Table([
                [Paragraph(label, styles['MetricLabel'])],
                [Paragraph(f"{value}{suffix}", styles['MetricValue'])]
            ], colWidths=[2.2*inch], rowHeights=[0.35*inch, 0.7*inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['gray-2'])),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            metric_cards.append(card)
        
        metrics_table = Table([metric_cards], colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
        story.append(metrics_table)
        
        story.append(Spacer(1, 0.25*inch))
        
        # Assessment text
        if risk_score > 70:
            assessment = "⚠️ This transaction exhibits elevated risk indicators requiring immediate attention."
            color = self.colors['danger']
        elif risk_score > 40:
            assessment = "📊 Moderate risk factors detected. Additional verification recommended."
            color = self.colors['warning']
        else:
            assessment = "✅ Transaction appears legitimate within normal risk parameters."
            color = self.colors['success']
        
        assessment_table = Table([[assessment]], colWidths=[7*inch], rowHeights=[0.5*inch])
        assessment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(color + '08')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor(color)),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(assessment_table)
        
        return story
    
    def _create_shap_waterfall(self, explanation: Dict, styles) -> List:
        """Create SHAP waterfall section"""
        story = []
        
        story.append(Paragraph("Feature Impact Analysis", styles['SectionTitle']))
        story.append(Paragraph("SHAP value breakdown by feature", styles['SectionSubtitle']))
        
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        chart_img = self._create_professional_waterfall(explanation)
        if chart_img:
            story.append(Image(chart_img, width=7*inch, height=3.8*inch))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Positive values increase risk | Negative values decrease risk", 
                              styles['DescriptionText']))
        
        return story
    
    def _create_professional_waterfall(self, explanation: Dict) -> Optional[io.BytesIO]:
        """Create clean professional waterfall chart"""
        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            return None
        
        # Prepare data
        features = []
        shap_values = []
        for feature, data in feature_importance.items():
            feature_name = feature.replace('_', ' ').title()
            if len(feature_name) > 25:
                feature_name = feature_name[:22] + '...'
            features.append(feature_name)
            shap_val = data.get('shap_value', 0) if isinstance(data, dict) else data
            shap_values.append(shap_val)
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:10]
        features = [features[i] for i in sorted_idx]
        shap_values = [shap_values[i] for i in sorted_idx]
        
        # Colors
        colors = ['#FF3B5C' if v > 0 else '#00C9A7' for v in shap_values]
        
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
        fig.patch.set_facecolor('white')
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, shap_values, color=colors, height=0.6, 
                      edgecolor='white', linewidth=1.5, alpha=0.8)
        
        # Labels
        for bar, val in zip(bars, shap_values):
            x_pos = val + 0.02 if val > 0 else val - 0.05
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
                   va='center', ha=ha, fontsize=8, fontweight='bold')
        
        ax.axvline(x=0, color='#9CA3AF', linewidth=1, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('SHAP Value', fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E2E8F0')
        ax.spines['bottom'].set_color('#E2E8F0')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        return buf
    
    def _create_feature_impact(self, explanation: Dict, styles) -> List:
        """Create feature impact section"""
        story = []
        
        story.append(Paragraph("Risk Factor Analysis", styles['SectionTitle']))
        story.append(Paragraph("Top contributing features to risk score", styles['SectionSubtitle']))
        
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        chart_img = self._create_professional_impact_chart(explanation)
        if chart_img:
            story.append(Image(chart_img, width=7*inch, height=3.8*inch))
        
        return story
    
    def _create_professional_impact_chart(self, explanation: Dict) -> Optional[io.BytesIO]:
        """Create clean impact chart"""
        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            return None
        
        features = []
        impacts = []
        for feature, data in feature_importance.items():
            feature_name = feature.replace('_', ' ').title()
            if len(feature_name) > 25:
                feature_name = feature_name[:22] + '...'
            features.append(feature_name)
            impact = data.get('impact_pct', 0) if isinstance(data, dict) else data * 100
            impacts.append(abs(impact))
        
        sorted_idx = np.argsort(impacts)[::-1][:8]
        features = [features[i] for i in sorted_idx]
        impacts = [impacts[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
        fig.patch.set_facecolor('white')
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(impacts)))
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, impacts, color=colors, height=0.6, 
                      edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, impacts):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                   va='center', ha='left', fontsize=8, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Impact (%)', fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E2E8F0')
        ax.spines['bottom'].set_color('#E2E8F0')
        ax.set_xlim(0, max(impacts) * 1.1)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        return buf
    
    def _create_risk_decomposition(self, explanation: Dict, styles) -> List:
        """Create risk decomposition section"""
        story = []
        
        story.append(Paragraph("Risk Decomposition", styles['SectionTitle']))
        story.append(Paragraph("Balance of positive and negative risk factors", styles['SectionSubtitle']))
        
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        chart_img = self._create_clean_donut(explanation)
        if chart_img:
            story.append(Image(chart_img, width=6*inch, height=3.8*inch))
        
        return story
    
    def _create_clean_donut(self, explanation: Dict) -> Optional[io.BytesIO]:
        """Create clean donut chart"""
        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            return None
        
        pos_sum = 0
        neg_sum = 0
        for feature, data in feature_importance.items():
            shap_val = data.get('shap_value', 0) if isinstance(data, dict) else data
            if shap_val > 0:
                pos_sum += abs(shap_val)
            else:
                neg_sum += abs(shap_val)
        
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')
        fig.patch.set_facecolor('white')
        
        sizes = [pos_sum, neg_sum]
        colors = ['#FF3B5C', '#00C9A7']
        labels = ['Risk Increasing', 'Risk Decreasing']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'fontsize': 9},
                                          pctdistance=0.75)
        
        centre_circle = Circle((0, 0), 0.55, fc='white', linewidth=1.5, edgecolor='#E2E8F0')
        ax.add_artist(centre_circle)
        
        total = pos_sum + neg_sum
        ax.text(0, 0, f'{total:.2f}', ha='center', va='center',
               fontsize=16, fontweight='bold', color='#1F2937')
        ax.text(0, -0.12, 'Total Impact', ha='center', va='center',
               fontsize=8, color='#6B7280')
        
        for wedge in wedges:
            wedge.set_edgecolor('white')
            wedge.set_linewidth(2)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        return buf
    
    def _create_detailed_analysis(self, explanation: Dict, styles) -> List:
        """Create detailed analysis table"""
        story = []
        
        story.append(Paragraph("Detailed Feature Analysis", styles['SectionTitle']))
        story.append(Paragraph("Complete breakdown with SHAP values and impact", styles['SectionSubtitle']))
        
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        feature_importance = explanation.get('feature_importance', {})
        
        # Table data
        table_data = [["Feature", "SHAP Value", "Impact", "Direction"]]
        
        for feature, data in list(feature_importance.items())[:12]:
            if isinstance(data, dict):
                shap_val = data.get('shap_value', 0)
                impact = data.get('impact_pct', 0)
                direction = "▲" if shap_val > 0 else "▼"
            else:
                shap_val = data
                impact = data * 100
                direction = "▲" if shap_val > 0 else "▼"
            
            feature_name = feature.replace('_', ' ').title()
            if len(feature_name) > 30:
                feature_name = feature_name[:27] + '...'
            
            table_data.append([
                feature_name,
                f"{shap_val:+.4f}",
                f"{abs(impact):.1f}%",
                direction
            ])
        
        # Create table with clean styling
        table = Table(table_data, colWidths=[3.5*inch, 1.2*inch, 1*inch, 0.8*inch])
        
        table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), HexColor(self.colors['gray-8'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor(self.colors['text-inverse'])),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            
            # Body
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor(self.colors['gray-4'])),
            
            # Row striping
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [HexColor(self.colors['gray-1']), HexColor(self.colors['gray-2'])]),
        ]))
        
        story.append(table)
        
        return story
    
    def _create_ai_explanations(self, explanation: Dict, styles) -> List:
        """Create AI explanations section"""
        story = []
        
        story.append(Paragraph("AI Model Interpretation", styles['SectionTitle']))
        story.append(Paragraph("SHAP analysis natural language explanation", styles['SectionSubtitle']))
        
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        natural_lang = explanation.get('natural_language', '')
        
        # Explanation box
        lines = natural_lang.split('\n')[:15]
        for line in lines:
            if line.strip():
                story.append(Paragraph(f"• {line}", styles['DescriptionText']))
        
        story.append(Spacer(1, 0.15*inch))
        
        # Confidence
        confidence = explanation.get('confidence', 0.8)
        confidence_text = f"Model Confidence: {confidence:.1%}"
        confidence_table = Table([[confidence_text]], colWidths=[7*inch], rowHeights=[0.4*inch])
        confidence_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['gray-2'])),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(confidence_table)
        
        return story
    
    def _create_recommendations(self, explanation: Dict, styles) -> List:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("Recommendations", styles['SectionTitle']))
        story.append(Paragraph("Actionable next steps based on risk assessment", styles['SectionSubtitle']))
        
        line = Table([[' ']], colWidths=[1*inch], rowHeights=[2])
        line.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['secondary']))]))
        story.append(line)
        story.append(Spacer(1, 0.2*inch))
        
        recommendations = explanation.get('recommendations', [
            "Review transaction for suspicious patterns",
            "Verify merchant credentials and history",
            "Check for velocity in recent transactions",
            "Consider additional verification steps"
        ])
        
        for i, rec in enumerate(recommendations[:8], 1):
            # Priority styling
            if any(word in rec.upper() for word in ['URGENT', 'BLOCK', 'IMMEDIATE']):
                priority = "HIGH"
                color = self.colors['danger']
            elif any(word in rec.upper() for word in ['REVIEW', 'VERIFY']):
                priority = "MEDIUM"
                color = self.colors['warning']
            else:
                priority = "LOW"
                color = self.colors['success']
            
            rec_table = Table([
                [f"{i}.", f"[{priority} PRIORITY]", rec]
            ], colWidths=[0.3*inch, 1*inch, 5.2*inch])
            rec_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TEXTCOLOR', (1, 0), (1, 0), HexColor(color)),
                ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(rec_table)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_divider = Table([[' ']], colWidths=[7.5*inch], rowHeights=[1])
        footer_divider.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor(self.colors['gray-4']))]))
        story.append(footer_divider)
        story.append(Spacer(1, 0.15*inch))
        
        footer_text = f"""
        FinGuard AI Enterprise Platform | Built by HASSAN SUBHANI<br/>
        Report ID: {datetime.now().strftime('%Y%m%d-%H%M%S')} | Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        """
        story.append(Paragraph(footer_text, styles['FooterText']))
        
        return story