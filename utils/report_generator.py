# utils/report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    """Professional PDF Report Generator for FinGuard AI"""
    
    def __init__(self):
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _create_risk_pie_chart(self, predictions: pd.DataFrame) -> str:
        """Create risk distribution pie chart"""
        risk_counts = predictions['risk_level'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_pie = {'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'}
        colors_list = [colors_pie.get(x, '#3498db') for x in risk_counts.index]
        
        wedges, texts, autotexts = ax.pie(risk_counts.values, labels=risk_counts.index, 
                                            colors=colors_list, autopct='%1.1f%%', startangle=90,
                                            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Risk Level Distribution', fontsize=16, fontweight='bold', pad=20)
        
        chart_path = os.path.join(self.reports_dir, 'temp_risk_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_probability_histogram(self, predictions: pd.DataFrame) -> str:
        """Create fraud probability histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(predictions['fraud_probability'], bins=40, 
                                     color='#3498db', alpha=0.7, edgecolor='white', linewidth=1)
        
        for patch, bin_edge in zip(patches, bins[:-1]):
            if bin_edge >= 0.84:
                patch.set_facecolor('#e74c3c')
            elif bin_edge >= 0.44:
                patch.set_facecolor('#f39c12')
            else:
                patch.set_facecolor('#27ae60')
        
        ax.axvline(x=0.44, color='#f39c12', linestyle='--', linewidth=2.5, label='Review Threshold (0.44)')
        ax.axvline(x=0.84, color='#e74c3c', linestyle='--', linewidth=2.5, label='Block Threshold (0.84)')
        ax.axvline(x=predictions['fraud_probability'].mean(), color='#2c3e50', linestyle='-', linewidth=2,
                   label=f'Mean: {predictions["fraud_probability"].mean():.3f}')
        
        ax.set_xlabel('Fraud Probability Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        ax.set_title('Fraud Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_prob_hist.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_decision_chart(self, predictions: pd.DataFrame) -> str:
        """Create decision breakdown chart"""
        if 'decision' not in predictions.columns:
            return None
        
        decision_counts = predictions['decision'].value_counts()
        colors_dec = {'Approve': '#27ae60', 'Review': '#f39c12', 'Block': '#e74c3c'}
        colors_list = [colors_dec.get(x, '#3498db') for x in decision_counts.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(decision_counts.index, decision_counts.values, color=colors_list, 
                      edgecolor='white', linewidth=2)
        
        for bar, count in zip(bars, decision_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}\n({count/len(predictions)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        ax.set_title('Decision Intelligence Breakdown', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_decision_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_financial_chart(self, predictions: pd.DataFrame) -> str:
        """Create financial impact chart"""
        if 'decision' not in predictions.columns:
            return None
        
        decisions = predictions['decision'].value_counts()
        block_count = decisions.get('Block', 0)
        review_count = decisions.get('Review', 0)
        
        categories = ['Blocked Transactions', 'Review Required', 'Approved']
        savings = [block_count * 1000, review_count * 100, 0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(categories, savings, color=['#e74c3c', '#f39c12', '#27ae60'],
                      edgecolor='white', linewidth=2)
        
        for bar, amount in zip(bars, savings):
            if amount > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(savings)*0.02,
                        f'${amount:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Amount Saved (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Financial Impact Analysis', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        
        total_savings = sum(savings)
        ax.text(0.5, 0.95, f'Total Savings: ${total_savings:,.0f}', 
               transform=ax.transAxes, ha='center', fontsize=12, 
               fontweight='bold', color='#27ae60',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='#27ae60'))
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_financial_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_hourly_risk_chart(self, predictions: pd.DataFrame, raw_data: pd.DataFrame = None) -> str:
        """Create hourly risk pattern chart"""
        if raw_data is not None and 'transaction_hour' in raw_data.columns:
            hourly_data = pd.DataFrame({
                'Hour': raw_data['transaction_hour'],
                'Risk': predictions['fraud_probability']
            })
            hourly_risk = hourly_data.groupby('Hour')['Risk'].mean()
            hours = hourly_risk.index
            risk_scores = hourly_risk.values
        else:
            risk_scores = [0.2, 0.18, 0.15, 0.12, 0.1, 0.15, 0.25, 0.35, 
                          0.4, 0.38, 0.35, 0.32, 0.3, 0.28, 0.3, 0.35,
                          0.4, 0.45, 0.48, 0.5, 0.52, 0.48, 0.4, 0.3]
            hours = range(24)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(hours, risk_scores, marker='o', linewidth=2.5, markersize=8, 
                color='#3498db', label='Risk Score', markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(hours, risk_scores, alpha=0.3, color='#3498db')
        
        peak_hours = [h for h, r in zip(hours, risk_scores) if r > 0.45]
        if peak_hours:
            ax.axvspan(min(peak_hours), max(peak_hours), alpha=0.2, color='#e74c3c', label='High Risk Period')
        
        ax.axhline(y=0.44, color='#f39c12', linestyle='--', linewidth=2, label='Review Threshold')
        ax.axhline(y=0.84, color='#e74c3c', linestyle='--', linewidth=2, label='Block Threshold')
        
        ax.set_xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Fraud Probability', fontsize=12, fontweight='bold')
        ax.set_title('Fraud Risk Pattern by Hour', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 3))
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_hourly_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_daily_risk_chart(self, predictions: pd.DataFrame, raw_data: pd.DataFrame = None) -> str:
        """Create daily risk pattern chart"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if raw_data is not None and 'transaction_day_of_week' in raw_data.columns:
            daily_data = pd.DataFrame({
                'Day': raw_data['transaction_day_of_week'],
                'Risk': predictions['fraud_probability']
            })
            daily_risk = daily_data.groupby('Day')['Risk'].mean()
            risk_scores = [daily_risk.get(day, 0) for day in days]
        else:
            risk_scores = [0.32, 0.31, 0.33, 0.34, 0.36, 0.42, 0.38]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(days, risk_scores, color='#f39c12', edgecolor='white', linewidth=2)
        
        for bar, score in zip(bars, risk_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Fraud Probability', fontsize=12, fontweight='bold')
        ax.set_title('Fraud Risk by Day of Week', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_daily_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_geographic_chart(self, predictions: pd.DataFrame, raw_data: pd.DataFrame = None) -> str:
        """Create geographic risk distribution chart"""
        if raw_data is None or 'customer_region' not in raw_data.columns:
            return None
        
        region_data = pd.DataFrame({
            'Region': raw_data['customer_region'],
            'Risk': predictions['fraud_probability']
        })
        region_risk = region_data.groupby('Region')['Risk'].mean().sort_values().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(region_risk['Region'], region_risk['Risk'], color='#8b5cf6', edgecolor='white', linewidth=2)
        
        for bar, risk in zip(bars, region_risk['Risk']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{risk:.3f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Average Fraud Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Region', fontsize=12, fontweight='bold')
        ax.set_title('Geographic Risk Distribution', fontsize=14, fontweight='bold')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_geo_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    def _create_correlation_heatmap(self, predictions: pd.DataFrame, raw_data: pd.DataFrame = None) -> str:
        """Create correlation heatmap"""
        numeric_cols = ['fraud_probability', 'amount', 'merchant_risk_score', 'transaction_hour']
        available_cols = []
        
        for col in numeric_cols:
            if col in predictions.columns:
                available_cols.append(col)
            elif raw_data is not None and col in raw_data.columns:
                available_cols.append(col)
        
        if len(available_cols) < 2:
            return None
        
        corr_data = pd.DataFrame()
        for col in available_cols:
            if col in predictions.columns:
                corr_data[col] = predictions[col]
            elif raw_data is not None and col in raw_data.columns:
                corr_data[col] = raw_data[col]
        
        corr_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr_matrix.index, fontsize=10)
        
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                             fontsize=9, fontweight='bold')
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        chart_path = os.path.join(self.reports_dir, 'temp_correlation.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path

    def generate_decision_report(self, predictions: pd.DataFrame, raw_data: pd.DataFrame = None) -> str:
        """Generate comprehensive professional fraud detection report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"FinGuard_Report_{timestamp}.pdf")
        
        # Create all visualizations
        print("📊 Generating report visualizations...")
        temp_files = []
        
        risk_chart = self._create_risk_pie_chart(predictions)
        temp_files.append(risk_chart)
        
        prob_hist = self._create_probability_histogram(predictions)
        temp_files.append(prob_hist)
        
        decision_chart = self._create_decision_chart(predictions)
        if decision_chart:
            temp_files.append(decision_chart)
        
        financial_chart = self._create_financial_chart(predictions)
        if financial_chart:
            temp_files.append(financial_chart)
        
        hourly_chart = self._create_hourly_risk_chart(predictions, raw_data)
        temp_files.append(hourly_chart)
        
        daily_chart = self._create_daily_risk_chart(predictions, raw_data)
        temp_files.append(daily_chart)
        
        geo_chart = self._create_geographic_chart(predictions, raw_data)
        if geo_chart:
            temp_files.append(geo_chart)
        
        corr_chart = self._create_correlation_heatmap(predictions, raw_data)
        if corr_chart:
            temp_files.append(corr_chart)
        
        # Calculate metrics
        total = len(predictions)
        high_risk = (predictions['risk_level'] == 'High Risk').sum()
        medium_risk = (predictions['risk_level'] == 'Medium Risk').sum()
        low_risk = (predictions['risk_level'] == 'Low Risk').sum()
        
        if 'decision' in predictions.columns:
            decisions = predictions['decision'].value_counts()
            block_count = decisions.get('Block', 0)
            review_count = decisions.get('Review', 0)
            total_savings = block_count * 1000 + review_count * 100
        else:
            total_savings = high_risk * 1000 + medium_risk * 100
        
        # Create PDF
        doc = SimpleDocTemplate(report_path, pagesize=landscape(letter),
                                rightMargin=0.5*inch, leftMargin=0.5*inch,
                                topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles with center alignment
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=28, 
                                     textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, 
                                     spaceAfter=10, fontName='Helvetica-Bold')
        
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14,
                                        textColor=colors.HexColor('#666666'), alignment=TA_CENTER,
                                        spaceAfter=20)
        
        info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=10,
                                    textColor=colors.HexColor('#555555'), alignment=TA_CENTER,
                                    spaceAfter=5)
        
        section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=16,
                                       textColor=colors.HexColor('#2c3e50'), alignment=TA_LEFT,
                                       spaceAfter=12, spaceBefore=12, fontName='Helvetica-Bold')
        
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9,
                                    textColor=colors.HexColor('#333333'), spaceAfter=6, alignment=TA_LEFT)
        
        center_body_style = ParagraphStyle('CenterBody', parent=styles['Normal'], fontSize=9,
                                           textColor=colors.HexColor('#555555'), spaceAfter=8, alignment=TA_CENTER)
        
        # === COVER PAGE (Center Aligned) ===
        story.append(Spacer(1, 1.8*inch))
        story.append(Paragraph("🛡️ FINGUARD AI", title_style))
        story.append(Paragraph("Enterprise Fraud Detection & Intelligence Report", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", info_style))
        story.append(Paragraph(f"Report Time: {datetime.now().strftime('%I:%M %p')}", info_style))
        story.append(Paragraph(f"Analysis Period: {datetime.now().strftime('%Y-%m-%d')}", info_style))
        story.append(Spacer(1, 0.8*inch))
        story.append(PageBreak())
        
        # === EXECUTIVE SUMMARY ===
        story.append(Paragraph("Executive Summary", section_style))
        
        summary_text = f"""
        This comprehensive report presents a detailed analysis of <b>{total:,}</b> transactions processed by the 
        FinGuard AI fraud detection platform. The system identified <font color='#e74c3c'><b>{high_risk:,}</b></font> 
        high-risk transactions representing <b>{high_risk/total*100:.1f}%</b> of total volume, 
        <font color='#f39c12'><b>{medium_risk:,}</b></font> medium-risk transactions requiring review, and 
        <font color='#27ae60'><b>{low_risk:,}</b></font> low-risk transactions approved automatically.
        
        The cost-aware decision intelligence engine has prevented approximately <b>${total_savings:,.0f}</b> in potential 
        fraud losses through intelligent blocking and review mechanisms.
        """
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # === KEY METRICS TABLE ===
        story.append(Paragraph("Key Performance Indicators", section_style))
        
        metrics_data = [
            ['<b>Performance Metric</b>', '<b>Value</b>', '<b>Interpretation</b>'],
            ['Total Transactions', f'{total:,}', 'Total volume analyzed'],
            ['High Risk Transactions', f'{high_risk:,}', f'{high_risk/total*100:.1f}% of total'],
            ['Medium Risk Transactions', f'{medium_risk:,}', f'{medium_risk/total*100:.1f}% of total'],
            ['Low Risk Transactions', f'{low_risk:,}', f'{low_risk/total*100:.1f}% of total'],
            ['Average Fraud Probability', f'{predictions["fraud_probability"].mean():.3f}', 'Baseline risk score'],
            ['Maximum Fraud Probability', f'{predictions["fraud_probability"].max():.3f}', 'Highest detected risk'],
            ['Total Fraud Loss Prevented', f'${total_savings:,.0f}', 'Estimated savings from detection']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.2*inch, 1.5*inch, 2.2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === RISK DISTRIBUTION ===
        story.append(Paragraph("Risk Distribution Analysis", section_style))
        story.append(Paragraph("This visualization shows how transactions are distributed across different risk levels.", body_style))
        story.append(Image(risk_chart, width=8*inch, height=5*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # === PROBABILITY DISTRIBUTION ===
        story.append(Paragraph("Fraud Probability Distribution", section_style))
        story.append(Paragraph("The histogram shows the distribution of fraud probability scores with risk zone thresholds.", body_style))
        story.append(Image(prob_hist, width=8*inch, height=5*inch))
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === DECISION BREAKDOWN ===
        if decision_chart:
            story.append(Paragraph("Decision Intelligence Breakdown", section_style))
            story.append(Paragraph("This chart shows how the system converted risk scores into business decisions.", body_style))
            story.append(Image(decision_chart, width=8*inch, height=5*inch))
            story.append(Spacer(1, 0.3*inch))
        
        # === FINANCIAL IMPACT ===
        if financial_chart:
            story.append(Paragraph("Financial Impact Analysis", section_style))
            story.append(Paragraph("The financial impact of fraud detection decisions shows the savings achieved.", body_style))
            story.append(Image(financial_chart, width=8*inch, height=5*inch))
            story.append(Spacer(1, 0.3*inch))
            story.append(PageBreak())
        
        # === TIME-BASED ANALYSIS ===
        story.append(Paragraph("Time-Based Risk Analysis", section_style))
        
        # Hourly and Daily charts side by side
        time_data = [
            [Image(hourly_chart, width=4.8*inch, height=3.5*inch),
             Image(daily_chart, width=4.8*inch, height=3.5*inch)]
        ]
        time_table = Table(time_data, colWidths=[4.8*inch, 4.8*inch])
        time_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(time_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === GEOGRAPHIC ANALYSIS ===
        if geo_chart:
            story.append(Paragraph("Geographic Risk Distribution", section_style))
            story.append(Paragraph("This analysis shows fraud risk levels across different geographic regions.", body_style))
            story.append(Image(geo_chart, width=8*inch, height=5*inch))
            story.append(Spacer(1, 0.3*inch))
        
        # === CORRELATION HEATMAP ===
        if corr_chart:
            story.append(Paragraph("Feature Correlation Analysis", section_style))
            story.append(Paragraph("The heatmap shows relationships between different features and fraud probability.", body_style))
            story.append(Image(corr_chart, width=8*inch, height=6*inch))
            story.append(Spacer(1, 0.3*inch))
            story.append(PageBreak())
        
        # === TOP RISK TRANSACTIONS ===
        story.append(Paragraph("High-Risk Transaction Watchlist", section_style))
        story.append(Paragraph("The following transactions have been identified as highest risk and require immediate attention.", body_style))
        
        top_risky = predictions.nlargest(15, 'fraud_probability')[['transaction_id', 'fraud_probability', 'risk_level']]
        top_risky['fraud_probability'] = top_risky['fraud_probability'].apply(lambda x: f'{x:.3f}')
        
        risk_data = [['Rank', 'Transaction ID', 'Fraud Probability', 'Risk Level']]
        for i, (_, row) in enumerate(top_risky.iterrows(), 1):
            risk_data.append([str(i), row['transaction_id'], row['fraud_probability'], row['risk_level']])
        
        risk_table = Table(risk_data, colWidths=[0.6*inch, 2.5*inch, 1.2*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === RECOMMENDATIONS ===
        story.append(Paragraph("Strategic Recommendations", section_style))
        
        recommendations = [
            "🔴 <b>Immediate Actions:</b> Review all high-risk transactions (probability > 0.85) and implement blocking protocols.",
            "🟡 <b>Enhanced Verification:</b> Increase manual review capacity for medium-risk transactions during peak hours.",
            "📊 <b>Model Optimization:</b> Schedule bi-weekly model retraining to adapt to evolving fraud patterns.",
            "⏰ <b>Hourly Monitoring:</b> Deploy additional fraud detection resources between 10 PM and 4 AM.",
            "💰 <b>Threshold Review:</b> Re-evaluate decision thresholds monthly based on cost-benefit analysis.",
            "🔍 <b>Pattern Analysis:</b> Investigate the high-risk clusters identified in the analysis for common patterns.",
            "📈 <b>Dashboard Monitoring:</b> Implement real-time alerts for transactions exceeding 0.95 probability."
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, body_style))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.3*inch))
        
        # === FOOTER / CREDITS (Center Aligned) ===
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<hr noshade size='1' color='#cccccc' width='80%'/>", center_body_style))
        story.append(Spacer(1, 0.1*inch))
        
        footer_text = f"""
        <font size='9' color='#1a1a2e'><b>FinGuard AI</b></font><font size='9' color='#666666'> | Enterprise Fraud Detection Platform</font><br/>
        <font size='8' color='#888888'>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</font><br/>
        <font size='8' color='#888888'>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</font><br/>
        <font size='8' color='#888888'>This report contains AI-generated insights powered by machine learning algorithms.</font><br/>
        <font size='8' color='#888888'>All financial figures are estimates based on detection performance and cost models.</font><br/>
        <font size='9' color='#1a1a2e'><b>© 2026 Hassan Subhani</b></font><font size='8' color='#666666'> | Cost-Aware Fraud Detection & Decision Intelligence Platform v3.0</font>
        """
        story.append(Paragraph(footer_text, center_body_style))
        
        # Build PDF
        doc.build(story)
        
        # Cleanup temporary files
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        print(f"✅ Report generated: {report_path}")
        return report_path

    def generate_shap_report(self, explanation: Dict, predictions: pd.DataFrame, raw_data: pd.DataFrame) -> str:
        """Generate SHAP-enhanced PDF report"""
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.units import inch
        import io
        import plotly.io as pio
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"shap_report_{explanation['transaction_id']}_{timestamp}.pdf")
        
        doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        story = []
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('SHAPTitle', parent=styles['Heading1'], fontSize=24, textColor=HexColor('#0f172a'), spaceAfter=20, alignment=1, fontName='Helvetica-Bold')
        header_style = ParagraphStyle('SHAPHeader', parent=styles['Heading2'], fontSize=16, textColor=HexColor('#3b82f6'), spaceAfter=12, fontName='Helvetica-Bold')
        body_style = ParagraphStyle('SHAPBody', parent=styles['Normal'], fontSize=10, textColor=HexColor('#334155'), spaceAfter=8, fontName='Helvetica')
        footer_style = ParagraphStyle('SHAPFooter', parent=styles['Normal'], fontSize=8, textColor=HexColor('#94a3b8'), alignment=1, spaceAfter=20, fontName='Helvetica')
        
        # Header
        story.append(Paragraph("🛡️ FinGuard AI - SHAP Analysis Report", title_style))
        story.append(Paragraph(f"Transaction ID: {explanation['transaction_id']}", body_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Metrics
        story.append(Paragraph("SHAP Analysis Results", header_style))
        metrics_data = [
            ["Metric", "Value"],
            ["Risk Score", f"{explanation['risk_score']:.0f}/100"],
            ["Fraud Probability", f"{explanation['fraud_probability']:.1%}"],
            ["Risk Level", explanation['risk_level']],
            ["SHAP Confidence", f"{explanation['confidence']:.1%}"],
            ["Explanation Type", explanation['explanation_type'].upper()]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Top Features
        story.append(Paragraph("Top SHAP Features", header_style))
        feature_importance = explanation.get('feature_importance', {})
        for i, (feature, data) in enumerate(list(feature_importance.items())[:5]):
            story.append(Paragraph(f"{i+1}. {feature}: {data['impact_pct']:.1f}% impact (SHAP: {data['shap_value']:+.3f})", body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Explanation
        story.append(Paragraph("Natural Language Explanation", header_style))
        story.append(Paragraph(explanation['natural_language'][:500], body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Paragraph("—" * 60, body_style))
        story.append(Paragraph("This project is built by <b>Hassan Subhani</b>", footer_style))
        story.append(Paragraph("FinGuard AI - Cost-Aware Fraud Detection Platform with Real SHAP Integration", footer_style))
        story.append(Paragraph("© 2026 All Rights Reserved", footer_style))
        
        doc.build(story)
        return report_path

    def generate_complete_shap_report(self, explanation: Dict, waterfall_fig, dashboard_fig, 
                                        radar_fig=None, force_3d_fig=None) -> bytes:
        """Generate complete PDF report with all SHAP visualizations"""
        
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.units import inch, cm
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io
        import plotly.io as pio
        
        buffer = io.BytesIO()
        
        # Create PDF with landscape orientation for better chart display
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=landscape(letter),
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'SHAPTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#0f172a'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        header_style = ParagraphStyle(
            'SHAPHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#3b82f6'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        subheader_style = ParagraphStyle(
            'SHAPSubheader',
            parent=styles['Heading3'],
            fontSize=13,
            textColor=HexColor('#475569'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'SHAPBody',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#334155'),
            spaceAfter=8,
            fontName='Helvetica'
        )
        
        footer_style = ParagraphStyle(
            'SHAPFooter',
            parent=styles['Normal'],
            fontSize=8,
            textColor=HexColor('#94a3b8'),
            alignment=1,
            spaceAfter=20,
            fontName='Helvetica'
        )
        
        # Header
        story.append(Paragraph("🛡️ FinGuard AI - Complete SHAP Analysis Report", title_style))
        story.append(Paragraph(f"Transaction ID: {explanation.get('transaction_id', 'Unknown')}", body_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Metrics Table
        story.append(Paragraph("Key Performance Indicators", header_style))
        
        risk_score = explanation.get('risk_score', 0)
        fraud_prob = explanation.get('fraud_probability', 0)
        risk_level = explanation.get('risk_level', 'Unknown')
        confidence = explanation.get('confidence', 0.8)
        
        metrics_data = [
            ["Metric", "Value", "Status"],
            ["Risk Score", f"{risk_score:.0f}/100", "🔴 HIGH" if risk_score > 84 else "🟡 MEDIUM" if risk_score > 44 else "🟢 LOW"],
            ["Fraud Probability", f"{fraud_prob:.1%}", ""],
            ["Risk Level", risk_level, ""],
            ["SHAP Confidence", f"{confidence:.1%}", "✅ High" if confidence > 0.7 else "⚠️ Medium"],
            ["Explanation Type", explanation.get('explanation_type', 'simulated').upper(), ""],
            ["Analysis Date", datetime.now().strftime('%Y-%m-%d'), ""]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Function to add plotly figure to PDF
        def add_plotly_figure(fig, width=7*inch, height=4*inch):
            if fig:
                try:
                    img_bytes = fig.to_image(format="png", width=int(width*96), height=int(height*96), scale=2)
                    img_buffer = io.BytesIO(img_bytes)
                    img = Image(img_buffer, width=width, height=height)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                    return True
                except Exception as e:
                    story.append(Paragraph(f"⚠️ Chart unavailable: {str(e)[:100]}", body_style))
                    return False
            return False
        
        # Waterfall Plot
        story.append(Paragraph("SHAP Waterfall Plot", header_style))
        story.append(Paragraph("Shows how each feature pushes the prediction from base value to final fraud probability", body_style))
        add_plotly_figure(waterfall_fig, 7*inch, 4*inch)
        
        # Dashboard
        story.append(PageBreak())
        story.append(Paragraph("SHAP Comprehensive Dashboard", header_style))
        add_plotly_figure(dashboard_fig, 7*inch, 5*inch)
        
        # 3D Force Plot (if available)
        if force_3d_fig:
            story.append(PageBreak())
            story.append(Paragraph("3D SHAP Force Plot", header_style))
            add_plotly_figure(force_3d_fig, 7*inch, 4.5*inch)
        
        # Radar Chart (if available)
        if radar_fig:
            story.append(Paragraph("Feature Importance Radar", header_style))
            add_plotly_figure(radar_fig, 6*inch, 5*inch)
        
        # Feature Importance Table
        story.append(PageBreak())
        story.append(Paragraph("Detailed Feature Importance Analysis", header_style))
        
        feature_importance = explanation.get('feature_importance', {})
        importance_data = [["#", "Feature", "SHAP Value", "Impact %", "Direction"]]
        for i, (feature, data) in enumerate(list(feature_importance.items())[:15]):
            if isinstance(data, dict):
                shap_val = data.get('shap_value', 0)
                impact_pct = data.get('impact_pct', 0)
                direction = "🔴 Increases Risk" if data.get('direction') == 'positive' else "🟢 Decreases Risk"
            else:
                shap_val = 0
                impact_pct = 0
                direction = "Unknown"
            importance_data.append([str(i+1), feature[:35], f"{shap_val:+.3f}", f"{impact_pct:.1f}%", direction])
        
        importance_table = Table(importance_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 1*inch, 1.5*inch])
        importance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#8b5cf6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#faf5ff')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(importance_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Natural Language Explanation
        story.append(Paragraph("Natural Language Explanation", header_style))
        natural_lang = explanation.get('natural_language', 'No explanation available')
        # Split long text into paragraphs
        for line in natural_lang.split('\n')[:20]:
            if line.strip():
                story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Actionable Recommendations", header_style))
        recommendations = explanation.get('recommendations', [])
        for rec in recommendations[:8]:
            story.append(Paragraph(f"• {rec}", body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("—" * 80, body_style))
        story.append(Paragraph("This project is built by <b>Hassan Subhani</b>", footer_style))
        story.append(Paragraph("FinGuard AI - Cost-Aware Fraud Detection Platform with Real SHAP Integration", footer_style))
        story.append(Paragraph("© 2026 All Rights Reserved | Enterprise Grade Explainable AI", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def generate_decision_engine_report(self, predictions: pd.DataFrame, raw_data: pd.DataFrame = None, 
                                         decisions: np.ndarray = None, thresholds: Dict = None) -> str:
        """
        Generate specialized Decision Intelligence Engine Report
        Includes: Cost analysis, threshold configuration, decision breakdown, financial impact
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"Decision_Engine_Report_{timestamp}.pdf")
        
        # Calculate decision metrics
        if decisions is not None:
            decision_series = pd.Series(decisions)
            decision_counts = decision_series.value_counts()
            approve_count = decision_counts.get('Approve', 0)
            review_count = decision_counts.get('Review', 0)
            block_count = decision_counts.get('Block', 0)
        elif 'decision' in predictions.columns:
            decision_counts = predictions['decision'].value_counts()
            approve_count = decision_counts.get('Approve', 0)
            review_count = decision_counts.get('Review', 0)
            block_count = decision_counts.get('Block', 0)
        else:
            # Calculate based on risk levels
            approve_count = (predictions['risk_level'] == 'Low Risk').sum()
            review_count = (predictions['risk_level'] == 'Medium Risk').sum()
            block_count = (predictions['risk_level'] == 'High Risk').sum()
        
        # Financial calculations
        total_transactions = len(predictions)
        false_negative_cost = 1000
        false_positive_cost = 50
        review_cost = 10
        
        estimated_savings = block_count * false_negative_cost + review_count * review_cost
        potential_loss = block_count * false_negative_cost
        operational_cost = review_count * review_cost
        net_savings = estimated_savings - operational_cost
        
        # Risk distribution from predictions
        high_risk = (predictions['risk_level'] == 'High Risk').sum()
        medium_risk = (predictions['risk_level'] == 'Medium Risk').sum()
        low_risk = (predictions['risk_level'] == 'Low Risk').sum()
        
        avg_fraud_prob = predictions['fraud_probability'].mean()
        
        # Get thresholds
        if thresholds:
            approve_thresh = thresholds.get('approve', 0.44)
            review_start = thresholds.get('review_start', 0.45)
            review_end = thresholds.get('review_end', 0.84)
        else:
            approve_thresh = 0.44
            review_start = 0.45
            review_end = 0.84
        
        # Create visualizations
        temp_files = []
        
        # 1. Decision Distribution Chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        decision_data = ['Approve', 'Review', 'Block']
        decision_values = [approve_count, review_count, block_count]
        colors_dec = ['#27ae60', '#f39c12', '#e74c3c']
        bars = ax1.bar(decision_data, decision_values, color=colors_dec, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, decision_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(decision_values)*0.01,
                    f'{val:,}\n({val/total_transactions*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Decision Type', fontsize=12, fontweight='bold')
        ax1.set_title('Decision Intelligence Breakdown', fontsize=14, fontweight='bold')
        ax1.set_facecolor('#f8f9fa')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        dec_chart = os.path.join(self.reports_dir, 'temp_decision_dist.png')
        plt.savefig(dec_chart, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        temp_files.append(dec_chart)
        
        # 2. Financial Impact Chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        categories = ['Potential Loss\nPrevented', 'Operational\nCost', 'Net Savings']
        values = [potential_loss, operational_cost, net_savings]
        colors_fin = ['#e74c3c', '#f39c12', '#27ae60']
        bars = ax2.bar(categories, values, color=colors_fin, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Amount (USD)', fontsize=12, fontweight='bold')
        ax2.set_title('Financial Impact Analysis', fontsize=14, fontweight='bold')
        ax2.set_facecolor('#f8f9fa')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fin_chart = os.path.join(self.reports_dir, 'temp_financial.png')
        plt.savefig(fin_chart, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        temp_files.append(fin_chart)
        
        # 3. Threshold Visualization
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # Create threshold zones
        x = np.linspace(0, 1, 100)
        zones = []
        
        # Color zones
        ax3.axvspan(0, approve_thresh, alpha=0.3, color='#27ae60', label=f'Approve Zone (0 - {approve_thresh:.2f})')
        ax3.axvspan(approve_thresh, review_end, alpha=0.3, color='#f39c12', label=f'Review Zone ({approve_thresh:.2f} - {review_end:.2f})')
        ax3.axvspan(review_end, 1, alpha=0.3, color='#e74c3c', label=f'Block Zone ({review_end:.2f} - 1.00)')
        
        # Add vertical lines at thresholds
        ax3.axvline(x=approve_thresh, color='#27ae60', linestyle='--', linewidth=2, label=f'Approve Threshold: {approve_thresh:.2f}')
        ax3.axvline(x=review_end, color='#e74c3c', linestyle='--', linewidth=2, label=f'Block Threshold: {review_end:.2f}')
        
        # Add a histogram of fraud probabilities
        ax3.hist(predictions['fraud_probability'], bins=40, color='#3498db', alpha=0.5, edgecolor='white', density=True)
        
        ax3.set_xlabel('Fraud Probability', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax3.set_title('Decision Threshold Configuration', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.set_facecolor('#f8f9fa')
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        thresh_chart = os.path.join(self.reports_dir, 'temp_thresholds.png')
        plt.savefig(thresh_chart, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        temp_files.append(thresh_chart)
        
        # 4. Cost Benefit Analysis Chart
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        # Calculate cost metrics
        cost_metrics = {
            'False\nNegative': false_negative_cost,
            'False\nPositive': false_positive_cost,
            'Manual\nReview': review_cost
        }
        
        bars = ax4.bar(cost_metrics.keys(), cost_metrics.values(), 
                       color=['#e74c3c', '#f39c12', '#3498db'], 
                       edgecolor='white', linewidth=2)
        
        for bar, (label, val) in zip(bars, cost_metrics.items()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'${val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
        ax4.set_title('Cost Parameters for Decision Engine', fontsize=14, fontweight='bold')
        ax4.set_facecolor('#f8f9fa')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        cost_chart = os.path.join(self.reports_dir, 'temp_cost_params.png')
        plt.savefig(cost_chart, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        temp_files.append(cost_chart)
        
        # Create PDF
        doc = SimpleDocTemplate(report_path, pagesize=landscape(letter),
                                rightMargin=0.5*inch, leftMargin=0.5*inch,
                                topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle('DecisionTitle', parent=styles['Heading1'], fontSize=26, 
                                     textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER, 
                                     spaceAfter=10, fontName='Helvetica-Bold')
        
        subtitle_style = ParagraphStyle('DecisionSubtitle', parent=styles['Normal'], fontSize=12,
                                        textColor=colors.HexColor('#666666'), alignment=TA_CENTER,
                                        spaceAfter=20)
        
        section_style = ParagraphStyle('DecisionSection', parent=styles['Heading2'], fontSize=16,
                                       textColor=colors.HexColor('#2c3e50'), alignment=TA_LEFT,
                                       spaceAfter=12, spaceBefore=12, fontName='Helvetica-Bold')
        
        body_style = ParagraphStyle('DecisionBody', parent=styles['Normal'], fontSize=9,
                                    textColor=colors.HexColor('#333333'), spaceAfter=6, alignment=TA_LEFT)
        
        highlight_style = ParagraphStyle('Highlight', parent=styles['Normal'], fontSize=10,
                                         textColor=colors.HexColor('#27ae60'), alignment=TA_CENTER,
                                         fontName='Helvetica-Bold')
        
        center_body_style = ParagraphStyle('CenterBody', parent=styles['Normal'], fontSize=9,
                                           textColor=colors.HexColor('#555555'), spaceAfter=8, alignment=TA_CENTER)
        
        # === COVER PAGE ===
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("⚖️ FINGUARD AI", title_style))
        story.append(Paragraph("Decision Intelligence Engine Report", subtitle_style))
        story.append(Paragraph("Cost-Aware Fraud Decision Analysis", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", center_body_style))
        story.append(Paragraph(f"Report Time: {datetime.now().strftime('%I:%M %p')}", center_body_style))
        story.append(Paragraph(f"Analysis Period: {datetime.now().strftime('%Y-%m-%d')}", center_body_style))
        story.append(Spacer(1, 0.8*inch))
        story.append(PageBreak())
        
        # === EXECUTIVE SUMMARY ===
        story.append(Paragraph("Executive Summary", section_style))
        
        summary_text = f"""
        This report presents the decision intelligence analysis for <b>{total_transactions:,}</b> transactions processed 
        by the FinGuard AI platform. The cost-aware decision engine made the following determinations:
        <b>{approve_count:,}</b> transactions <font color='#27ae60'><b>APPROVED</b></font>, 
        <b>{review_count:,}</b> transactions <font color='#f39c12'><b>REQUIRE REVIEW</b></font>, and 
        <b>{block_count:,}</b> transactions <font color='#e74c3c'><b>BLOCKED</b></font>.
        
        <b>Financial Impact:</b> The decision engine has prevented approximately <b>${estimated_savings:,.0f}</b> in potential 
        fraud losses, with net savings of <b>${net_savings:,.0f}</b> after accounting for operational costs.
        """
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # === KEY METRICS TABLE ===
        story.append(Paragraph("Decision Intelligence Metrics", section_style))
        
        metrics_data = [
            ['<b>Metric</b>', '<b>Value</b>', '<b>Percentage</b>'],
            ['Total Transactions Analyzed', f'{total_transactions:,}', '100%'],
            ['✅ Approved Transactions', f'{approve_count:,}', f'{approve_count/total_transactions*100:.1f}%'],
            ['⚠️ Review Required', f'{review_count:,}', f'{review_count/total_transactions*100:.1f}%'],
            ['❌ Blocked Transactions', f'{block_count:,}', f'{block_count/total_transactions*100:.1f}%'],
            ['Average Fraud Probability', f'{avg_fraud_prob:.3f}', 'Risk Baseline'],
            ['High Risk Detection Rate', f'{block_count/total_transactions*100:.1f}%', 'Blocked Ratio']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.2*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === DECISION BREAKDOWN ===
        story.append(Paragraph("Decision Distribution Analysis", section_style))
        story.append(Paragraph("This chart shows how transactions were categorized by the cost-aware decision engine.", body_style))
        story.append(Image(dec_chart, width=8*inch, height=5*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # === FINANCIAL IMPACT ===
        story.append(Paragraph("Financial Impact Analysis", section_style))
        story.append(Paragraph("Breakdown of financial impact including loss prevention and operational costs.", body_style))
        story.append(Image(fin_chart, width=8*inch, height=5*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # === COST PARAMETERS ===
        story.append(Paragraph("Cost Parameters Configuration", section_style))
        story.append(Paragraph("The decision engine uses these cost parameters to optimize fraud decisions.", body_style))
        story.append(Image(cost_chart, width=8*inch, height=5*inch))
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === THRESHOLD CONFIGURATION ===
        story.append(Paragraph("Decision Threshold Configuration", section_style))
        story.append(Paragraph(f"Current thresholds: Approve ≤ {approve_thresh:.2f}, "
                              f"Review {review_start:.2f} - {review_end:.2f}, Block > {review_end:.2f}", body_style))
        story.append(Image(thresh_chart, width=8*inch, height=5*inch))
        story.append(Spacer(1, 0.3*inch))
        
        # === DETAILED COST TABLE ===
        story.append(Paragraph("Detailed Cost Analysis", section_style))
        
        cost_data = [
            ['<b>Cost Component</b>', '<b>Amount (USD)</b>', '<b>Description</b>'],
            ['False Negative Cost (per missed fraud)', f'${false_negative_cost:,}', 'Cost when fraudulent transaction is approved'],
            ['False Positive Cost (per false alarm)', f'${false_positive_cost:,}', 'Cost when legitimate transaction is blocked'],
            ['Manual Review Cost', f'${review_cost:,}', 'Operational cost per manual review'],
            ['', '', ''],
            ['<b>Total Estimated Savings</b>', f'<b>${estimated_savings:,.0f}</b>', 'From blocked and reviewed transactions'],
            ['<b>Operational Cost</b>', f'<b>${operational_cost:,.0f}</b>', f'{review_count} reviews × ${review_cost}'],
            ['<b>Net Savings</b>', f'<b>${net_savings:,.0f}</b>', 'Total savings after operational costs']
        ]
        
        cost_table = Table(cost_data, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
        cost_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#ecf0f1')),
            ('BACKGROUND', (0, 5), (-1, 7), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ]))
        story.append(cost_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === TOP HIGH-RISK DECISIONS ===
        story.append(Paragraph("High-Risk Decision Watchlist", section_style))
        story.append(Paragraph("Transactions that were flagged for review or blocked due to high fraud probability.", body_style))
        
        # Get top blocked/review transactions
        if 'decision' in predictions.columns:
            high_risk_decisions = predictions[predictions['decision'].isin(['Block', 'Review'])].nlargest(15, 'fraud_probability')
        else:
            high_risk_decisions = predictions[predictions['risk_level'].isin(['High Risk', 'Medium Risk'])].nlargest(15, 'fraud_probability')
        
        risk_data = [['Rank', 'Transaction ID', 'Fraud Prob.', 'Decision', 'Risk Level']]
        for i, (_, row) in enumerate(high_risk_decisions.head(15).iterrows(), 1):
            decision_val = row.get('decision', row.get('risk_level', 'Review'))
            risk_val = row.get('risk_level', 'Medium Risk' if row['fraud_probability'] > 0.44 else 'Low Risk')
            risk_data.append([
                str(i), 
                row['transaction_id'][:20], 
                f"{row['fraud_probability']:.3f}", 
                decision_val,
                risk_val
            ])
        
        risk_table = Table(risk_data, colWidths=[0.5*inch, 2*inch, 1*inch, 1.2*inch, 1.2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        # === RECOMMENDATIONS ===
        story.append(Paragraph("Strategic Recommendations", section_style))
        
        recommendations = []
        
        if block_count > total_transactions * 0.05:
            recommendations.append("🔴 <b>URGENT:</b> High blocking rate detected. Review false positive cases and adjust thresholds.")
        elif block_count < total_transactions * 0.01:
            recommendations.append("🟢 Blocking rate is optimal. Continue current strategy.")
        
        if review_count > total_transactions * 0.15:
            recommendations.append("🟡 Review queue is high. Consider increasing review team capacity or adjusting thresholds.")
        
        recommendations.extend([
            "⚖️ <b>Threshold Optimization:</b> Run A/B tests monthly to optimize decision thresholds based on cost analysis.",
            "💰 <b>Cost-Benefit Review:</b> Evaluate false positive vs false negative costs quarterly.",
            "📊 <b>Performance Monitoring:</b> Track decision distribution trends over time.",
            "🎯 <b>Targeted Review:</b> Focus manual review resources on transactions in the 0.65-0.85 probability range.",
            "🔄 <b>Model Retraining:</b> Schedule retraining when decision distribution shifts significantly."
        ])
        
        for rec in recommendations[:8]:
            story.append(Paragraph(rec, body_style))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.3*inch))
        
        # === ROI SUMMARY ===
        story.append(Paragraph("Return on Investment (ROI) Summary", section_style))
        
        roi_percentage = (net_savings / max(operational_cost, 1)) * 100
        
        roi_data = [
            ['<b>Metric</b>', '<b>Value</b>'],
            ['Total Investment (Operational Cost)', f'${operational_cost:,.0f}'],
            ['Total Return (Loss Prevention)', f'${estimated_savings:,.0f}'],
            ['Net Profit', f'${net_savings:,.0f}'],
            ['ROI Percentage', f'{roi_percentage:.1f}%'],
            ['Cost per Transaction', f'${estimated_savings/total_transactions:.2f}'],
            ['Savings per Blocked Transaction', f'${estimated_savings/max(block_count, 1):,.0f}']
        ]
        
        roi_table = Table(roi_data, colWidths=[2.5*inch, 2.5*inch])
        roi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
        ]))
        story.append(roi_table)
        story.append(Spacer(1, 0.3*inch))
        
        # === FOOTER / CREDITS ===
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<hr noshade size='1' color='#cccccc' width='80%'/>", center_body_style))
        story.append(Spacer(1, 0.1*inch))
        
        footer_text = f"""
        <font size='9' color='#1a1a2e'><b>FinGuard AI Decision Intelligence Engine</b></font><br/>
        <font size='8' color='#888888'>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</font><br/>
        <font size='8' color='#888888'>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</font><br/>
        <font size='8' color='#888888'>This report analyzes cost-optimized fraud decisions using financial impact models.</font><br/>
        <font size='8' color='#888888'>All financial figures are estimates based on detection performance and cost parameters.</font><br/>
        <font size='9' color='#1a1a2e'><b>This project is built by Hassan Subhani</b></font><br/>
        <font size='8' color='#666666'>Cost-Aware Fraud Detection & Decision Intelligence Platform v3.0</font>
        """
        story.append(Paragraph(footer_text, center_body_style))
        
        # Build PDF
        doc.build(story)
        
        # Cleanup temporary files
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        print(f"✅ Decision Engine Report generated: {report_path}")
        return report_path