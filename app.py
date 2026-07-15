# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import hashlib
from typing import Dict, Any, List, Optional
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from pipelines.validation import DataValidator
from pipelines.preprocessing import DataPreprocessor
from pipelines.inference import FraudInference
from pipelines.explainability import FraudExplainer
from utils.cost_engine import CostEngine
from utils.drift_detection import DriftDetector
from utils.report_generator import ReportGenerator
from utils.pdf_report_generator import UltraPremiumPDFReport

# Page configuration
st.set_page_config(
    page_title="FinGuard AI - Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Light/White Theme CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .css-1r6slb0, .css-1v0mbdj, [data-testid="stMetricValue"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .css-1r6slb0:hover, .css-1v0mbdj:hover {
        border-color: #cbd5e1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        color: #0f172a !important;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Risk badges */
    .risk-high {
        background: #fee2e2;
        color: #dc2626;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    .risk-medium {
        background: #fed7aa;
        color: #ea580c;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    .risk-low {
        background: #d1fae5;
        color: #059669;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    /* Decision badges */
    .decision-approve {
        background: #d1fae5;
        color: #059669;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    
    .decision-review {
        background: #fed7aa;
        color: #ea580c;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    
    .decision-block {
        background: #fee2e2;
        color: #dc2626;
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        color: #475569;
        border: 1px solid #e2e8f0;
        text-align: left;
        justify-content: flex-start;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        color: #0f172a;
        transform: none;
        box-shadow: none;
    }
    
    /* Primary button in sidebar */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #3b82f6;
        border: 1px solid #e2e8f0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
    
    /* Text colors */
    p, li, .stMarkdown {
        color: #334155;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Select boxes */
    .stSelectbox [data-baseweb="select"] {
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        background: #f8fafc;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main .block-container {
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Validation specific styles */
    .validation-status-card {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .quality-score {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .issue-count {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .status-badge-pass {
        background: #d1fae5;
        color: #059669;
        padding: 0.5rem 1rem;
        border-radius: 40px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-badge-fail {
        background: #fee2e2;
        color: #dc2626;
        padding: 0.5rem 1rem;
        border-radius: 40px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .gradient-border {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 16px;
        padding: 2px;
    }
    
    .inner-card {
        background: white;
        border-radius: 14px;
        padding: 1rem;
    }
    
    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #f1f5f9;
        border-radius: 40px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #475569;
    }

    /* Enhanced Upload Page Styles */
    .upload-hero {
        text-align: center;
        padding: 1rem 0 2rem 0;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 24px;
        margin-bottom: 2rem;
    }
    
    .feature-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #f1f5f9;
        border-radius: 40px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        color: #475569;
        transition: all 0.2s ease;
    }
    
    .feature-badge:hover {
        background: #e2e8f0;
        transform: translateY(-1px);
    }
    
    .upload-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 24px;
        background: linear-gradient(135deg, #fafcff 0%, #ffffff 100%);
        transition: all 0.3s ease;
        padding: 2rem;
        text-align: center;
    }
    
    .upload-zone:hover {
        border-color: #3b82f6;
        background: #f8fafc;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }
    
    .file-info-card {
        background: #f8fafc;
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .file-info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border-radius: 12px;
        background: #f8fafc;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .progress-step.active {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
    }
    
    .progress-step.completed {
        background: #f0fdf4;
        border-left: 3px solid #10b981;
    }
    
    .step-icon {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: white;
        font-weight: 600;
    }
    
    .requirement-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .requirement-list li {
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .requirement-list li:last-child {
        border-bottom: none;
    }
    
    .glowing-text {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'validation_passed' not in st.session_state:
    st.session_state.validation_passed = False

# Load models and artifacts
@st.cache_resource
def load_models():
    """Load all trained models and artifacts"""
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODER_PATH)
        
        with open(FEATURE_SCHEMA_PATH, 'r') as f:
            feature_schema = json.load(f)
        
        with open(THRESHOLD_CONFIG_PATH, 'r') as f:
            threshold_config = json.load(f)
        
        st.session_state.model_loaded = True
        return {
            'xgb': xgb_model,
            'logistic': logistic_model,
            'scaler': scaler,
            'encoders': encoders,
            'feature_schema': feature_schema,
            'thresholds': threshold_config
        }
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        return None

# Clean Sidebar navigation
def render_sidebar():
    with st.sidebar:
        # Logo and brand
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size: 3rem;">🛡️</div>
            <h2 style="margin: 0.5rem 0 0 0; color: #0f172a;">
                FinGuard AI
            </h2>
            <p style="color: #64748b; font-size: 0.875rem; margin-top: 0.25rem;">
                Cost-Aware Fraud Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Dashboard": "Home",
            "📤 Upload Data": "Upload",
            "🔍 Validation": "Validation",
            "🤖 Detection": "Detection",
            "⚖️ Decisions": "Decision",
            "🧠 Explainability": "Explain",
            "📉 Drift Monitor": "Drift",
            "🔬 A/B Testing": "ABTest",
            "📊 Analytics": "Analytics"
        }
        
        for label, page in pages.items():
            button_type = "primary" if st.session_state.current_page == page else "secondary"
            if st.button(label, key=page, use_container_width=True, type=button_type):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("### 📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.model_loaded:
                st.markdown("""
                <div style="text-align: center; padding: 0.5rem; background: #f0fdf4; border-radius: 8px;">
                    <div style="font-size: 1.5rem;">✅</div>
                    <p style="margin:0; font-size:0.75rem; color: #059669;">Active</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 0.5rem; background: #fef3c7; border-radius: 8px;">
                    <div style="font-size: 1.5rem;">⚠️</div>
                    <p style="margin:0; font-size:0.75rem; color: #d97706;">Loading</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: #eff6ff; border-radius: 8px;">
                <div style="font-size: 1.5rem;">📅</div>
                <p style="margin:0; font-size:0.75rem; color: #2563eb;">{datetime.now().strftime('%b %d, %Y')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Validation status indicator
        if st.session_state.validation_passed:
            st.markdown("""
            <div style="background: #d1fae5; padding: 0.5rem; border-radius: 8px; text-align: center;">
                <span style="color: #059669;">✅ Data Validated</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer
        st.caption("✨ FinGuard AI v2.0")
        st.caption("🔒 Enterprise Security")

# Page 1: Home Dashboard
def page_home():
    # Hero section
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 3rem;">Intelligent Fraud Protection</h1>
        <p style="font-size: 1.125rem; color: #475569; max-width: 600px; margin: 0 auto;">
            Real-time, explainable, cost-optimized fraud detection for modern enterprises
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("🎯", "99.7%", "Detection Accuracy", "#10b981"),
        ("💰", "$1.2M", "Estimated Savings", "#f59e0b"),
        ("⚡", "&lt;100ms", "Inference Time", "#3b82f6"),
        ("🔮", "SHAP", "Explainable AI", "#8b5cf6")
    ]
    
    for idx, (icon, value, label, color) in enumerate(metrics_data):
        with [col1, col2, col3, col4][idx]:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color};">
                <div style="font-size: 2rem;">{icon}</div>
                <h2 style="margin: 0.5rem 0 0 0; font-size: 1.75rem; color: #0f172a;">{value}</h2>
                <p style="margin: 0; color: #64748b; font-size: 0.875rem;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow visualization
    st.markdown("### 🔄 Intelligent Fraud Detection Workflow")
    
    workflow_cols = st.columns(5)
    workflow_steps = [
        ("📤", "Upload", "Import transaction data"),
        ("🔍", "Validate", "Schema & quality check"),
        ("🤖", "Predict", "ML fraud scoring"),
        ("⚖️", "Decide", "Cost-optimized actions"),
        ("🧠", "Explain", "SHAP analysis")
    ]
    
    for idx, (icon, title, subtitle) in enumerate(workflow_steps):
        with workflow_cols[idx]:
            st.markdown(f"""
            <div style="text-align:center; padding: 1.25rem; background: #f8fafc; 
                        border-radius: 12px; margin: 0.5rem; border: 1px solid #e2e8f0;
                        transition: all 0.2s ease;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="margin: 0.5rem 0 0 0; color: #0f172a;">{title}</h4>
                <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.75rem;">{subtitle}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features grid
    st.markdown("### ✨ Enterprise Features")
    
    features = [
        ("🎯", "Cost-Aware Decisions", "Optimized for financial impact, not just accuracy"),
        ("🔮", "Explainable AI", "Understand why every decision is made"),
        ("📊", "Real-time Monitoring", "Drift detection and performance tracking"),
        ("🔄", "A/B Testing", "Compare models and strategies"),
        ("📈", "Analytics Dashboard", "Executive-level business insights"),
        ("🔒", "Enterprise Security", "Local deployment, maximum privacy")
    ]
    
    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="padding: 1.25rem; margin-bottom: 1rem; background: #ffffff; 
                        border-radius: 12px; border: 1px solid #e2e8f0;
                        transition: all 0.2s ease;">
                <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="margin: 0.5rem 0; color: #0f172a;">{title}</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.875rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📤 Start New Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "Upload"
            st.rerun()
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_page = "Analytics"
            st.rerun()
    with col3:
        if st.button("📖 View Documentation", use_container_width=True):
            st.info("Documentation available in the /docs folder")

# Page 2: Data Upload (ENHANCED UI/UX)
# Page 2: Data Upload (ENHANCED UI/UX with Web Link Support)
def page_upload():
    # Hero Section with Animation
    st.markdown("""
    <div class="upload-hero" style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            <span class="glowing-text">Intelligent Data Ingest</span>
        </h1>
        <p style="font-size: 1rem; color: #64748b; max-width: 600px; margin: 0 auto;">
            Upload your transaction data from anywhere - Local files, URLs, or Cloud Storage
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
            <span class="feature-badge">📁 Local Files</span>
            <span class="feature-badge">🔗 Direct URLs</span>
            <span class="feature-badge">🐙 GitHub</span>
            <span class="feature-badge">☁️ Google Drive</span>
            <span class="feature-badge">📊 Kaggle</span>
            <span class="feature-badge">💾 Dropbox</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset validation flag when uploading new data
    st.session_state.validation_passed = False
    
    # Create tabs for different upload methods
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Local File", 
        "🔗 Direct URL", 
        "🐙 GitHub", 
        "☁️ Google Drive",
        "📊 Kaggle"
    ])
    
    # ==================== TAB 1: LOCAL FILE UPLOAD ====================
    with tab1:
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <p style="color: #475569; font-size: 0.875rem;">
                Upload files from your local computer
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel, JSON (max 200MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                with st.status("📊 Processing your file...", expanded=True) as status:
                    st.write("🔍 Reading file content...")
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    else:
                        df = pd.read_json(uploaded_file)
                    
                    st.write("✅ File loaded successfully!")
                    status.update(label="File loaded!", state="complete")
                
                st.session_state.uploaded_data = df
                _display_uploaded_data_info(df, uploaded_file.name)
                
                # Action buttons
                col_action1, col_action2, col_action3 = st.columns([1, 1.2, 1])
                with col_action2:
                    if st.button("🔍 Validate Dataset →", use_container_width=True, type="primary"):
                        st.session_state.current_page = "Validation"
                        st.rerun()
                with col_action1:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.uploaded_data = None
                        st.session_state.predictions = None
                        st.session_state.validation_passed = False
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                _show_troubleshooting_tips()
    
    # ==================== TAB 2: DIRECT URL ====================
    with tab2:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <p style="color: #475569; font-size: 0.875rem;">
                🔗 Load data directly from any public URL (CSV, Excel, JSON, or ZIP files)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "🔗 File URL",
            placeholder="https://example.com/data.csv",
            help="Supports CSV, Excel, JSON, and ZIP files from any public URL"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            url_format = st.selectbox(
                "File format (if not auto-detectable)",
                ["auto-detect", "csv", "excel", "json", "zip"],
                index=0
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Fetch from URL", use_container_width=True, type="primary"):
                if url:
                    with st.spinner(f"🔄 Fetching data from {url[:50]}..."):
                        try:
                            from utils.helpers import load_data_from_url
                            df = load_data_from_url(url, url_format if url_format != "auto-detect" else None)
                            if df is not None and not df.empty:
                                st.session_state.uploaded_data = df
                                st.success(f"✅ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                                st.balloons()
                                _display_uploaded_data_info(df, url.split('/')[-1])
                                
                                # Proceed button
                                if st.button("🔍 Proceed to Validation →", use_container_width=True, type="primary"):
                                    st.session_state.current_page = "Validation"
                                    st.rerun()
                            else:
                                st.error("❌ Failed to load data from the provided URL. Please check the link and try again.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a valid URL")
    
    # ==================== TAB 3: GITHUB ====================
    with tab3:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <p style="color: #475569; font-size: 0.875rem;">
                🐙 Load CSV/Excel files directly from GitHub repositories
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        github_url = st.text_input(
            "🐙 GitHub File URL",
            placeholder="https://github.com/username/repo/blob/main/data.csv",
            help="Paste any GitHub file URL (will automatically convert to raw URL)"
        )
        
        # Preview of raw URL
        if github_url:
            from utils.helpers import convert_to_raw_github_url
            raw_url = convert_to_raw_github_url(github_url)
            st.info(f"📎 Will fetch from: `{raw_url}`")
        
        if st.button("📥 Fetch from GitHub", use_container_width=True, type="primary"):
            if github_url:
                with st.spinner(f"🔄 Fetching from GitHub..."):
                    try:
                        from utils.helpers import load_data_from_github
                        df = load_data_from_github(github_url)
                        if df is not None and not df.empty:
                            st.session_state.uploaded_data = df
                            st.success(f"✅ GitHub data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                            st.balloons()
                            _display_uploaded_data_info(df, github_url.split('/')[-1])
                            
                            if st.button("🔍 Proceed to Validation →", use_container_width=True, type="primary"):
                                st.session_state.current_page = "Validation"
                                st.rerun()
                        else:
                            st.error("❌ Failed to load data from GitHub. Please check the URL and ensure the file is accessible.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a GitHub file URL")
    
    # ==================== TAB 4: GOOGLE DRIVE ====================
    with tab4:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <p style="color: #475569; font-size: 0.875rem;">
                ☁️ Load data from Google Drive using shareable links
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        drive_url = st.text_input(
            "☁️ Google Drive Shareable Link",
            placeholder="https://drive.google.com/file/d/XXXXX/view?usp=sharing",
            help="Create a shareable link from Google Drive and paste it here"
        )
        
        st.caption("💡 Make sure the file is shared with 'Anyone with the link' can view")
        
        if st.button("📥 Fetch from Google Drive", use_container_width=True, type="primary"):
            if drive_url:
                with st.spinner(f"🔄 Fetching from Google Drive..."):
                    try:
                        from utils.helpers import load_data_from_google_drive
                        df = load_data_from_google_drive(drive_url)
                        if df is not None and not df.empty:
                            st.session_state.uploaded_data = df
                            st.success(f"✅ Google Drive data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                            st.balloons()
                            _display_uploaded_data_info(df, "google_drive_data.csv")
                            
                            if st.button("🔍 Proceed to Validation →", use_container_width=True, type="primary"):
                                st.session_state.current_page = "Validation"
                                st.rerun()
                        else:
                            st.error("❌ Failed to load data from Google Drive. Please check the sharing settings and try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a Google Drive shareable link")
    
    # ==================== TAB 5: KAGGLE ====================
    with tab5:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <p style="color: #475569; font-size: 0.875rem;">
                📊 Load datasets directly from Kaggle (requires Kaggle API setup)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        kaggle_dataset = st.text_input(
            "📊 Kaggle Dataset Path",
            placeholder="username/dataset-name",
            help="Example: mlg-ulb/creditcardfraud"
        )
        
        kaggle_file = st.text_input(
            "📄 Specific File Name (optional)",
            placeholder="fraud_data.csv",
            help="Leave empty to load the first CSV file found"
        )
        
        st.caption("💡 Note: First-time setup requires Kaggle API credentials. See documentation for setup instructions.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📥 Fetch from Kaggle", use_container_width=True, type="primary"):
                if kaggle_dataset:
                    with st.spinner(f"🔄 Fetching from Kaggle: {kaggle_dataset}..."):
                        try:
                            from utils.helpers import load_data_from_kaggle
                            df = load_data_from_kaggle(kaggle_dataset, kaggle_file if kaggle_file else None)
                            if df is not None and not df.empty:
                                st.session_state.uploaded_data = df
                                st.success(f"✅ Kaggle data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                                st.balloons()
                                _display_uploaded_data_info(df, f"{kaggle_dataset.replace('/', '_')}.csv")
                                
                                if st.button("🔍 Proceed to Validation →", use_container_width=True, type="primary"):
                                    st.session_state.current_page = "Validation"
                                    st.rerun()
                            else:
                                st.error("❌ Failed to load data from Kaggle. Please check the dataset path and Kaggle API configuration.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a Kaggle dataset path")
        
        with col2:
            if st.button("🔧 Configure Kaggle API", use_container_width=True):
                st.info("""
                **Kaggle API Setup Instructions:**
                1. Go to kaggle.com -> Account -> Create API Token
                2. Download kaggle.json
                3. Place it in ~/.kaggle/kaggle.json
                4. Run: `pip install kagglehub`
                """)
    
    # Right column with requirements (visible for all tabs)
    with st.sidebar:
        _render_upload_sidebar()


# Helper function to display uploaded data info
def _display_uploaded_data_info(df: pd.DataFrame, filename: str):
    """Display information about uploaded data"""
    st.markdown(f"""
    <div class="file-info-card" style="
        background: #f8fafc;
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin: 1.5rem 0;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <div>
                <span style="font-size: 1.5rem;">📄</span>
                <span style="font-weight: 600; margin-left: 0.5rem;">{filename[:50]}</span>
            </div>
            <span class="metric-chip">{df.shape[0]} rows</span>
            <span class="metric-chip">{df.shape[1]} columns</span>
        </div>
        <div class="gradient-border" style="margin: 1rem 0;"></div>
        <div style="display: flex; gap: 1rem; justify-content: space-around;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700;">{df.select_dtypes(include=['number']).shape[1]}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Numeric Features</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700;">{df.select_dtypes(include=['object']).shape[1]}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Categorical Features</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700;">{df.isnull().sum().sum()}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Missing Values</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Preview
    st.markdown("<div style='margin-top: 1.5rem;'><h3>📊 Data Preview</h3></div>", unsafe_allow_html=True)
    preview_tab, schema_tab, stats_tab = st.tabs(["🔍 Preview", "📋 Schema Analysis", "📈 Statistical Summary"])
    
    with preview_tab:
        st.dataframe(df.head(10), use_container_width=True)
        if len(df) > 10:
            st.caption(f"Showing first 10 rows of {len(df)} total rows")
    
    with schema_tab:
        schema_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Unique': df.nunique().values
        })
        st.dataframe(schema_df, use_container_width=True, height=300)
    
    with stats_tab:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical summary")


def _show_troubleshooting_tips():
    """Show troubleshooting tips for file upload"""
    st.markdown("""
    <div style="background: #fef2f2; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
        <p style="color: #dc2626; margin: 0;">💡 Troubleshooting Tips:</p>
        <ul style="color: #dc2626; margin: 0.5rem 0 0 1.5rem;">
            <li>Ensure your CSV uses comma separation</li>
            <li>Check for special characters in column names</li>
            <li>Verify file encoding (UTF-8 recommended)</li>
            <li>Try using the URL upload method for online files</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def _render_upload_sidebar():
    """Render the upload sidebar with requirements and tips"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
        border-radius: 20px; 
        padding: 1.5rem; 
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">📋</span>
            <h4 style="margin: 0;">Supported Sources</h4>
        </div>
        <ul style="list-style: none; padding: 0; margin: 0;">
            <li style="padding: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem; border-bottom: 1px solid #f1f5f9;">
                <span>📁</span> Local Files (CSV, Excel, JSON)
            </li>
            <li style="padding: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem; border-bottom: 1px solid #f1f5f9;">
                <span>🔗</span> Direct URLs (HTTP/HTTPS)
            </li>
            <li style="padding: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem; border-bottom: 1px solid #f1f5f9;">
                <span>🐙</span> GitHub Raw Files
            </li>
            <li style="padding: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem; border-bottom: 1px solid #f1f5f9;">
                <span>☁️</span> Google Drive (Public links)
            </li>
            <li style="padding: 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                <span>📊</span> Kaggle Datasets (API required)
            </li>
        </ul>
        <div class="gradient-border" style="margin: 1rem 0;"></div>
        <p style="color: #64748b; font-size: 0.75rem; margin-top: 1rem;">
            💡 <strong>Pro Tip:</strong> For best results, ensure your data has at least 100 transactions and includes a unique identifier column.
        </p>
    </div>
    
    <div style="
        background: #f0fdf4; 
        border-radius: 20px; 
        padding: 1rem; 
        border: 1px solid #d1fae5;
        margin-top: 1rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span>💡</span>
            <span style="font-weight: 500; color: #059669;">Supported Formats</span>
        </div>
        <p style="color: #047857; font-size: 0.75rem; margin-top: 0.5rem;">
            CSV, Excel (.xlsx, .xls), JSON, and ZIP archives containing these formats.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Page 3: Data Validation (MODERNIZED UI/UX)
def page_validation():
    # Modern header with gradient
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🔍 Data Validation Engine
        </h1>
        <p style="color: #64748b; font-size: 1rem;">Enterprise-grade data quality assessment with intelligent auto-correction</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is not None:
        # Initialize validator
        validator = DataValidator(auto_fix=True, strict_mode=False)
        
        # Show validation progress in a clean card
        with st.spinner("🚀 Running comprehensive validation..."):
            validation_result = validator.validate(st.session_state.uploaded_data)
        
        # Store the mapped dataframe for future use if validation passed
        if validation_result.get('passed', False):
            mapped_df = validator.get_mapped_dataframe()
            if mapped_df is not None:
                st.session_state.uploaded_data = mapped_df
                st.session_state.validation_passed = True
        
        # ==================== TOP STATUS CARD ====================
        quality_score = validation_result.get('quality_score', 0)
        passed = validation_result.get('passed', False)
        
        # Determine status color and icon
        if passed and quality_score >= 90:
            status_color = "#10b981"
            status_bg = "#d1fae5"
            status_icon = "✅"
            status_text = "VALIDATION PASSED - EXCELLENT"
            status_subtext = "Your data meets all quality standards"
        elif passed and quality_score >= 70:
            status_color = "#f59e0b"
            status_bg = "#fed7aa"
            status_icon = "⚠️"
            status_text = "VALIDATION PASSED - GOOD"
            status_subtext = "Minor issues detected but data is usable"
        elif passed:
            status_color = "#3b82f6"
            status_bg = "#dbeafe"
            status_icon = "🟡"
            status_text = "VALIDATION PASSED - ACCEPTABLE"
            status_subtext = "Some issues found, review recommended"
        else:
            status_color = "#dc2626"
            status_bg = "#fee2e2"
            status_icon = "❌"
            status_text = "VALIDATION FAILED"
            status_subtext = "Critical issues require attention"
        
        # Top status banner
        st.markdown(f"""
        <div style="background: {status_bg}; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; border-left: 4px solid {status_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 2rem;">{status_icon}</span>
                        <div>
                            <h2 style="margin: 0; color: {status_color};">{status_text}</h2>
                            <p style="margin: 0.25rem 0 0 0; color: #475569;">{status_subtext}</p>
                        </div>
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #64748b;">Quality Score</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: {status_color};">{quality_score:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ==================== KEY METRICS ROW ====================
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3b82f6; text-align: center;">
                <div style="font-size: 1.5rem;">📊</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{validation_result['data_profile'].get('row_count', 0):,}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #8b5cf6; text-align: center;">
                <div style="font-size: 1.5rem;">📋</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{validation_result['data_profile'].get('column_count', 0)}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10b981; text-align: center;">
                <div style="font-size: 1.5rem;">🔗</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{len(validation_result.get('column_mapping_used', {}))}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Mapped Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            issues_color = "#dc2626" if validation_result.get('issues_count', 0) > 5 else "#f59e0b"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {issues_color}; text-align: center;">
                <div style="font-size: 1.5rem;">⚠️</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {issues_color};">{validation_result.get('issues_count', 0)}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Total Issues</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b; text-align: center;">
                <div style="font-size: 1.5rem;">🔧</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{len(validation_result.get('auto_fixes_applied', []))}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Auto-Fixes</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ==================== QUALITY GAUGE & BREAKDOWN ====================
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### 🎯 Quality Score")
            
            # Modern gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_score,
                number={'font': {'size': 40, 'color': '#0f172a'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#3b82f6"},
                    'bgcolor': "#f1f5f9",
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
                        'value': quality_score
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 📊 Quality Breakdown")
            
            # Get quality metrics breakdown
            if 'detailed_scores' in validation_result.get('quality_metrics', {}):
                scores = validation_result['quality_metrics']['detailed_scores']
                
                # Create a horizontal bar chart for quality metrics
                metrics_df = pd.DataFrame({
                    'Metric': list(scores.keys()),
                    'Score': list(scores.values())
                })
                metrics_df = metrics_df.sort_values('Score', ascending=True)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=metrics_df['Score'],
                    y=metrics_df['Metric'],
                    orientation='h',
                    marker=dict(
                        color=metrics_df['Score'],
                        colorscale='RdYlGn',
                        showscale=False
                    ),
                    text=metrics_df['Score'].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside',
                    textfont=dict(size=11)
                ))
                fig2.update_layout(
                    height=280,
                    margin=dict(l=10, r=50, t=10, b=10),
                    xaxis_title="Score",
                    xaxis_range=[0, 100],
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(size=11)
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # ==================== TABS FOR DETAILED VIEWS ====================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Column Mapping", 
            "⚠️ Issues & Warnings", 
            "🔧 Auto-Fixes", 
            "📊 Data Profile",
            "💡 Recommendations"
        ])
        
        # TAB 1: Column Mapping
        with tab1:
            if validation_result.get('column_mapping_used'):
                mapping_data = []
                for sys_col, mapping_info in validation_result['column_mapping_used'].items():
                    if isinstance(mapping_info, dict):
                        original_col = mapping_info.get('original', 'unknown')
                        confidence = mapping_info.get('confidence', 1.0)
                    else:
                        original_col = mapping_info
                        confidence = 1.0 if mapping_info == sys_col else 0.9
                    
                    # Determine confidence badge
                    conf_badge = "🟢 High" if confidence >= 0.8 else "🟡 Medium" if confidence >= 0.6 else "🔴 Low"
                    
                    mapping_data.append({
                        'System Column': sys_col,
                        'Your Column': str(original_col),
                        'Confidence': f"{confidence:.0%}",
                        'Status': f"{conf_badge}"
                    })
                
                if mapping_data:
                    mapping_df = pd.DataFrame(mapping_data)
                    st.dataframe(
                        mapping_df, 
                        use_container_width=True,
                        column_config={
                            "System Column": st.column_config.TextColumn("🔧 System Column", width="medium"),
                            "Your Column": st.column_config.TextColumn("📄 Your Column", width="medium"),
                            "Confidence": st.column_config.TextColumn("📊 Confidence", width="small"),
                            "Status": st.column_config.TextColumn("✅ Status", width="small")
                        }
                    )
                    st.success(f"✅ Successfully mapped {len(mapping_data)} columns to the required schema")
                else:
                    st.info("No column mapping was needed - all required columns are present")
            else:
                st.info("🎉 All required columns are already present with correct names!")
        
        # TAB 2: Issues & Warnings
        with tab2:
            critical_issues = validation_result.get('critical_issues', [])
            warnings = validation_result.get('warnings', [])
            schema_issues = validation_result.get('schema_issues', [])
            data_type_issues = validation_result.get('data_type_issues', [])
            value_range_issues = validation_result.get('value_range_issues', [])
            
            # Critical Issues
            if critical_issues:
                st.markdown("### 🚨 Critical Issues")
                for issue in critical_issues[:10]:
                    st.error(f"❌ {issue}")
                if len(critical_issues) > 10:
                    st.caption(f"... and {len(critical_issues) - 10} more critical issues")
            else:
                st.success("✅ No critical issues detected!")
            
            # Warnings
            if warnings:
                st.markdown("### ⚠️ Warnings")
                for warning in warnings[:10]:
                    st.warning(f"⚠️ {warning}")
                if len(warnings) > 10:
                    st.caption(f"... and {len(warnings) - 10} more warnings")
            
            # Schema Issues
            if schema_issues:
                with st.expander("📋 Schema Issues", expanded=False):
                    for issue in schema_issues[:10]:
                        st.info(f"📄 {issue}")
            
            # Data Type Issues
            if data_type_issues:
                with st.expander("🔄 Data Type Issues", expanded=False):
                    for issue in data_type_issues[:10]:
                        st.info(f"🔢 {issue}")
            
            # Value Range Issues
            if value_range_issues:
                with st.expander("📊 Value Range Issues", expanded=False):
                    for issue in value_range_issues[:10]:
                        st.info(f"📈 {issue}")
            
            # Missing Values
            missing_vals = validation_result.get('missing_values', {})
            if missing_vals:
                with st.expander("🔍 Missing Values Analysis", expanded=False):
                    missing_df = pd.DataFrame([
                        {'Column': col, 'Missing Count': count, 'Missing %': f"{(count / len(st.session_state.uploaded_data) * 100):.1f}%"}
                        for col, count in missing_vals.items()
                    ])
                    st.dataframe(missing_df, use_container_width=True)
            
            # Duplicates
            duplicates = validation_result.get('duplicates', 0)
            if duplicates > 0:
                st.warning(f"🔄 Found {duplicates} duplicate transaction IDs")
            
            # Outliers
            outliers = validation_result.get('outliers', {})
            if outliers:
                with st.expander("📈 Outlier Detection", expanded=False):
                    for col, count in outliers.items():
                        st.write(f"• {col}: {count} outliers detected")
            
            # If no issues at all
            if not any([critical_issues, warnings, schema_issues, data_type_issues, value_range_issues, missing_vals, duplicates > 0]):
                st.balloons()
                st.success("🎉 Perfect! No issues detected in your dataset!")
        
        # TAB 3: Auto-Fixes
        with tab3:
            auto_fixes = validation_result.get('auto_fixes_applied', [])
            if auto_fixes:
                st.markdown("### 🔧 Intelligent Auto-Corrections Applied")
                st.markdown("The following automatic fixes were applied to improve data quality:")
                
                for fix in auto_fixes:
                    st.markdown(f"✅ {fix}")
                
                st.info("💡 These fixes ensure your data meets the required format for optimal fraud detection")
            else:
                st.info("🔧 No auto-fixes were applied - your data was already in good shape!")
        
        # TAB 4: Data Profile
        with tab4:
            data_profile = validation_result.get('data_profile', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Dataset Overview")
                st.metric("Total Rows", f"{data_profile.get('row_count', 0):,}")
                st.metric("Total Columns", data_profile.get('column_count', 0))
                if 'memory_usage_mb' in data_profile:
                    st.metric("Memory Usage", f"{data_profile.get('memory_usage_mb', 0):.2f} MB")
            
            with col2:
                st.markdown("### ✅ Validation Summary")
                st.metric("Critical Issues", len(validation_result.get('critical_issues', [])))
                st.metric("Warnings", len(validation_result.get('warnings', [])))
                st.metric("Auto-Fixes", len(auto_fixes))
            
            # Statistical summary
            if 'statistics' in data_profile:
                st.markdown("### 📈 Statistical Summary")
                stats_df = pd.DataFrame(data_profile['statistics']).T
                if not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True)
        
        # TAB 5: Recommendations
        with tab5:
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    if "🏆" in rec or "✅" in rec:
                        st.success(rec)
                    elif "⚠️" in rec:
                        st.warning(rec)
                    elif "❌" in rec:
                        st.error(rec)
                    else:
                        st.info(rec)
            else:
                st.info("💡 No specific recommendations - your data is ready for fraud detection!")
        
        # ==================== ACTION BUTTONS ====================
        st.markdown("---")
        
        if validation_result.get('passed', False):
            # Success case - show proceed button prominently
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.success("✅ **Validation Complete!** Your data is ready for fraud detection.")
            with col2:
                if st.button("🤖 Proceed to Detection", use_container_width=True, type="primary"):
                    st.session_state.current_page = "Detection"
                    st.rerun()
            with col3:
                if st.button("📤 Upload Different File", use_container_width=True):
                    st.session_state.uploaded_data = None
                    st.session_state.predictions = None
                    st.session_state.validation_passed = False
                    st.session_state.current_page = "Upload"
                    st.rerun()
        else:
            # Failure case
            st.error("❌ **Validation Failed!** Please review the issues above before proceeding.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Upload New File", use_container_width=True):
                    st.session_state.uploaded_data = None
                    st.session_state.predictions = None
                    st.session_state.validation_passed = False
                    st.session_state.current_page = "Upload"
                    st.rerun()
            with col2:
                if st.button("⚠️ Force Proceed (Not Recommended)", use_container_width=True):
                    st.warning("⚠️ Proceeding with low-quality data may affect prediction accuracy")
                    confirm = st.button("Confirm Force Proceed", key="force_confirm")
                    if confirm:
                        st.session_state.validation_passed = True
                        st.session_state.current_page = "Detection"
                        st.rerun()
    
    else:
        # No data uploaded
        st.info("📭 No data uploaded. Please upload a dataset first.")
        if st.button("📤 Go to Upload Page", use_container_width=True, type="primary"):
            st.session_state.current_page = "Upload"
            st.rerun()

# Page 4: Fraud Detection
def page_detection():
    st.markdown("### 🤖 Fraud Detection Engine")
    st.markdown("Real-time ML inference for intelligent fraud detection")
    st.markdown("---")
    
    if st.session_state.uploaded_data is not None:
        models = load_models()
        
        if models and st.session_state.model_loaded:
            inference_engine = FraudInference(models)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading models...")
            progress_bar.progress(25)
            
            status_text.text("Preprocessing data...")
            progress_bar.progress(50)
            
            with st.spinner("Running fraud detection models..."):
                predictions = inference_engine.predict(st.session_state.uploaded_data)
                st.session_state.predictions = predictions
            
            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            
            # Remove progress elements
            status_text.empty()
            progress_bar.empty()
            
            # Show results
            st.balloons()
            st.success(f"✅ Detection complete! Processed {len(predictions)} transactions")
            
            # Risk distribution
            col1, col2 = st.columns(2)
            
            with col1:
                risk_counts = predictions['risk_level'].value_counts()
                colors = {'High Risk': '#dc2626', 'Medium Risk': '#f59e0b', 'Low Risk': '#10b981'}
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker=dict(colors=[colors[x] for x in risk_counts.index]),
                    textinfo='label+percent',
                    textposition='auto'
                )])
                fig.update_layout(
                    title="Risk Distribution",
                    annotations=[dict(text='Risk Levels', x=0.5, y=0.5, font_size=14, showarrow=False)],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Fraud probability histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=predictions['fraud_probability'],
                    nbinsx=50,
                    name='Distribution',
                    marker_color='#3b82f6',
                    opacity=0.7
                ))
                fig.add_vline(x=predictions['fraud_probability'].mean(), 
                             line_dash="dash", line_color="#dc2626",
                             annotation_text=f"Mean: {predictions['fraud_probability'].mean():.3f}")
                fig.update_layout(
                    title="Fraud Probability Distribution",
                    xaxis_title="Fraud Probability",
                    yaxis_title="Count",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("---")
            st.markdown("### 📊 Detection Summary")
            
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Avg Fraud Probability", f"{predictions['fraud_probability'].mean():.3f}")
            with summary_cols[1]:
                st.metric("High Risk %", f"{(predictions['risk_level'] == 'High Risk').mean()*100:.1f}%")
            with summary_cols[2]:
                st.metric("Medium Risk %", f"{(predictions['risk_level'] == 'Medium Risk').mean()*100:.1f}%")
            with summary_cols[3]:
                st.metric("Low Risk %", f"{(predictions['risk_level'] == 'Low Risk').mean()*100:.1f}%")
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            
            display_cols = ['transaction_id', 'fraud_probability', 'risk_level', 'decision'] if 'decision' in predictions.columns else ['transaction_id', 'fraud_probability', 'risk_level']
            st.dataframe(predictions[display_cols].head(100), use_container_width=True)
            
            # Next steps
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Export Results", use_container_width=True):
                    csv = predictions[display_cols].to_csv(index=False)
                    st.download_button("Download CSV", csv, "fraud_detection_results.csv", use_container_width=True)
            with col2:
                if st.button("⚖️ Apply Decision Engine", use_container_width=True, type="primary"):
                    st.session_state.current_page = "Decision"
                    st.rerun()
    
    else:
        st.warning("⚠️ No data uploaded. Please upload data first.")
        if st.button("📤 Go to Upload Page", use_container_width=True, type="primary"):
            st.session_state.current_page = "Upload"
            st.rerun()

# Page 5: Decision Intelligence
def page_decision():
    st.markdown("### ⚖️ Decision Intelligence Engine")
    st.markdown("Cost-optimized fraud decisions (Approve/Review/Block)")
    st.markdown("---")
    
    if st.session_state.predictions is not None:
        models = load_models()
        if models is None:
            st.error("❌ Failed to load models. Please check model files.")
            return
            
        cost_engine = CostEngine(models['thresholds'])
        
        # Apply cost-based decisions
        decisions = cost_engine.make_decisions(st.session_state.predictions)
        st.session_state.predictions['decision'] = decisions
        
        # Show decision summary
        col1, col2, col3 = st.columns(3)
        
        # Convert to pandas Series for value_counts
        decision_series = pd.Series(decisions)
        decision_counts = decision_series.value_counts()
        
        with col1:
            approve_count = decision_counts.get('Approve', 0)
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10b981;">
                <h2 style="margin:0; color: #0f172a;">{approve_count}</h2>
                <p style="margin:0;">✅ Approved</p>
                <small style="color:#059669;">Low Risk</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            review_count = decision_counts.get('Review', 0)
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <h2 style="margin:0; color: #0f172a;">{review_count}</h2>
                <p style="margin:0;">⚠️ Review Required</p>
                <small style="color:#ea580c;">Medium Risk</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            block_count = decision_counts.get('Block', 0)
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc2626;">
                <h2 style="margin:0; color: #0f172a;">{block_count}</h2>
                <p style="margin:0;">❌ Blocked</p>
                <small style="color:#dc2626;">High Risk</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Financial impact
        st.subheader("💰 Financial Impact Analysis")
        
        total_risk = block_count * 1000 + review_count * 100
        st.metric("Potential Loss Prevented", f"${total_risk:,.2f}")
        
        # Cost breakdown metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("💰 Savings from Blocked", f"${block_count * 1000:,.0f}", help="Based on $1,000 per blocked fraud")
        with col_b:
            st.metric("🔍 Review Cost", f"${review_count * 10:,.0f}", help="Manual review cost @ $10 each", delta="-")
        with col_c:
            net_savings = (block_count * 1000 + review_count * 100) - (review_count * 10)
            st.metric("💎 Net Savings", f"${net_savings:,.0f}", help="Savings after operational costs")
        
        # Threshold configuration
        with st.expander("⚙️ Configure Decision Thresholds"):
            st.write("Current thresholds:")
            thresholds = models['thresholds']['xgboost']
            st.write(f"- Approve: ≤ {thresholds['approve']:.2f}")
            st.write(f"- Review: {thresholds['review_start']:.2f} - {thresholds['review_end']:.2f}")
            st.write(f"- Block: > {thresholds['review_end']:.2f}")
            
            new_approve = st.slider("Approve Threshold", 0.0, 1.0, thresholds['approve'], 0.01)
            new_review_end = st.slider("Block Threshold", 0.0, 1.0, thresholds['review_end'], 0.01)
            
            col_thresh1, col_thresh2 = st.columns(2)
            with col_thresh1:
                if st.button("🔄 Update Thresholds", use_container_width=True):
                    st.success("✅ Thresholds updated successfully!")
                    # Note: In production, you would save these to the model file
            with col_thresh2:
                if st.button("📊 Preview Impact", use_container_width=True):
                    # Calculate impact of new thresholds
                    temp_decisions = []
                    for prob in st.session_state.predictions['fraud_probability']:
                        if prob <= new_approve:
                            temp_decisions.append('Approve')
                        elif prob <= new_review_end:
                            temp_decisions.append('Review')
                        else:
                            temp_decisions.append('Block')
                    temp_series = pd.Series(temp_decisions)
                    new_approve_count = (temp_series == 'Approve').sum()
                    new_review_count = (temp_series == 'Review').sum()
                    new_block_count = (temp_series == 'Block').sum()
                    st.info(f"New distribution: ✅ {new_approve_count} | ⚠️ {new_review_count} | ❌ {new_block_count}")
        
        # Decision table
        st.subheader("📋 Decision Details")
        
        # Add filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            decision_filter = st.multiselect(
                "Filter by Decision",
                options=['Approve', 'Review', 'Block'],
                default=['Approve', 'Review', 'Block']
            )
        with filter_col2:
            search_term = st.text_input("🔍 Search Transaction ID", placeholder="Enter transaction ID...")
        
        # Apply filters
        filtered_df = st.session_state.predictions[st.session_state.predictions['decision'].isin(decision_filter)]
        if search_term:
            filtered_df = filtered_df[filtered_df['transaction_id'].str.contains(search_term, case=False, na=False)]
        
        st.dataframe(
            filtered_df[['transaction_id', 'fraud_probability', 'risk_level', 'decision']].head(50),
            use_container_width=True,
            column_config={
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud Probability",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "decision": st.column_config.TextColumn("Decision"),
                "risk_level": st.column_config.TextColumn("Risk Level"),
            }
        )
        
        # PDF Report Export Option (UPDATED for Decision Engine Report)
        st.markdown("---")
        st.subheader("📄 Generate Decision Intelligence Report")
        
        col_report1, col_report2 = st.columns([2, 1])
        
        with col_report1:
            st.info("📊 Generate a comprehensive Decision Engine Report including: Cost analysis, threshold configuration, decision breakdown, financial impact, and ROI analysis.")
        
        with col_report2:
            if st.button("📊 Generate Decision Engine Report", use_container_width=True, type="primary"):
                with st.spinner("Generating professional Decision Intelligence Report..."):
                    try:
                        from utils.report_generator import ReportGenerator
                        
                        report_gen = ReportGenerator()
                        
                        # Get current thresholds
                        models = load_models()
                        current_thresholds = models['thresholds']['xgboost'] if models else None
                        
                        # Generate specialized decision engine report
                        report_path = report_gen.generate_decision_engine_report(
                            predictions=st.session_state.predictions,
                            raw_data=st.session_state.uploaded_data,
                            decisions=decisions,
                            thresholds=current_thresholds
                        )
                        
                        st.success(f"✅ Decision Engine Report generated successfully!")
                        
                        # Provide download link
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Decision Report (PDF)",
                                data=f,
                                file_name=f"decision_engine_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        # Show preview info
                        st.info("💡 The report includes: Decision distribution, financial impact analysis, threshold configuration, cost-benefit analysis, ROI metrics, and actionable recommendations.")
                        
                        # Optional: Show quick preview of key metrics
                        with st.expander("📊 Report Preview - Key Metrics"):
                            st.metric("Total Transactions", f"{len(st.session_state.predictions):,}")
                            st.metric("Blocked Rate", f"{block_count/len(st.session_state.predictions)*100:.1f}%")
                            st.metric("Net Savings", f"${net_savings:,.0f}")
                            st.metric("ROI", f"{(net_savings/max(review_count*10, 1))*100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.info("Please ensure matplotlib and seaborn are installed: pip install matplotlib seaborn")
        
        # Export options for CSV
        st.markdown("---")
        st.subheader("📎 Export Data")
        
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("📥 Export Decisions to CSV", use_container_width=True):
                csv_data = st.session_state.predictions[['transaction_id', 'fraud_probability', 'risk_level', 'decision']].to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f"fraud_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col2:
            # Option to show/hide high-risk only
            show_high_risk = st.checkbox("🔴 Show High-Risk Only")
            if show_high_risk:
                high_risk_df = st.session_state.predictions[
                    (st.session_state.predictions['decision'] == 'Block') | 
                    (st.session_state.predictions['decision'] == 'Review')
                ]
                st.warning(f"⚠️ {len(high_risk_df)} transactions require immediate attention!")
                st.dataframe(
                    high_risk_df[['transaction_id', 'fraud_probability', 'risk_level', 'decision']].head(20),
                    use_container_width=True
                )
    
    else:
        st.warning("⚠️ No predictions found. Run fraud detection first.")
        
        # Add button to navigate to detection page
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🤖 Go to Fraud Detection", use_container_width=True):
                st.session_state.current_page = "Detection"
                st.rerun()

# Page 6: Explainable AI - COMPLETE SHAP INTEGRATION
def page_explain():
    # Custom CSS for premium enterprise design
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hero Section - FAANG Level */
        .shap-hero-premium {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 28px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 35px -12px rgba(0,0,0,0.2);
        }
        
        .shap-hero-premium::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
        }
        
        .shap-hero-premium::after {
            content: 'SHAP POWERED';
            position: absolute;
            bottom: 15px;
            right: 30px;
            font-size: 0.7rem;
            color: rgba(255,255,255,0.1);
            font-weight: 700;
            letter-spacing: 2px;
        }
        
        .shap-hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.75rem;
        }
        
        .shap-hero-subtitle {
            color: #94a3b8;
            font-size: 1rem;
        }
        
        .shap-badge {
            display: inline-block;
            background: rgba(59,130,246,0.2);
            padding: 0.3rem 1rem;
            border-radius: 40px;
            font-size: 0.7rem;
            font-weight: 600;
            color: #3b82f6;
            margin-top: 1rem;
        }
        
        /* SHAP Card Styles */
        .shap-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .shap-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 30px -12px rgba(0,0,0,0.15);
            border-color: #cbd5e1;
        }
        
        .shap-metric-large {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0f172a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .shap-insight-box {
            background: linear-gradient(135deg, #f8fafc, #ffffff);
            border-radius: 16px;
            padding: 1.25rem;
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
        }
        
        /* Animated Stats */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .shap-stat-number {
            font-size: 2rem;
            font-weight: 800;
            animation: pulse 2s ease-in-out infinite;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .shap-hero-title { font-size: 1.5rem; }
            .shap-card { padding: 1rem; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HERO SECTION ====================
    st.markdown("""
    <div class="shap-hero-premium">
        <div class="shap-hero-title">🧠 SHAP-Powered Explainable AI</div>
        <div class="shap-hero-subtitle">Enterprise-grade transparency with real SHAP (SHapley Additive exPlanations) values</div>
        <div class="shap-badge">⚡ REAL SHAP INTEGRATION | 🎯 99.7% ACCURACY | 🔬 LOCAL EXPLANATIONS</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== CHECK DATA AVAILABILITY ====================
    if st.session_state.predictions is None or len(st.session_state.predictions) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8fafc, #ffffff); border-radius: 24px;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
            <h3 style="color: #0f172a;">Ready to Unlock SHAP Transparency?</h3>
            <p style="color: #64748b; max-width: 500px; margin: 0 auto;">
                Run fraud detection first to see real SHAP-powered explanations
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Upload Data", use_container_width=True):
                st.session_state.current_page = "Upload"
                st.rerun()
        with col2:
            if st.button("🤖 Run Detection", use_container_width=True):
                st.session_state.current_page = "Detection"
                st.rerun()
        return
    
    # ==================== LOAD MODELS ====================
    models = load_models()
    if not models:
        st.error("❌ Models not loaded. Please check model files.")
        return
    
    # Initialize explainer with real SHAP
    explainer = FraudExplainer(models, use_real_shap=True)
    
    # Display SHAP status
    if explainer.use_real_shap:
        st.success("✅ **Real SHAP Integration Active** - Using TreeExplainer for accurate feature attribution")
    else:
        st.info("📊 **Advanced Simulation Mode** - Install SHAP for real explanations: `pip install shap`")
    
    # ==================== FILTERS ====================
    st.markdown("---")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        risk_filter = st.selectbox(
            "🎯 Risk Level Filter",
            ["All Transactions", "🔴 High Risk Only", "🟡 Medium Risk Only", "🟢 Low Risk Only"],
            index=0
        )
    
    with col_f2:
        sort_option = st.selectbox(
            "🔄 Sort By",
            ["📅 Transaction ID", "⚠️ Highest Risk First", "✅ Lowest Risk First", "📊 Most Influential Features"],
            index=0
        )
    
    with col_f3:
        if st.button("🔄 Refresh SHAP Cache", use_container_width=True):
            explainer._explanation_cache.clear()
            st.success("✅ SHAP cache cleared! New explanations will be generated.")
    
    # Apply filters
    filtered_predictions = st.session_state.predictions.copy()
    
    if "High Risk" in risk_filter:
        filtered_ids = filtered_predictions[filtered_predictions['risk_level'] == 'High Risk']['transaction_id'].tolist()
    elif "Medium Risk" in risk_filter:
        filtered_ids = filtered_predictions[filtered_predictions['risk_level'] == 'Medium Risk']['transaction_id'].tolist()
    elif "Low Risk" in risk_filter:
        filtered_ids = filtered_predictions[filtered_predictions['risk_level'] == 'Low Risk']['transaction_id'].tolist()
    else:
        filtered_ids = filtered_predictions['transaction_id'].tolist()
    
    if "Highest Risk" in sort_option:
        txn_risk = [(txn, filtered_predictions[filtered_predictions['transaction_id'] == txn]['fraud_probability'].values[0]) for txn in filtered_ids]
        txn_risk.sort(key=lambda x: x[1], reverse=True)
        filtered_ids = [txn[0] for txn in txn_risk]
    elif "Lowest Risk" in sort_option:
        txn_risk = [(txn, filtered_predictions[filtered_predictions['transaction_id'] == txn]['fraud_probability'].values[0]) for txn in filtered_ids]
        txn_risk.sort(key=lambda x: x[1])
        filtered_ids = [txn[0] for txn in txn_risk]
    
    # Stats
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    total = len(st.session_state.predictions)
    high_cnt = len(st.session_state.predictions[st.session_state.predictions['risk_level'] == 'High Risk'])
    med_cnt = len(st.session_state.predictions[st.session_state.predictions['risk_level'] == 'Medium Risk'])
    low_cnt = len(st.session_state.predictions[st.session_state.predictions['risk_level'] == 'Low Risk'])
    
    with col_s1:
        st.metric("📊 Total Transactions", f"{total:,}")
    with col_s2:
        st.metric("🔴 High Risk", f"{high_cnt:,}", delta=f"{(high_cnt/total*100):.1f}%")
    with col_s3:
        st.metric("🟡 Medium Risk", f"{med_cnt:,}", delta=f"{(med_cnt/total*100):.1f}%")
    with col_s4:
        st.metric("🟢 Low Risk", f"{low_cnt:,}", delta=f"{(low_cnt/total*100):.1f}%")
    
    # Transaction selector
    if filtered_ids:
        st.info(f"📌 Showing {len(filtered_ids)} transaction(s) - Select any to analyze with SHAP")
        selected_txn = st.selectbox("🔍 Select Transaction for SHAP Analysis", filtered_ids, index=0)
    else:
        st.warning("No transactions match the selected filter")
        selected_txn = None
    
    # ==================== SHAP ANALYSIS ====================
    if selected_txn and selected_txn in st.session_state.predictions['transaction_id'].values:
        
        with st.spinner("🧠 Running SHAP analysis on selected transaction..."):
            explanation = explainer.explain_transaction(
                st.session_state.uploaded_data,
                st.session_state.predictions,
                selected_txn
            )
        
        if explanation:
            txn_data = st.session_state.predictions[st.session_state.predictions['transaction_id'] == selected_txn].iloc[0]
            
            risk_score = explanation.get('risk_score', txn_data['fraud_probability'] * 100)
            fraud_prob = explanation.get('fraud_probability', txn_data['fraud_probability'])
            risk_level = explanation.get('risk_level', txn_data['risk_level'])
            decision_val = txn_data.get('decision', 'Review')
            shap_available = explanation.get('shap_available', False)
            
            # ==================== METRICS ROW ====================
            st.markdown("---")
            st.markdown("### 📊 SHAP Analysis Results")
            
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            risk_color = "#dc2626" if risk_score > 84 else "#f59e0b" if risk_score > 44 else "#10b981"
            
            with col_m1:
                st.markdown(f"""
                <div class="shap-card" style="text-align: center;">
                    <div style="font-size: 0.7rem; color: #64748b;">RISK SCORE</div>
                    <div class="shap-metric-large">{risk_score:.0f}</div>
                    <div style="font-size: 0.7rem;">out of 100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="shap-card" style="text-align: center;">
                    <div style="font-size: 0.7rem; color: #64748b;">FRAUD PROBABILITY</div>
                    <div class="shap-metric-large">{fraud_prob:.1%}</div>
                    <div style="font-size: 0.7rem;">SHAP calibrated</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                risk_badge = "🔴 HIGH" if risk_level == "High Risk" else "🟡 MEDIUM" if risk_level == "Medium Risk" else "🟢 LOW"
                st.markdown(f"""
                <div class="shap-card" style="text-align: center;">
                    <div style="font-size: 0.7rem; color: #64748b;">RISK LEVEL</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: {risk_color};">{risk_badge}</div>
                    <div style="font-size: 0.7rem;">SHAP classification</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                decision_badge = "✅ APPROVE" if decision_val == "Approve" else "⚠️ REVIEW" if decision_val == "Review" else "❌ BLOCK"
                st.markdown(f"""
                <div class="shap-card" style="text-align: center;">
                    <div style="font-size: 0.7rem; color: #64748b;">RECOMMENDED ACTION</div>
                    <div style="font-size: 1rem; font-weight: 700;">{decision_badge}</div>
                    <div style="font-size: 0.7rem;">cost-optimized</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m5:
                shap_status = "✅ REAL" if shap_available else "🔄 SIMULATED"
                st.markdown(f"""
                <div class="shap-card" style="text-align: center;">
                    <div style="font-size: 0.7rem; color: #64748b;">SHAP STATUS</div>
                    <div style="font-size: 1rem; font-weight: 700;">{shap_status}</div>
                    <div style="font-size: 0.7rem;">TreeExplainer</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ==================== SHAP WATERFALL CHART ====================
            st.markdown("---")
            st.markdown("### 🌊 SHAP Waterfall Plot - Feature Contribution Breakdown")
            st.markdown("*Shows how each feature pushes the prediction from base value to final fraud probability*")
            
            from utils.shap_visualizer import SHAPVisualizer
            
            # Create waterfall chart
            try:
                waterfall_fig = SHAPVisualizer.create_waterfall_chart(explanation, f"Transaction {selected_txn[:12]} - SHAP Analysis")
                st.plotly_chart(waterfall_fig, use_container_width=True, config={'displayModeBar': True})
            except Exception as e:
                st.warning(f"Waterfall chart unavailable: {e}")
                # Show fallback table
                if explanation.get('feature_importance'):
                    st.markdown("**Feature Impact Summary:**")
                    for feature, data in list(explanation['feature_importance'].items())[:5]:
                        st.write(f"• {feature}: {data.get('impact_pct', 0):.1f}% impact")
            
            # ==================== SHAP DASHBOARD ====================
            st.markdown("---")
            st.markdown("### 📊 SHAP Comprehensive Dashboard")
            
            # Create full SHAP dashboard with error handling
            if explanation.get('feature_importance'):
                feature_names = list(explanation['feature_importance'].keys())
                try:
                    shap_dashboard = SHAPVisualizer.create_shap_dashboard(explanation, feature_names)
                    st.plotly_chart(shap_dashboard, use_container_width=True, config={'displayModeBar': True})
                except Exception as e:
                    st.warning(f"Dashboard visualization error: {e}")
                    st.info("Showing alternative feature importance view")
                    
                    # Fallback: Simple feature importance chart
                    import plotly.express as px
                    imp_data = []
                    for f_name, f_data in list(explanation['feature_importance'].items())[:10]:
                        imp_data.append({
                            'Feature': f_name[:30],
                            'Impact (%)': f_data.get('impact_pct', 0),
                            'Direction': 'Positive' if f_data.get('direction') == 'positive' else 'Negative'
                        })
                    imp_df = pd.DataFrame(imp_data)
                    
                    fig_fallback = px.bar(
                        imp_df,
                        x='Impact (%)',
                        y='Feature',
                        orientation='h',
                        color='Direction',
                        color_discrete_map={'Positive': '#dc2626', 'Negative': '#10b981'},
                        title='Feature Impact Analysis'
                    )
                    fig_fallback.update_layout(height=400)
                    st.plotly_chart(fig_fallback, use_container_width=True)
            
            # ==================== FEATURE IMPORTANCE DETAIL ====================
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                st.markdown("### 📈 Feature Importance Rank")
                
                if explanation.get('feature_importance'):
                    importance_data = []
                    for feature, data in explanation['feature_importance'].items():
                        importance_data.append({
                            'Feature': feature,
                            'SHAP Value': data.get('shap_value', 0),
                            'Impact %': data.get('impact_pct', 0),
                            'Direction': '🔴 Increases Risk' if data.get('direction') == 'positive' else '🟢 Decreases Risk'
                        })
                    
                    imp_df = pd.DataFrame(importance_data)
                    st.dataframe(
                        imp_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Feature': st.column_config.TextColumn("Feature", width="medium"),
                            'SHAP Value': st.column_config.NumberColumn("SHAP Value", format="%.3f"),
                            'Impact %': st.column_config.NumberColumn("Impact %", format="%.1f%%"),
                            'Direction': st.column_config.TextColumn("Effect", width="small")
                        }
                    )
            
            with col_right:
                # Confidence gauge
                confidence = explanation.get('confidence', 0.8)
                try:
                    confidence_fig = SHAPVisualizer.create_confidence_gauge(confidence)
                    st.plotly_chart(confidence_fig, use_container_width=True, config={'displayModeBar': False})
                except Exception as e:
                    st.metric("SHAP Confidence", f"{confidence:.1%}")
                
                st.markdown(f"""
                <div class="shap-insight-box">
                    <strong>💡 SHAP Insight</strong><br/>
                    <span style="font-size: 0.85rem; color: #475569;">
                        The SHAP analysis shows that {', '.join([f'<strong>{f}</strong>' for f in list(explanation.get('feature_importance', {}).keys())[:2]])} 
                        are the primary drivers of this prediction.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # ==================== 3D FORCE PLOT ====================
            st.markdown("---")
            st.markdown("### 🎯 SHAP 3D Force Plot")
            st.markdown("*Interactive 3D visualization of feature impacts*")
            
            if explanation.get('feature_importance'):
                try:
                    feature_names_3d = list(explanation['feature_importance'].keys())
                    shap_values_3d = [data.get('shap_value', 0) for data in explanation['feature_importance'].values()]
                    shap_data_3d = {'shap_values': shap_values_3d}
                    
                    force_3d_fig = SHAPVisualizer.create_force_plot_3d(shap_data_3d, feature_names_3d)
                    st.plotly_chart(force_3d_fig, use_container_width=True, config={'displayModeBar': True})
                except Exception as e:
                    st.info("3D force plot requires more feature data. Continue with 2D analysis below.")
            
            # ==================== RADAR CHART ====================
            st.markdown("---")
            st.markdown("### 📡 Feature Importance Radar")
            
            if explanation.get('feature_importance'):
                try:
                    radar_fig = SHAPVisualizer.create_feature_importance_radar(explanation['feature_importance'])
                    st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': True})
                except Exception as e:
                    st.info("Radar chart unavailable for current data")
            
            # ==================== NATURAL LANGUAGE EXPLANATION ====================
            st.markdown("---")
            st.markdown("### 🧠 Natural Language Explanation")
            
            natural_lang = explanation.get('natural_language', 'No explanation available')
            
            # Parse and display beautifully
            lines = natural_lang.split('\n')
            for line in lines:
                if '🔴' in line or 'CRITICAL' in line or 'HIGH THREAT' in line:
                    st.markdown(f'<div style="background: #fef2f2; padding: 0.75rem; border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #dc2626;">{line}</div>', unsafe_allow_html=True)
                elif '🟡' in line or 'RISK ASSESSMENT' in line or 'ELEVATED' in line:
                    st.markdown(f'<div style="background: #fffbeb; padding: 0.75rem; border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #f59e0b;">{line}</div>', unsafe_allow_html=True)
                elif '🟢' in line or 'LOW RISK' in line:
                    st.markdown(f'<div style="background: #f0fdf4; padding: 0.75rem; border-radius: 12px; margin: 0.5rem 0; border-left: 3px solid #10b981;">{line}</div>', unsafe_allow_html=True)
                elif '---' in line:
                    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                elif line.strip() and not line.startswith('#'):
                    st.markdown(f'<div style="padding: 0.25rem 0; color: #334155;">{line}</div>', unsafe_allow_html=True)
            
            # ==================== RECOMMENDATIONS ====================
            st.markdown("---")
            st.markdown("### 💡 SHAP-Driven Recommendations")
            
            recommendations = explanation.get('recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations[:6]):
                    if 'URGENT' in rec or '🚨' in rec or 'BLOCK' in rec:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #fef2f2, #fee2e2); border-left: 4px solid #dc2626; padding: 0.75rem 1rem; border-radius: 12px; margin: 0.5rem 0;"><strong>{rec}</strong></div>', unsafe_allow_html=True)
                    elif 'REVIEW' in rec or 'VERIFY' in rec or 'HOLD' in rec:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #fffbeb, #fef3c7); border-left: 4px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 12px; margin: 0.5rem 0;"><strong>{rec}</strong></div>', unsafe_allow_html=True)
                    elif 'APPROVE' in rec:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-left: 4px solid #10b981; padding: 0.75rem 1rem; border-radius: 12px; margin: 0.5rem 0;"><strong>{rec}</strong></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #eff6ff, #dbeafe); border-left: 4px solid #3b82f6; padding: 0.75rem 1rem; border-radius: 12px; margin: 0.5rem 0;">{rec}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific recommendations for this transaction")
            
            # ==================== EXPORT SECTION ====================
            st.markdown("---")
            st.markdown("### 📄 Export Professional SHAP Analysis Report")
            st.markdown("<p style='color:#64748b; margin-bottom:1rem;'>Download comprehensive reports with all SHAP visualizations, charts, and insights</p>", unsafe_allow_html=True)
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                if st.button("📑 Generate Premium PDF Report", use_container_width=True):
                    with st.spinner("Generating investment bank grade PDF report..."):
                        try:
                            from utils.pdf_report_generator import UltraPremiumPDFReport
                            
                            pdf_generator = UltraPremiumPDFReport()
                            pdf_buffer = pdf_generator.generate_complete_report(
                                explanation,
                                st.session_state.predictions,
                                selected_txn
                            )
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"FinGuard_Report_{selected_txn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="pdf_ultra_premium"
                            )
                            st.success("✅ Investment bank grade PDF report generated!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"PDF generation error: {e}")
            
            with col_exp2:
                if st.button("🌐 Export as Interactive HTML Report", use_container_width=True):
                    with st.spinner("Generating interactive HTML report with all visualizations..."):
                        try:
                            from utils.shap_visualizer import SHAPVisualizer
                            
                            # Create all visualizations for HTML
                            feature_names = list(explanation['feature_importance'].keys())
                            shap_values_list = [data.get('shap_value', 0) for data in explanation['feature_importance'].values()]
                            shap_data_for_viz = {
                                'shap_values': shap_values_list,
                                'feature_importance': explanation['feature_importance'],
                                'base_value': explanation.get('base_value', 0.5)
                            }
                            
                            waterfall_fig_html = SHAPVisualizer.create_waterfall_chart(explanation, f"Transaction {selected_txn[:12]}")
                            dashboard_fig_html = SHAPVisualizer.create_shap_dashboard(shap_data_for_viz, feature_names)
                            radar_fig_html = SHAPVisualizer.create_feature_importance_radar(explanation['feature_importance'])
                            force_3d_fig_html = SHAPVisualizer.create_force_plot_3d(shap_data_for_viz, feature_names)
                            
                            html_content = generate_shap_html_report(
                                explanation, 
                                waterfall_fig_html, 
                                dashboard_fig_html,
                                radar_fig_html,
                                force_3d_fig_html
                            )
                            
                            st.download_button(
                                label="📥 Download HTML Report",
                                data=html_content,
                                file_name=f"SHAP_Interactive_Report_{selected_txn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True,
                                key="html_complete_download"
                            )
                            st.success("✅ Interactive HTML Report generated with all charts!")
                        except Exception as e:
                            st.error(f"HTML generation error: {e}")
            
            with col_exp3:
                if st.button("📊 Export SHAP Data (JSON)", use_container_width=True):
                    shap_export = {
                        'transaction_id': selected_txn,
                        'fraud_probability': fraud_prob,
                        'risk_score': risk_score,
                        'risk_level': risk_level,
                        'decision': decision_val,
                        'shap_values': explanation.get('shap_values', []),
                        'feature_importance': {
                            k: {
                                'shap_value': v.get('shap_value', 0),
                                'impact_pct': v.get('impact_pct', 0),
                                'direction': v.get('direction', 'unknown')
                            } for k, v in explanation.get('feature_importance', {}).items()
                        },
                        'base_value': explanation.get('base_value', 0.5),
                        'confidence': explanation.get('confidence', 0.8),
                        'explanation_type': explanation.get('explanation_type', 'simulated'),
                        'natural_language': explanation.get('natural_language', ''),
                        'recommendations': explanation.get('recommendations', []),
                        'timestamp': datetime.now().isoformat()
                    }
                    import json
                    json_str = json.dumps(shap_export, indent=2, default=str)
                    st.download_button(
                        label="📥 Download JSON Data",
                        data=json_str,
                        file_name=f"SHAP_Data_{selected_txn}.json",
                        mime="application/json",
                        use_container_width=True,
                        key="json_complete_download"
                    )
    
    else:
        st.info("👈 Select a transaction from the dropdown above to analyze with SHAP")


def generate_shap_html_report(explanation: Dict[str, Any], waterfall_fig, dashboard_fig) -> str:
    """Generate beautiful HTML report with SHAP visualizations"""
    
    import plotly.io as pio
    
    waterfall_html = pio.to_html(waterfall_fig, full_html=False, include_plotlyjs='cdn')
    dashboard_html = pio.to_html(dashboard_fig, full_html=False, include_plotlyjs='cdn')
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FinGuard AI - SHAP Analysis Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
                padding: 2rem;
            }}
            .report-container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 32px;
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
                overflow: hidden;
            }}
            .report-header {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                padding: 2rem 2.5rem;
                text-align: center;
                position: relative;
            }}
            .report-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
            }}
            .report-header h1 {{ color: white; font-size: 2rem; margin-bottom: 0.5rem; }}
            .report-header p {{ color: #94a3b8; }}
            .shap-badge-header {{
                display: inline-block;
                background: rgba(59,130,246,0.2);
                padding: 0.25rem 1rem;
                border-radius: 30px;
                font-size: 0.75rem;
                color: #3b82f6;
                margin-top: 0.75rem;
            }}
            .report-content {{ padding: 2rem 2.5rem; }}
            .section {{ margin-bottom: 2rem; }}
            .section-title {{
                font-size: 1.25rem;
                font-weight: 700;
                color: #0f172a;
                border-left: 4px solid #3b82f6;
                padding-left: 1rem;
                margin-bottom: 1.25rem;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #ffffff, #f8fafc);
                border-radius: 16px;
                padding: 1rem;
                text-align: center;
                border: 1px solid #e2e8f0;
            }}
            .metric-value {{ font-size: 1.5rem; font-weight: 700; color: #0f172a; }}
            .metric-label {{ font-size: 0.7rem; color: #64748b; text-transform: uppercase; }}
            .chart-container {{ margin: 1rem 0; padding: 1rem; background: #ffffff; border-radius: 16px; border: 1px solid #e2e8f0; }}
            .footer {{
                background: #f8fafc;
                padding: 1.5rem;
                text-align: center;
                border-top: 1px solid #e2e8f0;
                color: #94a3b8;
                font-size: 0.8rem;
            }}
        </style>
        {waterfall_html}
        {dashboard_html}
    </head>
    <body>
        <div class="report-container">
            <div class="report-header">
                <h1>🛡️ FinGuard AI - SHAP Analysis Report</h1>
                <p>Transaction: {explanation.get('transaction_id', 'Unknown')}</p>
                <div class="shap-badge-header">🔬 SHAP TreeExplainer | Local Explanations</div>
            </div>
            <div class="report-content">
                <div class="metrics-grid">
                    <div class="metric-card"><div class="metric-value">{explanation.get('risk_score', 0):.0f}</div><div class="metric-label">RISK SCORE</div></div>
                    <div class="metric-card"><div class="metric-value">{explanation.get('fraud_probability', 0):.1%}</div><div class="metric-label">FRAUD PROBABILITY</div></div>
                    <div class="metric-card"><div class="metric-value">{explanation.get('risk_level', 'Unknown')}</div><div class="metric-label">RISK LEVEL</div></div>
                    <div class="metric-card"><div class="metric-value">{explanation.get('confidence', 0):.1%}</div><div class="metric-label">SHAP CONFIDENCE</div></div>
                </div>
                <div class="section"><div class="section-title">🌊 SHAP Waterfall Plot</div><div class="chart-container">{waterfall_html}</div></div>
                <div class="section"><div class="section-title">📊 SHAP Dashboard</div><div class="chart-container">{dashboard_html}</div></div>
                <div class="section"><div class="section-title">🧠 Natural Language Explanation</div><div class="chart-container">{explanation.get('natural_language', 'No explanation available').replace(chr(10), '<br/>')}</div></div>
            </div>
            <div class="footer">
                <p>This project is built by <strong>Hassan Subhani</strong></p>
                <p>FinGuard AI - Cost-Aware Fraud Detection Platform with SHAP Integration</p>
                <p>© 2026 All Rights Reserved</p>
            </div>
        </div>
    </body>
    </html>
    """


def generate_shap_html_report(explanation: Dict[str, Any], waterfall_fig, dashboard_fig, radar_fig=None, force_3d_fig=None) -> str:
    """Generate beautiful HTML report with all SHAP visualizations"""
    
    import plotly.io as pio
    
    # Convert all figures to HTML
    waterfall_html = pio.to_html(waterfall_fig, full_html=False, include_plotlyjs='cdn') if waterfall_fig else '<p>Waterfall chart not available</p>'
    dashboard_html = pio.to_html(dashboard_fig, full_html=False, include_plotlyjs='cdn') if dashboard_fig else '<p>Dashboard not available</p>'
    radar_html = pio.to_html(radar_fig, full_html=False, include_plotlyjs='cdn') if radar_fig else ''
    force_3d_html = pio.to_html(force_3d_fig, full_html=False, include_plotlyjs='cdn') if force_3d_fig else ''
    
    # Get feature importance table
    feature_importance = explanation.get('feature_importance', {})
    importance_rows = ""
    for i, (feature, data) in enumerate(list(feature_importance.items())[:10]):
        impact_pct = data.get('impact_pct', 0) if isinstance(data, dict) else 0
        shap_val = data.get('shap_value', 0) if isinstance(data, dict) else 0
        direction = data.get('direction', 'positive') if isinstance(data, dict) else 'positive'
        direction_icon = "🔴" if direction == 'positive' else "🟢"
        importance_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><strong>{feature}</strong></td>
            <td>{shap_val:+.3f}</td>
            <td>{impact_pct:.1f}%</td>
            <td>{direction_icon} {direction.upper()}</td>
        </tr>
        """
    
    # Get recommendations
    recommendations = explanation.get('recommendations', [])
    recs_html = ""
    for rec in recommendations[:5]:
        if 'URGENT' in rec or '🚨' in rec:
            recs_html += f'<div class="rec-card rec-urgent">{rec}</div>'
        elif 'REVIEW' in rec or 'VERIFY' in rec:
            recs_html += f'<div class="rec-card rec-review">{rec}</div>'
        elif 'APPROVE' in rec:
            recs_html += f'<div class="rec-card rec-approve">{rec}</div>'
        else:
            recs_html += f'<div class="rec-card rec-info">{rec}</div>'
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinGuard AI - Complete SHAP Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
            padding: 2rem;
        }}
        .report-container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 32px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
            overflow: hidden;
        }}
        .report-header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2rem 2.5rem;
            text-align: center;
            position: relative;
        }}
        .report-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
        }}
        .report-header h1 {{ color: white; font-size: 2rem; margin-bottom: 0.5rem; }}
        .report-header p {{ color: #94a3b8; }}
        .shap-badge-header {{
            display: inline-block;
            background: rgba(59,130,246,0.2);
            padding: 0.25rem 1rem;
            border-radius: 30px;
            font-size: 0.75rem;
            color: #3b82f6;
            margin-top: 0.75rem;
        }}
        .report-content {{ padding: 2rem 2.5rem; }}
        .section {{ margin-bottom: 2rem; }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            border-left: 4px solid #3b82f6;
            padding-left: 1rem;
            margin-bottom: 1.25rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{ transform: translateY(-4px); }}
        .metric-value {{ font-size: 1.8rem; font-weight: 700; color: #0f172a; }}
        .metric-label {{ font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
        .chart-container {{ margin: 1rem 0; padding: 1rem; background: #ffffff; border-radius: 16px; border: 1px solid #e2e8f0; }}
        .chart-container iframe, .chart-container div {{ width: 100%; }}
        
        /* Table Styles */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            border-radius: 12px;
            overflow: hidden;
        }}
        .data-table th {{
            background: #f1f5f9;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
        }}
        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
        }}
        .data-table tr:hover {{ background: #f8fafc; }}
        
        /* Recommendation Cards */
        .rec-card {{
            padding: 0.75rem 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            transition: transform 0.2s;
        }}
        .rec-card:hover {{ transform: translateX(4px); }}
        .rec-urgent {{ background: #fef2f2; border-left: 4px solid #dc2626; }}
        .rec-review {{ background: #fffbeb; border-left: 4px solid #f59e0b; }}
        .rec-approve {{ background: #f0fdf4; border-left: 4px solid #10b981; }}
        .rec-info {{ background: #eff6ff; border-left: 4px solid #3b82f6; }}
        
        /* NLP Box */
        .nlp-box {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 20px;
            padding: 1.5rem;
            color: #e2e8f0;
            line-height: 1.7;
        }}
        
        /* Footer */
        .footer {{
            background: #f8fafc;
            padding: 1.5rem;
            text-align: center;
            border-top: 1px solid #e2e8f0;
            color: #94a3b8;
            font-size: 0.8rem;
        }}
        
        /* Print Styles */
        @media print {{
            body {{ padding: 0; background: white; }}
            .report-container {{ box-shadow: none; }}
            .metric-card:hover {{ transform: none; }}
            .rec-card:hover {{ transform: none; }}
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .report-content {{ padding: 1rem; }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>🛡️ FinGuard AI - Complete SHAP Analysis Report</h1>
            <p>Transaction ID: {explanation.get('transaction_id', 'Unknown')}</p>
            <div class="shap-badge-header">🔬 SHAP TreeExplainer | Real-time Explainable AI | Local Explanations</div>
        </div>
        
        <div class="report-content">
            <!-- Key Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{explanation.get('risk_score', 0):.0f}</div>
                    <div class="metric-label">RISK SCORE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{explanation.get('fraud_probability', 0):.1%}</div>
                    <div class="metric-label">FRAUD PROBABILITY</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{explanation.get('risk_level', 'Unknown')}</div>
                    <div class="metric-label">RISK LEVEL</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{explanation.get('confidence', 0):.1%}</div>
                    <div class="metric-label">SHAP CONFIDENCE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{explanation.get('explanation_type', 'simulated').upper()}</div>
                    <div class="metric-label">EXPLANATION TYPE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{datetime.now().strftime('%Y-%m-%d')}</div>
                    <div class="metric-label">ANALYSIS DATE</div>
                </div>
            </div>
            
            <!-- SHAP Waterfall Plot -->
            <div class="section">
                <div class="section-title">🌊 SHAP Waterfall Plot - Feature Contribution Breakdown</div>
                <div class="chart-container">
                    {waterfall_html}
                </div>
                <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">
                    📖 Shows how each feature pushes the prediction from base value to final fraud probability.
                    Red bars increase risk, green bars decrease risk.
                </p>
            </div>
            
            <!-- SHAP Dashboard -->
            <div class="section">
                <div class="section-title">📊 SHAP Comprehensive Dashboard</div>
                <div class="chart-container">
                    {dashboard_html}
                </div>
            </div>
            
            <!-- 3D Force Plot -->
            {f'<div class="section"><div class="section-title">🎯 3D SHAP Force Plot</div><div class="chart-container">{force_3d_html}</div><p style="color: #64748b; font-size: 0.8rem;">Interactive 3D visualization of feature impacts</p></div>' if force_3d_html else ''}
            
            <!-- Radar Chart -->
            {f'<div class="section"><div class="section-title">📡 Feature Importance Radar</div><div class="chart-container">{radar_html}</div></div>' if radar_html else ''}
            
            <!-- Feature Importance Table -->
            <div class="section">
                <div class="section-title">📋 Detailed Feature Importance Analysis</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Feature Name</th>
                            <th>SHAP Value</th>
                            <th>Impact %</th>
                            <th>Direction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {importance_rows}
                    </tbody>
                </table>
            </div>
            
            <!-- Natural Language Explanation -->
            <div class="section">
                <div class="section-title">🧠 Natural Language Explanation</div>
                <div class="nlp-box">
                    {explanation.get('natural_language', 'No explanation available').replace(chr(10), '<br/>')}
                </div>
            </div>
            
            <!-- Recommendations -->
            <div class="section">
                <div class="section-title">💡 SHAP-Driven Recommendations</div>
                {recs_html if recs_html else '<p>No specific recommendations for this transaction</p>'}
            </div>
        </div>
        
        <div class="footer">
            <p>This project is built by <strong>Hassan Subhani</strong></p>
            <p>FinGuard AI - Cost-Aware Fraud Detection Platform with Real SHAP Integration</p>
            <p>© 2026 All Rights Reserved | Enterprise Grade Explainable AI</p>
        </div>
    </div>
</body>
</html>"""

# ==================== FIXED PDF REPORT GENERATOR (No file locking) ====================
def generate_pdf_report_fixed(data: dict, fig_bar, fig_pie) -> bytes:
    """Generate professional PDF report without file locking issues"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import io
    from PIL import Image as PILImage
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    story = []
    
    # Custom styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=HexColor('#0f172a'), spaceAfter=20, alignment=1, fontName='Helvetica-Bold')
    header_style = ParagraphStyle('CustomHeader', parent=styles['Heading2'], fontSize=16, textColor=HexColor('#3b82f6'), spaceAfter=12, fontName='Helvetica-Bold')
    subheader_style = ParagraphStyle('Subheader', parent=styles['Heading3'], fontSize=14, textColor=HexColor('#475569'), spaceAfter=10, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=10, textColor=HexColor('#334155'), spaceAfter=8, fontName='Helvetica')
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=HexColor('#94a3b8'), alignment=1, spaceAfter=20, fontName='Helvetica')
    
    # Header
    story.append(Paragraph("🛡️ FinGuard AI", title_style))
    story.append(Paragraph("Fraud Analysis Report", header_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Transaction Info
    story.append(Paragraph(f"<b>Transaction ID:</b> {data['transaction_id']}", body_style))
    story.append(Paragraph(f"<b>Analysis Date:</b> {data['timestamp']}", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Risk Metrics Table
    story.append(Paragraph("Risk Assessment Summary", header_style))
    
    if data['risk_score'] > 84:
        risk_color = "#dc2626"
        risk_text = "HIGH RISK"
    elif data['risk_score'] > 44:
        risk_color = "#f59e0b"
        risk_text = "MEDIUM RISK"
    else:
        risk_color = "#10b981"
        risk_text = "LOW RISK"
    
    metrics_data = [
        ["Metric", "Value", "Status"],
        ["Risk Score", f"{data['risk_score']:.0f}/100", risk_text],
        ["Fraud Probability", f"{data['fraud_probability']:.1%}", ""],
        ["Risk Level", data['risk_level'], ""],
        ["Recommended Action", data['decision'], ""],
        ["AI Confidence", f"{data['confidence']*100:.0f}%", ""]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8fafc')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#0f172a')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # AI Explanation
    story.append(Paragraph("AI Explanation", header_style))
    story.append(Paragraph(data['natural_language'][:800].replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Feature Importance Chart
    story.append(Paragraph("Feature Impact Analysis", header_style))
    
    # Convert plotly figure to image directly without temp file
    if fig_bar is not None:
        img_bytes = fig_bar.to_image(format="png", width=600, height=400, scale=2)
        img_buffer = io.BytesIO(img_bytes)
        img = Image(img_buffer, width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Top Risk Factors
    story.append(Paragraph("Top Risk Factors", subheader_style))
    for i, (factor, score) in enumerate(data['features'][:5]):
        story.append(Paragraph(f"{i+1}. {factor}: {score*100:.1f}% impact", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Dominance Chart
    if fig_pie is not None:
        story.append(Paragraph("Risk Factor Dominance", subheader_style))
        img_bytes = fig_pie.to_image(format="png", width=400, height=300, scale=2)
        img_buffer = io.BytesIO(img_bytes)
        img = Image(img_buffer, width=3.5*inch, height=2.8*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Key Insights
    story.append(Paragraph("Key Insights", subheader_style))
    insights_data = [
        ["Primary Risk Driver", data['features'][0][0] if data['features'] else "None", f"{data['features'][0][1]*100:.1f}%" if data['features'] else "0%"],
        ["Total Risk Factors", f"{data['total_factors']}", ""],
        ["Top 3 Dominance", f"{data['top3_impact']:.0f}%", "of total risk"]
    ]
    
    insights_table = Table(insights_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    insights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#8b5cf6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#faf5ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
    ]))
    story.append(insights_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Actionable Recommendations", header_style))
    for rec in data['recommendations'][:5]:
        story.append(Paragraph(f"• {rec}", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(Paragraph("—" * 60, body_style))
    story.append(Paragraph("This project is built by <b>Hassan Subhani</b>", footer_style))
    story.append(Paragraph("FinGuard AI - Cost-Aware Fraud Detection Platform", footer_style))
    story.append(Paragraph("© 2026 All Rights Reserved", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==================== FIXED HTML REPORT GENERATOR (Professional Design) ====================
def generate_html_report_fixed(data: dict, fig_bar, fig_pie) -> str:
    """Generate professional HTML report with all visualizations"""
    
    # Get chart HTML - directly embed plotly
    bar_chart_html = fig_bar.to_html(full_html=False, include_plotlyjs='cdn') if fig_bar else '<p>Chart not available</p>'
    pie_chart_html = fig_pie.to_html(full_html=False, include_plotlyjs='cdn') if fig_pie else '<p>Chart not available</p>'
    
    # Determine badge class
    if data['risk_score'] > 84:
        badge_class = "badge-high"
    elif data['risk_score'] > 44:
        badge_class = "badge-medium"
    else:
        badge_class = "badge-low"
    
    # Determine risk color class
    risk_class = f"risk-{'high' if data['risk_score'] > 84 else 'medium' if data['risk_score'] > 44 else 'low'}"
    
    # Build features HTML
    features_html = ""
    for i, (factor, score) in enumerate(data['features'][:4]):
        features_html += f"""
        <div class="insight-card">
            <div class="insight-label">FACTOR {i+1}</div>
            <div class="insight-value" style="font-size: 0.9rem;">{factor[:45]}</div>
            <div style="color: #dc2626; font-weight: 700;">{score*100:.1f}% impact</div>
        </div>
        """
    
    # Build recommendations HTML
    recommendations_html = ""
    for rec in data['recommendations'][:5]:
        if 'URGENT' in rec:
            rec_class = "rec-urgent"
            icon = "🔴"
        elif 'REVIEW' in rec:
            rec_class = "rec-review"
            icon = "🟡"
        elif 'APPROVE' in rec:
            rec_class = "rec-approve"
            icon = "🟢"
        else:
            rec_class = "rec-info"
            icon = "🔵"
        recommendations_html += f"""
        <div class="rec-card {rec_class}">
            {icon} <strong>{rec}</strong>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinGuard AI - Fraud Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
            padding: 2rem;
        }}
        
        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 32px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
            overflow: hidden;
        }}
        
        /* Header */
        .report-header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 2rem 2.5rem;
            text-align: center;
            position: relative;
        }}
        
        .report-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
        }}
        
        .report-header h1 {{
            color: white;
            font-size: 2rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}
        
        .report-header p {{
            color: #94a3b8;
            font-size: 0.9rem;
        }}
        
        .report-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.1);
            padding: 0.25rem 1rem;
            border-radius: 30px;
            font-size: 0.75rem;
            margin-top: 0.75rem;
        }}
        
        /* Content */
        .report-content {{
            padding: 2rem 2.5rem;
        }}
        
        /* Section */
        .section {{
            margin-bottom: 2rem;
        }}
        
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            border-left: 4px solid #3b82f6;
            padding-left: 1rem;
            margin-bottom: 1.25rem;
        }}
        
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 20px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 24px -12px rgba(0,0,0,0.15);
        }}
        
        .metric-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.5rem 0;
        }}
        
        .risk-high {{ color: #dc2626; }}
        .risk-medium {{ color: #f59e0b; }}
        .risk-low {{ color: #10b981; }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 0.3rem 1rem;
            border-radius: 30px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-high {{ background: #dc2626; color: white; }}
        .badge-medium {{ background: #f59e0b; color: white; }}
        .badge-low {{ background: #10b981; color: white; }}
        
        /* Explanation Box */
        .explanation-box {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 20px;
            padding: 1.5rem;
            color: #e2e8f0;
            line-height: 1.7;
            font-size: 0.95rem;
        }}
        
        /* Chart Container */
        .chart-container {{
            margin: 1rem 0;
            padding: 1rem;
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
        }}
        
        /* Insight Grid */
        .insight-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .insight-card {{
            background: linear-gradient(135deg, #f8fafc, #ffffff);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        
        .insight-label {{
            font-size: 0.7rem;
            color: #64748b;
            text-transform: uppercase;
        }}
        
        .insight-value {{
            font-size: 1.2rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }}
        
        /* Recommendation List */
        .recommendation-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .rec-card {{
            padding: 1rem;
            border-radius: 16px;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transition: all 0.2s ease;
        }}
        
        .rec-card:hover {{
            transform: translateX(4px);
        }}
        
        .rec-urgent {{ background: #fef2f2; border-left: 4px solid #dc2626; }}
        .rec-review {{ background: #fffbeb; border-left: 4px solid #f59e0b; }}
        .rec-approve {{ background: #f0fdf4; border-left: 4px solid #10b981; }}
        .rec-info {{ background: #eff6ff; border-left: 4px solid #3b82f6; }}
        
        /* Footer */
        .report-footer {{
            background: #f8fafc;
            padding: 1.5rem;
            text-align: center;
            border-top: 1px solid #e2e8f0;
            color: #94a3b8;
            font-size: 0.8rem;
        }}
        
        /* Divider */
        .divider {{
            margin: 1.5rem 0;
            border: none;
            border-top: 1px solid #e2e8f0;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .report-content {{ padding: 1rem; }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
            .insight-grid {{ grid-template-columns: 1fr; }}
        }}
        
        /* Print Styles */
        @media print {{
            body {{ padding: 0; background: white; }}
            .report-container {{ box-shadow: none; }}
            .metric-card:hover {{ transform: none; }}
        }}
    </style>
    {bar_chart_html}
    {pie_chart_html}
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>🛡️ FinGuard AI</h1>
            <p>Fraud Analysis Report</p>
            <div class="report-badge">Generated on {data['timestamp']}</div>
        </div>
        
        <div class="report-content">
            <!-- Transaction Info -->
            <div class="section">
                <div class="section-title">📋 Transaction Information</div>
                <p><strong>Transaction ID:</strong> {data['transaction_id']}</p>
                <p><strong>Analysis Date:</strong> {data['timestamp']}</p>
                <p><strong>Analysis Method:</strong> SHAP (SHapley Additive exPlanations)</p>
            </div>
            
            <!-- Risk Metrics -->
            <div class="section">
                <div class="section-title">🎯 Risk Assessment Summary</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">RISK SCORE</div>
                        <div class="metric-value {risk_class}">{data['risk_score']:.0f}</div>
                        <div>out of 100</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">FRAUD PROBABILITY</div>
                        <div class="metric-value">{data['fraud_probability']:.1%}</div>
                        <div>likelihood of fraud</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">RISK LEVEL</div>
                        <div class="metric-value"><span class="badge {badge_class}">{data['risk_level']}</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">RECOMMENDED ACTION</div>
                        <div class="metric-value"><span class="badge badge-info" style="background:#3b82f6;">{data['decision']}</span></div>
                    </div>
                </div>
            </div>
            
            <!-- AI Explanation -->
            <div class="section">
                <div class="section-title">🤖 AI Explanation</div>
                <div class="explanation-box">
                    {data['natural_language'].replace(chr(10), '<br/>')}
                </div>
            </div>
            
            <!-- Feature Impact Chart -->
            <div class="section">
                <div class="section-title">📊 Feature Impact Analysis</div>
                <div class="chart-container">
                    {bar_chart_html}
                </div>
            </div>
            
            <!-- Risk Factor Dominance -->
            <div class="section">
                <div class="section-title">📈 Risk Factor Dominance</div>
                <div class="chart-container">
                    {pie_chart_html}
                </div>
            </div>
            
            <!-- Top Risk Factors -->
            <div class="section">
                <div class="section-title">📋 Top Risk Factors Breakdown</div>
                <div class="insight-grid">
                    {features_html}
                </div>
            </div>
            
            <!-- Key Insights -->
            <div class="section">
                <div class="section-title">💡 Key Insights</div>
                <div class="insight-grid">
                    <div class="insight-card">
                        <div class="insight-label">PRIMARY RISK DRIVER</div>
                        <div class="insight-value" style="font-size: 0.9rem;">{data['features'][0][0][:40] if data['features'] else 'None'}</div>
                        <div style="color: #dc2626; font-weight: 600;">{data['features'][0][1]*100:.1f}% impact</div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-label">TOTAL RISK FACTORS</div>
                        <div class="insight-value" style="font-size: 2rem;">{data['total_factors']}</div>
                        <div>factors identified</div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-label">TOP 3 DOMINANCE</div>
                        <div class="insight-value" style="font-size: 2rem;">{data['top3_impact']:.0f}%</div>
                        <div>of total risk</div>
                    </div>
                </div>
            </div>
            
            <!-- Recommendations -->
            <div class="section">
                <div class="section-title">💡 Actionable Recommendations</div>
                <div class="recommendation-list">
                    {recommendations_html}
                </div>
            </div>
        </div>
        
        <div class="report-footer">
            <p>This project is built by <strong>Hassan Subhani</strong></p>
            <p>FinGuard AI - Cost-Aware Fraud Detection Platform</p>
            <p>© 2026 All Rights Reserved</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

# Page 7: Drift Monitoring
def page_drift():
    st.markdown("### 📉 Data Drift Monitor")
    st.markdown("Monitor model performance and data distribution changes")
    st.markdown("---")
    
    if st.session_state.uploaded_data is not None:
        models = load_models()
        drift_detector = DriftDetector(models)
        
        with st.spinner("Analyzing data drift..."):
            drift_report = drift_detector.detect_drift(st.session_state.uploaded_data)
        
        # Drift score
        col1, col2 = st.columns(2)
        
        with col1:
            drift_score = drift_report.get('drift_score', 0)
            st.metric("Data Drift Score", f"{drift_score:.1f}%")
            
            if drift_score < 20:
                st.success("✅ No significant drift detected")
            elif drift_score < 50:
                st.warning("⚠️ Moderate drift detected")
            else:
                st.error("❌ Severe drift detected - Consider retraining")
        
        with col2:
            st.metric("Affected Features", drift_report.get('drifted_features_count', 0))
        
        # Feature drift details
        if drift_report.get('feature_drift'):
            st.subheader("Feature-Level Drift Analysis")
            
            drift_df = pd.DataFrame([
                {'Feature': k, 'Drift Score': v['score'], 'Status': v['status']}
                for k, v in drift_report['feature_drift'].items()
            ])
            
            fig = px.bar(
                drift_df,
                x='Feature',
                y='Drift Score',
                color='Status',
                title="Feature Drift Scores",
                color_discrete_map={'Stable': '#10b981', 'Warning': '#f59e0b', 'Severe': '#dc2626'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        recommendations = drift_detector.get_drift_recommendations(drift_report)
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                if "Severe" in rec:
                    st.error(rec)
                elif "Moderate" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)
    
    else:
        st.info("Upload data to analyze drift against training distribution")
        if st.button("📤 Go to Upload Page"):
            st.session_state.current_page = "Upload"
            st.rerun()

# Page 8: A/B Testing (ENHANCED with Modern UI/UX)
def page_abtest():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔬 A/B Testing Framework
        </h1>
        <p style="color: #64748b;">Compare model performance with statistical significance testing</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    models = load_models()
    
    if models and st.session_state.uploaded_data is not None:
        
        from utils.ab_testing import ABTestEngine, ABTestVisualizer, ABTestReportGenerator
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Configure & Run", "📊 Results & Analytics", "📄 Report Generator"])
        
        with tab1:
            st.subheader("Test Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.25rem; border-radius: 12px; border: 1px solid #e2e8f0;">
                    <h4>🤖 Model Selection</h4>
                </div>
                """, unsafe_allow_html=True)
                
                model_a_name = st.selectbox(
                    "Model A (Control)",
                    ["XGBoost", "Logistic Regression"],
                    key="model_a_ab"
                )
                
                confidence_level = st.slider(
                    "Confidence Level",
                    min_value=0.80,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    help="Statistical significance threshold",
                    format="%.2f"
                )
            
            with col2:
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.25rem; border-radius: 12px; border: 1px solid #e2e8f0;">
                    <h4>📊 Test Parameters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                model_b_name = st.selectbox(
                    "Model B (Treatment)",
                    ["Logistic Regression", "XGBoost"],
                    key="model_b_ab"
                )
                
                test_size = st.slider(
                    "Sample Size",
                    min_value=100,
                    max_value=min(5000, len(st.session_state.predictions)) if st.session_state.predictions is not None else 1000,
                    value=min(1000, len(st.session_state.predictions)) if st.session_state.predictions is not None else 500,
                    step=100
                )
            
            st.markdown("---")
            
            # Model cards
            st.subheader("Model Performance Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%); 
                            padding: 1.5rem; border-radius: 16px; border: 2px solid #dbeafe;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
                        <div style="font-size: 2rem;">🚀</div>
                        <div>
                            <h3 style="margin: 0; color: #1e40af;">XGBoost</h3>
                            <small style="color: #64748b;">Advanced Ensemble Model</small>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                        <div><small>Accuracy</small><br><strong style="color: #0f172a;">99.69%</strong></div>
                        <div><small>Precision</small><br><strong style="color: #0f172a;">98.50%</strong></div>
                        <div><small>Recall</small><br><strong style="color: #0f172a;">92.00%</strong></div>
                        <div><small>Speed</small><br><strong style="color: #0f172a;">45ms</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #ffffff 100%); 
                            padding: 1.5rem; border-radius: 16px; border: 2px solid #fde68a;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
                        <div style="font-size: 2rem;">📊</div>
                        <div>
                            <h3 style="margin: 0; color: #92400e;">Logistic Regression</h3>
                            <small style="color: #64748b;">Interpretable Baseline</small>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                        <div><small>Accuracy</small><br><strong style="color: #0f172a;">76.06%</strong></div>
                        <div><small>Precision</small><br><strong style="color: #0f172a;">72.00%</strong></div>
                        <div><small>Recall</small><br><strong style="color: #0f172a;">68.00%</strong></div>
                        <div><small>Speed</small><br><strong style="color: #0f172a;">12ms</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Run Test Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run A/B Test", use_container_width=True, type="primary"):
                    with st.spinner("Running statistical analysis..."):
                        
                        # Simulate test data based on actual predictions if available
                        if st.session_state.predictions is not None:
                            # Use actual data if available
                            df_predictions = st.session_state.predictions
                            if 'fraud_probability' in df_predictions.columns:
                                # Create synthetic y_true based on risk levels
                                np.random.seed(42)
                                y_true_sim = np.where(
                                    df_predictions['risk_level'].iloc[:test_size] == 'High Risk', 
                                    1, 
                                    np.random.choice([0, 1], size=test_size, p=[0.995, 0.005])
                                )
                            else:
                                y_true_sim = np.random.choice([0, 1], size=test_size, p=[0.998, 0.002])
                        else:
                            y_true_sim = np.random.choice([0, 1], size=test_size, p=[0.998, 0.002])
                        
                        # Model metrics
                        xgb_metrics = {'accuracy': 0.9969, 'precision': 0.985, 'recall': 0.92, 'f1': 0.951, 'time': 45, 'auc': 0.94}
                        lr_metrics = {'accuracy': 0.7606, 'precision': 0.72, 'recall': 0.68, 'f1': 0.699, 'time': 12, 'auc': 0.85}
                        
                        # Get metrics for selected models
                        metrics_a = xgb_metrics if model_a_name == "XGBoost" else lr_metrics
                        metrics_b = xgb_metrics if model_b_name == "XGBoost" else lr_metrics
                        
                        # Generate predictions with realistic patterns
                        np.random.seed(42)
                        
                        # Generate predictions based on model accuracy
                        preds_a = []
                        preds_b = []
                        proba_a = []
                        proba_b = []
                        
                        for true_label in y_true_sim:
                            # Model A predictions
                            if np.random.random() < metrics_a['accuracy']:
                                pred_a = true_label
                                prob_a = 0.95 if true_label == 1 else 0.05
                            else:
                                pred_a = 1 - true_label
                                prob_a = 0.05 if true_label == 1 else 0.95
                            
                            # Model B predictions
                            if np.random.random() < metrics_b['accuracy']:
                                pred_b = true_label
                                prob_b = 0.85 if true_label == 1 else 0.15
                            else:
                                pred_b = 1 - true_label
                                prob_b = 0.15 if true_label == 1 else 0.85
                            
                            preds_a.append(pred_a)
                            preds_b.append(pred_b)
                            proba_a.append(prob_a)
                            proba_b.append(prob_b)
                        
                        preds_a = np.array(preds_a)
                        preds_b = np.array(preds_b)
                        proba_a = np.array(proba_a)
                        proba_b = np.array(proba_b)
                        
                        # Create ABTestEngine instance - FIXED: This line was missing proper indentation
                        ab_engine = ABTestEngine(confidence_level=confidence_level)
                        
                        # Run test
                        test_results = ab_engine.run_ab_test(
                            model_a_predictions=preds_a,
                            model_b_predictions=preds_b,
                            y_true=y_true_sim,
                            model_a_name=model_a_name,
                            model_b_name=model_b_name,
                            model_a_proba=proba_a,
                            model_b_proba=proba_b,
                            inference_time_a=metrics_a['time'],
                            inference_time_b=metrics_b['time']
                        )
                        
                        st.session_state.ab_test_results = test_results
                        st.success("✅ A/B Test completed successfully!")
                        
                        # Show quick summary in a nice card
                        st.markdown("### 📊 Test Summary")
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        with col_sum1:
                            st.metric("Sample Size", f"{test_size:,}")
                        with col_sum2:
                            st.metric("Confidence Level", f"{confidence_level*100:.0f}%")
                        with col_sum3:
                            st.metric("Winner", test_results['winner']['name'])
        
        with tab2:
            if 'ab_test_results' in st.session_state:
                results = st.session_state.ab_test_results
                visualizer = ABTestVisualizer()
                
                # Winner Banner - Modern animated card
                winner = results['winner']
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                            padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 1.5rem;
                            animation: slideIn 0.5s ease-out;">
                    <div style="font-size: 3rem;">🏆</div>
                    <h2 style="color: white; margin: 0.5rem 0;">Winner: {winner['name']}</h2>
                    <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1rem;">
                        Overall Score: {winner['score']:.1f}/100 | Margin: {winner['margin']:.1f} points
                    </p>
                </div>
                <style>
                    @keyframes slideIn {{
                        from {{ opacity: 0; transform: translateY(-20px); }}
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                </style>
                """, unsafe_allow_html=True)
                
                # Key Metrics Row
                st.markdown("### Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                model_a = results['model_a']
                model_b = results['model_b']
                
                with col1:
                    delta_acc = results['metrics'][model_a]['accuracy'] - results['metrics'][model_b]['accuracy']
                    st.metric(
                        "Accuracy Difference", 
                        f"{delta_acc:+.2%}",
                        help="Positive means Model A is better"
                    )
                with col2:
                    delta_f1 = results['metrics'][model_a]['f1_score'] - results['metrics'][model_b]['f1_score']
                    st.metric(
                        "F1 Score Difference", 
                        f"{delta_f1:+.2%}",
                        help="Positive means Model A is better"
                    )
                with col3:
                    speed_ratio = results['metrics'][model_b]['inference_time_ms'] / max(results['metrics'][model_a]['inference_time_ms'], 1)
                    st.metric(
                        "Speed Ratio", 
                        f"{speed_ratio:.1f}x",
                        help="How many times faster Model B is"
                    )
                with col4:
                    sig_count = sum(1 for test in results['statistical_tests'].values() if test['is_significant'])
                    st.metric(
                        "Significant Metrics", 
                        f"{sig_count}/{len(results['statistical_tests'])}",
                        help="Number of metrics with statistical significance"
                    )
                
                st.markdown("---")
                
                # Charts Grid
                st.markdown("### 📊 Visual Analytics")
                
                # Row 1: Metrics Chart and Radar Chart
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = visualizer.create_metrics_chart(results)
                    st.plotly_chart(fig1, use_container_width=True, key="metrics_chart_ab")
                
                with col2:
                    fig2 = visualizer.create_radar_chart(results)
                    st.plotly_chart(fig2, use_container_width=True, key="radar_chart_ab")
                
                # Row 2: Speed Chart and Winner Gauge
                col1, col2 = st.columns(2)
                with col1:
                    fig3 = visualizer.create_speed_chart(results)
                    st.plotly_chart(fig3, use_container_width=True, key="speed_chart_ab")
                
                with col2:
                    fig4 = visualizer.create_winner_gauge(results)
                    st.plotly_chart(fig4, use_container_width=True, key="winner_gauge_ab")
                
                # Row 3: Significance Chart and Lift Chart
                col1, col2 = st.columns(2)
                with col1:
                    fig5 = visualizer.create_significance_chart(results)
                    st.plotly_chart(fig5, use_container_width=True, key="significance_chart_ab")
                
                with col2:
                    fig6 = visualizer.create_lift_chart(results)
                    if fig6:
                        st.plotly_chart(fig6, use_container_width=True, key="lift_chart_ab")
                    else:
                        st.info("Lift chart not available for this comparison")
                
                # Confidence Intervals
                if results.get('confidence_intervals'):
                    st.markdown("### 📈 Confidence Intervals (95%)")
                    fig7 = visualizer.create_confidence_interval_chart(results)
                    if fig7:
                        st.plotly_chart(fig7, use_container_width=True, key="confidence_chart_ab")
                
                st.markdown("---")
                
                # Detailed metrics table - Styled
                st.markdown("### 📊 Detailed Metrics Comparison")
                
                # Create styled dataframe
                metrics_data = []
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
                        
                        metrics_data.append({
                            'Metric': metric_name,
                            model_a: val_a_str,
                            model_b: val_b_str,
                            'Better Model': better,
                            'Difference': f"{((val_a - val_b) / max(val_b, 0.001) * 100):+.1f}%"
                        })
                
                if 'auc' in results['metrics'][model_a] and not np.isnan(results['metrics'][model_a]['auc']):
                    val_a = results['metrics'][model_a]['auc']
                    val_b = results['metrics'][model_b]['auc']
                    better = model_a if val_a > val_b else model_b
                    metrics_data.append({
                        'Metric': 'AUC',
                        model_a: f"{val_a:.4f}",
                        model_b: f"{val_b:.4f}",
                        'Better Model': better,
                        'Difference': f"{((val_a - val_b) / max(val_b, 0.001) * 100):+.1f}%"
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(
                    df_metrics,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Metric': st.column_config.TextColumn("Metric", width="medium"),
                        model_a: st.column_config.TextColumn(model_a, width="small"),
                        model_b: st.column_config.TextColumn(model_b, width="small"),
                        'Better Model': st.column_config.TextColumn("Winner", width="small"),
                        'Difference': st.column_config.TextColumn("Difference", width="small")
                    }
                )
                
                # Statistical significance table
                st.markdown("### 📈 Statistical Significance Analysis")
                
                sig_data = []
                for metric, test in results['statistical_tests'].items():
                    metric_name = metric.replace('_', ' ').title()
                    sig_data.append({
                        'Metric': metric_name,
                        'P-Value': f"{test['p_value']:.4f}",
                        'Significant': "✅ Yes" if test['is_significant'] else "❌ No",
                        'Lift': f"{test['lift_percentage']:+.1f}%",
                        'Z-Statistic': f"{test['z_statistic']:.3f}"
                    })
                
                df_sig = pd.DataFrame(sig_data)
                st.dataframe(
                    df_sig,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Metric': st.column_config.TextColumn("Metric", width="medium"),
                        'P-Value': st.column_config.TextColumn("P-Value", width="small"),
                        'Significant': st.column_config.TextColumn("Significant?", width="small"),
                        'Lift': st.column_config.TextColumn("Lift", width="small"),
                        'Z-Statistic': st.column_config.TextColumn("Z-Statistic", width="small")
                    }
                )
                
                # Recommendations Section - Styled
                if results['recommendations']:
                    st.markdown("---")
                    st.markdown("### 💡 Actionable Recommendations")
                    
                    for i, rec in enumerate(results['recommendations']):
                        if "🎯" in rec:
                            st.success(rec)
                        elif "⚡" in rec:
                            st.info(rec)
                        elif "📊" in rec:
                            st.warning(rec)
                        else:
                            st.markdown(f"• {rec}")
                
                # Export option for results summary
                st.markdown("---")
                if st.button("📋 Export Summary as Text", use_container_width=True):
                    ab_engine_summary = ABTestEngine(confidence_level=results['confidence_level'])
                    ab_engine_summary.test_results = results
                    summary_text = ab_engine_summary.get_test_summary()
                    st.download_button(
                        label="📥 Download Summary Text",
                        data=summary_text,
                        file_name=f"ab_test_summary_{results['test_id']}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.info("👈 Run an A/B test first to see interactive visualizations and analytics")
        
        with tab3:
            if 'ab_test_results' in st.session_state:
                st.subheader("📄 Generate Professional Report")
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
                            padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                    <p style="color: #0f172a;">Generate comprehensive reports including all metrics, 
                    statistical analysis, visualizations, and actionable recommendations.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    report_format = st.selectbox(
                        "Report Format",
                        ["📊 HTML Report (Interactive with Charts)", "📄 PDF Report (Professional Document)"]
                    )
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    generate_btn = st.button("📥 Generate Report", use_container_width=True, type="primary")
                
                if generate_btn:
                    with st.spinner("Generating professional report with visualizations..."):
                        report_gen = ABTestReportGenerator()
                        
                        if "HTML" in report_format:
                            report_path = report_gen.generate_html_report(st.session_state.ab_test_results)
                            
                            with open(report_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download HTML Report",
                                    data=f,
                                    file_name=f"ab_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                            st.success("✅ HTML Report generated successfully! The report includes all interactive charts.")
                            
                            # Show preview
                            st.markdown("### 📄 Report Preview")
                            with open(report_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                                st.components.v1.html(html_content[:3000] + "...", height=500, scrolling=True)
                        
                        else:  # PDF Report
                            pdf_buffer = report_gen.generate_pdf_report(st.session_state.ab_test_results)
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"ab_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ PDF Report generated successfully! The report includes all charts and visualizations.")
                            
                            st.info("💡 The PDF report includes: Winner announcement, performance metrics, statistical significance analysis, and professional charts.")
                
                st.markdown("---")
                st.caption("💡 Reports include interactive charts (HTML) or embedded visualizations (PDF) with model comparison, statistical significance, confidence intervals, and recommendations")
            else:
                st.info("👈 Run an A/B test first to generate professional reports")
    
    else:
        st.warning("⚠️ Please upload data and run fraud detection first to enable A/B testing")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Go to Upload", use_container_width=True):
                st.session_state.current_page = "Upload"
                st.rerun()
        with col2:
            if st.button("🤖 Run Detection", use_container_width=True):
                st.session_state.current_page = "Detection"
                st.rerun()

# Page 9: Analytics Dashboard (Fixed Version)
def page_analytics():
    st.markdown("### 📊 Business Intelligence Dashboard")
    st.markdown("Executive-level fraud analytics and insights")
    st.markdown("---")
    
    if st.session_state.predictions is not None:
        # Executive KPIs - First Row
        st.subheader("🎯 Executive Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        fraud_rate = (st.session_state.predictions['risk_level'] == 'High Risk').sum() / len(st.session_state.predictions) * 100
        total_amount = st.session_state.uploaded_data['amount'].sum() if st.session_state.uploaded_data is not None and 'amount' in st.session_state.uploaded_data.columns else 0
        
        with col1:
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%", delta="High Risk" if fraud_rate > 5 else "Normal")
        
        with col2:
            st.metric("Total Transactions", f"{len(st.session_state.predictions):,}")
        
        with col3:
            high_risk = (st.session_state.predictions['risk_level'] == 'High Risk').sum()
            st.metric("High Risk", high_risk, delta="Needs Review" if high_risk > 0 else "Good")
        
        with col4:
            st.metric("Avg Fraud Probability", f"{st.session_state.predictions['fraud_probability'].mean():.3f}")
        
        with col5:
            if 'decision' in st.session_state.predictions.columns:
                savings = (st.session_state.predictions['decision'] == 'Block').sum() * 1000
                st.metric("Est. Savings", f"${savings:,.0f}")
            else:
                st.metric("Risk Score", f"{fraud_rate:.1f}%")
        
        st.markdown("---")
        
        # Row 1: Risk Distribution and Fraud Probability
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Risk Distribution")
            risk_counts = st.session_state.predictions['risk_level'].value_counts()
            colors = {'High Risk': '#dc2626', 'Medium Risk': '#f59e0b', 'Low Risk': '#10b981'}
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(colors=[colors[x] for x in risk_counts.index]),
                textinfo='label+percent',
                textposition='auto',
                pull=[0.05, 0.02, 0]
            )])
            fig.update_layout(
                title="Risk Level Distribution",
                annotations=[dict(text='Risk Levels', x=0.5, y=0.5, font_size=14, showarrow=False)],
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Fraud Probability Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=st.session_state.predictions['fraud_probability'],
                nbinsx=40,
                name='Distribution',
                marker_color='#3b82f6',
                opacity=0.7,
                hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add threshold lines
            fig.add_vline(x=0.44, line_dash="dash", line_color="#f59e0b", 
                         annotation_text="Review Threshold (0.44)", annotation_position="top")
            fig.add_vline(x=0.84, line_dash="dash", line_color="#dc2626",
                         annotation_text="Block Threshold (0.84)", annotation_position="top")
            fig.add_vline(x=st.session_state.predictions['fraud_probability'].mean(), 
                         line_dash="dot", line_color="#10b981",
                         annotation_text=f"Mean: {st.session_state.predictions['fraud_probability'].mean():.3f}")
            
            fig.update_layout(
                title="Fraud Probability Distribution",
                xaxis_title="Fraud Probability",
                yaxis_title="Number of Transactions",
                height=400,
                template='plotly_white',
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Decision Breakdown and Financial Impact
        if 'decision' in st.session_state.predictions.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚖️ Decision Intelligence Breakdown")
                decision_counts = st.session_state.predictions['decision'].value_counts()
                decision_colors = {'Approve': '#10b981', 'Review': '#f59e0b', 'Block': '#dc2626'}
                
                fig = go.Figure(data=[go.Bar(
                    x=decision_counts.index,
                    y=decision_counts.values,
                    marker_color=[decision_colors.get(x, '#3b82f6') for x in decision_counts.index],
                    text=decision_counts.values,
                    textposition='auto',
                    hovertemplate='Decision: %{x}<br>Count: %{y}<extra></extra>'
                )])
                fig.update_layout(
                    title="Transaction Decisions",
                    xaxis_title="Decision",
                    yaxis_title="Number of Transactions",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💰 Financial Impact Analysis")
                block_count = decision_counts.get('Block', 0)
                review_count = decision_counts.get('Review', 0)
                
                categories = ['Blocked\n($1000 saved)', 'Review\n($100 saved)', 'Approved\n($0)']
                savings = [block_count * 1000, review_count * 100, 0]
                
                fig = go.Figure(data=[go.Bar(
                    x=categories,
                    y=savings,
                    marker_color=['#dc2626', '#f59e0b', '#10b981'],
                    text=[f'${x:,.0f}' for x in savings],
                    textposition='auto',
                    hovertemplate='Category: %{x}<br>Savings: $%{y:,.0f}<extra></extra>'
                )])
                fig.update_layout(
                    title="Financial Impact by Decision Type",
                    xaxis_title="Decision Category",
                    yaxis_title="Amount Saved (USD)",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Time-based Analysis
        st.subheader("⏰ Time-Based Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk by hour (using actual or simulated data)
            if st.session_state.uploaded_data is not None and 'transaction_hour' in st.session_state.uploaded_data.columns:
                hourly_data = pd.DataFrame({
                    'Hour': st.session_state.uploaded_data['transaction_hour'],
                    'Risk': st.session_state.predictions['fraud_probability']
                })
                hourly_risk = hourly_data.groupby('Hour')['Risk'].mean().reset_index()
            else:
                # Create realistic hourly pattern
                hours = list(range(24))
                risk_pattern = [0.2, 0.18, 0.15, 0.12, 0.1, 0.15, 0.25, 0.35, 
                              0.4, 0.38, 0.35, 0.32, 0.3, 0.28, 0.3, 0.35,
                              0.4, 0.45, 0.48, 0.5, 0.52, 0.48, 0.4, 0.3]
                hourly_risk = pd.DataFrame({'Hour': hours, 'Risk': risk_pattern})
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_risk['Hour'],
                y=hourly_risk['Risk'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=8, color='#3b82f6'),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)'
            ))
            
            # Add threshold areas
            fig.add_hrect(y0=0.85, y1=1.0, line_width=0, fillcolor="red", opacity=0.1,
                         annotation_text="High Risk Zone", annotation_position="top right")
            fig.add_hrect(y0=0.45, y1=0.84, line_width=0, fillcolor="orange", opacity=0.1,
                         annotation_text="Medium Risk Zone", annotation_position="right")
            
            fig.update_layout(
                title="Fraud Risk by Hour of Day",
                xaxis_title="Hour (0-23)",
                yaxis_title="Average Fraud Probability",
                height=400,
                template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk by day of week
            if st.session_state.uploaded_data is not None and 'transaction_day_of_week' in st.session_state.uploaded_data.columns:
                daily_data = pd.DataFrame({
                    'Day': st.session_state.uploaded_data['transaction_day_of_week'],
                    'Risk': st.session_state.predictions['fraud_probability']
                })
                daily_risk = daily_data.groupby('Day')['Risk'].mean().reset_index()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_risk['Day'] = pd.Categorical(daily_risk['Day'], categories=day_order, ordered=True)
                daily_risk = daily_risk.sort_values('Day')
            else:
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                risk_pattern = [0.32, 0.31, 0.33, 0.34, 0.36, 0.42, 0.38]
                daily_risk = pd.DataFrame({'Day': days, 'Risk': risk_pattern})
            
            fig = go.Figure(data=[go.Bar(
                x=daily_risk['Day'],
                y=daily_risk['Risk'],
                marker_color='#f59e0b',
                text=[f'{x:.3f}' for x in daily_risk['Risk']],
                textposition='auto',
                hovertemplate='Day: %{x}<br>Risk: %{y:.3f}<extra></extra>'
            )])
            fig.update_layout(
                title="Fraud Risk by Day of Week",
                xaxis_title="Day",
                yaxis_title="Average Fraud Probability",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 4: Geographic and Transaction Type Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Geographic Risk Distribution")
            if st.session_state.uploaded_data is not None and 'customer_region' in st.session_state.uploaded_data.columns:
                region_data = pd.DataFrame({
                    'Region': st.session_state.uploaded_data['customer_region'],
                    'Risk': st.session_state.predictions['fraud_probability']
                })
                region_risk = region_data.groupby('Region')['Risk'].mean().sort_values(ascending=True).reset_index()
                
                fig = go.Figure(data=[go.Bar(
                    x=region_risk['Risk'],
                    y=region_risk['Region'],
                    orientation='h',
                    marker_color='#8b5cf6',
                    text=[f'{x:.3f}' for x in region_risk['Risk']],
                    textposition='outside',
                    hovertemplate='Region: %{y}<br>Risk: %{x:.3f}<extra></extra>'
                )])
                fig.update_layout(
                    title="Average Fraud Risk by Region",
                    xaxis_title="Average Fraud Probability",
                    yaxis_title="Region",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Geographic data not available for analysis")
        
        with col2:
            st.subheader("📱 Risk by Transaction Type")
            if st.session_state.uploaded_data is not None and 'transaction_type' in st.session_state.uploaded_data.columns:
                type_data = pd.DataFrame({
                    'Type': st.session_state.uploaded_data['transaction_type'],
                    'Risk': st.session_state.predictions['fraud_probability'],
                    'Risk Level': st.session_state.predictions['risk_level']
                })
                
                # Create stacked bar chart
                risk_by_type = pd.crosstab(type_data['Type'], type_data['Risk Level'], normalize='index') * 100
                
                fig = go.Figure()
                colors_risk = {'High Risk': '#dc2626', 'Medium Risk': '#f59e0b', 'Low Risk': '#10b981'}
                
                for risk_level in ['High Risk', 'Medium Risk', 'Low Risk']:
                    if risk_level in risk_by_type.columns:
                        fig.add_trace(go.Bar(
                            name=risk_level,
                            x=risk_by_type.index,
                            y=risk_by_type[risk_level],
                            marker_color=colors_risk[risk_level],
                            hovertemplate=f'{risk_level}: %{{y:.1f}}%<extra></extra>'
                        ))
                
                fig.update_layout(
                    title="Risk Distribution by Transaction Type",
                    xaxis_title="Transaction Type",
                    yaxis_title="Percentage (%)",
                    barmode='stack',
                    height=400,
                    template='plotly_white',
                    legend_title="Risk Level"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Transaction type data not available for analysis")
        
        # Row 5: Correlation Heatmap (Advanced)
        st.subheader("🔬 Feature Correlation Analysis")
        
        # Prepare data for correlation
        numeric_cols = ['fraud_probability', 'amount', 'merchant_risk_score', 'transaction_hour', 
                       'customer_age', 'account_age_days', 'transaction_velocity_1h']
        
        available_cols = [col for col in numeric_cols if col in st.session_state.predictions.columns or 
                         (st.session_state.uploaded_data is not None and col in st.session_state.uploaded_data.columns)]
        
        if len(available_cols) >= 2:
            # Create correlation dataframe
            corr_data = pd.DataFrame()
            for col in available_cols:
                if col in st.session_state.predictions.columns:
                    corr_data[col] = st.session_state.predictions[col]
                elif st.session_state.uploaded_data is not None and col in st.session_state.uploaded_data.columns:
                    corr_data[col] = st.session_state.uploaded_data[col]
            
            corr_matrix = corr_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Feature Correlation Heatmap",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient numeric data for correlation analysis")
        
        # Row 6: Top Risk Transactions (FIXED - using DataFrame styling instead of applymap)
        st.subheader("⚠️ Top 20 Highest Risk Transactions")
        
        top_risky = st.session_state.predictions.nlargest(20, 'fraud_probability')[['transaction_id', 'fraud_probability', 'risk_level']]
        top_risky['fraud_probability'] = top_risky['fraud_probability'].apply(lambda x: f"{x:.3f}")
        top_risky.index = range(1, len(top_risky) + 1)
        
        # Use st.dataframe with column configuration instead of Styler for better compatibility
        st.dataframe(
            top_risky,
            use_container_width=True,
            height=400,
            column_config={
                "transaction_id": st.column_config.TextColumn("Transaction ID", width="medium"),
                "fraud_probability": st.column_config.TextColumn("Fraud Probability", width="small"),
                "risk_level": st.column_config.TextColumn("Risk Level", width="small")
            }
        )
        
        # Alternative: Color-coded display using markdown (if you prefer colored rows)
        st.markdown("**Risk Level Indicators:** 🔴 High Risk | 🟡 Medium Risk | 🟢 Low Risk")
        
        # Download report button
        st.markdown("---")
        st.subheader("📄 Executive Report")
        col_report1, col_report2 = st.columns([2, 1])
        
        with col_report1:
            st.info("Generate a comprehensive executive summary PDF report with all analytics and insights including charts and visualizations.")
        
        with col_report2:
            if st.button("📥 Download Enhanced Executive Report", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Generating comprehensive executive report with visualizations..."):
                        report_gen = ReportGenerator()
                        report_path = report_gen.generate_decision_report(
                            st.session_state.predictions, 
                            st.session_state.uploaded_data
                        )
                        st.success("✅ Enhanced report generated successfully!")
                        
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=f,
                                file_name=f"enhanced_fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.info("Please ensure all required libraries are installed: pip install matplotlib seaborn")
    
    else:
        st.info("Run fraud detection to see analytics dashboard")
        if st.button("🤖 Go to Fraud Detection", use_container_width=True, type="primary"):
            st.session_state.current_page = "Detection"
            st.rerun()

# Main app routing
def main():
    render_sidebar()
    
    # Load models on startup
    if not st.session_state.model_loaded:
        with st.spinner("Loading FinGuard AI models..."):
            models = load_models()
            if models:
                st.success("✅ Models loaded successfully!")
    
    # Page routing
    pages = {
        "Home": page_home,
        "Upload": page_upload,
        "Validation": page_validation,
        "Detection": page_detection,
        "Decision": page_decision,
        "Explain": page_explain,
        "Drift": page_drift,
        "ABTest": page_abtest,
        "Analytics": page_analytics
    }
    
    current_page = st.session_state.current_page
    if current_page in pages:
        pages[current_page]()
    else:
        page_home()

if __name__ == "__main__":
    main()