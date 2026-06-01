# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data", "training")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
DASHBOARDS_DIR = os.path.join(BASE_DIR, "dashboards")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(DASHBOARDS_DIR, exist_ok=True)

# Training data file paths
TRAINING_DATA_WITH_FRAUD = os.path.join(DATA_DIR, "fraud_detection_dataset_with_fraud_column.csv")
TRAINING_DATA_WITHOUT_FRAUD = os.path.join(DATA_DIR, "fraud_detection_dataset_without_fraud.csv")

# Model artifact paths
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
LOGISTIC_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.pkl")
FEATURE_SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")
THRESHOLD_CONFIG_PATH = os.path.join(MODELS_DIR, "threshold_config.json")
DATA_HASH_PATH = os.path.join(MODELS_DIR, "data_hash.txt")

# Model hyperparameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5,  # Handle class imbalance
    'random_state': 42,
    'eval_metric': 'logloss',
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'n_jobs': -1
}

LOGISTIC_PARAMS = {
    'C': 0.1,
    'class_weight': 'balanced',
    'random_state': 42,
    'max_iter': 1000,
    'solver': 'lbfgs',
    'penalty': 'l2'
}

# Cost parameters for decision engine
COST_PARAMS = {
    'false_negative_cost': 1000,  # Cost of missing a fraudulent transaction
    'false_positive_cost': 50,     # Cost of blocking a legitimate transaction
    'review_cost': 10,              # Cost of manual review
    'fraud_loss_multiplier': 1.5    # Additional multiplier for fraud losses
}

# Default decision thresholds (will be optimized during training)
DEFAULT_THRESHOLDS = {
    'approve': 0.44,      # Transactions with probability <= 0.44 are approved
    'review_start': 0.45, # Transactions between 0.45 and 0.84 go for review
    'review_end': 0.84,   # Transactions above 0.84 are blocked
    'block': 1.0
}

# Feature categories
NUMERICAL_FEATURES = [
    'transaction_hour', 'amount', 'customer_age', 'account_age_days',
    'transaction_velocity_1h', 'transaction_velocity_24h',
    'avg_transaction_amount_7d', 'deviation_from_avg_amount',
    'merchant_risk_score', 'previous_fraud_count'
]

CATEGORICAL_FEATURES = [
    'transaction_day_of_week', 'transaction_type', 'customer_region',
    'income_level', 'device_type'
]

BINARY_FEATURES = [
    'is_international', 'is_new_device', 'is_night_transaction', 'high_amount_flag'
]

# All features for training
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Columns to drop during preprocessing
COLUMNS_TO_DROP = [
    'transaction_id', 'customer_id', 'transaction_timestamp', 
    'true_risk_score', 'fraud'
]

# Required columns for validation
REQUIRED_COLUMNS = [
    'transaction_id', 'amount', 'transaction_type', 'customer_region',
    'device_type', 'merchant_risk_score', 'transaction_hour',
    'transaction_day_of_week', 'customer_age', 'account_age_days',
    'transaction_velocity_1h', 'transaction_velocity_24h',
    'avg_transaction_amount_7d', 'deviation_from_avg_amount',
    'is_international', 'is_new_device', 'previous_fraud_count',
    'is_night_transaction', 'high_amount_flag'
]

# Expected data types for validation
EXPECTED_DATA_TYPES = {
    'transaction_id': 'object',
    'amount': 'float64',
    'transaction_type': 'object',
    'customer_region': 'object',
    'device_type': 'object',
    'merchant_risk_score': 'float64',
    'transaction_hour': 'int64',
    'customer_age': 'int64',
    'account_age_days': 'int64',
    'is_international': 'int64',
    'is_new_device': 'int64',
    'previous_fraud_count': 'int64',
    'is_night_transaction': 'int64',
    'high_amount_flag': 'int64'
}

# Risk level definitions
RISK_LEVELS = {
    'low': {'min': 0, 'max': 0.44, 'color': '#10b981', 'action': 'Approve'},
    'medium': {'min': 0.45, 'max': 0.84, 'color': '#f59e0b', 'action': 'Review'},
    'high': {'min': 0.85, 'max': 1.0, 'color': '#dc2626', 'action': 'Block'}
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'FinGuard AI - Fraud Detection Platform',
    'page_icon': '🛡️',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': 'dark',
    'refresh_interval': 60,  # seconds
    'max_display_rows': 1000,
    'chart_height': 400,
    'confidence_threshold': 0.5
}

# Report configuration
REPORT_CONFIG = {
    'company_name': 'FinGuard AI',
    'company_logo': None,
    'default_currency': 'USD',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'include_shap_plots': True,
    'include_cost_analysis': True,
    'max_transactions_in_report': 500
}

# Drift detection configuration
DRIFT_CONFIG = {
    'psi_threshold': 0.1,      # Population Stability Index threshold
    'ks_threshold': 0.05,       # Kolmogorov-Smirnov test threshold
    'correlation_threshold': 0.7, # Feature correlation threshold
    'window_size': 1000,         # Rolling window size for drift detection
    'alert_on_drift': True,      # Send alerts when drift detected
    'auto_retrain_threshold': 0.3 # Drift score that triggers retraining
}

# A/B Testing configuration
AB_TEST_CONFIG = {
    'models_to_compare': ['xgboost', 'logistic'],
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cost_saved'],
    'traffic_split': 50,  # Percentage for test model
    'minimum_samples': 1000,
    'confidence_level': 0.95,
    'test_duration_days': 7
}

# API configuration (for future use)
API_CONFIG = {
    'enable_api': False,
    'api_port': 8000,
    'api_key_required': False,
    'rate_limit': 100,  # requests per minute
    'cors_origins': ['http://localhost:8501']
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': os.path.join(BASE_DIR, 'logs', 'finguard.log'),
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}

# Create logs directory
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# Application version
APP_VERSION = '1.0.0'
APP_NAME = 'FinGuard AI'
APP_DESCRIPTION = 'Cost-Aware Fraud Detection & Decision Intelligence Platform'

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.75,
    'min_precision': 0.70,
    'min_recall': 0.65,
    'min_roc_auc': 0.80,
    'max_inference_time_ms': 100
}

# Supported file formats for upload
SUPPORTED_FORMATS = {
    'csv': ['csv'],
    'excel': ['xlsx', 'xls'],
    'json': ['json'],
    'sql': ['db', 'sqlite']
}

# Maximum file size (in MB)
MAX_FILE_SIZE_MB = 100

# Sample data configuration
SAMPLE_DATA_CONFIG = {
    'generate_on_startup': False,
    'num_samples': 1000,
    'fraud_ratio': 0.01,  # 1% fraud rate
    'random_seed': 42
}

# Export configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'excel', 'json', 'pdf'],
    'compression': False,
    'include_metadata': True,
    'max_export_rows': 10000
}

def get_model_info():
    """Get information about trained models"""
    return {
        'primary_model': 'XGBoost',
        'baseline_model': 'Logistic Regression',
        'features_count': len(ALL_FEATURES),
        'version': APP_VERSION,
        'training_date': None,  # Will be set during training
        'last_updated': None
    }

def get_system_status():
    """Get current system status"""
    return {
        'models_loaded': os.path.exists(XGB_MODEL_PATH),
        'artifacts_ready': all([
            os.path.exists(XGB_MODEL_PATH),
            os.path.exists(SCALER_PATH),
            os.path.exists(ENCODER_PATH),
            os.path.exists(FEATURE_SCHEMA_PATH)
        ]),
        'training_data_available': all([
            os.path.exists(TRAINING_DATA_WITH_FRAUD),
            os.path.exists(TRAINING_DATA_WITHOUT_FRAUD)
        ]),
        'reports_directory': os.path.exists(REPORTS_DIR),
        'version': APP_VERSION
    }

# Utility function to validate paths
def validate_paths():
    """Validate all required paths exist"""
    paths = {
        'Data Directory': DATA_DIR,
        'Models Directory': MODELS_DIR,
        'Reports Directory': REPORTS_DIR,
        'Training Data (with fraud)': TRAINING_DATA_WITH_FRAUD,
        'Training Data (without fraud)': TRAINING_DATA_WITHOUT_FRAUD
    }
    
    status = {}
    for name, path in paths.items():
        status[name] = os.path.exists(path)
    
    return status

# Print configuration summary (for debugging)
if __name__ == "__main__":
    print("=" * 60)
    print(f"{APP_NAME} - Configuration Summary")
    print("=" * 60)
    print(f"Version: {APP_VERSION}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Reports Directory: {REPORTS_DIR}")
    print(f"Features Count: {len(ALL_FEATURES)}")
    print(f"Numerical Features: {len(NUMERICAL_FEATURES)}")
    print(f"Categorical Features: {len(CATEGORICAL_FEATURES)}")
    print(f"Binary Features: {len(BINARY_FEATURES)}")
    print("-" * 60)
    print("System Status:")
    for key, value in get_system_status().items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    print("=" * 60)