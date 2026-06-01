# utils/helpers.py
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, Any, List, Optional
import requests
import io
import zipfile
import os
import tempfile
import re

# ==================== EXISTING FUNCTIONS (Keep as is) ====================

def calculate_data_hash(df: pd.DataFrame) -> str:
    """Calculate hash of dataframe for version tracking"""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

def get_current_timestamp() -> str:
    """Get current timestamp in standard format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value * 100:.1f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division to avoid division by zero"""
    return numerator / denominator if denominator != 0 else default

def detect_file_type(filename: str) -> str:
    """Detect file type from extension"""
    if filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif filename.endswith('.json'):
        return 'json'
    else:
        return 'unknown'

def get_feature_statistics(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
    """Calculate statistics for given features"""
    stats = {}
    for col in feature_columns:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            }
    return stats

def validate_numerical_range(value: float, min_val: float, max_val: float) -> bool:
    """Check if value is within valid numerical range"""
    return min_val <= value <= max_val

def create_sample_data(num_samples: int = 100) -> pd.DataFrame:
    """Create sample transaction data for testing"""
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(num_samples)],
        'amount': np.random.uniform(10, 5000, num_samples),
        'transaction_type': np.random.choice(['online', 'POS', 'mobile', 'in-store'], num_samples),
        'customer_region': np.random.choice(['NA', 'EU', 'APAC', 'LATAM', 'MEA'], num_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], num_samples),
        'merchant_risk_score': np.random.uniform(0, 1, num_samples),
        'transaction_hour': np.random.randint(0, 24, num_samples),
        'customer_age': np.random.randint(18, 80, num_samples),
        'account_age_days': np.random.randint(1, 3650, num_samples),
        'transaction_velocity_1h': np.random.randint(0, 10, num_samples),
        'transaction_velocity_24h': np.random.randint(0, 50, num_samples),
        'is_international': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'is_new_device': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'is_night_transaction': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'high_amount_flag': np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
    })
    
    return sample_data

def log_message(message: str, level: str = "INFO") -> None:
    """Log message with timestamp"""
    timestamp = get_current_timestamp()
    print(f"[{timestamp}] [{level}] {message}")


# ==================== NEW FUNCTIONS FOR WEB LINK SUPPORT ====================

def load_data_from_url(url: str, file_format: str = None) -> Optional[pd.DataFrame]:
    """
    Load data from a direct URL (CSV, Excel, JSON, or ZIP)
    
    Args:
        url: Direct URL to the data file
        file_format: Optional forced format ('csv', 'excel', 'json', 'zip')
        
    Returns:
        DataFrame or None if failed
    """
    try:
        # Send GET request with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv,application/json,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*'
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        # Detect file type
        content_type = response.headers.get('content-type', '').lower()
        
        # Use specified format if provided
        if file_format:
            detected_format = file_format
        else:
            # Auto-detect from URL or content-type
            url_lower = url.lower()
            if 'csv' in url_lower or 'csv' in content_type:
                detected_format = 'csv'
            elif 'xlsx' in url_lower or 'xls' in url_lower or 'excel' in content_type:
                detected_format = 'excel'
            elif 'json' in url_lower or 'json' in content_type:
                detected_format = 'json'
            elif 'zip' in url_lower or 'zip' in content_type:
                detected_format = 'zip'
            else:
                detected_format = 'csv'  # Default to CSV
        
        # Read based on detected format
        if detected_format == 'csv':
            # Try different encodings
            try:
                df = pd.read_csv(io.BytesIO(response.content))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            except:
                df = pd.read_csv(io.BytesIO(response.content), encoding='latin1')
                
        elif detected_format == 'excel':
            df = pd.read_excel(io.BytesIO(response.content))
            
        elif detected_format == 'json':
            df = pd.read_json(io.BytesIO(response.content))
            
        elif detected_format == 'zip':
            # Handle zip files
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find first CSV/Excel file
                found_file = None
                for file_name in z.namelist():
                    if file_name.endswith('.csv'):
                        found_file = file_name
                        break
                    elif file_name.endswith('.xlsx'):
                        found_file = file_name
                        break
                
                if found_file:
                    with z.open(found_file) as f:
                        if found_file.endswith('.csv'):
                            df = pd.read_csv(f)
                        else:
                            df = pd.read_excel(f)
                else:
                    raise ValueError("No CSV/Excel file found in zip archive")
        else:
            raise ValueError(f"Unsupported file format: {detected_format}")
        
        # Basic validation
        if df is None or df.empty:
            raise ValueError("Loaded data is empty")
        
        return df
        
    except requests.exceptions.Timeout:
        print(f"Timeout error loading {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_data_from_github(url: str) -> Optional[pd.DataFrame]:
    """
    Load data from GitHub repository
    
    Args:
        url: GitHub file URL (blob or raw)
        
    Returns:
        DataFrame or None if failed
    """
    # Convert to raw URL
    raw_url = convert_to_raw_github_url(url)
    return load_data_from_url(raw_url)


def load_data_from_google_drive(url: str) -> Optional[pd.DataFrame]:
    """
    Load data from Google Drive shareable link
    
    Args:
        url: Google Drive shareable link
        
    Returns:
        DataFrame or None if failed
    """
    direct_url = convert_google_drive_url(url)
    return load_data_from_url(direct_url)


def load_data_from_dropbox(url: str) -> Optional[pd.DataFrame]:
    """
    Load data from Dropbox shareable link
    
    Args:
        url: Dropbox shareable link
        
    Returns:
        DataFrame or None if failed
    """
    direct_url = convert_dropbox_url(url)
    return load_data_from_url(direct_url)


def load_data_from_kaggle(dataset_path: str, file_name: str = None) -> Optional[pd.DataFrame]:
    """
    Load data from Kaggle dataset
    
    Args:
        dataset_path: Kaggle dataset path (e.g., 'mlg-ulb/creditcardfraud')
        file_name: Optional specific file name
        
    Returns:
        DataFrame or None if failed
    """
    try:
        # Try using kagglehub first
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset_path)
            
            # Find the data file
            if file_name:
                file_path = os.path.join(path, file_name)
                if not os.path.exists(file_path):
                    # Try to find matching file
                    for f in os.listdir(path):
                        if file_name.lower() in f.lower():
                            file_path = os.path.join(path, f)
                            break
            else:
                # Look for CSV files
                files = [f for f in os.listdir(path) if f.endswith('.csv')]
                if not files:
                    files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
                if not files:
                    raise ValueError("No CSV/Excel files found in dataset")
                file_path = os.path.join(path, files[0])
            
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            return df
            
        except ImportError:
            # Fallback: Try using kaggle CLI
            import subprocess
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download using kaggle CLI
                cmd = f"kaggle datasets download {dataset_path} -p {tmpdir}"
                result = subprocess.run(cmd, shell=True, capture_output=True)
                
                if result.returncode != 0:
                    raise Exception("Kaggle CLI failed. Please install kaggle CLI and configure API key.")
                
                # Extract and load
                zip_files = [f for f in os.listdir(tmpdir) if f.endswith('.zip')]
                if not zip_files:
                    raise ValueError("No zip file downloaded")
                
                import zipfile
                with zipfile.ZipFile(os.path.join(tmpdir, zip_files[0]), 'r') as z:
                    z.extractall(tmpdir)
                
                # Find CSV/Excel file
                files = [f for f in os.listdir(tmpdir) if f.endswith('.csv') or f.endswith('.xlsx')]
                if not files:
                    raise ValueError("No CSV/Excel files found in downloaded dataset")
                
                file_path = os.path.join(tmpdir, files[0])
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            
            return df
            
    except Exception as e:
        print(f"Error loading from Kaggle: {e}")
        return None


def load_data_from_aws_s3(bucket: str, key: str, aws_access_key: str = None, aws_secret_key: str = None) -> Optional[pd.DataFrame]:
    """
    Load data from AWS S3 bucket
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        aws_access_key: Optional AWS access key
        aws_secret_key: Optional AWS secret key
        
    Returns:
        DataFrame or None if failed
    """
    try:
        import boto3
        from botocore.config import Config
        
        # Configure S3 client
        if aws_access_key and aws_secret_key:
            s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                config=Config(signature_version='s3v4')
            )
        else:
            s3 = boto3.client('s3')
        
        # Get object
        obj = s3.get_object(Bucket=bucket, Key=key)
        
        # Read based on file extension
        if key.endswith('.csv'):
            df = pd.read_csv(obj['Body'])
        elif key.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(obj['Body'])
        elif key.endswith('.json'):
            df = pd.read_json(obj['Body'])
        else:
            raise ValueError(f"Unsupported file format: {key}")
        
        return df
        
    except ImportError:
        print("boto3 not installed. Install with: pip install boto3")
        return None
    except Exception as e:
        print(f"Error loading from S3: {e}")
        return None


def convert_to_raw_github_url(url: str) -> str:
    """
    Convert GitHub URL to raw content URL
    
    Args:
        url: GitHub file URL (blob or raw)
        
    Returns:
        Raw GitHub URL
    """
    if 'github.com' in url and '/blob/' in url:
        # Convert blob URL to raw
        raw_url = url.replace('github.com', 'raw.githubusercontent.com')
        raw_url = raw_url.replace('/blob/', '/')
        return raw_url
    return url


def convert_google_drive_url(url: str) -> str:
    """
    Convert Google Drive shareable link to direct download link
    
    Args:
        url: Google Drive shareable link
        
    Returns:
        Direct download URL
    """
    # Extract file ID
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',      # Standard share link
        r'id=([a-zA-Z0-9_-]+)',           # id parameter
        r'd/([a-zA-Z0-9_-]+)',            # Alternative format
        r'uc\?export=download&id=([a-zA-Z0-9_-]+)'  # Already a download link
    ]
    
    file_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            break
    
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # If already a direct download link, return as is
    if 'drive.google.com/uc' in url or 'drive.usercontent.google.com' in url:
        return url
    
    return url


def convert_dropbox_url(url: str) -> str:
    """
    Convert Dropbox shareable link to direct download link
    
    Args:
        url: Dropbox shareable link
        
    Returns:
        Direct download URL
    """
    if 'dropbox.com' in url:
        if '?dl=0' in url:
            url = url.replace('?dl=0', '?dl=1')
        elif '?dl=1' not in url and '?raw=1' not in url:
            url += '?dl=1'
        elif 'www.dropbox.com' in url:
            url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    
    return url


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL
    
    Args:
        url: String to check
        
    Returns:
        True if valid URL, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return url_pattern.match(url) is not None


def get_file_extension_from_url(url: str) -> str:
    """
    Extract file extension from URL
    
    Args:
        url: URL string
        
    Returns:
        File extension (e.g., 'csv', 'xlsx', 'json')
    """
    # Remove query parameters
    url_path = url.split('?')[0]
    # Get extension
    if '.' in url_path:
        extension = url_path.split('.')[-1].lower()
        return extension
    return ''


def validate_url_accessibility(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Validate if a URL is accessible
    
    Args:
        url: URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'is_valid': False,
        'status_code': None,
        'content_type': None,
        'file_size_mb': None,
        'error': None
    }
    
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        result['status_code'] = response.status_code
        result['is_valid'] = 200 <= response.status_code < 300
        
        if result['is_valid']:
            result['content_type'] = response.headers.get('content-type', 'unknown')
            content_length = response.headers.get('content-length')
            if content_length:
                result['file_size_mb'] = round(int(content_length) / (1024 * 1024), 2)
        
    except requests.exceptions.Timeout:
        result['error'] = 'Connection timeout'
    except requests.exceptions.ConnectionError:
        result['error'] = 'Connection error'
    except Exception as e:
        result['error'] = str(e)
    
    return result