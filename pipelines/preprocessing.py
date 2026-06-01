# pipelines/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple, List, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Preprocesses transaction data for model inference with robust missing column handling"""
    
    def __init__(self, models: Dict[str, Any] = None, verbose: bool = False):
        """
        Initialize preprocessor
        
        Args:
            models: Dictionary containing trained models, scaler, encoders, and feature schema
            verbose: Whether to print preprocessing details
        """
        self.models = models
        self.scaler = models.get('scaler') if models else None
        self.encoders = models.get('encoders') if models else None
        self.feature_schema = models.get('feature_schema') if models else None
        self.verbose = verbose
        self.preprocessing_log = []
        
        # Define default values for missing columns
        self.default_values = {
            # Numeric defaults
            'transaction_hour': 12,
            'customer_age': 35,
            'account_age_days': 365,
            'transaction_velocity_1h': 0,
            'transaction_velocity_24h': 0,
            'avg_transaction_amount_7d': 500,
            'deviation_from_avg_amount': 0,
            'merchant_risk_score': 0.3,
            'previous_fraud_count': 0,
            'is_international': 0,
            'is_new_device': 0,
            'is_night_transaction': 0,
            'high_amount_flag': 0,
            'amount': 100,  # Default amount if missing
            
            # Categorical defaults
            'transaction_day_of_week': 'Monday',
            'transaction_type': 'online',
            'customer_region': 'unknown',
            'income_level': 'medium',
            'device_type': 'desktop'
        }
        
        # Define derived columns (can be calculated from others)
        self.derived_columns = {
            'transaction_hour': ['transaction_time', 'transaction_timestamp', 'timestamp'],
            'transaction_day_of_week': ['transaction_time', 'transaction_timestamp', 'timestamp']
        }
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline with missing column handling
        
        Args:
            df: Input dataframe to preprocess
            
        Returns:
            Preprocessed dataframe ready for model inference
        """
        self.preprocessing_log = []
        self._log("Starting preprocessing pipeline...")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Add default values for missing columns
        df_processed = self._add_default_columns(df_processed)
        
        # Handle datetime and extract time features if available
        df_processed = self._extract_time_features(df_processed)
        
        # Try to derive missing features from available data
        df_processed = self._derive_missing_features(df_processed)
        
        # Drop unnecessary columns
        df_processed = self._drop_unnecessary_columns(df_processed)
        
        # Encode categorical variables
        df_processed = self._encode_categorical(df_processed)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Handle infinite values
        df_processed = self._handle_infinite_values(df_processed)
        
        # Ensure correct feature order
        df_processed = self._align_features(df_processed)
        
        # Scale numerical features
        if self.scaler:
            df_processed = self._scale_features(df_processed)
        else:
            self._log("Warning: No scaler available. Using unscaled features.")
        
        self._log(f"Preprocessing complete. Output shape: {df_processed.shape}")
        
        return df_processed
    
    def _add_default_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default values for missing required columns"""
        
        # Get list of expected features if available
        if self.feature_schema and 'feature_columns' in self.feature_schema:
            expected_cols = self.feature_schema['feature_columns']
        else:
            expected_cols = list(self.default_values.keys())
        
        for col in expected_cols:
            if col not in df.columns:
                if col in self.default_values:
                    df[col] = self.default_values[col]
                    self._log(f"Added missing column '{col}' with default value: {self.default_values[col]}")
                elif col in self.derived_columns:
                    # Will be derived later
                    continue
                else:
                    # Unknown column, add with 0
                    df[col] = 0
                    self._log(f"Added missing column '{col}' with default value: 0")
        
        return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract useful time-based features from timestamp columns"""
        
        # Find timestamp column
        timestamp_col = None
        possible_timestamp_names = ['transaction_time', 'transaction_timestamp', 'timestamp', 
                                    'txn_time', 'created_at', 'date', 'datetime']
        
        for col in possible_timestamp_names:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            try:
                self._log(f"Found timestamp column: '{timestamp_col}'")
                
                # Convert to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Extract hour if not already present
                if 'transaction_hour' not in df.columns or df['transaction_hour'].isnull().all():
                    df['transaction_hour'] = df[timestamp_col].dt.hour.fillna(12).astype(int)
                    self._log("Extracted 'transaction_hour' from timestamp")
                
                # Extract day of week if not already present
                if 'transaction_day_of_week' not in df.columns or df['transaction_day_of_week'].isnull().all():
                    df['transaction_day_of_week'] = df[timestamp_col].dt.day_name().fillna('Monday')
                    self._log("Extracted 'transaction_day_of_week' from timestamp")
                
                # Extract is_night_transaction if not present
                if 'is_night_transaction' not in df.columns:
                    df['is_night_transaction'] = ((df['transaction_hour'] >= 22) | 
                                                  (df['transaction_hour'] <= 5)).astype(int)
                    self._log("Derived 'is_night_transaction' from transaction hour")
                
                # Drop original timestamp column
                df = df.drop(columns=[timestamp_col])
                
            except Exception as e:
                self._log(f"Warning: Could not extract time features: {str(e)}")
                # Set defaults
                if 'transaction_hour' not in df.columns:
                    df['transaction_hour'] = 12
                if 'transaction_day_of_week' not in df.columns:
                    df['transaction_day_of_week'] = 'Monday'
        else:
            self._log("No timestamp column found. Using default time values.")
            if 'transaction_hour' not in df.columns:
                df['transaction_hour'] = 12
            if 'transaction_day_of_week' not in df.columns:
                df['transaction_day_of_week'] = 'Monday'
        
        return df
    
    def _derive_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive missing features from available data"""
        
        # Derive high_amount_flag from amount if missing
        if 'high_amount_flag' not in df.columns and 'amount' in df.columns:
            # Flag amounts above 75th percentile or above $1000
            threshold = max(df['amount'].quantile(0.75), 1000)
            df['high_amount_flag'] = (df['amount'] > threshold).astype(int)
            self._log(f"Derived 'high_amount_flag' using threshold: ${threshold:.2f}")
        
        # Derive deviation_from_avg_amount if missing but have amount and avg
        if 'deviation_from_avg_amount' not in df.columns:
            if 'amount' in df.columns and 'avg_transaction_amount_7d' in df.columns:
                df['deviation_from_avg_amount'] = df['amount'] - df['avg_transaction_amount_7d']
                self._log("Derived 'deviation_from_avg_amount' from amount and avg_transaction_amount_7d")
            else:
                df['deviation_from_avg_amount'] = 0
        
        # Derive is_night_transaction if missing
        if 'is_night_transaction' not in df.columns and 'transaction_hour' in df.columns:
            df['is_night_transaction'] = ((df['transaction_hour'] >= 22) | 
                                          (df['transaction_hour'] <= 5)).astype(int)
            self._log("Derived 'is_night_transaction' from transaction_hour")
        
        # Set default income_level if missing
        if 'income_level' not in df.columns:
            df['income_level'] = 'medium'
            self._log("Set default 'income_level' to 'medium'")
        
        return df
    
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not needed for inference"""
        columns_to_drop = [
            'transaction_id', 'customer_id', 'true_risk_score', 'fraud',
            'predicted_label', 'fraud_probability', 'decision', 'risk_level',
            'is_fraudulent', 'prediction', 'score'
        ]
        
        dropped = []
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped.append(col)
        
        if dropped:
            self._log(f"Dropped unnecessary columns: {', '.join(dropped[:5])}")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using saved encoders or automatic encoding"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            self._log("No categorical columns to encode")
            return df
        
        for col in categorical_cols:
            try:
                if self.encoders and col in self.encoders:
                    # Use saved encoder from training
                    encoder = self.encoders[col]
                    df[col] = df[col].astype(str)
                    # Handle unknown categories by mapping to most frequent or 0
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                    )
                    self._log(f"Encoded '{col}' using saved encoder (classes: {len(encoder.classes_)})")
                else:
                    # Automatic encoding for unknown categories
                    df[col] = df[col].astype('category').cat.codes
                    self._log(f"Auto-encoded '{col}' with {df[col].nunique()} unique values")
            except Exception as e:
                self._log(f"Warning: Could not encode '{col}': {str(e)}. Setting to 0.")
                df[col] = 0
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numerical and categorical columns"""
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                
                # Use median for skewed distributions, mean for normal
                if col in ['amount', 'merchant_risk_score']:
                    fill_value = df[col].median() if not df[col].isnull().all() else 0
                else:
                    fill_value = df[col].mean() if not df[col].isnull().all() else 0
                
                df[col] = df[col].fillna(fill_value)
                self._log(f"Filled {missing_count} missing values in '{col}' with {fill_value:.2f}")
        
        # Handle categorical columns (should already be encoded)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                df[col] = df[col].fillna(mode_value)
                self._log(f"Filled {missing_count} missing values in '{col}' with '{mode_value}'")
        
        return df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in numerical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                self._log(f"Replaced {inf_count} infinite values in '{col}' with 0")
        
        return df
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature order matches training"""
        if self.feature_schema and 'feature_columns' in self.feature_schema:
            expected_cols = self.feature_schema['feature_columns']
            
            # Check for missing columns
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                self._log(f"Warning: Missing {len(missing_cols)} expected columns. Adding defaults.")
                for col in missing_cols:
                    df[col] = self.default_values.get(col, 0)
            
            # Check for extra columns
            extra_cols = [col for col in df.columns if col not in expected_cols]
            if extra_cols:
                self._log(f"Dropping {len(extra_cols)} extra columns not in feature schema")
                df = df.drop(columns=extra_cols)
            
            # Reorder columns to match training
            df = df[expected_cols]
            self._log(f"Aligned {len(df.columns)} features to training schema")
        else:
            self._log("No feature schema available. Using all numeric columns.")
            # Keep only numeric columns
            df = df.select_dtypes(include=[np.number])
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using saved scaler"""
        if self.scaler:
            try:
                # Ensure all columns are numeric
                df_numeric = df.select_dtypes(include=[np.number])
                
                if len(df_numeric.columns) != len(df.columns):
                    self._log(f"Warning: Converting {len(df.columns) - len(df_numeric.columns)} non-numeric columns")
                    for col in df.columns:
                        if col not in df_numeric.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df_numeric = df.select_dtypes(include=[np.number])
                
                # Scale the features
                scaled_array = self.scaler.transform(df_numeric)
                scaled_df = pd.DataFrame(scaled_array, columns=df_numeric.columns, index=df.index)
                
                self._log(f"Scaled {len(scaled_df.columns)} features using pre-trained scaler")
                return scaled_df
                
            except Exception as e:
                self._log(f"Error during scaling: {str(e)}. Returning unscaled features.")
                return df
        
        return df
    
    def _log(self, message: str):
        """Log preprocessing steps if verbose mode is enabled"""
        if self.verbose:
            print(f"[Preprocessor] {message}")
        self.preprocessing_log.append(message)
    
    def get_preprocessing_log(self) -> List[str]:
        """Get the preprocessing log for debugging"""
        return self.preprocessing_log
    
    def preprocess_with_fallback(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess with detailed fallback information
        
        Returns:
            Tuple of (preprocessed_df, metadata) where metadata contains preprocessing details
        """
        metadata = {
            'original_shape': df.shape,
            'original_columns': list(df.columns),
            'missing_columns_added': [],
            'derived_features': [],
            'defaults_used': [],
            'warnings': [],
            'success': True
        }
        
        try:
            # Track columns before preprocessing
            before_columns = set(df.columns)
            
            # Run preprocessing
            processed_df = self.preprocess(df)
            
            # Track what was added
            after_columns = set(processed_df.columns)
            metadata['missing_columns_added'] = list(after_columns - before_columns)
            
            # Check for any warnings
            for log in self.preprocessing_log:
                if 'Warning' in log:
                    metadata['warnings'].append(log)
                if 'default' in log.lower():
                    metadata['defaults_used'].append(log)
            
            metadata['final_shape'] = processed_df.shape
            metadata['success'] = True
            
        except Exception as e:
            metadata['success'] = False
            metadata['error'] = str(e)
            self._log(f"Preprocessing failed: {str(e)}")
            # Return empty dataframe as fallback
            processed_df = pd.DataFrame()
        
        return processed_df, metadata
    
    def validate_preprocessing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate if dataframe is ready for model inference
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': False,
            'has_nan': False,
            'has_inf': False,
            'feature_count': 0,
            'expected_features': 0,
            'missing_features': [],
            'issues': []
        }
        
        if df is None or df.empty:
            validation['issues'].append("DataFrame is empty")
            return validation
        
        # Check for NaN values
        if df.isnull().any().any():
            validation['has_nan'] = True
            validation['issues'].append("DataFrame contains NaN values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if np.isinf(df[numeric_cols]).any().any():
                validation['has_inf'] = True
                validation['issues'].append("DataFrame contains infinite values")
        
        # Check feature count
        validation['feature_count'] = len(df.columns)
        
        if self.feature_schema and 'feature_columns' in self.feature_schema:
            expected_cols = self.feature_schema['feature_columns']
            validation['expected_features'] = len(expected_cols)
            validation['missing_features'] = [col for col in expected_cols if col not in df.columns]
            
            if validation['missing_features']:
                validation['issues'].append(f"Missing {len(validation['missing_features'])} expected features")
        
        # Determine validity
        validation['is_valid'] = (
            not validation['has_nan'] and 
            not validation['has_inf'] and
            len(validation['missing_features']) == 0 and
            len(validation['issues']) == 0
        )
        
        return validation
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get expected feature names from the feature schema"""
        if self.feature_schema and 'feature_columns' in self.feature_schema:
            return self.feature_schema['feature_columns']
        return None
    
    def reset(self):
        """Reset the preprocessor state and logs"""
        self.preprocessing_log = []
        self._log("Preprocessor reset")


# Helper function for quick preprocessing without model artifacts
def quick_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick preprocessing for demo or testing without trained models
    
    Args:
        df: Input dataframe
        
    Returns:
        Preprocessed dataframe with minimal preprocessing
    """
    preprocessor = DataPreprocessor(models=None, verbose=False)
    
    # Add basic columns
    df = preprocessor._add_default_columns(df)
    df = preprocessor._extract_time_features(df)
    df = preprocessor._derive_missing_features(df)
    df = preprocessor._drop_unnecessary_columns(df)
    
    # Convert categorical to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # Handle missing values
    df = preprocessor._handle_missing_values(df)
    df = preprocessor._handle_infinite_values(df)
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    return df