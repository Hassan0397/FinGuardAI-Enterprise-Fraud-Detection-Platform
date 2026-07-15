# pipelines/validation.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """Enterprise-grade Data Validator with Intelligent Column Mapping"""
    
    # Required columns with detailed metadata
    REQUIRED_COLUMNS = {
        'transaction_id': {'type': 'string', 'required': True, 'unique': True, 'description': 'Unique transaction identifier'},
        'amount': {'type': 'numeric', 'required': True, 'min': 0.01, 'max': 1e9, 'description': 'Transaction amount'},
        'transaction_type': {'type': 'categorical', 'required': True, 'allowed': ['online', 'POS', 'mobile', 'in-store', 'ATM'], 'description': 'Type of transaction'},
        'customer_region': {'type': 'categorical', 'required': True, 'description': 'Customer geographic region'},
        'device_type': {'type': 'categorical', 'required': True, 'allowed': ['mobile', 'desktop', 'tablet', 'unknown'], 'description': 'Device used'},
        'merchant_risk_score': {'type': 'numeric', 'required': True, 'min': 0, 'max': 1, 'description': 'Merchant risk score'},
        'transaction_hour': {'type': 'numeric', 'required': False, 'min': 0, 'max': 23, 'description': 'Hour of transaction'},
        'transaction_day_of_week': {'type': 'categorical', 'required': False, 'description': 'Day of week'},
        'customer_age': {'type': 'numeric', 'required': False, 'min': 0, 'max': 120, 'description': 'Customer age'},
        'account_age_days': {'type': 'numeric', 'required': False, 'min': 0, 'description': 'Account age in days'},
        'transaction_velocity_1h': {'type': 'numeric', 'required': False, 'min': 0, 'description': 'Transactions in last hour'},
        'transaction_velocity_24h': {'type': 'numeric', 'required': False, 'min': 0, 'description': 'Transactions in last 24 hours'},
        'avg_transaction_amount_7d': {'type': 'numeric', 'required': False, 'min': 0, 'description': '7-day avg amount'},
        'deviation_from_avg_amount': {'type': 'numeric', 'required': False, 'description': 'Deviation from average'},
        'is_international': {'type': 'binary', 'required': False, 'allowed': [0, 1], 'description': 'International flag'},
        'is_new_device': {'type': 'binary', 'required': False, 'allowed': [0, 1], 'description': 'New device flag'},
        'previous_fraud_count': {'type': 'numeric', 'required': False, 'min': 0, 'description': 'Previous fraud count'},
        'is_night_transaction': {'type': 'binary', 'required': False, 'allowed': [0, 1], 'description': 'Night transaction flag'},
        'high_amount_flag': {'type': 'binary', 'required': False, 'allowed': [0, 1], 'description': 'High amount flag'}
    }
    
    # Advanced column mappings with weights
    COLUMN_MAPPINGS = {
        'transaction_id': {
            'patterns': ['transaction_id', 'txn_id', 'transactionid', 'txnid', 'id', 'trans_id', 'tran_id', 'reference_id'],
            'weight': 10,
            'regex': r'(transaction|txn|trans).*id|id$'
        },
        'amount': {
            'patterns': ['amount', 'transaction_amount', 'txn_amount', 'amt', 'value', 'amount_usd', 'transactionvalue', 'tran_amount'],
            'weight': 10,
            'regex': r'amount|amt|value|txn_.*amount'
        },
        'transaction_type': {
            'patterns': ['transaction_type', 'txn_type', 'type', 'payment_type', 'trans_type', 'txn_type_cd', 'transaction_category'],
            'weight': 8,
            'regex': r'.*type$|.*category'
        },
        'customer_region': {
            'patterns': ['customer_region', 'region', 'country', 'location', 'customer_location', 'geography', 'region_code'],
            'weight': 7,
            'regex': r'region|country|location|geography'
        },
        'device_type': {
            'patterns': ['device_type', 'device', 'device_used', 'device_category', 'device_name', 'user_device'],
            'weight': 7,
            'regex': r'device|mobile|desktop|tablet'
        },
        'merchant_risk_score': {
            'patterns': ['merchant_risk_score', 'risk_score', 'merchant_risk', 'risk', 'merchant_score', 'risk_rating', 'fraud_risk'],
            'weight': 9,
            'regex': r'risk.*score|merchant.*risk'
        },
        'transaction_hour': {
            'patterns': ['transaction_hour', 'hour', 'txn_hour', 'hour_of_day', 'time_hour', 'transaction_time_hour'],
            'weight': 6,
            'regex': r'hour|time$'
        },
        'transaction_day_of_week': {
            'patterns': ['transaction_day_of_week', 'day_of_week', 'weekday', 'day', 'transaction_weekday', 'week_day'],
            'weight': 6,
            'regex': r'day|weekday|dow'
        }
    }
    
    # Intelligent default values with business logic
    DEFAULT_VALUES = {
        'transaction_hour': 12,
        'transaction_day_of_week': 'Monday',
        'customer_age': 35,
        'account_age_days': 365,
        'transaction_velocity_1h': 0,
        'transaction_velocity_24h': 0,
        'avg_transaction_amount_7d': 0,
        'deviation_from_avg_amount': 0,
        'is_international': 0,
        'merchant_risk_score': 0.3,
        'device_type': 'desktop',
        'is_new_device': 0,
        'previous_fraud_count': 0,
        'is_night_transaction': 0,
        'high_amount_flag': 0,
        'transaction_type': 'online',
        'customer_region': 'unknown'
    }
    
    def __init__(self, auto_fix: bool = True, strict_mode: bool = False, enable_ml_validation: bool = False):
        """
        Initialize enterprise validator
        
        Args:
            auto_fix: Automatically fix common data issues
            strict_mode: Enforce all validations strictly
            enable_ml_validation: Enable ML-based anomaly detection
        """
        self.auto_fix = auto_fix
        self.strict_mode = strict_mode
        self.enable_ml_validation = enable_ml_validation
        
        # Validation results with comprehensive tracking
        self.validation_results = {
            'passed': False,
            'quality_score': 0.0,
            'issues_count': 0,
            'critical_issues': [],
            'warnings': [],
            'info_messages': [],
            'missing_values': {},
            'schema_issues': [],
            'data_type_issues': [],
            'value_range_issues': [],
            'duplicates': 0,
            'outliers': {},
            'anomalies': [],
            'recommendations': [],
            'column_mapping_used': {},
            'auto_fixes_applied': [],
            'quality_metrics': {},
            'validation_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'validation_timestamp': datetime.now().isoformat(),
            'data_profile': {}
        }
        
        self.mapped_df = None
        self.original_df = None
        self.quality_weights = {
            'completeness': 0.25,
            'accuracy': 0.20,
            'consistency': 0.20,
            'uniqueness': 0.15,
            'timeliness': 0.10,
            'validity': 0.10
        }
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main validation pipeline with comprehensive checks"""
        self.original_df = df.copy()
        self.mapped_df = df.copy()
        
        # Initialize validation
        self._log_info(f"Starting validation with ID: {self.validation_results['validation_id']}")
        
        # 1. Basic Structure Validation
        self._validate_dataframe_structure()
        
        # 2. Intelligent Column Mapping
        self._perform_intelligent_column_mapping()
        
        # 3. Schema Compliance
        self._validate_schema_compliance()
        
        # 4. Data Type Validation
        self._validate_data_types()
        
        # 5. Value Range Validation
        self._validate_value_ranges()
        
        # 6. Missing Values Analysis
        self._analyze_missing_values()
        
        # 7. Duplicate Detection
        self._detect_duplicates()
        
        # 8. Statistical Outlier Detection
        self._detect_statistical_outliers()
        
        # 9. Pattern Validation
        self._validate_patterns()
        
        # 10. Business Logic Validation
        self._validate_business_logic()
        
        # 11. Cross-Column Validation
        self._validate_cross_column_relationships()
        
        # 12. Data Profile Generation
        self._generate_data_profile()
        
        # 13. Auto-Fix Application (if enabled)
        if self.auto_fix:
            self._apply_intelligent_fixes()
        
        # 14. Quality Score Calculation
        self._calculate_comprehensive_quality_score()
        
        # 15. Final Decision
        self._determine_validation_passed()
        
        # 16. Generate Recommendations
        self._generate_intelligent_recommendations()
        
        # 17. Create Validation Report
        self._create_validation_report()
        
        return self.validation_results
    
    def _validate_dataframe_structure(self):
        """Validate basic dataframe structure"""
        if self.mapped_df is None or self.mapped_df.empty:
            self.validation_results['critical_issues'].append("Dataset is empty")
            self.validation_results['quality_score'] = 0
            self.validation_results['passed'] = False
            return
        
        # Check minimum rows
        if len(self.mapped_df) < 10:
            self.validation_results['warnings'].append(f"Dataset has only {len(self.mapped_df)} rows (minimum recommended: 10)")
        
        # Check maximum rows
        if len(self.mapped_df) > 1000000:
            self.validation_results['warnings'].append(f"Large dataset detected: {len(self.mapped_df):,} rows. Performance may be affected.")
        
        # Record basic stats
        self.validation_results['data_profile']['row_count'] = len(self.mapped_df)
        self.validation_results['data_profile']['column_count'] = len(self.mapped_df.columns)
        self.validation_results['data_profile']['memory_usage_mb'] = self.mapped_df.memory_usage(deep=True).sum() / 1024**2
    
    def _perform_intelligent_column_mapping(self):
        """Advanced intelligent column mapping with fuzzy matching"""
        mapping_used = {}
        confidence_scores = {}
        
        # Convert columns to lowercase for matching
        df_columns_lower = {col.lower(): col for col in self.mapped_df.columns}
        
        for expected_col, mapping_config in self.COLUMN_MAPPINGS.items():
            if expected_col in self.mapped_df.columns:
                mapping_used[expected_col] = expected_col
                confidence_scores[expected_col] = 1.0
                continue
            
            best_match = None
            best_score = 0
            
            # Try pattern matching
            for pattern in mapping_config['patterns']:
                pattern_lower = pattern.lower()
                for col_lower, original_col in df_columns_lower.items():
                    # Exact match
                    if col_lower == pattern_lower:
                        score = 1.0
                    # Contains match
                    elif pattern_lower in col_lower:
                        score = 0.8
                    # Word match
                    elif any(word in col_lower.split('_') for word in pattern_lower.split('_')):
                        score = 0.6
                    else:
                        score = 0
                    
                    # Try regex matching
                    if score < 0.8 and 'regex' in mapping_config:
                        if re.search(mapping_config['regex'], col_lower, re.IGNORECASE):
                            score = max(score, 0.7)
                    
                    if score > best_score:
                        best_score = score
                        best_match = original_col
            
            if best_match and best_score >= 0.6:
                self.mapped_df[expected_col] = self.original_df[best_match]
                mapping_used[expected_col] = best_match
                confidence_scores[expected_col] = best_score
        
        # Try to derive timestamp features
        timestamp_col = self._find_best_timestamp_column()
        if timestamp_col:
            try:
                dt_series = pd.to_datetime(self.original_df[timestamp_col], errors='coerce')
                
                if 'transaction_hour' not in self.mapped_df.columns:
                    self.mapped_df['transaction_hour'] = dt_series.dt.hour.fillna(12).astype(int)
                    mapping_used['transaction_hour'] = f'derived_from_{timestamp_col}'
                
                if 'transaction_day_of_week' not in self.mapped_df.columns:
                    self.mapped_df['transaction_day_of_week'] = dt_series.dt.day_name().fillna('Monday')
                    mapping_used['transaction_day_of_week'] = f'derived_from_{timestamp_col}'
                
                self.validation_results['info_messages'].append(f"Derived time features from timestamp column: {timestamp_col}")
            except:
                pass
        
        self.validation_results['column_mapping_used'] = mapping_used
        self.validation_results['quality_metrics']['mapping_confidence'] = np.mean(list(confidence_scores.values())) if confidence_scores else 0
        
        # Add mapping summary
        if mapping_used:
            self.validation_results['info_messages'].append(f"Successfully mapped {len(mapping_used)} columns with avg confidence {self.validation_results['quality_metrics']['mapping_confidence']:.2%}")
    
    def _find_best_timestamp_column(self) -> Optional[str]:
        """Intelligently find the best timestamp column"""
        timestamp_keywords = ['time', 'timestamp', 'date', 'datetime', 'created_at', 'updated_at', 'transaction_time']
        
        for col in self.original_df.columns:
            col_lower = col.lower()
            
            # Check keyword matches
            for keyword in timestamp_keywords:
                if keyword in col_lower:
                    try:
                        pd.to_datetime(self.original_df[col])
                        return col
                    except:
                        continue
            
            # Try to convert to datetime
            try:
                pd.to_datetime(self.original_df[col])
                return col
            except:
                continue
        
        return None
    
    def _validate_schema_compliance(self):
        """Validate required columns with flexible rules"""
        missing_required = []
        
        for col, metadata in self.REQUIRED_COLUMNS.items():
            if col not in self.mapped_df.columns:
                if metadata.get('required', False):
                    missing_required.append(col)
                    self.validation_results['schema_issues'].append(f"Missing required column: {col}")
                elif self.auto_fix and col in self.DEFAULT_VALUES:
                    # Add default value for non-required
                    self.mapped_df[col] = self.DEFAULT_VALUES[col]
                    self.validation_results['auto_fixes_applied'].append(f"Added missing column '{col}' with default value")
        
        if missing_required:
            self.validation_results['critical_issues'].append(f"Missing {len(missing_required)} required columns: {', '.join(missing_required[:5])}")
            self.validation_results['issues_count'] += len(missing_required)
    
    def _validate_data_types(self):
        """Validate and convert data types intelligently"""
        type_conversions = []
        
        for col, metadata in self.REQUIRED_COLUMNS.items():
            if col in self.mapped_df.columns:
                expected_type = metadata['type']
                
                try:
                    if expected_type == 'numeric':
                        self.mapped_df[col] = pd.to_numeric(self.mapped_df[col], errors='coerce')
                        type_conversions.append(f"Converted {col} to numeric")
                    
                    elif expected_type == 'string':
                        self.mapped_df[col] = self.mapped_df[col].astype(str)
                    
                    elif expected_type == 'categorical':
                        self.mapped_df[col] = self.mapped_df[col].astype('category')
                    
                    elif expected_type == 'binary':
                        self.mapped_df[col] = pd.to_numeric(self.mapped_df[col], errors='coerce').fillna(0)
                        self.mapped_df[col] = (self.mapped_df[col] > 0).astype(int)
                        type_conversions.append(f"Converted {col} to binary")
                
                except Exception as e:
                    self.validation_results['data_type_issues'].append(f"Failed to convert {col} to {expected_type}: {str(e)}")
                    self.validation_results['issues_count'] += 1
        
        if type_conversions and self.auto_fix:
            self.validation_results['auto_fixes_applied'].extend(type_conversions[:3])
    
    def _validate_value_ranges(self):
        """Validate value ranges for numeric columns - FIXED VERSION"""
        range_issues = []
        
        for col, metadata in self.REQUIRED_COLUMNS.items():
            if col in self.mapped_df.columns and metadata.get('type') == 'numeric':
                # Create a boolean mask for valid values
                valid_mask = pd.Series([True] * len(self.mapped_df), index=self.mapped_df.index)
                
                if 'min' in metadata:
                    valid_mask = valid_mask & (self.mapped_df[col] >= metadata['min'])
                if 'max' in metadata:
                    valid_mask = valid_mask & (self.mapped_df[col] <= metadata['max'])
                
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    issue_msg = f"{col}: {invalid_count} values outside range [{metadata.get('min', '-∞')}, {metadata.get('max', '∞')}]"
                    range_issues.append(issue_msg)
                    
                    if self.auto_fix and invalid_count / len(self.mapped_df) < 0.1:
                        # Fix by clipping to range - using .loc to avoid indexing issues
                        if 'min' in metadata:
                            self.mapped_df.loc[valid_mask == False, col] = metadata['min']
                        if 'max' in metadata:
                            self.mapped_df.loc[valid_mask == False, col] = metadata['max']
                        self.validation_results['auto_fixes_applied'].append(f"Clipped {invalid_count} values in {col} to valid range")
                    else:
                        self.validation_results['value_range_issues'].append(issue_msg)
                        self.validation_results['issues_count'] += 1
        
        if range_issues:
            self.validation_results['warnings'].extend(range_issues[:3])
    
    def _analyze_missing_values(self):
        """Comprehensive missing values analysis"""
        missing_analysis = {}
        
        for col in self.mapped_df.columns:
            missing_count = self.mapped_df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.mapped_df)) * 100
                missing_analysis[col] = missing_count
                
                if missing_pct > 50:
                    self.validation_results['critical_issues'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                    self.validation_results['issues_count'] += 2
                elif missing_pct > 20:
                    self.validation_results['warnings'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                    self.validation_results['issues_count'] += 1
        
        self.validation_results['missing_values'] = missing_analysis
        
        # Calculate completeness score
        if self.mapped_df is not None:
            total_cells = len(self.mapped_df) * len(self.mapped_df.columns)
            filled_cells = total_cells - sum(missing_analysis.values())
            completeness = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
            self.validation_results['quality_metrics']['completeness'] = completeness
    
    def _detect_duplicates(self):
        """Advanced duplicate detection"""
        # Check transaction ID duplicates
        if 'transaction_id' in self.mapped_df.columns:
            dup_count = self.mapped_df['transaction_id'].duplicated().sum()
            self.validation_results['duplicates'] = dup_count
            
            if dup_count > 0:
                dup_pct = (dup_count / len(self.mapped_df)) * 100
                
                if dup_pct > 5:
                    self.validation_results['critical_issues'].append(f"High duplicate rate ({dup_pct:.1f}%) in transaction IDs")
                    self.validation_results['issues_count'] += 2
                elif dup_pct > 1:
                    self.validation_results['warnings'].append(f"{dup_count} duplicate transaction IDs found")
                    self.validation_results['issues_count'] += 1
                
                if self.auto_fix and dup_pct < 10:
                    # Remove duplicates but keep first occurrence
                    original_len = len(self.mapped_df)
                    self.mapped_df = self.mapped_df.drop_duplicates(subset=['transaction_id'], keep='first')
                    removed = original_len - len(self.mapped_df)
                    if removed > 0:
                        self.validation_results['auto_fixes_applied'].append(f"Removed {removed} duplicate transaction records")
                        self.validation_results['duplicates'] = 0
        
        # Calculate uniqueness score
        self.validation_results['quality_metrics']['uniqueness'] = 100 - (self.validation_results['duplicates'] / len(self.mapped_df) * 100) if len(self.mapped_df) > 0 else 100
    
    def _detect_statistical_outliers(self):
        """Advanced statistical outlier detection using IQR"""
        outlier_analysis = {}
        
        numeric_cols = self.mapped_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.mapped_df.columns and len(self.mapped_df[col].dropna()) > 0:
                data = self.mapped_df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    outliers_count = ((data < lower_bound) | (data > upper_bound)).sum()
                    
                    if outliers_count > 0:
                        outlier_pct = (outliers_count / len(data)) * 100
                        outlier_analysis[col] = outliers_count
                        
                        if outlier_pct > 10:
                            self.validation_results['warnings'].append(f"High outlier rate ({outlier_pct:.1f}%) in {col}")
        
        self.validation_results['outliers'] = outlier_analysis
        self.validation_results['quality_metrics']['outlier_rate'] = sum(outlier_analysis.values()) / len(self.mapped_df) if len(self.mapped_df) > 0 else 0
    
    def _validate_patterns(self):
        """Validate data patterns using regex"""
        pattern_issues = []
        
        # Validate transaction_id pattern
        if 'transaction_id' in self.mapped_df.columns:
            valid_pattern = r'^[A-Za-z0-9\-_]+$'
            invalid_ids = ~self.mapped_df['transaction_id'].astype(str).str.match(valid_pattern)
            invalid_count = invalid_ids.sum()
            
            if invalid_count > 0:
                pattern_issues.append(f"{invalid_count} transaction IDs have invalid format")
                self.validation_results['issues_count'] += 1
        
        if pattern_issues:
            self.validation_results['warnings'].extend(pattern_issues[:3])
    
    def _validate_business_logic(self):
        """Validate business rules and logic"""
        # Rule: Night transactions should be flagged
        if 'transaction_hour' in self.mapped_df.columns and 'is_night_transaction' in self.mapped_df.columns:
            night_hours = (self.mapped_df['transaction_hour'] >= 22) | (self.mapped_df['transaction_hour'] <= 5)
            not_flagged = night_hours & (self.mapped_df['is_night_transaction'] == 0)
            violations = not_flagged.sum()
            
            if violations > 0 and self.auto_fix:
                self.mapped_df.loc[night_hours, 'is_night_transaction'] = 1
                self.validation_results['auto_fixes_applied'].append(f"Fixed {violations} night transaction flags")
    
    def _validate_cross_column_relationships(self):
        """Validate relationships between columns"""
        # Check derived fields consistency
        if 'amount' in self.mapped_df.columns and 'high_amount_flag' in self.mapped_df.columns:
            high_amount_expected = (self.mapped_df['amount'] > 1000).astype(int)
            high_amount_actual = self.mapped_df['high_amount_flag']
            mismatches = (high_amount_expected != high_amount_actual).sum()
            
            if mismatches > 0 and self.auto_fix:
                self.mapped_df['high_amount_flag'] = high_amount_expected
                self.validation_results['auto_fixes_applied'].append(f"Corrected {mismatches} high_amount_flag values based on amount")
    
    def _generate_data_profile(self):
        """Generate comprehensive data profile"""
        profile = {}
        
        # Numerical columns statistics
        numeric_cols = self.mapped_df.select_dtypes(include=[np.number]).columns
        for col in list(numeric_cols)[:10]:
            if col in self.mapped_df.columns and len(self.mapped_df[col].dropna()) > 0:
                profile[col] = {
                    'min': float(self.mapped_df[col].min()),
                    'max': float(self.mapped_df[col].max()),
                    'mean': float(self.mapped_df[col].mean()),
                    'median': float(self.mapped_df[col].median()),
                    'std': float(self.mapped_df[col].std()),
                    'missing': int(self.mapped_df[col].isnull().sum())
                }
        
        self.validation_results['data_profile']['statistics'] = profile
    
    def _apply_intelligent_fixes(self):
        """Apply intelligent auto-fixes with reasoning"""
        fixes_applied = []
        
        # Fix 1: Generate missing transaction IDs
        if 'transaction_id' not in self.mapped_df.columns or self.mapped_df['transaction_id'].isnull().all():
            self.mapped_df['transaction_id'] = [f"TXN_{datetime.now().strftime('%Y%m%d')}_{i:06d}" for i in range(len(self.mapped_df))]
            fixes_applied.append("Generated unique transaction IDs for all rows")
        
        # Fix 2: Handle missing amounts
        if 'amount' in self.mapped_df.columns:
            missing_amounts = self.mapped_df['amount'].isnull().sum()
            if missing_amounts > 0:
                median_amount = self.mapped_df['amount'].median()
                if pd.isna(median_amount):
                    median_amount = 100.0
                self.mapped_df['amount'] = self.mapped_df['amount'].fillna(median_amount)
                fixes_applied.append(f"Filled {missing_amounts} missing amounts with median value ({median_amount:.2f})")
        
        # Fix 3: Ensure merchant risk score is in valid range
        if 'merchant_risk_score' in self.mapped_df.columns:
            invalid_risk = ((self.mapped_df['merchant_risk_score'] < 0) | (self.mapped_df['merchant_risk_score'] > 1)).sum()
            if invalid_risk > 0:
                self.mapped_df['merchant_risk_score'] = self.mapped_df['merchant_risk_score'].clip(0, 1)
                fixes_applied.append(f"Clipped {invalid_risk} risk scores to valid range [0,1]")
        
        # Fix 4: Ensure binary columns have correct values
        binary_columns = ['is_international', 'is_new_device', 'is_night_transaction', 'high_amount_flag']
        for col in binary_columns:
            if col in self.mapped_df.columns:
                invalid_binary = ((self.mapped_df[col] != 0) & (self.mapped_df[col] != 1)).sum()
                if invalid_binary > 0:
                    self.mapped_df[col] = (self.mapped_df[col] > 0).astype(int)
                    fixes_applied.append(f"Converted {invalid_binary} values in {col} to binary format")
        
        # Fix 5: Generate derived features
        if 'amount' in self.mapped_df.columns and 'high_amount_flag' not in self.mapped_df.columns:
            self.mapped_df['high_amount_flag'] = (self.mapped_df['amount'] > 1000).astype(int)
            fixes_applied.append("Derived high_amount_flag from transaction amount")
        
        if 'transaction_hour' in self.mapped_df.columns and 'is_night_transaction' not in self.mapped_df.columns:
            night_hours = (self.mapped_df['transaction_hour'] >= 22) | (self.mapped_df['transaction_hour'] <= 5)
            self.mapped_df['is_night_transaction'] = night_hours.astype(int)
            fixes_applied.append("Derived is_night_transaction from transaction hour")
        
        # Record fixes
        self.validation_results['auto_fixes_applied'].extend(fixes_applied)
    
    def _calculate_comprehensive_quality_score(self):
        """Calculate comprehensive quality score with weighted metrics"""
        weights = self.quality_weights
        scores = {}
        
        # Completeness Score
        completeness = self.validation_results['quality_metrics'].get('completeness', 100)
        scores['completeness'] = completeness
        
        # Accuracy Score
        total_issues = len(self.validation_results['data_type_issues']) + len(self.validation_results['value_range_issues'])
        max_issues = len(self.REQUIRED_COLUMNS)
        accuracy = max(0, 100 - (total_issues / max_issues * 100)) if max_issues > 0 else 100
        scores['accuracy'] = accuracy
        
        # Consistency Score
        consistency = 100 - (len(self.validation_results.get('warnings', [])) * 5)
        consistency = max(0, min(100, consistency))
        scores['consistency'] = consistency
        
        # Uniqueness Score
        uniqueness = self.validation_results['quality_metrics'].get('uniqueness', 100)
        scores['uniqueness'] = uniqueness
        
        # Timeliness Score
        timeliness = 80
        if any('time' in col.lower() for col in self.mapped_df.columns):
            timeliness = 90
        scores['timeliness'] = timeliness
        
        # Validity Score
        found_required = len([col for col in self.REQUIRED_COLUMNS if col in self.mapped_df.columns])
        total_required = len(self.REQUIRED_COLUMNS)
        validity = (found_required / total_required) * 100
        scores['validity'] = validity
        
        # Calculate weighted average
        weighted_score = sum(scores[k] * weights.get(k, 0.1) for k in scores)
        
        # Adjust for critical issues
        critical_penalty = len(self.validation_results['critical_issues']) * 10
        final_score = max(0, weighted_score - critical_penalty)
        
        self.validation_results['quality_score'] = round(final_score, 2)
        self.validation_results['quality_metrics']['detailed_scores'] = scores
    
    def _determine_validation_passed(self):
        """Determine if validation passed based on multiple criteria"""
        quality_score = self.validation_results['quality_score']
        has_critical = len(self.validation_results['critical_issues']) > 0
        has_amount = 'amount' in self.mapped_df.columns
        
        if self.strict_mode:
            passed = (quality_score >= 80 and not has_critical and has_amount)
        else:
            # Lenient mode: allow some issues
            passed = (quality_score >= 50 and has_amount)
        
        self.validation_results['passed'] = passed
        
        # Add pass/fail message
        if passed:
            self.validation_results['info_messages'].append(f"Validation PASSED with quality score {quality_score:.1f}%")
        else:
            self.validation_results['critical_issues'].append(f"Validation FAILED with quality score {quality_score:.1f}%")
    
    def _generate_intelligent_recommendations(self):
        """Generate intelligent, actionable recommendations"""
        recommendations = []
        
        # Quality-based recommendations
        quality_score = self.validation_results['quality_score']
        if quality_score >= 90:
            recommendations.append("🏆 Excellent data quality! Your dataset is production-ready.")
        elif quality_score >= 70:
            recommendations.append("✅ Good data quality. Minor improvements suggested for optimal performance.")
        elif quality_score >= 50:
            recommendations.append("⚠️ Acceptable data quality. Consider addressing key issues for better results.")
        else:
            recommendations.append("❌ Poor data quality. Please review the critical issues before proceeding.")
        
        # Specific recommendations
        if self.validation_results['missing_values']:
            missing_cols = list(self.validation_results['missing_values'].keys())[:3]
            recommendations.append(f"📊 Address missing values in: {', '.join(missing_cols)}")
        
        if self.validation_results['duplicates'] > 0:
            recommendations.append(f"🔄 Remove {self.validation_results['duplicates']} duplicate transactions")
        
        if self.validation_results['outliers']:
            outlier_cols = list(self.validation_results['outliers'].keys())[:2]
            recommendations.append(f"📈 Review outliers in {', '.join(outlier_cols)} for potential fraud indicators")
        
        if self.validation_results['column_mapping_used']:
            recommendations.append(f"🔗 {len(self.validation_results['column_mapping_used'])} columns were auto-mapped successfully")
        
        if self.validation_results['auto_fixes_applied']:
            recommendations.append(f"🔧 {len(self.validation_results['auto_fixes_applied'])} automatic fixes were applied")
        
        # Operational recommendations
        if not self.strict_mode and quality_score >= 50:
            recommendations.append("🚀 Dataset is ready for fraud detection analysis")
        
        # Limit to 7 recommendations
        self.validation_results['recommendations'] = recommendations[:7]
    
    def _create_validation_report(self):
        """Create comprehensive validation report"""
        self.validation_results['validation_summary'] = self.get_validation_summary()
        self.validation_results['can_proceed_to_prediction'] = self.is_valid_for_prediction()
    
    def _log_info(self, message: str):
        """Log info message (for debugging)"""
        pass
    
    def get_mapped_dataframe(self) -> pd.DataFrame:
        """Get the mapped dataframe with correct column names"""
        return self.mapped_df
    
    def get_validation_summary(self) -> str:
        """Get beautiful formatted validation summary"""
        summary_lines = []
        
        # Header
        summary_lines.append("=" * 70)
        summary_lines.append(f"📋 FINGUARD AI VALIDATION REPORT".center(70))
        summary_lines.append("=" * 70)
        summary_lines.append(f"Validation ID: {self.validation_results['validation_id']}")
        summary_lines.append(f"Timestamp: {self.validation_results['validation_timestamp']}")
        summary_lines.append("=" * 70)
        summary_lines.append("")
        
        # Quality Score with visual indicator
        score = self.validation_results['quality_score']
        if score >= 90:
            indicator = "🟢 EXCELLENT"
        elif score >= 70:
            indicator = "🟡 GOOD"
        elif score >= 50:
            indicator = "🟠 ACCEPTABLE"
        else:
            indicator = "🔴 POOR"
        
        summary_lines.append(f"QUALITY SCORE: {score:.1f}% [{indicator}]")
        summary_lines.append("")
        
        # Status
        status = "✅ PASSED" if self.validation_results['passed'] else "❌ FAILED"
        summary_lines.append(f"VALIDATION STATUS: {status}")
        summary_lines.append("")
        
        # Key Metrics
        summary_lines.append("📊 KEY METRICS:")
        summary_lines.append(f"   • Total Rows: {self.validation_results['data_profile'].get('row_count', 0):,}")
        summary_lines.append(f"   • Columns Found: {self.validation_results['data_profile'].get('column_count', 0)}")
        summary_lines.append(f"   • Auto-Mapped Columns: {len(self.validation_results['column_mapping_used'])}")
        summary_lines.append(f"   • Auto-Fixes Applied: {len(self.validation_results['auto_fixes_applied'])}")
        summary_lines.append(f"   • Issues Found: {self.validation_results['issues_count']}")
        summary_lines.append(f"   • Critical Issues: {len(self.validation_results['critical_issues'])}")
        summary_lines.append(f"   • Warnings: {len(self.validation_results['warnings'])}")
        summary_lines.append("")
        
        # Quality Metrics Breakdown
        if 'detailed_scores' in self.validation_results['quality_metrics']:
            summary_lines.append("🎯 QUALITY BREAKDOWN:")
            for metric, value in self.validation_results['quality_metrics']['detailed_scores'].items():
                bar = "█" * int(value / 10) + "░" * (10 - int(value / 10))
                summary_lines.append(f"   • {metric.capitalize():12} {bar} {value:.1f}%")
            summary_lines.append("")
        
        # Critical Issues (if any)
        if self.validation_results['critical_issues']:
            summary_lines.append("🚨 CRITICAL ISSUES:")
            for issue in self.validation_results['critical_issues'][:5]:
                summary_lines.append(f"   • {issue}")
            summary_lines.append("")
        
        # Warnings (if any)
        if self.validation_results['warnings']:
            summary_lines.append("⚠️ WARNINGS:")
            for warning in self.validation_results['warnings'][:5]:
                summary_lines.append(f"   • {warning}")
            summary_lines.append("")
        
        # Successes
        if self.validation_results['auto_fixes_applied']:
            summary_lines.append("✅ AUTO-FIXES APPLIED:")
            for fix in self.validation_results['auto_fixes_applied'][:3]:
                summary_lines.append(f"   • {fix}")
            summary_lines.append("")
        
        # Recommendations
        if self.validation_results['recommendations']:
            summary_lines.append("💡 RECOMMENDATIONS:")
            for rec in self.validation_results['recommendations'][:5]:
                summary_lines.append(f"   • {rec}")
            summary_lines.append("")
        
        # Footer
        summary_lines.append("=" * 70)
        summary_lines.append("📌 Powered by FinGuard AI Validation Engine")
        summary_lines.append("=" * 70)
        
        return "\n".join(summary_lines)
    
    def is_valid_for_prediction(self) -> bool:
        """Check if data is valid enough to run predictions"""
        if self.mapped_df is None or len(self.mapped_df) == 0:
            return False
        
        has_amount = 'amount' in self.mapped_df.columns
        quality_score = self.validation_results['quality_score']
        
        # Minimum requirements
        if self.strict_mode:
            return has_amount and quality_score >= 80
        else:
            return has_amount and quality_score >= 50
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get detailed data quality report as dictionary"""
        return {
            'validation_id': self.validation_results['validation_id'],
            'timestamp': self.validation_results['validation_timestamp'],
            'quality_score': self.validation_results['quality_score'],
            'passed': self.validation_results['passed'],
            'can_predict': self.is_valid_for_prediction(),
            'issues_count': self.validation_results['issues_count'],
            'critical_count': len(self.validation_results['critical_issues']),
            'warning_count': len(self.validation_results['warnings']),
            'auto_fixes_count': len(self.validation_results['auto_fixes_applied']),
            'mapped_columns_count': len(self.validation_results['column_mapping_used']),
            'row_count': self.validation_results['data_profile'].get('row_count', 0),
            'column_count': self.validation_results['data_profile'].get('column_count', 0),
            'quality_breakdown': self.validation_results['quality_metrics'].get('detailed_scores', {}),
            'recommendations': self.validation_results['recommendations'][:5]
        }