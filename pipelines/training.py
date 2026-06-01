# pipelines/training.py
import pandas as pd
import numpy as np
import joblib
import json
import hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class FraudModelTrainer:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.logistic_model = None
        self.xgb_model = None
        self.feature_columns = None
        
    def load_data(self):
        """Load and merge training datasets"""
        print("📂 Loading training data...")
        
        # Load dataset with fraud labels - Using comma separator
        try:
            df_fraud = pd.read_csv(TRAINING_DATA_WITH_FRAUD, sep=',')
            print(f"   ✅ Loaded fraud dataset: {df_fraud.shape[0]} rows, {df_fraud.shape[1]} columns")
            print(f"   📋 Columns: {list(df_fraud.columns[:5])}...")  # Show first 5 columns
        except Exception as e:
            print(f"   ❌ Error loading fraud dataset: {e}")
            # Try alternative encoding
            df_fraud = pd.read_csv(TRAINING_DATA_WITH_FRAUD, sep=',', encoding='utf-8-sig')
            print(f"   ✅ Loaded fraud dataset with utf-8-sig: {df_fraud.shape[0]} rows, {df_fraud.shape[1]} columns")
        
        # Load dataset without fraud labels
        try:
            df_no_fraud = pd.read_csv(TRAINING_DATA_WITHOUT_FRAUD, sep=',')
            print(f"   ✅ Loaded non-fraud dataset: {df_no_fraud.shape[0]} rows, {df_no_fraud.shape[1]} columns")
        except Exception as e:
            print(f"   ❌ Error loading non-fraud dataset: {e}")
            df_no_fraud = pd.read_csv(TRAINING_DATA_WITHOUT_FRAUD, sep=',', encoding='utf-8-sig')
            print(f"   ✅ Loaded non-fraud dataset with utf-8-sig: {df_no_fraud.shape[0]} rows, {df_no_fraud.shape[1]} columns")
        
        # Check if 'fraud' column exists
        if 'fraud' not in df_fraud.columns:
            print(f"   ⚠️ 'fraud' column not found in fraud dataset!")
            print(f"   Available columns: {list(df_fraud.columns)}")
            # Try to find if there's a similar column
            for col in df_fraud.columns:
                if 'fraud' in col.lower():
                    print(f"   ℹ️ Found similar column: {col}")
        
        # Add fraud column to second dataset (all 0 since no fraud flag)
        df_no_fraud['fraud'] = 0
        
        # Combine datasets
        self.df = pd.concat([df_fraud, df_no_fraud], ignore_index=True)
        print(f"   ✅ Combined dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Check class distribution
        if 'fraud' in self.df.columns:
            fraud_count = self.df['fraud'].sum()
            legit_count = len(self.df) - fraud_count
            print(f"   📊 Class distribution:")
            print(f"      - Legitimate: {legit_count} ({legit_count/len(self.df)*100:.2f}%)")
            print(f"      - Fraudulent: {fraud_count} ({fraud_count/len(self.df)*100:.2f}%)")
        else:
            print(f"   ⚠️ Warning: 'fraud' column not found in combined dataset!")
            print(f"   Available columns: {list(self.df.columns)}")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess features for training"""
        print("\n🔧 Preprocessing data...")
        
        # Handle datetime
        if 'transaction_timestamp' in self.df.columns:
            try:
                # Try multiple datetime formats
                self.df['transaction_timestamp'] = pd.to_datetime(
                    self.df['transaction_timestamp'], 
                    format='%d/%m/%Y %H:%M', 
                    errors='coerce'
                )
                print(f"   ✅ Parsed transaction_timestamp")
            except:
                self.df['transaction_timestamp'] = pd.to_datetime(
                    self.df['transaction_timestamp'], 
                    errors='coerce'
                )
                print(f"   ✅ Parsed transaction_timestamp (auto format)")
        
        # Drop unnecessary columns
        columns_to_drop = []
        
        # Drop transaction_id as it's unique identifier
        if 'transaction_id' in self.df.columns:
            columns_to_drop.append('transaction_id')
            print(f"   🗑️ Dropping: transaction_id")
        
        # Drop customer_id as it's categorical with many unique values
        if 'customer_id' in self.df.columns:
            columns_to_drop.append('customer_id')
            print(f"   🗑️ Dropping: customer_id")
        
        # Drop transaction_timestamp after feature extraction
        if 'transaction_timestamp' in self.df.columns:
            columns_to_drop.append('transaction_timestamp')
            print(f"   🗑️ Dropping: transaction_timestamp")
        
        # Drop true_risk_score if exists (it's correlated with fraud)
        if 'true_risk_score' in self.df.columns:
            columns_to_drop.append('true_risk_score')
            print(f"   🗑️ Dropping: true_risk_score")
        
        # Separate features and target
        if 'fraud' not in self.df.columns:
            raise ValueError("'fraud' column not found in dataframe! Cannot proceed with training.")
        
        self.y = self.df['fraud'].copy()
        self.X = self.df.drop(columns=['fraud'], errors='ignore')
        
        # Drop unnecessary columns
        self.X = self.X.drop(columns=columns_to_drop, errors='ignore')
        
        print(f"   ✅ Removed {len(columns_to_drop)} unnecessary columns")
        print(f"   📊 Feature columns ({len(self.X.columns)}): {list(self.X.columns)}")
        
        # Handle categorical variables
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        print(f"   🔧 Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Convert to string and handle NaN
            self.X[col] = self.X[col].astype(str)
            # Fill 'nan' string with most frequent value
            self.X[col] = self.X[col].replace('nan', self.X[col].mode()[0] if len(self.X[col].mode()) > 0 else 'unknown')
            self.X[col] = le.fit_transform(self.X[col])
            self.label_encoders[col] = le
            print(f"   ✅ Encoded column: {col}")
        
        # Handle missing values in numerical columns
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.X[col].isnull().any():
                self.X[col] = self.X[col].fillna(self.X[col].median())
                print(f"   ✅ Filled missing values in: {col}")
        
        # Save feature columns for inference
        self.feature_columns = list(self.X.columns)
        
        # Remove any rows with NaN in target
        if self.y.isnull().any():
            print(f"   ⚠️ Found {self.y.isnull().sum()} NaN values in target. Removing them...")
            valid_idx = ~self.y.isnull()
            self.X = self.X[valid_idx]
            self.y = self.y[valid_idx]
        
        # Check if we have both classes
        unique_classes = self.y.unique()
        print(f"   📊 Unique classes in target: {unique_classes}")
        
        if len(unique_classes) < 2:
            print(f"   ⚠️ WARNING: Only one class found in target! This will cause issues.")
            print(f"   You need both legitimate (0) and fraudulent (1) transactions for training.")
        
        # Split data
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y if len(unique_classes) > 1 else None
            )
            print(f"   ✅ Training set: {self.X_train.shape[0]} rows")
            print(f"   ✅ Test set: {self.X_test.shape[0]} rows")
        except Exception as e:
            print(f"   ❌ Error in train_test_split: {e}")
            # Fallback without stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            print(f"   ✅ Training set (without stratification): {self.X_train.shape[0]} rows")
            print(f"   ✅ Test set: {self.X_test.shape[0]} rows")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   ✅ Features scaled successfully")
        print(f"   📊 Feature range after scaling: [{self.X_train_scaled.min():.2f}, {self.X_train_scaled.max():.2f}]")
        
        return True
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n📊 Training Logistic Regression...")
        
        # Check if we have both classes
        if len(np.unique(self.y_train)) < 2:
            print(f"   ⚠️ Cannot train Logistic Regression: Only one class in training data")
            print(f"   Creating dummy model...")
            # Create a dummy model that always predicts 0
            from sklearn.dummy import DummyClassifier
            self.logistic_model = DummyClassifier(strategy='constant', constant=0)
            self.logistic_model.fit(self.X_train_scaled, self.y_train)
            print(f"   ⚠️ Created dummy classifier (always predicts 0)")
            return self.logistic_model
        
        self.logistic_model = LogisticRegression(**LOGISTIC_PARAMS)
        self.logistic_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        y_pred = self.logistic_model.predict(self.X_test_scaled)
        y_pred_proba = self.logistic_model.predict_proba(self.X_test_scaled)[:, 1]
        
        print(f"   ✅ Logistic Regression trained")
        print(f"      - Accuracy: {self.logistic_model.score(self.X_test_scaled, self.y_test):.4f}")
        if len(np.unique(self.y_test)) > 1:
            print(f"      - ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        return self.logistic_model
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n🚀 Training XGBoost...")
        
        # Check if we have both classes
        if len(np.unique(self.y_train)) < 2:
            print(f"   ⚠️ Cannot train XGBoost: Only one class in training data")
            print(f"   Creating dummy model...")
            from sklearn.dummy import DummyClassifier
            self.xgb_model = DummyClassifier(strategy='constant', constant=0)
            self.xgb_model.fit(self.X_train_scaled, self.y_train)
            print(f"   ⚠️ Created dummy classifier (always predicts 0)")
            return self.xgb_model
        
        self.xgb_model = XGBClassifier(**XGB_PARAMS)
        self.xgb_model.fit(
            self.X_train_scaled, self.y_train,
            eval_set=[(self.X_test_scaled, self.y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.xgb_model.predict(self.X_test_scaled)
        y_pred_proba = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
        
        print(f"   ✅ XGBoost trained")
        print(f"      - Accuracy: {self.xgb_model.score(self.X_test_scaled, self.y_test):.4f}")
        if len(np.unique(self.y_test)) > 1:
            print(f"      - ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        return self.xgb_model
    
    def optimize_thresholds(self, model, model_name):
        """Optimize decision thresholds based on cost"""
        print(f"\n⚖️ Optimizing thresholds for {model_name}...")
        
        # Get prediction probabilities
        try:
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        except:
            print(f"   ⚠️ Cannot optimize thresholds for {model_name}")
            return {'approve': 0.44, 'review_start': 0.45, 'review_end': 0.84, 'block': 1.0}
        
        # Find optimal thresholds
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_thresholds = {'approve': 0.44, 'review_start': 0.45, 'review_end': 0.84}
        best_cost = float('inf')
        
        for approve_thresh in thresholds:
            for review_thresh in thresholds[thresholds > approve_thresh]:
                # Calculate decisions
                decisions = np.where(
                    y_pred_proba <= approve_thresh, 'approve',
                    np.where(y_pred_proba <= review_thresh, 'review', 'block')
                )
                
                # Calculate cost
                total_cost = 0
                for i, dec in enumerate(decisions):
                    actual = self.y_test.iloc[i] if isinstance(self.y_test, pd.Series) else self.y_test[i]
                    if actual == 1 and dec in ['approve', 'review']:
                        total_cost += COST_PARAMS['false_negative_cost']
                    elif actual == 0 and dec == 'block':
                        total_cost += COST_PARAMS['false_positive_cost']
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_thresholds = {
                        'approve': approve_thresh,
                        'review_start': approve_thresh + 0.01,
                        'review_end': review_thresh,
                        'block': 1.0
                    }
        
        print(f"   ✅ Optimal thresholds found:")
        print(f"      - Approve: ≤ {best_thresholds['approve']:.2f}")
        print(f"      - Review: {best_thresholds['review_start']:.2f} - {best_thresholds['review_end']:.2f}")
        print(f"      - Block: > {best_thresholds['review_end']:.2f}")
        if best_cost != float('inf'):
            print(f"      - Minimum cost: ${best_cost:,.2f}")
        
        return best_thresholds
    
    def save_artifacts(self):
        """Save all trained artifacts"""
        print("\n💾 Saving artifacts...")
        
        # Create models directory if not exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save models
        joblib.dump(self.xgb_model, XGB_MODEL_PATH)
        joblib.dump(self.logistic_model, LOGISTIC_MODEL_PATH)
        print(f"   ✅ Models saved to {MODELS_DIR}")
        
        # Save scaler
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"   ✅ Scaler saved")
        
        # Save label encoders
        joblib.dump(self.label_encoders, ENCODER_PATH)
        print(f"   ✅ Label encoders saved")
        
        # Save feature schema
        feature_schema = {
            'feature_columns': self.feature_columns,
            'categorical_columns': list(self.label_encoders.keys()),
            'numerical_columns': [col for col in self.feature_columns if col not in self.label_encoders.keys()],
            'expected_shape': len(self.feature_columns)
        }
        
        with open(FEATURE_SCHEMA_PATH, 'w') as f:
            json.dump(feature_schema, f, indent=4)
        print(f"   ✅ Feature schema saved")
        
        # Optimize and save thresholds
        print("\n🎯 Optimizing final thresholds...")
        xgb_thresholds = self.optimize_thresholds(self.xgb_model, "XGBoost")
        
        threshold_config = {
            'xgboost': xgb_thresholds,
            'logistic': self.optimize_thresholds(self.logistic_model, "Logistic Regression"),
            'selected_model': 'xgboost',
            'cost_params': COST_PARAMS,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(THRESHOLD_CONFIG_PATH, 'w') as f:
            json.dump(threshold_config, f, indent=4)
        print(f"   ✅ Threshold config saved")
        
        # Save data hash for drift detection
        data_hash = hashlib.md5(pd.util.hash_pandas_object(self.df).values.tobytes()).hexdigest()
        with open(DATA_HASH_PATH, 'w') as f:
            f.write(data_hash)
        print(f"   ✅ Data hash saved: {data_hash[:8]}...")
        
        return True
    
    def generate_training_report(self):
        """Generate training summary report"""
        print("\n📋 Training Summary Report")
        print("=" * 60)
        print(f"Dataset Size: {len(self.df)} transactions")
        print(f"Features Used: {len(self.feature_columns)}")
        if 'fraud' in self.df.columns:
            print(f"Fraud Rate: {self.y.sum()/len(self.y)*100:.2f}%")
        print("\nModel Performance:")
        try:
            print(f"  XGBoost - Accuracy: {self.xgb_model.score(self.X_test_scaled, self.y_test):.4f}")
            print(f"  Logistic - Accuracy: {self.logistic_model.score(self.X_test_scaled, self.y_test):.4f}")
        except:
            print(f"  Models trained successfully")
        print("=" * 60)
        
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        print("🚀 Starting FinGuard AI Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Preprocess
            self.preprocess_data()
            
            # Step 3: Train models
            self.train_logistic_regression()
            self.train_xgboost()
            
            # Step 4: Save artifacts
            self.save_artifacts()
            
            # Step 5: Generate report
            self.generate_training_report()
            
            print("\n✅ Training completed successfully!")
            print(f"📁 All artifacts saved to: {MODELS_DIR}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    trainer = FraudModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 FinGuard AI is ready for deployment!")
    else:
        print("\n⚠️ Training failed. Please check your data files.")