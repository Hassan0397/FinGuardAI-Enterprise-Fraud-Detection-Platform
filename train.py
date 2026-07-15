# train.py
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.training import FraudModelTrainer

def main():
    print("""
    ╔══════════════════════════════════════════╗
    ║     FinGuard AI Model Training v1.0      ║
    ║   Cost-Aware Fraud Detection Platform    ║
    ╚══════════════════════════════════════════╝
    """)
    
    trainer = FraudModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("✨ NEXT STEPS:")
        print("1. Run 'streamlit run app.py' to launch the dashboard")
        print("2. Upload transaction data for fraud detection")
        print("3. View explainable AI decisions")
        print("=" * 60)
    else:
        print("\n❌ Please check:")
        print("1. Both CSV files exist in data/training/ folder")
        print("2. File names match exactly")
        print("3. Files are tab-separated (\\t)")

if __name__ == "__main__":
    main()