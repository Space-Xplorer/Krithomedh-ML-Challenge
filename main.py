"""
Main Execution Script (v2 - MAE Optimized)
Run complete pipeline from preprocessing to submission.
All models now train with MAE objectives to match competition metric.
"""

import os
import sys
import time
import glob
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_data
from train_deberta import train_deberta
from train_baselines import train_all_baselines
from ensemble import validate_ensemble_oof, create_submission


def print_banner(text):
    """Print formatted banner."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def clean_old_models():
    """Delete old models trained with MSE loss."""
    print("Cleaning old models (trained with wrong MSE objective)...")
    model_dir = 'models'
    if os.path.exists(model_dir):
        old_files = glob.glob(os.path.join(model_dir, '*'))
        for f in old_files:
            os.remove(f)
            print(f"  Deleted: {os.path.basename(f)}")
        print(f"✅ Removed {len(old_files)} old model files")
    else:
        print("  No old models found")


def main():
    """Execute complete pipeline."""
    start_time = time.time()
    
    print_banner("ML CHALLENGE - MAE-OPTIMIZED PIPELINE v2")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Metric: MAE (Mean Absolute Error) — lower is better")
    print(f"Current best: 1096.45 MAE (2nd place)")
    print(f"Target: < 1089.21 MAE (beat 1st place)")
    
    # Phase 1: Check environment
    print_banner("PHASE 1: ENVIRONMENT CHECK")
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  PyTorch: {torch.__version__}")
    
    # Phase 2: Preprocessing
    print_banner("PHASE 2: DATA PREPROCESSING")
    if not os.path.exists('Data/train_processed.csv'):
        print("Running preprocessing...")
        preprocess_data()
    else:
        print("✓ Processed data already exists. Skipping preprocessing.")
    
    # Phase 3: Train DeBERTa Fold 0 (with L1Loss + AMP)
    print_banner("PHASE 3: TRAINING DeBERTa FOLD 0 (L1Loss + AMP)")
    if not os.path.exists('models/deberta_fold0.pth'):
        print("Training DeBERTa on Fold 0 with MAE objective...")
        train_deberta(fold=0, use_features=True, epochs=3, batch_size=12,
                     gradient_accumulation_steps=2)
    else:
        print("✓ DeBERTa Fold 0 model already exists. Skipping.")
    
    # Phase 4a: Train DeBERTa Fold 1
    print_banner("PHASE 4a: TRAINING DeBERTa FOLD 1 (L1Loss + AMP)")
    if not os.path.exists('models/deberta_fold1.pth'):
        print("Training DeBERTa on Fold 1 with MAE objective...")
        train_deberta(fold=1, use_features=True, epochs=3, batch_size=12,
                     gradient_accumulation_steps=2)
    else:
        print("✓ DeBERTa Fold 1 model already exists. Skipping.")
    
    # Phase 4b: Train baseline models (with MAE objectives)
    print_banner("PHASE 4b: TRAINING BASELINES (MAE Objectives)")
    if not os.path.exists('models/lightgbm_fold0.pkl'):
        print("Training LightGBM (objective=mae) and XGBoost (objective=reg:absoluteerror)...")
        train_all_baselines(fold=0)
    else:
        print("✓ Baseline models already exist. Skipping.")
    
    # Phase 5: Ensemble validation (OOF — no data leakage)
    print_banner("PHASE 5: ENSEMBLE VALIDATION (OOF - No Leakage)")
    print("Using Out-of-Fold predictions to eliminate data leakage...")
    weights, oof_mae = validate_ensemble_oof()
    print(f"\n✅ Ensemble OOF MAE: {oof_mae:.2f}")
    
    # Phase 7: Create submission
    print_banner("PHASE 7: CREATING SUBMISSION")
    submission = create_submission()
    
    # Summary
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print_banner("PIPELINE COMPLETE!")
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")
    print(f"Submission file: submissions/Solution.csv")
    print(f"\n📊 Key Changes in v2:")
    print(f"  ✅ DeBERTa trained with L1Loss (MAE) instead of MSELoss")
    print(f"  ✅ LightGBM trained with objective='mae'")
    print(f"  ✅ XGBoost trained with objective='reg:absoluteerror'")
    print(f"  ✅ Ensemble weights optimized for MAE (not RMSE)")
    print(f"  ✅ OOF validation — no data leakage")
    print(f"  ✅ AMP mixed precision for DeBERTa")
    print(f"  ✅ max_length=384 for more text context")
    print(f"\n🎯 OOF MAE: {oof_mae:.2f}")
    print(f"\n✅ Ready to submit! Good luck! 🚀")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML challenge pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing if already done')
    parser.add_argument('--only-submission', action='store_true',
                       help='Only create submission from existing models')
    
    args = parser.parse_args()
    
    if args.only_submission:
        print_banner("CREATING SUBMISSION ONLY")
        create_submission()
    else:
        main()
