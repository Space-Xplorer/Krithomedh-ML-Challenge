"""
Baseline Models Training (LightGBM & XGBoost)
Phase 4 of POA - Train gradient boosting baselines with TF-IDF
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
import os
import torch


def train_lightgbm(fold=0, max_features=50000, n_estimators=1000):
    """
    Train LightGBM model with TF-IDF features.
    
    Args:
        fold: Fold number to use as validation
        max_features: Maximum number of TF-IDF features
        n_estimators: Number of boosting rounds
    """
    print("="*60)
    print(f"TRAINING LightGBM MODEL (Fold {fold})")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading processed data...")
    train = pd.read_csv('Data/train_processed.csv')
    
    train_data = train[train['fold'] != fold].reset_index(drop=True)
    val_data = train[train['fold'] == fold].reset_index(drop=True)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # TF-IDF vectorization
    print(f"\n[2/5] Creating TF-IDF features (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    
    X_train = vectorizer.fit_transform(train_data['text'])
    X_val = vectorizer.transform(val_data['text'])
    y_train = train_data['score_log'].values
    y_val = val_data['score_log'].values
    
    print(f"✅ TF-IDF matrix created: {X_train.shape}")
    
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    device_type = 'gpu' if use_gpu else 'cpu'
    
    print(f"\n[3/5] Training LightGBM (device={device_type})...")
    
    # Train model
    lgb_params = {
        'n_estimators': n_estimators,
        'objective': 'mae',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'device': device_type,
        'random_state': 42,
        'verbose': -1
    }
    
    model = LGBMRegressor(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[
            # early_stopping(50),
            # log_evaluation(100)
        ]
    )
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    val_preds_log = model.predict(X_val)
    val_mae_log = mean_absolute_error(y_val, val_preds_log)
    
    val_preds_original = np.expm1(np.clip(val_preds_log, 0, 20))
    val_actuals_original = np.expm1(y_val)
    val_mae_original = mean_absolute_error(val_actuals_original, val_preds_original)
    
    print(f"Val MAE (log): {val_mae_log:.4f}")
    print(f"Val MAE (original): {val_mae_original:.2f}")
    
    # Save model
    print("\n[5/5] Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/lightgbm_fold{fold}.pkl')
    joblib.dump(vectorizer, f'models/tfidf_vectorizer_fold{fold}.pkl')
    
    print(f"✅ Model saved: models/lightgbm_fold{fold}.pkl")
    print(f"✅ Vectorizer saved: models/tfidf_vectorizer_fold{fold}.pkl")
    
    return model, vectorizer, val_mae_log


def train_xgboost(fold=0, max_features=50000, n_estimators=1000):
    """
    Train XGBoost model with TF-IDF features.
    
    Args:
        fold: Fold number to use as validation
        max_features: Maximum number of TF-IDF features
        n_estimators: Number of boosting rounds
    """
    print("="*60)
    print(f"TRAINING XGBoost MODEL (Fold {fold})")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading processed data...")
    train = pd.read_csv('Data/train_processed.csv')
    
    train_data = train[train['fold'] != fold].reset_index(drop=True)
    val_data = train[train['fold'] == fold].reset_index(drop=True)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Check if vectorizer already exists
    vectorizer_path = f'models/tfidf_vectorizer_fold{fold}.pkl'
    if os.path.exists(vectorizer_path):
        print(f"\n[2/5] Loading existing TF-IDF vectorizer...")
        vectorizer = joblib.load(vectorizer_path)
        X_train = vectorizer.transform(train_data['text'])
        X_val = vectorizer.transform(val_data['text'])
    else:
        print(f"\n[2/5] Creating TF-IDF features (max_features={max_features})...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        X_train = vectorizer.fit_transform(train_data['text'])
        X_val = vectorizer.transform(val_data['text'])
        joblib.dump(vectorizer, vectorizer_path)
    
    y_train = train_data['score_log'].values
    y_val = val_data['score_log'].values
    
    print(f"✅ TF-IDF matrix: {X_train.shape}")
    
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    
    print(f"\n[3/5] Training XGBoost (GPU={'enabled' if use_gpu else 'disabled'})...")
    
    # Configure parameters
    xgb_params = {
        'n_estimators': n_estimators,
        'objective': 'reg:absoluteerror',
        'learning_rate': 0.01,
        'max_depth': 6,
        'random_state': 42,
        'verbosity': 0
    }
    
    if use_gpu:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['gpu_id'] = 0
        print("  Using GPU acceleration (tree_method='gpu_hist')")
    else:
        xgb_params['tree_method'] = 'hist'
    
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    val_preds_log = model.predict(X_val)
    val_mae_log = mean_absolute_error(y_val, val_preds_log)
    
    val_preds_original = np.expm1(np.clip(val_preds_log, 0, 20))
    val_actuals_original = np.expm1(y_val)
    val_mae_original = mean_absolute_error(val_actuals_original, val_preds_original)
    
    print(f"Val MAE (log): {val_mae_log:.4f}")
    print(f"Val MAE (original): {val_mae_original:.2f}")
    
    # Save model
    print("\n[5/5] Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/xgboost_fold{fold}.pkl')
    
    print(f"✅ Model saved: models/xgboost_fold{fold}.pkl")
    
    return model, vectorizer, val_mae_log


def train_all_baselines(fold=0):
    """Train both LightGBM and XGBoost models."""
    print("\n" + "="*60)
    print("PHASE 4: TRAINING BASELINE MODELS")
    print("="*60)
    
    # Train LightGBM
    print("\n[Model 1/2] LightGBM")
    lgb_model, vectorizer, lgb_mae = train_lightgbm(fold=fold)
    
    # Train XGBoost
    print("\n[Model 2/2] XGBoost")
    xgb_model, _, xgb_mae = train_xgboost(fold=fold)
    
    # Summary
    print("\n" + "="*60)
    print("BASELINE MODELS TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults:")
    print(f"  LightGBM MAE (log): {lgb_mae:.4f}")
    print(f"  XGBoost MAE (log): {xgb_mae:.4f}")
    
    return lgb_model, xgb_model, vectorizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--model', type=str, choices=['lightgbm', 'xgboost', 'both'], 
                       default='both', help='Which model to train')
    parser.add_argument('--max-features', type=int, default=50000, help='Max TF-IDF features')
    parser.add_argument('--n-estimators', type=int, default=1000, help='Number of estimators')
    
    args = parser.parse_args()
    
    if args.model == 'lightgbm':
        train_lightgbm(args.fold, args.max_features, args.n_estimators)
    elif args.model == 'xgboost':
        train_xgboost(args.fold, args.max_features, args.n_estimators)
    else:
        train_all_baselines(args.fold)
