"""
Baseline Models Training (LightGBM & XGBoost v3)
Phase 4 of POA - Train gradient boosting baselines
XGBoost v3: Feature Fusion (TF-IDF + 22 handcrafted features including is_numeric_id)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from scipy.sparse import hstack, csr_matrix
import joblib
import os
import torch
import sys

# Import feature columns
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_feature_columns


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
            early_stopping(50, verbose=True),
            log_evaluation(100)
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


def train_xgboost(fold=0, max_features=25000, n_estimators=2000):
    """
    Train XGBoost v3 with FUSION features (TF-IDF + Tabular + ID type).
    
    This upgrade targets sub-1085 MAE by recognizing that Numeric IDs (median=526)
    and Alphanumeric IDs (median=22) are fundamentally different populations.
    
    Args:
        fold: Fold number to use as validation
        max_features: Maximum number of TF-IDF features (25k, reduced from 50k)
        n_estimators: Number of boosting rounds (2000, increased for fusion)
    """
    print("="*60)
    print(f"TRAINING XGBoost v3 MODEL (Fold {fold}) - FEATURE FUSION")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading processed data...")
    train = pd.read_csv('Data/train_processed.csv')
    
    train_data = train[train['fold'] != fold].reset_index(drop=True)
    val_data = train[train['fold'] == fold].reset_index(drop=True)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Analyze ID distribution
    print("\n[2/6] Analyzing ID type distribution...")
    train_numeric = train_data['is_numeric_id'].mean()
    val_numeric = val_data['is_numeric_id'].mean()
    print(f"  Train: {train_numeric:.1%} numeric IDs, {1-train_numeric:.1%} alphanumeric")
    print(f"  Val: {val_numeric:.1%} numeric IDs, {1-val_numeric:.1%} alphanumeric")
    
    # TF-IDF features
    print(f"\n[3/6] Creating TF-IDF features (max_features={max_features})...")
    vectorizer_path = f'models/tfidf_vectorizer_fold{fold}.pkl'
    if os.path.exists(vectorizer_path):
        print("  Loading existing vectorizer...")
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("  Creating new vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents='unicode'
        )
    
    X_train_tfidf = vectorizer.fit_transform(train_data['text'])
    X_val_tfidf = vectorizer.transform(val_data['text'])
    print(f"  TF-IDF matrix: {X_train_tfidf.shape}")
    
    # Tabular features (22 features including is_numeric_id)
    print("\n[4/6] Extracting tabular features (22 features)...")
    feature_cols = get_feature_columns()
    print(f"  Features: {len(feature_cols)} total (including is_numeric_id)")
    
    # Normalize tabular features
    scaler = StandardScaler()
    X_train_tab = scaler.fit_transform(train_data[feature_cols].values)
    X_val_tab = scaler.transform(val_data[feature_cols].values)
    
    # Convert to sparse and combine
    X_train_tab_sparse = csr_matrix(X_train_tab)
    X_val_tab_sparse = csr_matrix(X_val_tab)
    
    X_train = hstack([X_train_tfidf, X_train_tab_sparse])
    X_val = hstack([X_val_tfidf, X_val_tab_sparse])
    
    print(f"  Combined feature matrix: {X_train.shape}")
    print(f"    - TF-IDF: {X_train_tfidf.shape[1]} features")
    print(f"    - Tabular: {len(feature_cols)} features")
    
    y_train = train_data['score_log'].values
    y_val = val_data['score_log'].values
    
    # Train model
    print(f"\n[5/6] Training XGBoost v3 (CPU, Feature Fusion)...")
    
    xgb_params = {
        'n_estimators': n_estimators,
        'objective': 'reg:absoluteerror',
        'learning_rate': 0.02,  # Reduced for more estimators
        'max_depth': 7,  # Increased for richer features
        'tree_method': 'hist',
        'n_jobs': -1,
        'reg_alpha': 0.1,  # L1 regularization for outlier robustness
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 100
    }
    
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Evaluate
    print("\n[6/6] Evaluating model...")
    val_preds_log = model.predict(X_val)
    val_mae_log = mean_absolute_error(y_val, val_preds_log)
    
    val_preds_original = np.expm1(np.clip(val_preds_log, 0, 20))
    val_actuals_original = np.expm1(y_val)
    val_mae_original = mean_absolute_error(val_actuals_original, val_preds_original)
    
    print(f"Val MAE (log): {val_mae_log:.4f}")
    print(f"Val MAE (original): {val_mae_original:.2f}")
    
    # Save model and metadata
    print("\nSaving model and metadata...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/xgboost_fold{fold}.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(scaler, f'models/xgb_scaler_fold{fold}.pkl')
    joblib.dump(feature_cols, f'models/xgb_features_fold{fold}.pkl')
    
    print(f"✅ Model saved: models/xgboost_fold{fold}.pkl")
    print(f"✅ Scaler saved: models/xgb_scaler_fold{fold}.pkl")
    print(f"✅ Feature config saved: models/xgb_features_fold{fold}.pkl")
    
    return model, vectorizer, val_mae_log


def train_all_baselines(fold=0):
    """Train both LightGBM and XGBoost models."""
    print("\n" + "="*60)
    print("PHASE 4: TRAINING BASELINE MODELS")
    print("="*60)
    
    # Train LightGBM
    print("\n[Model 1/2] LightGBM")
    lgb_model, vectorizer, lgb_mae = train_lightgbm(fold=fold)
    
    # Train XGBoost v3
    print("\n[Model 2/2] XGBoost v3 (Feature Fusion)")
    xgb_model, _, xgb_mae = train_xgboost(fold=fold)
    
    # Summary
    print("\n" + "="*60)
    print("BASELINE MODELS TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults:")
    print(f"  LightGBM MAE (log): {lgb_mae:.4f}")
    print(f"  XGBoost v3 MAE (log): {xgb_mae:.4f}")
    
    return lgb_model, xgb_model, vectorizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--model', type=str, choices=['lightgbm', 'xgboost', 'both'], 
                       default='both', help='Which model to train')
    parser.add_argument('--max-features', type=int, default=25000, help='Max TF-IDF features')
    parser.add_argument('--n-estimators', type=int, default=2000, help='Number of estimators')
    
    args = parser.parse_args()
    
    if args.model == 'lightgbm':
        train_lightgbm(args.fold, 50000, args.n_estimators)
    elif args.model == 'xgboost':
        train_xgboost(args.fold, args.max_features, args.n_estimators)
    else:
        train_all_baselines(args.fold)
