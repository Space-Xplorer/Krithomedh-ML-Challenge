"""
Ensemble and Prediction Module (v2 - MAE Optimized, No Data Leakage)
- OOF (Out-of-Fold) prediction approach eliminates data leakage
- MAE-based weight optimization matches competition metric
- Proper cross-validation: each model only predicts on its held-out fold
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import joblib
import os
import sys

# Import custom modules
sys.path.append(os.path.dirname(__file__))
from train_deberta import DeBERTaRegressor, ScoreDataset


def load_deberta_model(model_path, device='cuda'):
    """Load trained DeBERTa model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_features = checkpoint['n_features']
    
    model = DeBERTaRegressor(n_features=n_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('scaler'), checkpoint.get('feature_cols')


def predict_deberta(model, texts, features, tokenizer, device, batch_size=16):
    """Generate predictions using DeBERTa model with AMP."""
    dataset = ScoreDataset(texts, np.zeros(len(texts)), tokenizer, features=features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    use_amp = device.type == 'cuda'
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            feats = batch.get('features')
            if feats is not None:
                feats = feats.to(device)
            
            with autocast(enabled=use_amp):
                preds = model(input_ids, attention_mask, feats)
            predictions.extend(preds.float().cpu().numpy())
    
    return np.array(predictions)


def optimize_ensemble_weights(predictions_list, y_true, model_names=None):
    """
    Optimize ensemble weights using MAE (competition metric).
    
    Args:
        predictions_list: List of prediction arrays from different models
        y_true: True target values (log space)
        model_names: Names for display
    
    Returns:
        Optimal weights array
    """
    print("\n[Optimizing Ensemble Weights (MAE)]")
    
    predictions_array = np.array(predictions_list)
    
    def objective(weights):
        """Minimize MAE in ORIGINAL space — the actual competition metric."""
        ensemble_pred_log = np.average(predictions_array, axis=0, weights=weights)
        ensemble_pred_orig = np.expm1(np.clip(ensemble_pred_log, 0, 20))
        y_true_orig = np.expm1(y_true)
        return mean_absolute_error(y_true_orig, ensemble_pred_orig)
    
    # Constraints: weights sum to 1, all weights >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(len(predictions_list))]
    
    # Initial guess: equal weights
    x0 = np.ones(len(predictions_list)) / len(predictions_list)
    
    # Optimize with multiple restarts for robustness
    best_result = None
    best_mae = float('inf')
    
    for seed in range(5):
        np.random.seed(seed)
        if seed == 0:
            init = x0
        else:
            init = np.random.dirichlet(np.ones(len(predictions_list)))
        
        result = minimize(
            objective,
            init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.fun < best_mae:
            best_mae = result.fun
            best_result = result
    
    optimal_weights = best_result.x
    final_mae = best_result.fun
    
    if model_names is None:
        model_names = [f'Model {i}' for i in range(len(optimal_weights))]
    
    print("Optimized Weights:")
    for name, weight in zip(model_names, optimal_weights):
        print(f"  {name}: {weight:.4f}")
    print(f"\nEnsemble MAE (original space): {final_mae:.2f}")
    
    return optimal_weights


def validate_ensemble_oof():
    """
    Validate ensemble using proper Out-of-Fold (OOF) predictions.
    NO DATA LEAKAGE: each model only predicts on data it was NOT trained on.
    
    DeBERTa Fold 0 → trained on folds {1,2,3,4} → predicts fold 0
    DeBERTa Fold 1 → trained on folds {0,2,3,4} → predicts fold 1
    LightGBM Fold 0 → trained on folds {1,2,3,4} → predicts fold 0
    XGBoost Fold 0 → trained on folds {1,2,3,4} → predicts fold 0
    
    For proper ensemble OOF, we use the overlapping fold (fold 0) where 
    DeBERTa Fold 0, LightGBM, and XGBoost all have held-out predictions.
    Then use fold 1 for DeBERTa Fold 1 separately.
    """
    print("="*60)
    print("PHASE 5: ENSEMBLE VALIDATION (OOF - No Data Leakage)")
    print("="*60)
    
    # Load data
    train = pd.read_csv('Data/train_processed.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    # =========================================================
    # Step 1: Generate OOF predictions for each model
    # =========================================================
    print("\n[1/4] Generating Out-of-Fold predictions...")
    
    # --- DeBERTa Fold 0: predicts on fold 0 (its held-out fold) ---
    print("\n  DeBERTa Fold 0 → predicting on fold 0 (OOF)...")
    deberta0, scaler0, feature_cols = load_deberta_model('models/deberta_fold0.pth', device)
    fold0_data = train[train['fold'] == 0].reset_index(drop=True)
    
    if feature_cols and scaler0:
        fold0_features_d0 = scaler0.transform(fold0_data[feature_cols].values)
    else:
        fold0_features_d0 = None
    
    oof_deberta0 = predict_deberta(deberta0, fold0_data['text'].tolist(),
                                    fold0_features_d0, tokenizer, device)
    y_fold0 = fold0_data['score_log'].values
    del deberta0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- DeBERTa Fold 1: predicts on fold 1 (its held-out fold) ---
    print("  DeBERTa Fold 1 → predicting on fold 1 (OOF)...")
    deberta1, scaler1, _ = load_deberta_model('models/deberta_fold1.pth', device)
    fold1_data = train[train['fold'] == 1].reset_index(drop=True)
    
    if feature_cols and scaler1:
        fold1_features_d1 = scaler1.transform(fold1_data[feature_cols].values)
    else:
        fold1_features_d1 = None
    
    oof_deberta1 = predict_deberta(deberta1, fold1_data['text'].tolist(),
                                    fold1_features_d1, tokenizer, device)
    y_fold1 = fold1_data['score_log'].values
    del deberta1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- LightGBM Fold 0: predicts on fold 0 (its held-out fold) ---
    print("  LightGBM → predicting on fold 0 (OOF)...")
    lgb_model = joblib.load('models/lightgbm_fold0.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer_fold0.pkl')
    X_fold0 = vectorizer.transform(fold0_data['text'])
    oof_lgb_fold0 = lgb_model.predict(X_fold0)
    
    # --- XGBoost Fold 0: predicts on fold 0 (its held-out fold) ---
    print("  XGBoost → predicting on fold 0 (OOF)...")
    xgb_model = joblib.load('models/xgboost_fold0.pkl')
    oof_xgb_fold0 = xgb_model.predict(X_fold0)
    
    # =========================================================
    # Step 2: Evaluate individual model OOF performance
    # =========================================================
    print("\n[2/4] Individual model OOF performance (MAE, original space):")
    
    def calc_mae_original(y_true_log, y_pred_log):
        y_true_orig = np.expm1(y_true_log)
        y_pred_orig = np.expm1(np.clip(y_pred_log, 0, 20))
        return mean_absolute_error(y_true_orig, y_pred_orig)
    
    mae_d0 = calc_mae_original(y_fold0, oof_deberta0)
    mae_d1 = calc_mae_original(y_fold1, oof_deberta1)
    mae_lgb = calc_mae_original(y_fold0, oof_lgb_fold0)
    mae_xgb = calc_mae_original(y_fold0, oof_xgb_fold0)
    
    print(f"  DeBERTa Fold 0 OOF MAE: {mae_d0:.2f}")
    print(f"  DeBERTa Fold 1 OOF MAE: {mae_d1:.2f}")
    print(f"  LightGBM OOF MAE: {mae_lgb:.2f}")
    print(f"  XGBoost OOF MAE: {mae_xgb:.2f}")
    
    # =========================================================
    # Step 3: Optimize ensemble weights on fold 0 
    # (where DeBERTa0, LGB, XGB all have OOF predictions)
    # Then also include DeBERTa1's performance to set its weight
    # =========================================================
    print("\n[3/4] Optimizing ensemble weights on fold 0 OOF predictions...")
    
    # Strategy: Optimize 3-model weights on fold 0 (DeBERTa0, LGB, XGB)
    # Then separately determine DeBERTa1 vs DeBERTa0 blend weight
    
    # First: optimize DeBERTa0 + LGB + XGB on fold 0
    predictions_fold0 = [oof_deberta0, oof_lgb_fold0, oof_xgb_fold0]
    model_names_fold0 = ['DeBERTa Fold 0', 'LightGBM', 'XGBoost']
    weights_fold0 = optimize_ensemble_weights(predictions_fold0, y_fold0, model_names_fold0)
    
    # Now we need to set the final 4-model weights
    # Since DeBERTa F0 and F1 are trained on different folds but same architecture,
    # we blend them equally and distribute the DeBERTa weight
    deberta_total_weight = weights_fold0[0]  # DeBERTa share from fold0 optimization
    
    # Final weights: split DeBERTa weight equally between fold 0 and fold 1
    final_weights = np.array([
        deberta_total_weight / 2,   # DeBERTa Fold 0
        deberta_total_weight / 2,   # DeBERTa Fold 1
        weights_fold0[1],            # LightGBM
        weights_fold0[2]             # XGBoost
    ])
    
    # Normalize to sum to 1
    final_weights = final_weights / final_weights.sum()
    
    print("\n[4/4] Final 4-model ensemble weights:")
    model_names = ['DeBERTa Fold 0', 'DeBERTa Fold 1', 'LightGBM', 'XGBoost']
    for name, weight in zip(model_names, final_weights):
        print(f"  {name}: {weight:.4f}")
    
    # Estimate overall ensemble MAE from fold 0 metrics
    fold0_ensemble = np.average(predictions_fold0, axis=0, weights=weights_fold0)
    fold0_mae = calc_mae_original(y_fold0, fold0_ensemble)
    print(f"\nFold 0 Ensemble MAE (original): {fold0_mae:.2f}")
    
    # Save weights
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_weights, 'models/ensemble_weights.pkl')
    print("\n✅ Weights saved: models/ensemble_weights.pkl")
    
    return final_weights, fold0_mae


def create_submission(weights=None):
    """
    Create final submission file.
    Phase 7 of POA.
    """
    print("="*60)
    print("PHASE 7: CREATING SUBMISSION")
    print("="*60)
    
    # Load test data
    print("\n[1/5] Loading test data...")
    test = pd.read_csv('Data/test_processed.csv')
    print(f"Test size: {len(test)}")
    
    # Load weights
    if weights is None:
        weights = joblib.load('models/ensemble_weights.pkl')
        print("✅ Loaded ensemble weights")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("\n[2/5] Loading models...")
    deberta0, scaler0, feature_cols = load_deberta_model('models/deberta_fold0.pth', device)
    deberta1, scaler1, _ = load_deberta_model('models/deberta_fold1.pth', device)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    lgb_model = joblib.load('models/lightgbm_fold0.pkl')
    xgb_model = joblib.load('models/xgboost_fold0.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer_fold0.pkl')
    print("✅ All models loaded")
    
    # Generate predictions
    print("\n[3/5] Generating predictions...")
    
    # Prepare features
    if feature_cols:
        test_features0 = scaler0.transform(test[feature_cols].values)
        test_features1 = scaler1.transform(test[feature_cols].values)
    else:
        test_features0 = None
        test_features1 = None
    
    print("  DeBERTa Fold 0...")
    pred_deberta0 = predict_deberta(deberta0, test['text'].tolist(),
                                    test_features0, tokenizer, device)
    del deberta0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("  DeBERTa Fold 1...")
    pred_deberta1 = predict_deberta(deberta1, test['text'].tolist(),
                                    test_features1, tokenizer, device)
    del deberta1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("  LightGBM...")
    X_test = vectorizer.transform(test['text'])
    pred_lgb = lgb_model.predict(X_test)
    
    print("  XGBoost...")
    pred_xgb = xgb_model.predict(X_test)
    
    # Ensemble predictions
    print("\n[4/5] Creating ensemble...")
    predictions_list = [pred_deberta0, pred_deberta1, pred_lgb, pred_xgb]
    
    print("  Weights being applied:")
    model_names = ['DeBERTa Fold 0', 'DeBERTa Fold 1', 'LightGBM', 'XGBoost']
    for name, w in zip(model_names, weights):
        print(f"    {name}: {w:.4f}")
    
    ensemble_pred_log = np.average(predictions_list, axis=0, weights=weights)
    
    # Convert to original space with clipping
    ensemble_pred_log = np.clip(ensemble_pred_log, 0, 20)
    ensemble_pred = np.expm1(ensemble_pred_log)
    
    # Clip negative predictions
    ensemble_pred = np.clip(ensemble_pred, 0, None)
    
    # Round to integers (scores are integers in the dataset)
    ensemble_pred = np.round(ensemble_pred).astype(int)
    
    print(f"\n  Prediction stats:")
    print(f"    Range: [{ensemble_pred.min()}, {ensemble_pred.max()}]")
    print(f"    Median: {np.median(ensemble_pred):.0f}")
    print(f"    Mean: {np.mean(ensemble_pred):.0f}")
    print(f"    P1: {np.percentile(ensemble_pred, 1):.0f}")
    print(f"    P99: {np.percentile(ensemble_pred, 99):.0f}")
    
    # Create submission
    print("\n[5/5] Creating submission file...")
    submission = pd.DataFrame({
        'id': test['id'],
        'score': ensemble_pred
    })
    
    # Validate
    assert submission.shape[0] == 15000, f"Expected 15000 rows, got {submission.shape[0]}"
    assert list(submission.columns) == ['id', 'score'], f"Wrong columns: {submission.columns.tolist()}"
    assert submission['score'].min() >= 0, "Negative scores detected!"
    assert submission.isnull().sum().sum() == 0, "Missing values detected!"
    
    # Save as Solution.csv (competition format)
    os.makedirs('submissions', exist_ok=True)
    submission.to_csv('submissions/Solution.csv', index=False)
    
    print("\n✅ SUBMISSION CREATED!")
    print(f"  File: submissions/Solution.csv")
    print(f"  Shape: {submission.shape}")
    print(f"  Score range: [{submission['score'].min()}, {submission['score'].max()}]")
    print(f"  Median score: {submission['score'].median():.0f}")
    
    return submission


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble and submission')
    parser.add_argument('--action', type=str, choices=['validate', 'submit', 'both'],
                       default='both', help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action in ['validate', 'both']:
        weights, avg_mae = validate_ensemble_oof()
        print(f"\n🎯 Ensemble validation complete. OOF MAE: {avg_mae:.2f}")
    
    if args.action in ['submit', 'both']:
        submission = create_submission()
        print("\n🚀 Ready for submission!")
