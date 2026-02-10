"""
Data Preprocessing and Feature Engineering
Phase 2 of POA - Load, clean, extract features, log transform, create folds
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib
import os


def extract_features(df):
    """
    Extract 20+ handcrafted features from text.
    
    Features:
    - 8 length features
    - 6 punctuation features
    - 2 capitalization features
    - 5 content markers
    """
    print("Extracting features...")
    
    # Length features
    df['title_len'] = df['title'].str.len()
    df['body_len'] = df['body'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['body_word_count'] = df['body'].str.split().str.len()
    df['total_word_count'] = df['title_word_count'] + df['body_word_count']
    
    # Binary flag
    df['has_body'] = (df['body'] != '').astype(int)
    
    # Ratio features
    df['title_body_ratio'] = df['title_len'] / (df['body_len'] + 1)
    
    # Average word length
    df['avg_word_len'] = df.apply(
        lambda row: np.mean([len(w) for w in (row['title'] + ' ' + row['body']).split()]) 
        if len((row['title'] + ' ' + row['body']).split()) > 0 else 0,
        axis=1
    )
    
    # Punctuation features
    df['title_question_marks'] = df['title'].str.count(r'\?')
    df['title_exclamation_marks'] = df['title'].str.count(r'!')
    df['body_question_marks'] = df['body'].str.count(r'\?')
    df['body_exclamation_marks'] = df['body'].str.count(r'!')
    df['total_question_marks'] = df['title_question_marks'] + df['body_question_marks']
    df['total_exclamation_marks'] = df['title_exclamation_marks'] + df['body_exclamation_marks']
    
    # Capitalization features (title)
    df['title_upper_ratio'] = df['title'].apply(
        lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0
    )
    df['title_caps_words'] = df['title'].apply(
        lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1)
    )
    
    # Special content markers
    df['title_starts_with_aita'] = df['title'].str.lower().str.startswith('aita').astype(int)
    df['has_edit'] = df['body'].str.lower().str.contains('edit:', na=False).astype(int)
    df['has_update'] = df['body'].str.lower().str.contains('update:', na=False).astype(int)
    df['has_tldr'] = df['body'].str.lower().str.contains('tl;?dr', na=False).astype(int)
    
    # Ellipsis (suspense/trailing)
    df['ellipsis_count'] = (df['title'].str.count(r'\.\.\.') + df['body'].str.count(r'\.\.\.')).astype(int)
    
    # ID type feature (CRITICAL for segmented ensemble)
    df['is_numeric_id'] = df['id'].apply(lambda x: 1 if str(x).isdigit() else 0)
    
    print("✅ Features extracted:")
    print(f"  - Length features: title_len, body_len, word counts (8)")
    print(f"  - Punctuation: question marks, exclamation marks (6)")
    print(f"  - Capitalization: upper_ratio, caps_words (2)")
    print(f"  - Content markers: AITA, edit, update, tldr, ellipsis (5)")
    print(f"  - ID type: is_numeric_id (1) — CRITICAL for segmentation")
    
    return df


def preprocess_data():
    """
    Main preprocessing pipeline:
    1. Load train and test data
    2. Handle missing bodies
    3. Extract features
    4. Apply log transformation
    5. Create stratified folds
    6. Save processed data
    """
    print("="*60)
    print("PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    train = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')
    print(f"Loaded train: {train.shape}, test: {test.shape}")
    
    # Handle missing bodies
    print("\n[2/6] Handling missing bodies...")
    train['body'] = train['body'].fillna('')
    test['body'] = test['body'].fillna('')
    missing_pct = (train['body'] == '').mean()
    print(f"Missing bodies handled. Train missing: {(train['body'] == '').sum()} ({missing_pct:.2%})")
    
    # Extract features
    print("\n[3/6] Extracting features...")
    train = extract_features(train)
    test = extract_features(test)
    
    # Create combined text for models
    print("\n[4/6] Creating combined text...")
    train['text'] = train['title'] + ' [SEP] ' + train['body']
    test['text'] = test['title'] + ' [SEP] ' + test['body']
    print("✅ Combined text created with [SEP] separator")
    
    # Log transform target
    print("\n[5/6] Applying log transformation to target...")
    train['score_log'] = np.log1p(train['score'])
    
    print(f"\nOriginal score stats:")
    print(f"  Mean: {train['score'].mean():.2f}, Std: {train['score'].std():.2f}")
    print(f"  Skewness: {train['score'].skew():.2f}")
    print(f"\nLog-transformed score stats:")
    print(f"  Mean: {train['score_log'].mean():.2f}, Std: {train['score_log'].std():.2f}")
    print(f"  Skewness: {train['score_log'].skew():.2f}")
    print("✅ Skewness reduced significantly!")
    
    # Create stratified folds
    print("\n[6/6] Creating stratified folds...")
    train['score_bin'] = pd.qcut(train['score_log'], q=10, labels=False, duplicates='drop')
    train['strat_key'] = train['score_bin'].astype(str) + '_' + train['has_body'].astype(str)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train['fold'] = -1
    
    for fold_id, (_, val_idx) in enumerate(skf.split(train, train['strat_key'])):
        train.loc[val_idx, 'fold'] = fold_id
    
    # Validate fold distribution
    print("\nFold validation:")
    for fold in range(5):
        fold_data = train[train['fold'] == fold]
        missing_pct = (fold_data['body'] == '').mean()
        print(f"  Fold {fold}: {len(fold_data)} samples, {missing_pct:.2%} missing bodies")
    
    # Drop temporary columns
    train = train.drop(['score_bin', 'strat_key'], axis=1)
    
    # Save processed data
    print("\n[7/6] Saving processed data...")
    os.makedirs('Data', exist_ok=True)
    train.to_csv('Data/train_processed.csv', index=False)
    test.to_csv('Data/test_processed.csv', index=False)
    
    print("\n✅ PREPROCESSING COMPLETE!")
    print(f"  - Data/train_processed.csv ({train.shape})")
    print(f"  - Data/test_processed.csv ({test.shape})")
    print(f"  - Features: {len([col for col in train.columns if col not in ['id', 'title', 'body', 'text', 'score', 'score_log', 'fold']])} engineered features")
    
    return train, test


def get_feature_columns():
    """Return list of feature column names."""
    return [
        'title_len', 'body_len', 'title_word_count', 'body_word_count',
        'total_word_count', 'has_body', 'title_body_ratio', 'avg_word_len',
        'title_question_marks', 'title_exclamation_marks', 
        'body_question_marks', 'body_exclamation_marks',
        'total_question_marks', 'total_exclamation_marks',
        'title_upper_ratio', 'title_caps_words',
        'title_starts_with_aita', 'has_edit', 'has_update', 'has_tldr',
        'ellipsis_count', 'is_numeric_id'
    ]


if __name__ == "__main__":
    train, test = preprocess_data()
    
    # Print feature summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    feature_cols = get_feature_columns()
    print(f"Total features: {len(feature_cols)}")
    print("\nFeature list:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
