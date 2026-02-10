"""
DeBERTa Training Module (v2 - MAE Optimized)
- HuberLoss (smooth MAE) for robust convergence
- Mean Pooling over all tokens (not just [CLS])
- Mixed Precision (AMP) for 2x memory efficiency
- Gradient accumulation for larger effective batch size
- max_length=384 for more context (enabled by AMP memory savings)
- MAE tracking in original space as primary metric
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os
from tqdm import tqdm


class ScoreDataset(Dataset):
    """PyTorch Dataset for score prediction."""
    
    def __init__(self, texts, scores, tokenizer, max_length=384, features=None):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = features
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(self.scores[idx], dtype=torch.float)
        }
        
        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
        
        return item


class DeBERTaRegressor(nn.Module):
    """DeBERTa model with optional feature concatenation."""
    
    def __init__(self, n_features=0):
        super().__init__()
        self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        self.dropout = nn.Dropout(0.1)
        
        # Input size: 768 (DeBERTa) + n_features (handcrafted)
        input_size = 768 + n_features
        self.regressor = nn.Linear(input_size, 1)
        
        self.n_features = n_features
    
    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean Pooling — uses ALL token representations, not just [CLS]
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Concatenate with handcrafted features if provided
        if self.n_features > 0 and features is not None:
            pooled = torch.cat([pooled, features], dim=1)
        
        x = self.dropout(pooled)
        return self.regressor(x).squeeze()


def train_deberta(fold=0, use_features=True, epochs=3, batch_size=12, learning_rate=2e-5,
                  gradient_accumulation_steps=2):
    """
    Train DeBERTa model on specified fold.
    
    Args:
        fold: Fold number to use as validation (0-4)
        use_features: Whether to concatenate handcrafted features
        epochs: Number of training epochs
        batch_size: Batch size for training (effective = batch_size * grad_accum)
        learning_rate: Learning rate for optimizer
        gradient_accumulation_steps: Steps to accumulate gradients
    """
    print("="*60)
    print(f"PHASE 3: TRAINING DeBERTa MODEL (Fold {fold})")
    print("="*60)
    
    # Load processed data
    print("\n[1/8] Loading processed data...")
    train = pd.read_csv('Data/train_processed.csv')
    
    # Split train/val
    train_data = train[train['fold'] != fold].reset_index(drop=True)
    val_data = train[train['fold'] == fold].reset_index(drop=True)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Feature columns
    feature_cols = [
        'title_len', 'body_len', 'title_word_count', 'body_word_count',
        'total_word_count', 'has_body', 'title_body_ratio', 'avg_word_len',
        'title_question_marks', 'title_exclamation_marks',
        'body_question_marks', 'body_exclamation_marks',
        'total_question_marks', 'total_exclamation_marks',
        'title_upper_ratio', 'title_caps_words',
        'title_starts_with_aita', 'has_edit', 'has_update', 'has_tldr',
        'ellipsis_count'
    ]
    
    # Extract and normalize features
    train_features = None
    val_features = None
    scaler = None
    
    if use_features:
        print(f"\n[2/8] Extracting and normalizing {len(feature_cols)} features...")
        train_features_raw = train_data[feature_cols].values
        val_features_raw = val_data[feature_cols].values
        
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features_raw)
        val_features = scaler.transform(val_features_raw)
        print(f"✅ Features normalized (mean=0, std=1)")
    else:
        print("\n[2/8] Skipping features (text-only model)")
    
    # Create datasets
    print("\n[3/8] Creating datasets and dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    train_dataset = ScoreDataset(
        train_data['text'].tolist(),
        train_data['score_log'].values,
        tokenizer,
        features=train_features
    )
    
    val_dataset = ScoreDataset(
        val_data['text'].tolist(),
        val_data['score_log'].values,
        tokenizer,
        features=val_features
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=2, pin_memory=True)
    effective_batch = batch_size * gradient_accumulation_steps
    print(f"✅ Datasets created (batch={batch_size}, effective={effective_batch} via grad accum)")
    
    # Initialize model
    print("\n[4/8] Initializing model...")
    n_features = len(feature_cols) if use_features else 0
    model = DeBERTaRegressor(n_features=n_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model: DeBERTa-v3-base + {n_features} features")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ WARNING: Training on CPU will be VERY slow!")
    
    # Optimizer and scheduler
    print("\n[5/8] Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    criterion = nn.HuberLoss(delta=1.0)  # Smooth MAE — MSE for small errors, MAE for large
    
    use_amp = torch.cuda.is_available()
    amp_scaler = GradScaler('cuda', enabled=use_amp)
    
    print(f"✅ Loss: HuberLoss (delta=1.0) — smooth MAE for better convergence")
    print(f"✅ Optimizer: AdamW (lr={learning_rate})")
    print(f"✅ Scheduler: Linear warmup ({num_warmup_steps} steps)")
    print(f"✅ Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    print(f"✅ Gradient Accumulation: {gradient_accumulation_steps} steps")
    
    # Training loop
    print("\n[6/8] Starting training...")
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc='Training')
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            with autocast('cuda', enabled=use_amp):
                predictions = model(input_ids, attention_mask, features)
                loss = criterion(predictions, scores) / gradient_accumulation_steps
            
            amp_scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                scores = batch['score'].to(device)
                features = batch.get('features')
                if features is not None:
                    features = features.to(device)
                
                with autocast('cuda', enabled=use_amp):
                    predictions = model(input_ids, attention_mask, features)
                val_predictions.extend(predictions.float().cpu().numpy())
                val_actuals.extend(scores.cpu().numpy())
        
        val_predictions = np.array(val_predictions)
        val_actuals = np.array(val_actuals)
        
        # Calculate MAE in log space
        val_mae_log = mean_absolute_error(val_actuals, val_predictions)
        
        # Calculate MAE in original space (competition metric!)
        val_predictions_original = np.expm1(np.clip(val_predictions, 0, 20))
        val_actuals_original = np.expm1(val_actuals)
        val_mae_original = mean_absolute_error(val_actuals_original, val_predictions_original)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss (MAE log): {avg_train_loss:.4f}")
        print(f"  Val MAE (log): {val_mae_log:.4f}")
        print(f"  Val MAE (original): {val_mae_original:.2f}")
        
        # Save best model (track MAE in original space — the actual metric)
        if val_mae_original < best_val_mae:
            best_val_mae = val_mae_original
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae_log': val_mae_log,
                'val_mae_original': val_mae_original,
                'scaler': scaler,
                'n_features': n_features,
                'feature_cols': feature_cols if use_features else None
            }, f'models/deberta_fold{fold}.pth')
            print(f"  ✅ Model saved! Best val MAE (original): {best_val_mae:.2f}")
    
    print("\n[7/8] Training complete!")
    print(f"Best Val MAE (original): {best_val_mae:.2f}")
    print(f"Model saved: models/deberta_fold{fold}.pth")
    
    return model, tokenizer, scaler, best_val_mae


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeBERTa model')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--no-features', action='store_true', help='Train without handcrafted features')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--grad-accum', type=int, default=2, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    train_deberta(
        fold=args.fold,
        use_features=not args.no_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum
    )
