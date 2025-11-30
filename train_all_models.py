"""
Unified training script for all models
Supports: LSTM, GRU, Transformer, XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_fetcher import CCXTDataFetcher
from config import (
    SYMBOL, TIMEFRAME, DATA_LIMIT, TRAIN_TEST_SPLIT,
    MIN_DATA_POINTS, SEQUENCE_LENGTH
)
import os
import sys

# Import all models
from models.lstm.model import LSTMTrainer
from models.gru.model import GRUTrainer
from models.transformer.model import TransformerTrainer
from models.xgboost.model import XGBoostTrainer


MODEL_MAP = {
    'lstm': LSTMTrainer,
    'gru': GRUTrainer,
    'transformer': TransformerTrainer,
    'xgboost': XGBoostTrainer
}


def train_model(model_type='lstm'):
    """
    Train a specific model
    
    Args:
        model_type: 'lstm', 'gru', 'transformer', or 'xgboost'
    """
    if model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_MAP.keys())}")
    
    print("=" * 60)
    print(f"{model_type.upper()} Training for Crypto Trading Signals")
    print("=" * 60)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n[1/4] Fetching data...")
    fetcher = CCXTDataFetcher()
    data = fetcher.fetch_ohlcv(limit=DATA_LIMIT)
    
    if data is None or len(data) < MIN_DATA_POINTS:
        print(f"✗ Insufficient data. Need at least {MIN_DATA_POINTS} points, got {len(data) if data is not None else 0}")
        return None
    
    print(f"✓ Fetched {len(data)} candles")
    
    # Step 2: Prepare features
    print("\n[2/4] Preparing features...")
    trainer = MODEL_MAP[model_type]()
    features = trainer.prepare_features(data)
    
    print(f"✓ Created {len(trainer.feature_engineer.feature_columns)} features")
    print(f"  Features: {', '.join(trainer.feature_engineer.feature_columns[:10])}...")
    
    # Step 3: Create sequences
    print("\n[3/4] Creating sequences...")
    X, y = trainer.create_sequences(features)
    
    print(f"✓ Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-TRAIN_TEST_SPLIT, shuffle=False
    )
    
    print(f"\n  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # Step 4: Build and train model
    print("\n[4/4] Building and training model...")
    model = trainer.build_model()
    
    if hasattr(model, '__str__'):
        print("\nModel Architecture:")
        print(model)
    
    print("\nTraining model...")
    history = trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    
    train_pred = trainer.predict(X_train)
    test_pred = trainer.predict(X_test)
    
    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Test MAE:  {test_mae:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    
    # Save model
    model_dir = f"models/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    
    if model_type == 'xgboost':
        model_path = f"{model_dir}/{model_type}_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.json"
    else:
        model_path = f"{model_dir}/{model_type}_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.pt"
    
    trainer.save_model(model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    return trainer, history


def train_all_models():
    """Train all models and compare"""
    print("Training all models for comparison...\n")
    
    results = {}
    
    for model_type in ['xgboost', 'gru', 'lstm', 'transformer']:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}\n")
        
        try:
            trainer, history = train_model(model_type)
            if trainer:
                results[model_type] = {
                    'trainer': trainer,
                    'history': history,
                    'status': 'success'
                }
        except Exception as e:
            print(f"✗ Error training {model_type}: {e}")
            results[model_type] = {
                'status': 'failed',
                'error': str(e)
            }
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for model_type, result in results.items():
        status = result.get('status', 'unknown')
        print(f"{model_type.upper()}: {status}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train crypto trading models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'gru', 'transformer', 'xgboost', 'all'],
        default='lstm',
        help='Model to train (default: lstm)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models()
    else:
        train_model(args.model)

