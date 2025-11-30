"""
Training script for Multivariate LSTM model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_fetcher import CCXTDataFetcher
from multivariate_lstm import MultivariateLSTMTrainer
from config import (
    SYMBOL, TIMEFRAME, DATA_LIMIT, TRAIN_TEST_SPLIT,
    MIN_DATA_POINTS, SEQUENCE_LENGTH
)
import os


def train_multivariate_lstm():
    """Main training function"""
    
    print("=" * 60)
    print("Multivariate LSTM Training for Crypto Trading Signals")
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
    print("\n[2/4] Preparing multivariate features...")
    lstm = MultivariateLSTMTrainer()
    features = lstm.prepare_multivariate_features(data)
    
    print(f"✓ Created {len(lstm.feature_columns)} features")
    print(f"  Features: {', '.join(lstm.feature_columns[:10])}...")
    
    # Step 3: Create sequences
    print("\n[3/4] Creating sequences...")
    X, y = lstm.create_sequences(features)
    
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
    model = lstm.build_model()
    
    print("\nModel Architecture:")
    print(model)
    
    print("\nTraining model...")
    history = lstm.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    
    train_pred = lstm.predict(X_train)
    test_pred = lstm.predict(X_test)
    
    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Test MAE:  {test_mae:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/multivariate_lstm_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.pt"
    lstm.save_model(model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    return lstm, history


if __name__ == "__main__":
    train_multivariate_lstm()

