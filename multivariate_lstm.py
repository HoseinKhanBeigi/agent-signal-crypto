"""
Multivariate LSTM Model for Crypto Price Prediction using PyTorch
Uses multiple features (OHLCV + technical indicators) for prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import pickle
import os
from config import (
    SEQUENCE_LENGTH, PREDICTION_STEPS, FEATURES,
    LSTM_UNITS, DROPOUT_RATE, DENSE_UNITS, LEARNING_RATE,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT
)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultivariateLSTM(nn.Module):
    """Multivariate LSTM model for crypto price prediction"""
    
    def __init__(self, n_features, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE, 
                 dense_units=DENSE_UNITS):
        super(MultivariateLSTM, self).__init__()
        
        self.n_features = n_features
        self.lstm_units = lstm_units
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_size = n_features
        
        for i, units in enumerate(lstm_units):
            self.lstm_layers.append(
                nn.LSTM(input_size, units, batch_first=True)
            )
            input_size = units
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.dense1 = nn.Linear(lstm_units[-1], dense_units)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dense_units, 1)
    
    def forward(self, x):
        # LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:  # Apply dropout except last layer
                x = self.dropout(x)
        
        # Take the last output from the sequence
        x = x[:, -1, :]
        
        # Dense layers
        x = self.dropout(x)
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        
        return x.squeeze(-1)


class MultivariateLSTMTrainer:
    """Wrapper class for training and using Multivariate LSTM"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.n_features = None
        self.feature_columns = None
        
        print(f"Using device: {self.device}")
        
    def prepare_multivariate_features(self, df):
        """
        Prepare multivariate features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        # Base OHLCV features
        base_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages (short-term for 5-15min trading)
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        
        # RSI (short periods for short-term trading)
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'], fast=8, slow=17, signal=9)
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            df = pd.concat([df, bb], axis=1)
            df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
            df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # ATR (volatility)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action
        df['high_low_ratio'] = df['high'] / df['low']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        # Momentum
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        
        # Drop NaN rows
        df = df.dropna()
        
        # Select feature columns (exclude timestamp and target)
        exclude = ['timestamp']
        self.feature_columns = [col for col in df.columns if col not in exclude]
        
        return df
    
    def create_sequences(self, data, target_col='close'):
        """
        Create sequences for multivariate LSTM
        
        Args:
            data: DataFrame with features
            target_col: Column to predict
            
        Returns:
            X (sequences), y (targets)
        """
        # Get feature matrix
        feature_data = data[self.feature_columns].values
        
        # Scale features
        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
        
        # Get target (next period return)
        target = data[target_col].values
        
        # Create sequences
        X, y = [], []
        
        for i in range(SEQUENCE_LENGTH, len(feature_data_scaled) - PREDICTION_STEPS + 1):
            # Input sequence: [sequence_length, n_features]
            X.append(feature_data_scaled[i-SEQUENCE_LENGTH:i])
            
            # Target: future return
            future_price = target[i + PREDICTION_STEPS - 1]
            current_price = target[i - 1]
            future_return = (future_price / current_price) - 1
            y.append(future_return)
        
        X = np.array(X)
        y = np.array(y)
        
        self.n_features = X.shape[2]
        
        return X, y
    
    def build_model(self, n_features=None):
        """
        Build multivariate LSTM model
        
        Args:
            n_features: Number of features (auto-detected if None)
        """
        if n_features is None:
            n_features = self.n_features
        
        if n_features is None:
            raise ValueError("n_features must be provided or set from data")
        
        self.model = MultivariateLSTM(
            n_features=n_features,
            lstm_units=LSTM_UNITS,
            dropout_rate=DROPOUT_RATE,
            dense_units=DENSE_UNITS
        ).to(self.device)
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            self.build_model()
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model_temp.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        # Load best model
                        self.model.load_state_dict(torch.load('best_model_temp.pt'))
                        os.remove('best_model_temp.pt')
                        break
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted returns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def predict_next(self, data):
        """
        Predict next period return from latest data
        
        Args:
            data: DataFrame with features (last SEQUENCE_LENGTH rows)
            
        Returns:
            Predicted return
        """
        # Prepare features
        features = data[self.feature_columns].values
        features_scaled = self.feature_scaler.transform(features)
        
        # Create sequence (last SEQUENCE_LENGTH rows)
        if len(features_scaled) < SEQUENCE_LENGTH:
            raise ValueError(f"Need at least {SEQUENCE_LENGTH} data points")
        
        sequence = features_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)
        
        # Predict
        prediction = self.predict(sequence)
        
        return prediction[0]
    
    def save_model(self, filepath):
        """Save model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), filepath)
        
        # Save scalers and metadata
        scaler_path = filepath.replace('.pt', '_scalers.pkl')
        metadata = {
            'feature_scaler': self.feature_scaler,
            'scaler': self.scaler,
            'n_features': self.n_features,
            'feature_columns': self.feature_columns,
            'lstm_units': LSTM_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'dense_units': DENSE_UNITS
        }
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {filepath}")
        print(f"Scalers saved to {scaler_path}")
    
    def load_model(self, filepath):
        """Load model"""
        # Load metadata and scalers
        scaler_path = filepath.replace('.pt', '_scalers.pkl')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_scaler = metadata['feature_scaler']
        self.scaler = metadata['scaler']
        self.n_features = metadata['n_features']
        self.feature_columns = metadata['feature_columns']
        
        # Build and load model
        self.build_model(self.n_features)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {filepath}")


# Alias for backward compatibility
MultivariateLSTM = MultivariateLSTMTrainer


if __name__ == "__main__":
    # Test model
    from data_fetcher import CCXTDataFetcher
    
    print("Testing Multivariate LSTM (PyTorch)...")
    
    # Fetch data
    fetcher = CCXTDataFetcher()
    data = fetcher.fetch_ohlcv(limit=500)
    
    if data is not None:
        # Prepare features
        lstm = MultivariateLSTMTrainer()
        features = lstm.prepare_multivariate_features(data)
        
        print(f"\nFeatures shape: {features.shape}")
        print(f"Number of features: {len(lstm.feature_columns)}")
        
        # Create sequences
        X, y = lstm.create_sequences(features)
        
        print(f"\nSequences shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        
        # Build model
        model = lstm.build_model()
        print(f"\nModel architecture:")
        print(model)
