"""
Configuration for CCXT + Multivariate LSTM Crypto Trading System
"""

# Trading Configuration
SYMBOL = 'BTC/USDT'  # Trading pair
TIMEFRAME = '5m'  # Primary timeframe: '5m' or '15m'
EXCHANGE_NAME = 'binance'  # Exchange (binance, coinbase, etc.)

# Multivariate LSTM Parameters
SEQUENCE_LENGTH = 60  # Look back 60 candles (5 hours for 5m, 15 hours for 15m)
PREDICTION_STEPS = 1  # Predict next 1 candle ahead
FEATURES = ['open', 'high', 'low', 'close', 'volume']  # Base OHLCV features

# Model Architecture
LSTM_UNITS = [128, 64, 32]  # LSTM layers with units
DROPOUT_RATE = 0.2
DENSE_UNITS = 16
LEARNING_RATE = 0.001

# Training Parameters
EPOCHS = 100
BATCH_SIZE = 32  # Increase to 64 or 128 if using GPU for faster training
VALIDATION_SPLIT = 0.2
TRAIN_TEST_SPLIT = 0.8

# Data Parameters
DATA_LIMIT = 2000  # Number of candles to fetch
MIN_DATA_POINTS = 500  # Minimum data points required

# Signal Generation
BUY_THRESHOLD = 0.02  # 2% predicted price increase
SELL_THRESHOLD = -0.02  # 2% predicted price decrease
MIN_CONFIDENCE = 0.015  # Minimum 1.5% movement to generate signal

