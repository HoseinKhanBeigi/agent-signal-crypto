# Crypto Trading Signal AI - Multiple Models

AI-powered cryptocurrency trading signal system using **CCXT** for data fetching and **multiple AI models** for price prediction. Optimized for **5-15 minute trading timeframes**.

## 🚀 Features

- **CCXT Integration**: Fetches real-time crypto data from multiple exchanges
- **4 AI Models**: LSTM, GRU, Transformer, and XGBoost
- **Short-term Trading**: Optimized for 5-15 minute timeframes
- **Real-time Signals**: Generate BUY/SELL/HOLD signals every 5-15 minutes
- **Multiple Features**: Uses 20+ technical indicators for better predictions
- **Model Comparison**: Train and compare all models

## 📋 Requirements

- Python 3.8+ (tested on Python 3.12+)
- PyTorch 2.1+ (for LSTM, GRU, Transformer)
- XGBoost 2.0+ (for XGBoost model)
- CCXT library for crypto data

## 🔧 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `config.py` (optional):
   - Trading pair (default: BTC/USDT)
   - Timeframe (default: 5m)
   - Model parameters

## 🚀 Training Options

### Option 1: Train on Vast.ai (Recommended for Mac without GPU)

If you have a Mac without GPU (like M1 MacBook with 8GB RAM), train on [vast.ai](https://cloud.vast.ai/):

```bash
# 1. Upload code to vast.ai
./upload_to_vast.sh <vast-ai-ip>

# 2. SSH and train on vast.ai
ssh root@<vast-ai-ip>
cd agent-signal-crypto
pip install -r requirements.txt
python train_all_models.py --model all

# 3. Download trained models to Mac
./download_models.sh <vast-ai-ip>

# 4. Use models on Mac (no GPU needed for predictions)
python main.py signal --model gru
```

**Cost**: ~$0.15 to train all models (~15 minutes)  
**See**: `VAST_AI_TRAINING.md` for complete guide

### Option 2: Train Locally (if you have GPU)

If you have a GPU (NVIDIA or Apple Silicon), train locally:
```bash
python main.py train --model all
```

## 🎯 Usage

### 1. Train Models

Train a specific model:

```bash
# Train LSTM
python main.py train --model lstm

# Train GRU (faster, often better)
python main.py train --model gru

# Train Transformer (best accuracy, slower)
python main.py train --model transformer

# Train XGBoost (fastest)
python main.py train --model xgboost

# Train ALL models for comparison
python main.py train --model all
```

Or use the training script directly:
```bash
python train_all_models.py --model lstm
```

### 2. Generate Single Signal

Get a trading signal using a specific model:

```bash
# Using LSTM
python main.py signal --model lstm

# Using GRU
python main.py signal --model gru

# Using Transformer
python main.py signal --model transformer

# Using XGBoost
python main.py signal --model xgboost
```

### 3. Continuous Signal Generation

Run continuous signal generation (updates every 5 minutes):

```bash
python main.py continuous --model lstm
```

### 4. Custom Symbol/Timeframe

```bash
# Train on ETH/USDT with 15-minute timeframe using GRU
python main.py train --model gru --symbol ETH/USDT --timeframe 15m

# Generate signal for custom pair
python main.py signal --model gru --symbol ETH/USDT --timeframe 15m
```

## 📊 Available Models

### 1. **LSTM** (Long Short-Term Memory)
- **Architecture**: 3 LSTM layers (128→64→32 units)
- **Speed**: Medium training, fast prediction
- **Best For**: General sequence learning

### 2. **GRU** (Gated Recurrent Unit) ⭐ Recommended
- **Architecture**: 3 GRU layers (128→64→32 units)
- **Speed**: Fast training, fast prediction
- **Best For**: Faster alternative to LSTM, often better performance

### 3. **Transformer** (Attention-based) 🏆 Best Accuracy
- **Architecture**: Multi-head attention, 3 encoder layers
- **Speed**: Slow training, medium prediction
- **Best For**: Best accuracy, long-range dependencies

### 4. **XGBoost** (Gradient Boosting) ⚡ Fastest
- **Architecture**: Gradient boosting trees
- **Speed**: Very fast training, very fast prediction
- **Best For**: Quick baseline, feature-based learning

All models use:
- **Input**: 60 candles × 20+ features
- **Output**: Predicted return (next 5-15 minutes)
- **Features**: OHLCV + RSI, MACD, Bollinger Bands, ATR, Volume indicators, etc.

## 🎛️ Configuration

Edit `config.py` to customize:

```python
SYMBOL = 'BTC/USDT'        # Trading pair
TIMEFRAME = '5m'            # 5m or 15m
SEQUENCE_LENGTH = 60        # Look back 60 candles
BUY_THRESHOLD = 0.02        # 2% predicted increase = BUY
SELL_THRESHOLD = -0.02      # 2% predicted decrease = SELL
```

## 📈 Signal Interpretation

- **BUY**: Predicted price increase ≥ 2%
- **SELL**: Predicted price decrease ≥ 2%
- **HOLD**: Predicted change < 2% (or below confidence threshold)

## 🔍 How It Works

1. **Data Fetching**: CCXT fetches OHLCV data from exchange
2. **Feature Engineering**: Creates 20+ technical indicators
3. **Sequence Creation**: Builds sequences of 60 candles
4. **Prediction**: LSTM predicts next period return
5. **Signal Generation**: Converts prediction to BUY/SELL/HOLD

## ⚠️ Disclaimer

This is for educational purposes. Cryptocurrency trading involves risk. Always:
- Test thoroughly on historical data
- Use proper risk management
- Never invest more than you can afford to lose
- Consider transaction fees and slippage

## 📁 Project Structure

```
agent-signal-crypto/
├── config.py                    # Configuration
├── data_fetcher.py              # CCXT data fetching
├── train_all_models.py           # Unified training script
├── signal_generator_all.py       # Unified signal generator
├── main.py                       # Main execution script
├── requirements.txt              # Dependencies
├── models/
│   ├── shared/
│   │   └── feature_engineer.py  # Shared feature engineering
│   ├── lstm/
│   │   └── model.py              # LSTM model
│   ├── gru/
│   │   └── model.py              # GRU model
│   ├── transformer/
│   │   └── model.py              # Transformer model
│   └── xgboost/
│       └── model.py              # XGBoost model
└── models/                       # Saved trained models (created after training)
    ├── lstm/
    ├── gru/
    ├── transformer/
    └── xgboost/
```

## 🛠️ Troubleshooting

**Model not found error:**
- Train the model first: `python main.py train`

**Insufficient data:**
- Increase `DATA_LIMIT` in `config.py`
- Check internet connection for CCXT

**Low prediction accuracy:**
- Train for more epochs
- Adjust `SEQUENCE_LENGTH` and model architecture
- Try different timeframes or symbols

## 📝 License

MIT License - Use at your own risk

