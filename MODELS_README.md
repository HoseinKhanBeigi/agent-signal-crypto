# Models Comparison Guide

## Quick Model Selection

| Model | Training Time | Prediction Speed | Accuracy | Best For |
|-------|--------------|------------------|----------|----------|
| **XGBoost** | ~30 seconds | Very Fast | Good | Quick baseline, fast iteration |
| **GRU** | ~5-10 min | Fast | Good | Balanced speed/accuracy ⭐ |
| **LSTM** | ~10-20 min | Fast | Good | Traditional sequence learning |
| **Transformer** | ~30-60 min | Medium | Best | Maximum accuracy 🏆 |

## Model Details

### XGBoost
- **Type**: Gradient Boosting Trees
- **Input**: Flattened sequences (60 × 25 = 1500 features)
- **Pros**: Very fast, interpretable, less overfitting
- **Cons**: Doesn't model sequences naturally
- **Use When**: You need quick results, want baseline

### GRU
- **Type**: Gated Recurrent Unit (simpler than LSTM)
- **Input**: Sequences (60 timesteps × 25 features)
- **Pros**: Faster than LSTM, often better accuracy
- **Cons**: Less memory capacity than LSTM
- **Use When**: You want good balance of speed and accuracy ⭐

### LSTM
- **Type**: Long Short-Term Memory
- **Input**: Sequences (60 timesteps × 25 features)
- **Pros**: Good sequence modeling, well-established
- **Cons**: Slower than GRU, similar accuracy
- **Use When**: You prefer traditional RNN approach

### Transformer
- **Type**: Attention-based Transformer
- **Input**: Sequences (60 timesteps × 25 features)
- **Pros**: Best accuracy, attention mechanism, long-range dependencies
- **Cons**: Slowest training, more complex
- **Use When**: You want maximum accuracy, have time to train 🏆

## Training Recommendations

1. **Start with XGBoost**: Get quick baseline (~30 seconds)
2. **Try GRU**: Good balance, often best practical choice
3. **Compare with Transformer**: If you have time and want best accuracy
4. **Use LSTM**: If you prefer traditional approach

## Example Usage

```bash
# Quick test with XGBoost
python main.py train --model xgboost
python main.py signal --model xgboost

# Best practical choice (GRU)
python main.py train --model gru
python main.py signal --model gru

# Maximum accuracy (Transformer)
python main.py train --model transformer
python main.py signal --model transformer

# Compare all models
python main.py train --model all
```

## Performance Tips

- **XGBoost**: Great for quick experiments, feature importance analysis
- **GRU**: Best for production (fast + accurate)
- **Transformer**: Best for final model if accuracy is critical
- **LSTM**: Good middle ground, well-tested

