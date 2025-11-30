# Training on Vast.ai GPU Cloud

Guide for training models on [vast.ai](https://cloud.vast.ai/) for faster training.

## When to Use Vast.ai

✅ **Use Vast.ai if:**
- You don't have a GPU locally
- Training Transformer takes too long (>30 min)
- You want to train multiple models quickly
- You're experimenting with larger datasets

❌ **Don't need Vast.ai if:**
- You have a local GPU (code auto-detects it)
- XGBoost training is fast enough (~30 sec)
- Training time is acceptable on your machine

## Setup Steps

### 1. Create Vast.ai Account
1. Go to https://cloud.vast.ai/
2. Sign up / Login
3. Add payment method (pay-as-you-go)

### 2. Rent a GPU Instance
1. Click "Create" → "Instance"
2. Choose GPU:
   - **RTX 3090** or **RTX 4090**: Best for training (~$0.50-1.00/hour)
   - **RTX 3060**: Budget option (~$0.20-0.40/hour)
   - **A100**: Fastest but expensive (~$2-3/hour)
3. Select instance with:
   - CUDA support
   - Python 3.8+
   - At least 20GB storage

### 3. Connect to Instance
```bash
# SSH into the instance (vast.ai provides connection details)
ssh root@<instance-ip>
```

### 4. Setup Environment on Vast.ai

```bash
# Update system
apt-get update

# Install Python and pip
apt-get install -y python3 python3-pip git

# Clone your repository (or upload files)
git clone <your-repo-url>
cd agent-signal-crypto

# Or upload files via SCP
# scp -r /path/to/agent-signal-crypto root@<instance-ip>:~/

# Install dependencies
pip3 install -r requirements.txt

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 5. Train Models

```bash
# Train on GPU (will auto-detect)
python3 train_all_models.py --model transformer  # Best for GPU
python3 train_all_models.py --model gru
python3 train_all_models.py --model lstm

# Or use main.py
python3 main.py train --model transformer
```

### 6. Download Trained Models

```bash
# From your local machine, download models
scp -r root@<instance-ip>:~/agent-signal-crypto/models/ ./models/
```

## Cost Estimation

For your dataset (~2000 candles):

| Model | CPU Time | GPU Time (Vast.ai) | Cost (RTX 3090 @ $0.50/hr) |
|-------|----------|-------------------|---------------------------|
| XGBoost | ~30 sec | ~30 sec | $0.004 |
| GRU | ~10 min | ~2 min | $0.017 |
| LSTM | ~20 min | ~4 min | $0.033 |
| Transformer | ~60 min | ~5 min | $0.042 |

**Total for all models: ~$0.10** (very cheap!)

## Alternative: Local GPU Check

Your code already supports GPU! Check if you have one:

```bash
# Check if PyTorch detects GPU
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"
```

If it says `True`, you don't need vast.ai - just train locally!

## Quick Start Script for Vast.ai

Create this file on vast.ai instance:

```bash
#!/bin/bash
# setup_vast.sh

echo "Setting up crypto trading models on Vast.ai..."

# Install dependencies
pip3 install -r requirements.txt

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Train all models
echo "Training all models..."
python3 main.py train --model all

echo "Done! Models saved in ./models/"
```

Run: `bash setup_vast.sh`

## Tips

1. **Start with XGBoost**: Train it first (~30 sec) to verify setup
2. **Monitor GPU usage**: `nvidia-smi` to see GPU utilization
3. **Save money**: Stop instance when not training
4. **Batch size**: Increase `BATCH_SIZE` in config.py for faster GPU training (32 → 64 or 128)
5. **Download models**: Always download trained models before stopping instance

## Troubleshooting

**GPU not detected?**
```bash
# Check CUDA installation
nvidia-smi
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory?**
- Reduce `BATCH_SIZE` in config.py (32 → 16)
- Reduce `SEQUENCE_LENGTH` (60 → 40)

**Connection issues?**
- Use vast.ai web terminal instead of SSH
- Check firewall settings

## Summary

- **Vast.ai is useful** if you don't have a GPU
- **Very cheap** (~$0.10 to train all models)
- **10-20x faster** for Transformer
- **Your code already supports GPU** - just works on vast.ai!

