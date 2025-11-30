# Training on Vast.ai (Complete Guide)

Since you don't want to train on your Mac, here's how to train everything on vast.ai.

## Quick Start

### 1. Setup Vast.ai Instance

1. Go to https://cloud.vast.ai/
2. Sign up / Login
3. Click "Create" → "Instance"
4. **Select Template: "PyTorch (Vast)"** ⭐
   - This has PyTorch pre-installed!
   - Has CUDA support
   - Has SSH access
5. Choose GPU:
   - **RTX 3090** (~$0.50-0.70/hour) - Recommended
   - **RTX 4090** (~$0.80-1.20/hour) - Fastest
   - **RTX 3060** (~$0.20-0.40/hour) - Budget option
6. Filter settings:
   - Price: $0.20 - $1.00/hour
   - GPU Count: 1
   - Per GPU RAM: 8 GB minimum (16 GB+ recommended)
7. Click "Create"

### 2. Upload Your Code

**Option A: Using Git (Recommended)**
```bash
# On vast.ai instance terminal
git clone <your-repo-url>
cd agent-signal-crypto
```

**Option B: Using SCP (from your Mac)**
```bash
# From your Mac terminal
cd /Users/hossein/Documents/agent-signal-crypto
scp -r . root@<vast-ai-ip>:/root/agent-signal-crypto/
```

**Option C: Using Vast.ai Web Interface**
- Use the file upload feature in vast.ai dashboard
- Upload your entire project folder

### 3. Install Dependencies on Vast.ai

```bash
# SSH into vast.ai instance
ssh root@<vast-ai-ip>

# Navigate to project
cd agent-signal-crypto

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Train Models

```bash
# Train all models (recommended)
python train_all_models.py --model all

# Or train individually
python train_all_models.py --model xgboost   # ~30 seconds
python train_all_models.py --model gru        # ~2-3 minutes
python train_all_models.py --model lstm       # ~4-5 minutes
python train_all_models.py --model transformer # ~5-10 minutes
```

### 5. Download Trained Models

```bash
# From your Mac, download the trained models
scp -r root@<vast-ai-ip>:/root/agent-signal-crypto/models/ ./models/
```

## Cost Estimate

For your dataset (~2000 candles):

| Model | Training Time | Cost (RTX 3090 @ $0.60/hr) |
|-------|--------------|---------------------------|
| XGBoost | ~30 sec | $0.005 |
| GRU | ~2-3 min | $0.03 |
| LSTM | ~4-5 min | $0.05 |
| Transformer | ~5-10 min | $0.10 |
| **All Models** | **~15 min** | **~$0.15** |

**Total: Less than $0.20 to train everything!**

## Quick Training Script

Create this file on vast.ai instance: `quick_train.sh`

```bash
#!/bin/bash
# Quick training script for vast.ai

echo "Starting training on Vast.ai GPU..."

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Train all models
echo "Training all models..."
python train_all_models.py --model all

echo "Training complete! Models saved in ./models/"
echo "Download with: scp -r root@<ip>:/root/agent-signal-crypto/models/ ./models/"
```

Run: `bash quick_train.sh`

## Step-by-Step Workflow

### From Your Mac:

1. **Prepare code** (already done ✅)
2. **Upload to vast.ai**:
   ```bash
   cd /Users/hossein/Documents/agent-signal-crypto
   scp -r . root@<vast-ai-ip>:/root/agent-signal-crypto/
   ```

3. **SSH into vast.ai**:
   ```bash
   ssh root@<vast-ai-ip>
   ```

4. **On vast.ai instance**:
   ```bash
   cd agent-signal-crypto
   pip install -r requirements.txt
   python train_all_models.py --model all
   ```

5. **Download models to Mac**:
   ```bash
   # From your Mac
   scp -r root@<vast-ai-ip>:/root/agent-signal-crypto/models/ ./models/
   ```

6. **Use models on Mac**:
   ```bash
   python main.py signal --model gru
   ```

## Tips

1. **Start with XGBoost**: Train it first (~30 sec) to verify everything works
2. **Monitor GPU**: Use `nvidia-smi` to see GPU usage
3. **Save money**: Stop instance immediately after training
4. **Keep models**: Always download models before stopping instance
5. **Batch size**: Can increase to 64-128 on GPU (faster training)

## Troubleshooting

**Can't connect via SSH?**
- Use vast.ai web terminal instead
- Check firewall settings

**Out of memory?**
- Reduce BATCH_SIZE in config.py (16 → 8)
- Train models one at a time

**Models not downloading?**
- Check file paths
- Use `ls -la models/` to verify files exist

## Summary

✅ **Upload code to vast.ai**  
✅ **Train all models (~15 min, ~$0.15)**  
✅ **Download models to Mac**  
✅ **Use models locally for predictions**

You only need vast.ai for training. Once models are downloaded, you can generate signals on your Mac without any GPU!

