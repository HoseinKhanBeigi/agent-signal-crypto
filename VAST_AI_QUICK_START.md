# Vast.ai Quick Start Guide

## Step 1: Select Template

**Choose: "PyTorch (Vast)"** ✅

This template has:
- ✅ PyTorch pre-installed
- ✅ CUDA support
- ✅ SSH access
- ✅ Jupyter notebook (optional)
- ✅ Everything you need for training

**Alternative:** "NVIDIA CUDA" (if PyTorch template not available)

## Step 2: Configure Instance

After selecting "PyTorch (Vast)" template:

1. **Filter Settings:**
   - ✅ Show Secure Cloud Only: ON
   - Price: $0.20 - $1.00/hour (RTX 3090/4090 range)
   - GPU Count: 1 (you only need 1 GPU)
   - Per GPU RAM: At least 8 GB (16 GB+ recommended)
   - Min Cuda Version: 11.4 or higher

2. **Select a GPU:**
   - **RTX 3090** (~$0.50-0.70/hr) - Best value ⭐
   - **RTX 4090** (~$0.80-1.20/hr) - Fastest
   - **RTX 3060** (~$0.20-0.40/hr) - Budget option

3. **Click "Create"**

## Step 3: Get Connection Info

After instance is created, you'll see:
- **SSH Command**: `ssh root@<ip-address>`
- **Password**: (if needed)
- **Jupyter URL**: (optional, for web interface)

**Save the IP address!** You'll need it for upload/download scripts.

## Step 4: Upload Your Code

From your Mac terminal:

```bash
cd /Users/hossein/Documents/agent-signal-crypto
./upload_to_vast.sh <vast-ai-ip>
```

Or manually:
```bash
scp -r . root@<vast-ai-ip>:/root/agent-signal-crypto/
```

## Step 5: Train Models

SSH into the instance:
```bash
ssh root@<vast-ai-ip>
```

Then:
```bash
cd agent-signal-crypto

# PyTorch is already installed, but install other dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Train all models
python train_all_models.py --model all
```

## Step 6: Download Models

From your Mac:
```bash
./download_models.sh <vast-ai-ip>
```

Or manually:
```bash
scp -r root@<vast-ai-ip>:/root/agent-signal-crypto/models/ ./models/
```

## Step 7: Stop Instance

**Important:** Stop the instance immediately after training to save money!

In vast.ai dashboard:
- Click on your instance
- Click "Stop" or "Destroy"

## Template Comparison

| Template | Best For | Pre-installed |
|----------|----------|---------------|
| **PyTorch (Vast)** ⭐ | Your use case | PyTorch, CUDA |
| NVIDIA CUDA | Base setup | CUDA only |
| Ubuntu 22.04 VM | Full OS | Ubuntu only |
| TensorFlow CUDA | TensorFlow projects | TensorFlow |

## Troubleshooting

**"No rentable instances" error?**
- Broaden price range
- Try different GPU types
- Uncheck "Show Secure Cloud Only" (if comfortable)

**Can't connect via SSH?**
- Use vast.ai web terminal
- Check if instance is running
- Verify IP address

**PyTorch not found?**
- The template should have it, but if not:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## Cost Estimate

With RTX 3090 at ~$0.60/hour:
- Training all models: ~15 minutes = **~$0.15**
- Very affordable!

## Summary

1. ✅ Select **"PyTorch (Vast)"** template
2. ✅ Choose RTX 3090 or 4090 GPU
3. ✅ Upload code: `./upload_to_vast.sh <ip>`
4. ✅ Train: `python train_all_models.py --model all`
5. ✅ Download: `./download_models.sh <ip>`
6. ✅ Stop instance to save money

That's it! Your models will be trained on GPU in ~15 minutes for ~$0.15.

