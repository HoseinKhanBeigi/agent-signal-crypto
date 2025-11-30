#!/bin/bash
# Script to download trained models from vast.ai
# Usage: ./download_models.sh <vast-ai-ip>

if [ -z "$1" ]; then
    echo "Usage: ./download_models.sh <vast-ai-ip>"
    echo "Example: ./download_models.sh 123.45.67.89"
    exit 1
fi

VAST_IP=$1
LOCAL_MODELS_DIR="./models"
REMOTE_MODELS_DIR="/root/agent-signal-crypto/models"

echo "Downloading models from vast.ai instance at $VAST_IP..."
echo "Remote: $REMOTE_MODELS_DIR"
echo "Local: $LOCAL_MODELS_DIR"
echo ""

# Create local models directory if it doesn't exist
mkdir -p "$LOCAL_MODELS_DIR"

# Download models
scp -r root@$VAST_IP:$REMOTE_MODELS_DIR/* "$LOCAL_MODELS_DIR/"

echo ""
echo "✅ Download complete!"
echo "Models are now in: $LOCAL_MODELS_DIR"
echo ""
echo "You can now use models on your Mac:"
echo "  python main.py signal --model gru"
echo "  python main.py signal --model transformer"

