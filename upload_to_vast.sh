#!/bin/bash
# Script to upload project to vast.ai instance
# Usage: ./upload_to_vast.sh <vast-ai-ip>

if [ -z "$1" ]; then
    echo "Usage: ./upload_to_vast.sh <vast-ai-ip>"
    echo "Example: ./upload_to_vast.sh 123.45.67.89"
    exit 1
fi

VAST_IP=$1
PROJECT_DIR="/Users/hossein/Documents/agent-signal-crypto"
REMOTE_DIR="/root/agent-signal-crypto"

echo "Uploading project to vast.ai instance at $VAST_IP..."
echo "Project: $PROJECT_DIR"
echo "Remote: $REMOTE_DIR"
echo ""

# Upload project files
scp -r "$PROJECT_DIR" root@$VAST_IP:/root/

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps:"
echo "1. SSH into instance: ssh root@$VAST_IP"
echo "2. cd agent-signal-crypto"
echo "3. pip install -r requirements.txt"
echo "4. python train_all_models.py --model all"
echo "5. Download models: scp -r root@$VAST_IP:/root/agent-signal-crypto/models/ ./models/"

