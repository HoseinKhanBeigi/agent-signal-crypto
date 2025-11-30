"""
Quick script to check if GPU is available for training
"""

import torch

def check_gpu():
    """Check GPU availability and info"""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print("✅ GPU is AVAILABLE!")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("\n💡 You can train locally - no need for vast.ai!")
        print("   Just run: python main.py train --model <model_type>")
    else:
        print("❌ GPU is NOT available")
        print("   Training will use CPU (slower)")
        print("\n💡 Options:")
        print("   1. Train XGBoost (fast on CPU, ~30 seconds)")
        print("   2. Use vast.ai for GPU training (see VAST_AI_SETUP.md)")
        print("   3. Train on CPU (will be slower but works)")
    
    print("=" * 60)
    
    return cuda_available


if __name__ == "__main__":
    check_gpu()

