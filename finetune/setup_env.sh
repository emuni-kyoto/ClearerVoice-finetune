#!/bin/bash
# One-time environment setup for ClearerVoice finetuning

set -e

echo "=== Setting up ClearerVoice Finetuning Environment ==="

# Create conda environment
echo "Creating conda environment..."
conda create -n clearervoice python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clearervoice

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install ClearerVoice requirements
echo "Installing ClearerVoice requirements..."
pip install -r requirements.txt

# Install additional packages for synthetic data generation
echo "Installing data generation packages..."
pip install google-cloud-texttospeech google-genai audiomentations soundfile tqdm pandas

# Create necessary directories
mkdir -p pretrained_models
mkdir -p checkpoints
mkdir -p synthetic_data

# Download pretrained model for 8kHz
echo "Downloading pretrained models..."
wget -O pretrained_models/MossFormer2_SS_8K.pth \
    "https://modelscope.cn/models/modelscope/ClearerVoice-Studio/resolve/master/pretrained_models/MossFormer2_SS_8K.pth"

# Also get 16kHz model if needed
wget -O pretrained_models/MossFormer2_SS_16K.pth \
    "https://modelscope.cn/models/modelscope/ClearerVoice-Studio/resolve/master/pretrained_models/MossFormer2_SS_16K.pth"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your API credentials:"
echo "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'"
echo "   export GOOGLE_API_KEY='your-gemini-api-key'"
echo ""
echo "2. Run training:"
echo "   bash finetune/run_training.sh 1000 8000"