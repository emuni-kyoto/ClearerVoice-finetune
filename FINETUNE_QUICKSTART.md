# ClearerVoice Finetuning Quick Start

## 1. Create GCP Instance
```bash
gcloud compute instances create clearervoice-train \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB
```

## 2. SSH and Setup
```bash
# SSH into instance
gcloud compute ssh clearervoice-train --zone=us-central1-a

sudo apt-get install git-lfs

git clone https://github.com/emuni-kyoto/ClearerVoice-finetune.git
cd ClearerVoice-finetune
mkdir audio_datasets_emuni

gcsfuse audio_datasets_emuni "$HOME/ClearerVoice-finetune/audio_datasets_emuni"

git config --global user.name "Shinnosuke Uesaka"

git config --global user.email "shinnosuke.uesaka@gmail.com"


curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
uv pip install torch torchvision torchaudio --torch-backend=cu124
uv pip install -r requirements.txt



# Set your API keys
export GOOGLE_API_KEY="your-gemini-api-key"

# Optional: Set wandb key for experiment tracking
export WANDB_API_KEY="your-wandb-api-key"
```

## 3. Generate Data & Train
```bash
# Generate synthetic data and start training
# Usage: bash finetune/run_training.sh <num_conversations> <sample_rate>
bash finetune/run_training.sh 1000 8000
```

## 4. Monitor Progress
```bash
# In another terminal
tail -f checkpoints/finetune_*/train.log

# If using wandb, check your wandb dashboard
```

## 5. Test Model
```bash
# After training
python finetune/test_separation.py path/to/mixed_audio.wav --checkpoint checkpoints/finetune_*/best_model.pth
```

That's it! Model will be in `checkpoints/finetune_*/best_model.pth`