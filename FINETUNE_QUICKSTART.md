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

# Clone this repository
git clone https://github.com/YOUR_ORG/ClearerVoice-finetune.git
cd ClearerVoice-finetune

# Run one-time setup
bash finetune/setup_env.sh

# Set your API keys
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
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