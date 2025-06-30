#!/bin/bash
# Generate synthetic data and run finetuning with comprehensive error checking

set -e

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check arguments
if [ $# -lt 2 ]; then
    print_error "Usage: bash finetune/run_training.sh <num_conversations> <sample_rate>"
    echo "Example: bash finetune/run_training.sh 1000 8000"
    echo "Sample rates: 8000 or 16000"
    exit 1
fi

NUM_CONVERSATIONS=$1
SAMPLE_RATE=$2

# Validate sample rate
if [ "$SAMPLE_RATE" != "8000" ] && [ "$SAMPLE_RATE" != "16000" ]; then
    print_error "Sample rate must be 8000 or 16000"
    exit 1
fi

# Activate environment
print_info "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clearervoice || {
    print_error "Failed to activate conda environment 'clearervoice'"
    echo "Please run: bash finetune/setup_env.sh first"
    exit 1
}

# Check API keys
if [ -z "$GOOGLE_API_KEY" ]; then
    print_error "GOOGLE_API_KEY not set"
    echo "Please run: export GOOGLE_API_KEY='your-key'"
    exit 1
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
    print_error "Google Cloud credentials not found"
    echo "Please set GOOGLE_APPLICATION_CREDENTIALS or run: gcloud auth application-default login"
    exit 1
fi

# Check if wandb is configured
if [ -z "$WANDB_API_KEY" ]; then
    print_info "WANDB_API_KEY not set. Wandb logging will be disabled or prompt for login."
    echo "To enable wandb logging, run: export WANDB_API_KEY='your-key'"
    USE_WANDB=0
else
    print_success "WANDB_API_KEY found. Wandb logging enabled."
    USE_WANDB=1
fi

# Set up paths
WORK_DIR=$(pwd)
DATA_DIR="${WORK_DIR}/synthetic_data/ss_${SAMPLE_RATE}hz"
EXP_NAME="finetune_${SAMPLE_RATE}hz_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints/${EXP_NAME}"

print_info "=== Configuration ==="
echo "Conversations: $NUM_CONVERSATIONS"
echo "Sample rate: $SAMPLE_RATE Hz"
echo "Data directory: $DATA_DIR"
echo "Experiment name: $EXP_NAME"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Check if pretrained model exists
PRETRAINED_MODEL="${WORK_DIR}/pretrained_models/MossFormer2_SS_${SAMPLE_RATE:0:2}K.pth"
if [ ! -f "$PRETRAINED_MODEL" ]; then
    print_error "Pretrained model not found: $PRETRAINED_MODEL"
    echo "Please run: bash finetune/setup_env.sh first"
    exit 1
fi

# Generate synthetic data
print_info "=== Generating Synthetic Conversations ==="
python finetune/generate_conversation_dataset_clearervoice.py \
    --output_dir "$DATA_DIR" \
    --num_conversations "$NUM_CONVERSATIONS" \
    --sample_rate "$SAMPLE_RATE" \
    --language ja \
    --target_duration 30 \
    --volume_variation 0.5 || {
    print_error "Data generation failed!"
    exit 1
}

# Verify data generation
if [ ! -f "$DATA_DIR/train.scp" ] || [ ! -f "$DATA_DIR/valid.scp" ] || [ ! -f "$DATA_DIR/test.scp" ]; then
    print_error "SCP files not found in $DATA_DIR"
    exit 1
fi

# Count samples
TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.scp")
VALID_COUNT=$(wc -l < "$DATA_DIR/valid.scp")
TEST_COUNT=$(wc -l < "$DATA_DIR/test.scp")

if [ "$TRAIN_COUNT" -eq 0 ]; then
    print_error "No training samples generated!"
    exit 1
fi

print_success "Data generation complete!"
echo "Train: $TRAIN_COUNT samples"
echo "Valid: $VALID_COUNT samples"
echo "Test: $TEST_COUNT samples"
echo ""

# Verify first line of SCP files
print_info "Verifying SCP file format..."
FIRST_LINE=$(head -n1 "$DATA_DIR/train.scp")
NUM_FIELDS=$(echo "$FIRST_LINE" | wc -w)
if [ "$NUM_FIELDS" -ne 3 ]; then
    print_error "Invalid SCP format. Expected 3 fields, got $NUM_FIELDS"
    echo "First line: $FIRST_LINE"
    exit 1
fi

# Check if audio files exist
FIRST_MIX=$(echo "$FIRST_LINE" | cut -d' ' -f1)
if [ ! -f "$FIRST_MIX" ]; then
    print_error "Audio file not found: $FIRST_MIX"
    exit 1
fi
print_success "SCP file format verified"

# Create config directory if not exists
mkdir -p "${WORK_DIR}/config/train"

# Create config file for this run
CONFIG_FILE="${WORK_DIR}/config/train/${EXP_NAME}.yaml"
if [ "$SAMPLE_RATE" == "8000" ]; then
    CONFIG_TEMPLATE="${WORK_DIR}/finetune/config_template_8000.yaml"
else
    CONFIG_TEMPLATE="${WORK_DIR}/finetune/config_template_16000.yaml"
fi

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    print_error "Config template not found: $CONFIG_TEMPLATE"
    exit 1
fi

cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"

# Update paths in config with absolute paths
print_info "Updating configuration file..."
sed -i "s|PLACEHOLDER_EXP_NAME|${EXP_NAME}|g" "$CONFIG_FILE"
sed -i "s|PLACEHOLDER_CHECKPOINT_DIR|${CHECKPOINT_DIR}|g" "$CONFIG_FILE"
sed -i "s|PLACEHOLDER_TRAIN_SCP|${DATA_DIR}/train.scp|g" "$CONFIG_FILE"
sed -i "s|PLACEHOLDER_VALID_SCP|${DATA_DIR}/valid.scp|g" "$CONFIG_FILE"
sed -i "s|PLACEHOLDER_TEST_SCP|${DATA_DIR}/test.scp|g" "$CONFIG_FILE"

# Verify config update
if grep -q "PLACEHOLDER" "$CONFIG_FILE"; then
    print_error "Failed to update all placeholders in config file"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Start training
print_info "=== Starting Training ==="
echo "Config: $CONFIG_FILE"
echo "Logs will be in: ${CHECKPOINT_DIR}/train.log"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')" 

export CUDA_VISIBLE_DEVICES=0

# Run training with output to both console and log file
print_info "Starting training process..."
python train/speech_separation/train.py \
    --config "$CONFIG_FILE" \
    --use_wandb "$USE_WANDB" \
    --wandb_project "clearervoice-finetune" \
    2>&1 | tee "${CHECKPOINT_DIR}/train.log" || {
    print_error "Training failed! Check logs at: ${CHECKPOINT_DIR}/train.log"
    exit 1
}

print_success "=== Training Complete ==="
echo "Model saved in: ${CHECKPOINT_DIR}/"
echo ""

# Check for best model
if [ -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
    print_success "Best model: ${CHECKPOINT_DIR}/best_model.pth"
else
    # Find latest checkpoint
    LATEST_CKPT=$(ls -t "${CHECKPOINT_DIR}"/checkpoint_epoch_*.pth 2>/dev/null | head -n1)
    if [ -n "$LATEST_CKPT" ]; then
        print_info "Latest checkpoint: $LATEST_CKPT"
    else
        print_error "No model checkpoints found!"
    fi
fi

echo ""
echo "To test the model:"
echo "python finetune/test_separation.py <audio_file> --checkpoint ${CHECKPOINT_DIR}/best_model.pth"