# Wandb Integration for ClearerVoice Finetuning

This guide explains how to use Weights & Biases (wandb) to track your finetuning experiments.

## Setup

1. **Install wandb** (already added to requirements.txt):
   ```bash
   pip install wandb
   ```

2. **Create a wandb account** at https://wandb.ai/

3. **Get your API key** from https://wandb.ai/settings

4. **Set your API key**:
   ```bash
   export WANDB_API_KEY='your-api-key-here'
   ```

## Usage

### Automatic Integration

Wandb logging is automatically enabled when you run the training script with a valid API key:

```bash
# With wandb logging (online)
export WANDB_API_KEY='your-api-key'
bash finetune/run_training.sh 1000 8000
```

### Manual Control

You can control wandb usage with environment variables:

```bash
# Disable wandb logging
bash finetune/run_training.sh 1000 8000
# (Don't set WANDB_API_KEY)

# Use wandb in offline mode
export WANDB_MODE=offline
bash finetune/run_training.sh 1000 8000
```

## Tracked Metrics

The following metrics are automatically logged to wandb:

### Epoch-level metrics:
- `train_loss`: Average training loss for the epoch
- `val_loss`: Validation loss
- `test_loss`: Test loss (if test set provided)
- `learning_rate`: Current learning rate
- `best_val_loss`: Best validation loss so far
- `val_no_improvement`: Number of epochs without improvement

### Step-level metrics (logged every `print_freq` steps):
- `step_loss`: Current batch loss
- `batch_time`: Time per batch in seconds
- `learning_rate`: Current learning rate
- `step`: Global step count

### Summary metrics:
- `best_val_loss`: Best validation loss achieved
- `best_epoch`: Epoch where best validation was achieved

## Testing Wandb Integration

Before starting a full training run, test the wandb integration:

```bash
python test_wandb_integration.py
```

This will:
- Verify wandb is installed correctly
- Test metric logging
- Create a test run in your wandb project

## Viewing Results

1. **Online**: Visit https://wandb.ai/your-username/clearervoice-finetune
2. **Compare runs**: Use wandb's UI to compare different hyperparameters
3. **Download data**: Export metrics as CSV/JSON for further analysis

## Customizing Project Name

You can change the wandb project name:

```bash
python train/speech_separation/train.py \
    --config config.yaml \
    --wandb_project "my-custom-project"
```

## Troubleshooting

### "wandb: ERROR Abnormal program exit"
- Check your internet connection
- Verify your API key is correct
- Try running in offline mode: `export WANDB_MODE=offline`

### "No module named wandb"
- Run: `pip install wandb`
- Activate your conda environment: `conda activate clearervoice`

### Metrics not showing up
- Check that `use_wandb` is set to 1
- Verify you're on the main process (not a distributed worker)
- Check the console output for wandb initialization messages

## Advanced Usage

### Resume a run

If training is interrupted, wandb will automatically resume logging to the same run when you restart with `--train_from_last_checkpoint 1`.

### Log additional metrics

Add custom metrics in `solver.py`:

```python
if self.use_wandb:
    wandb.log({
        "custom_metric": value,
        "step": self.step
    })
```

### Save model artifacts

Best models are automatically saved to wandb. You can download them from the wandb UI under the "Files" tab of your run.