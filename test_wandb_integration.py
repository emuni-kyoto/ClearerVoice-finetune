#!/usr/bin/env python
"""
Test script to verify wandb integration is working properly
Run this before starting full training to ensure wandb logging is configured correctly
"""

import os
import sys
import wandb
import torch
import numpy as np

def test_wandb_integration():
    print("Testing wandb integration for ClearerVoice finetuning...")
    
    # Check if wandb is installed
    try:
        import wandb
        print("✓ wandb is installed")
    except ImportError:
        print("✗ wandb is not installed. Run: pip install wandb")
        return False
    
    # Check if API key is set
    if os.environ.get('WANDB_API_KEY'):
        print("✓ WANDB_API_KEY is set")
    else:
        print("! WANDB_API_KEY not set. You'll be prompted to login or can run offline")
    
    # Test wandb initialization
    try:
        # Initialize a test run
        run = wandb.init(
            project="clearervoice-finetune-test",
            name="test-integration",
            config={
                "test": True,
                "learning_rate": 0.001,
                "batch_size": 8,
                "epochs": 10
            },
            mode="offline" if not os.environ.get('WANDB_API_KEY') else "online"
        )
        print("✓ wandb initialization successful")
        
        # Test logging metrics
        for epoch in range(5):
            train_loss = np.random.uniform(0.5, 2.0)
            val_loss = np.random.uniform(0.4, 1.8)
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": 0.001 * (0.5 ** (epoch // 2))
            })
        
        print("✓ Metric logging successful")
        
        # Test logging summary statistics
        wandb.run.summary["best_val_loss"] = 0.42
        wandb.run.summary["best_epoch"] = 3
        print("✓ Summary logging successful")
        
        # Finish the run
        wandb.finish()
        print("✓ wandb run finished successfully")
        
        print("\n✅ All wandb integration tests passed!")
        print("\nYou can now run the full training with wandb tracking enabled.")
        print("Make sure to set WANDB_API_KEY for online tracking:")
        print("  export WANDB_API_KEY='your-api-key'")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during wandb testing: {e}")
        return False

def test_training_imports():
    """Test that training scripts can import with wandb"""
    print("\nTesting training script imports...")
    
    try:
        sys.path.append('train/speech_separation')
        # Don't actually import solver as it needs the full model setup
        # Just check that the imports would work
        import yamlargparse
        print("✓ Training dependencies available")
        return True
    except Exception as e:
        print(f"✗ Error importing training dependencies: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("ClearerVoice Wandb Integration Test")
    print("="*60)
    
    # Run tests
    wandb_ok = test_wandb_integration()
    imports_ok = test_training_imports()
    
    if wandb_ok and imports_ok:
        print("\n✅ All tests passed! Wandb integration is ready.")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)