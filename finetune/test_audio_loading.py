#!/usr/bin/env python3
"""Test script to verify audio loading from GCS works correctly."""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_conversation_from_real_audio import load_audio_from_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_loading():
    """Test loading audio from different path types."""
    
    # Test paths
    test_paths = [
        # Mount path example (will be converted to gs://audio_datasets_emuni/podcast/audio_files_processed/...)
        "/mnt/disks/podcast/audio_files_processed/ja/8da70e4c-2c46-4c00-9311-abf7124bfd64/3b6c8bee_2784_4724_9560_675b338a4576.wav",
        # Direct GCS path example
        "gs://audio_datasets_emuni/podcast/audio_files_processed/ja/8da70e4c-2c46-4c00-9311-abf7124bfd64/3b6c8bee_2784_4724_9560_675b338a4576.wav",
        # You can add more test paths here
    ]
    
    bucket_name = None  # Will be auto-detected based on path type
    
    for path in test_paths:
        logger.info(f"\nTesting path: {path}")
        try:
            audio, sample_rate = load_audio_from_path(path, logger, bucket_name)
            logger.info(f"✓ Successfully loaded audio")
            logger.info(f"  Shape: {audio.shape}")
            logger.info(f"  Sample rate: {sample_rate} Hz")
            logger.info(f"  Duration: {audio.shape[-1] / sample_rate:.2f} seconds")
        except Exception as e:
            logger.error(f"✗ Failed to load audio: {e}")

if __name__ == "__main__":
    logger.info("Testing audio loading functionality...")
    logger.info("\nMake sure you have authenticated with Google Cloud:")
    logger.info("  gcloud auth application-default login")
    logger.info("\nOr set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    logger.info("=" * 50)
    
    test_audio_loading()