#!/usr/bin/env python3
"""Test soundfile write functionality."""

import numpy as np
import soundfile as sf
from pathlib import Path


def test_soundfile_write():
    """Test basic soundfile write operations."""
    
    print("Testing soundfile write operations...")
    
    # Create test audio
    sample_rate = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Test 1: Write to current directory
    try:
        test_file = "test_audio.wav"
        sf.write(test_file, audio, sample_rate)
        print(f"✓ Successfully wrote to {test_file}")
        Path(test_file).unlink()  # Clean up
    except Exception as e:
        print(f"✗ Failed to write to current directory: {e}")
    
    # Test 2: Write to subdirectory
    try:
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        test_file = output_dir / "test_audio.wav"
        sf.write(str(test_file), audio, sample_rate)
        print(f"✓ Successfully wrote to {test_file}")
        test_file.unlink()  # Clean up
        output_dir.rmdir()
    except Exception as e:
        print(f"✗ Failed to write to subdirectory: {e}")
    
    # Test 3: Write with absolute path
    try:
        output_dir = Path("test_output").absolute()
        output_dir.mkdir(exist_ok=True)
        test_file = output_dir / "test_audio.wav"
        sf.write(str(test_file), audio, sample_rate)
        print(f"✓ Successfully wrote to absolute path: {test_file}")
        test_file.unlink()  # Clean up
        output_dir.rmdir()
    except Exception as e:
        print(f"✗ Failed to write to absolute path: {e}")
    
    # Test 4: Check audio array properties
    print(f"\nAudio properties:")
    print(f"- Shape: {audio.shape}")
    print(f"- Dtype: {audio.dtype}")
    print(f"- Min: {audio.min():.3f}, Max: {audio.max():.3f}")
    print(f"- Contains NaN: {np.any(np.isnan(audio))}")
    print(f"- Contains Inf: {np.any(np.isinf(audio))}")


if __name__ == "__main__":
    test_soundfile_write()