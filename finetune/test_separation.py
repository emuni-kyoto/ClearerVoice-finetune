#!/usr/bin/env python3
"""Test finetuned speech separation model."""

import argparse
import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Add parent directory to path to import from train modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_separation(audio_path, checkpoint_path, output_dir='test_output'):
    """Run speech separation on an audio file."""
    
    # Import model based on checkpoint name
    if '8k' in checkpoint_path.lower() or '8000' in checkpoint_path:
        from network import MossFormer2_SS_8K as ModelClass
        target_sr = 8000
    else:
        from network import MossFormer2_SS_16K as ModelClass
        target_sr = 16000
    
    print(f"Using {target_sr}Hz model")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ModelClass(
        encoder_kernel_size=16,
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        num_spks=2
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    
    # Load and preprocess audio
    audio, sr = sf.read(audio_path)
    print(f"Loaded audio: {audio.shape}, {sr}Hz")
    
    # Convert to mono if needed
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Resample if needed
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        print(f"Resampled to {sr}Hz")
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Process audio
    with torch.no_grad():
        # Convert to tensor
        mixture = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        
        # For long audio, process in segments
        max_length_samples = 60 * sr  # 60 seconds max
        
        if mixture.shape[1] > max_length_samples:
            print(f"Audio longer than 60s, processing in segments...")
            # This is a simplified version - the actual implementation would use overlap-add
            separated = []
            for i in range(0, mixture.shape[1], max_length_samples):
                segment = mixture[:, i:i+max_length_samples]
                sep_segment = model(segment)
                separated.append(sep_segment)
            separated = torch.cat(separated, dim=2)
        else:
            separated = model(mixture)
    
    # Convert to numpy
    separated_np = separated.cpu().numpy()
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(audio_path).stem
    
    for i in range(separated_np.shape[0]):
        output_path = os.path.join(output_dir, f"{base_name}_speaker{i+1}.wav")
        sf.write(output_path, separated_np[i].squeeze(), sr)
        print(f"Saved: {output_path}")
    
    # Also save the normalized mixture for comparison
    mix_path = os.path.join(output_dir, f"{base_name}_mixture_normalized.wav")
    sf.write(mix_path, audio, sr)
    print(f"Saved normalized mixture: {mix_path}")
    
    return separated_np, sr


def main():
    parser = argparse.ArgumentParser(description="Test speech separation model")
    parser.add_argument("audio_file", help="Path to mixed audio file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="test_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run separation
    try:
        test_separation(args.audio_file, args.checkpoint, args.output_dir)
        print("\nSeparation complete!")
    except Exception as e:
        print(f"Error during separation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()