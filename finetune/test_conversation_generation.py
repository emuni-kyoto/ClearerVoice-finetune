#!/usr/bin/env python3
"""Test script to verify conversation generation improvements."""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from generate_conversation_from_real_audio import (
    create_conversation_from_segments,
    split_audio_into_segments
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_conversation')

def create_test_segments(sample_rate=8000):
    """Create test audio segments with different durations."""
    # Create various duration segments
    durations = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    
    speaker1_segments = []
    speaker2_segments = []
    
    for i, dur in enumerate(durations):
        # Create tone for speaker1 (lower frequency)
        t = np.linspace(0, dur, int(dur * sample_rate))
        freq1 = 200 + i * 20  # Varying frequency
        segment1 = 0.3 * np.sin(2 * np.pi * freq1 * t)
        speaker1_segments.append(segment1.astype(np.float32))
        
        # Create tone for speaker2 (higher frequency)
        freq2 = 400 + i * 30  # Different frequency
        segment2 = 0.3 * np.sin(2 * np.pi * freq2 * t)
        speaker2_segments.append(segment2.astype(np.float32))
    
    return speaker1_segments, speaker2_segments

def visualize_conversation(mixed, speaker1, speaker2, sample_rate, output_path):
    """Create visualization of the conversation."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    time = np.arange(len(mixed)) / sample_rate
    
    # Plot mixed audio
    axes[0].plot(time, mixed, 'b-', alpha=0.7)
    axes[0].set_title('Mixed Audio')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Plot speaker1
    axes[1].plot(time, speaker1, 'g-', alpha=0.7)
    axes[1].set_title('Speaker 1')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    # Plot speaker2
    axes[2].plot(time, speaker2, 'r-', alpha=0.7)
    axes[2].set_title('Speaker 2')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")

def analyze_audio_coverage(audio, sample_rate, name):
    """Analyze how much of the audio duration is filled."""
    total_duration = len(audio) / sample_rate
    
    # Find segments with audio (above threshold)
    threshold = 0.001
    audio_mask = np.abs(audio) > threshold
    
    # Calculate coverage
    audio_samples = np.sum(audio_mask)
    coverage = audio_samples / len(audio) * 100
    
    # Find the last audio sample
    last_audio_idx = 0
    for i in range(len(audio) - 1, -1, -1):
        if np.abs(audio[i]) > threshold:
            last_audio_idx = i
            break
    
    last_audio_time = last_audio_idx / sample_rate
    end_silence = total_duration - last_audio_time
    
    logger.info(f"\n{name} Analysis:")
    logger.info(f"  Total duration: {total_duration:.2f}s")
    logger.info(f"  Audio coverage: {coverage:.1f}%")
    logger.info(f"  Last audio at: {last_audio_time:.2f}s")
    logger.info(f"  End silence: {end_silence:.2f}s")
    
    return {
        'coverage': coverage,
        'last_audio_time': last_audio_time,
        'end_silence': end_silence
    }

def main():
    """Run test generation."""
    output_dir = Path("test_conversation_output")
    output_dir.mkdir(exist_ok=True)
    
    sample_rate = 8000
    target_duration = 30.0
    
    logger.info("Creating test segments...")
    speaker1_segments, speaker2_segments = create_test_segments(sample_rate)
    
    logger.info(f"Speaker1: {len(speaker1_segments)} segments")
    logger.info(f"Speaker2: {len(speaker2_segments)} segments")
    
    # Test conversation generation
    logger.info("\nGenerating conversation...")
    mixed, speaker1, speaker2 = create_conversation_from_segments(
        speaker1_segments, speaker2_segments,
        sample_rate, target_duration=target_duration,
        logger=logger
    )
    
    # Save audio files
    logger.info("\nSaving audio files...")
    sf.write(output_dir / "test_mixed.wav", mixed, sample_rate)
    sf.write(output_dir / "test_speaker1.wav", speaker1, sample_rate)
    sf.write(output_dir / "test_speaker2.wav", speaker2, sample_rate)
    
    # Analyze coverage
    logger.info("\nAnalyzing audio coverage...")
    mixed_stats = analyze_audio_coverage(mixed, sample_rate, "Mixed Audio")
    s1_stats = analyze_audio_coverage(speaker1, sample_rate, "Speaker 1")
    s2_stats = analyze_audio_coverage(speaker2, sample_rate, "Speaker 2")
    
    # Create visualization
    visualize_conversation(mixed, speaker1, speaker2, sample_rate, 
                         output_dir / "conversation_visualization.png")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY:")
    logger.info(f"Target duration: {target_duration}s")
    logger.info(f"Mixed audio filled to: {mixed_stats['last_audio_time']:.2f}s ({mixed_stats['last_audio_time']/target_duration*100:.1f}%)")
    logger.info(f"End silence: {mixed_stats['end_silence']:.2f}s")
    
    if mixed_stats['end_silence'] > 1.0:
        logger.warning("⚠️  Significant silence at the end - conversation not filled completely!")
    else:
        logger.info("✅ Conversation filled successfully to the end!")

if __name__ == "__main__":
    main()