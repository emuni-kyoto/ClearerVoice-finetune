#!/usr/bin/env python3
"""Test script for speaker embedding functionality."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.generate_conversation_from_real_audio import (
    extract_speaker_embedding,
    compute_embedding_similarity
)


def test_speaker_embeddings():
    """Test speaker embedding extraction and similarity computation."""
    
    print("Testing speaker embedding functionality...")
    
    # Create synthetic audio signals with different characteristics
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speaker 1: Lower pitch, slower modulation
    audio1 = np.sin(2 * np.pi * 120 * t) * np.sin(2 * np.pi * 3 * t)
    
    # Speaker 2: Similar to speaker 1 (for hard sample)
    audio2 = np.sin(2 * np.pi * 130 * t) * np.sin(2 * np.pi * 3.5 * t)
    
    # Speaker 3: Very different (higher pitch, faster modulation)
    audio3 = np.sin(2 * np.pi * 250 * t) * np.sin(2 * np.pi * 8 * t)
    
    # Add some noise to make it more realistic
    audio1 += np.random.normal(0, 0.01, len(audio1))
    audio2 += np.random.normal(0, 0.01, len(audio2))
    audio3 += np.random.normal(0, 0.01, len(audio3))
    
    # Extract embeddings
    print("\nExtracting speaker embeddings...")
    embedding1 = extract_speaker_embedding(audio1, sample_rate)
    embedding2 = extract_speaker_embedding(audio2, sample_rate)
    embedding3 = extract_speaker_embedding(audio3, sample_rate)
    
    print(f"Embedding dimension: {len(embedding1)}")
    
    # Compute similarities
    print("\nComputing similarities...")
    sim_12 = compute_embedding_similarity(embedding1, embedding2)
    sim_13 = compute_embedding_similarity(embedding1, embedding3)
    sim_23 = compute_embedding_similarity(embedding2, embedding3)
    
    print(f"Similarity between similar speakers (1-2): {sim_12:.3f}")
    print(f"Similarity between different speakers (1-3): {sim_13:.3f}")
    print(f"Similarity between different speakers (2-3): {sim_23:.3f}")
    
    # Verify that similar speakers have higher similarity
    if sim_12 > sim_13 and sim_12 > sim_23:
        print("\n✓ Test PASSED: Similar speakers have higher similarity score")
    else:
        print("\n✗ Test FAILED: Expected similar speakers to have higher similarity")
    
    # Test embedding consistency
    print("\nTesting embedding consistency...")
    embedding1_repeat = extract_speaker_embedding(audio1, sample_rate)
    consistency = compute_embedding_similarity(embedding1, embedding1_repeat)
    print(f"Consistency score (same audio): {consistency:.3f}")
    
    if consistency > 0.99:
        print("✓ Test PASSED: Embeddings are consistent for same audio")
    else:
        print("✗ Test WARNING: Embeddings may not be fully deterministic")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_speaker_embeddings()