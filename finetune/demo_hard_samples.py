#!/usr/bin/env python3
"""Demo script to show hard sample selection functionality."""

import numpy as np
from generate_conversation_from_real_audio import (
    extract_speaker_embedding,
    compute_embedding_similarity
)


def demo_hard_samples():
    """Demonstrate hard sample selection for speaker pairs."""
    
    print("Demo: Hard Sample Selection for Speech Separation\n")
    print("Creating synthetic speakers with varying similarity...")
    
    # Create 6 synthetic speakers with different characteristics
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    speakers = {}
    
    # Group 1: Low pitch speakers (similar)
    speakers['Speaker_A'] = np.sin(2 * np.pi * 110 * t) * np.sin(2 * np.pi * 3 * t)
    speakers['Speaker_B'] = np.sin(2 * np.pi * 120 * t) * np.sin(2 * np.pi * 3.2 * t)
    
    # Group 2: Medium pitch speakers (similar)
    speakers['Speaker_C'] = np.sin(2 * np.pi * 180 * t) * np.sin(2 * np.pi * 5 * t)
    speakers['Speaker_D'] = np.sin(2 * np.pi * 190 * t) * np.sin(2 * np.pi * 5.2 * t)
    
    # Group 3: High pitch speakers (dissimilar to others)
    speakers['Speaker_E'] = np.sin(2 * np.pi * 280 * t) * np.sin(2 * np.pi * 8 * t)
    speakers['Speaker_F'] = np.sin(2 * np.pi * 320 * t) * np.sin(2 * np.pi * 10 * t)
    
    # Add realistic noise
    for name in speakers:
        speakers[name] += np.random.normal(0, 0.01, len(speakers[name]))
    
    # Extract embeddings
    print("\nExtracting speaker embeddings...")
    embeddings = {}
    for name, audio in speakers.items():
        embeddings[name] = extract_speaker_embedding(audio, sample_rate)
    
    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = {}
    speaker_list = list(speakers.keys())
    
    for i, sp1 in enumerate(speaker_list):
        for j, sp2 in enumerate(speaker_list):
            if i < j:
                sim = compute_embedding_similarity(embeddings[sp1], embeddings[sp2])
                similarity_matrix[(sp1, sp2)] = sim
                similarity_matrix[(sp2, sp1)] = sim
    
    # Display similarity matrix
    print("\nSpeaker Similarity Matrix:")
    print("         ", end="")
    for sp in speaker_list:
        print(f"{sp:>10}", end="")
    print()
    
    for sp1 in speaker_list:
        print(f"{sp1:>10}", end="")
        for sp2 in speaker_list:
            if sp1 == sp2:
                print(f"{'1.000':>10}", end="")
            else:
                sim = similarity_matrix[(sp1, sp2)]
                print(f"{sim:>10.3f}", end="")
        print()
    
    # Demonstrate hard/easy sample selection
    print("\n\nDemonstrating Sample Selection:")
    print("-" * 50)
    
    # Get all pairs sorted by similarity
    all_pairs = [(s1, s2, sim) for (s1, s2), sim in similarity_matrix.items() 
                 if s1 < s2]
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Select hard samples (top 30%)
    hard_threshold = int(len(all_pairs) * 0.3)
    hard_samples = all_pairs[:hard_threshold]
    
    print("\nHARD SAMPLES (30% most similar pairs):")
    for s1, s2, sim in hard_samples:
        print(f"  {s1} <-> {s2}: similarity = {sim:.3f}")
    
    # Select easy samples (bottom 50%)
    easy_threshold = int(len(all_pairs) * 0.5)
    easy_samples = all_pairs[-easy_threshold:]
    
    print("\nEASY SAMPLES (50% least similar pairs):")
    for s1, s2, sim in easy_samples:
        print(f"  {s1} <-> {s2}: similarity = {sim:.3f}")
    
    print("\n" + "="*50)
    print("Explanation:")
    print("- Hard samples: Similar speakers (high similarity score)")
    print("  These are challenging for separation models")
    print("- Easy samples: Dissimilar speakers (low similarity score)")
    print("  These are easier for separation models")
    print("- By mixing 30% hard + 70% easy samples, the model")
    print("  learns to handle both easy and challenging cases")


if __name__ == "__main__":
    demo_hard_samples()