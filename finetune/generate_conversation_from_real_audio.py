#!/usr/bin/env python3
"""Generate conversation datasets from real single-speaker audio files.

This script:
1. Loads preprocessed parquet files from GCP containing single-speaker audio
2. Efficiently loads files one-by-one until sufficient data is collected
3. Handles audio paths: GCS URLs (gs://...), mount paths (/mnt/disks/...), and local paths
4. Extracts lightweight speaker embeddings using MFCC-based features
5. Selects samples with only one speaker
6. Splits audio from different speakers into segments based on:
   - Word timestamps and pauses (if available in parquet)
   - VAD-based pause detection (fallback)
   - Random segmentation (when --no-vad flag is used)
7. Creates synthetic conversations by overlapping segments from different speakers
8. Supports hard/easy sample generation based on speaker similarity
9. Saves in ClearerVoice format for finetuning

Features:
- Speaker embedding extraction using fast MFCC-based features
- Hard sample generation: Conversations between similar speakers
- Easy sample generation: Conversations between dissimilar speakers
- Diverse speaker pair selection to avoid repetition
- Configurable hard sample percentage (default: 30%)

Usage:
    python generate_conversation_from_real_audio.py \
        --output_dir /home/shinnosukeuesaka/ClearerVoice-finetune/audio_datasets_emuni/audio_separation_data/3000_samples \
        --bucket_name audio_datasets_emuni \
        --prefix podcast/parquet_processed/train \
        --num_conversations 3000 \
        --sample_rate 8000 \
        --max_samples 3000 \
        --hard_sample_percentage 0.3
    
Usage for testing:
    python generate_conversation_from_real_audio.py \
      --output_dir ./real_conversation_data \
      --bucket_name audio_datasets_emuni \
      --prefix podcast/parquet_processed/train \
      --num_conversations 10 \
        --max_samples 10

Optional flags:
    --streaming: Enable streaming mode for processing (experimental)
    --max_samples: Maximum number of single-speaker samples to load (default: 1000)
    --no-vad: Disable VAD and use random segmentation (useful when VAD fails)
    --hard_sample_percentage: Percentage of hard samples with similar speakers (0.0-1.0, default: 0.3)
    --pause_threshold: Minimum pause duration in seconds to split segments when using word timestamps (default: 0.5)

Note: Audio files from GCS are cached locally in /tmp/audio_downloads/ for efficiency.
"""

import argparse
import gc
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import librosa
import numpy as np
import soundfile as sf
import torchaudio
import webrtcvad
from google.cloud import storage
from scipy import signal
from scipy.spatial.distance import cosine
from tqdm import tqdm


# Setup logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "generation.log"
    
    # Create logger
    logger = logging.getLogger('real_conversation_generator')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def extract_speaker_embedding(audio: np.ndarray, sample_rate: int, logger: logging.Logger = None) -> np.ndarray:
    """Extract lightweight speaker embedding using MFCC features.
    
    This is a fast, cheap method that captures speaker characteristics.
    Returns a fixed-size embedding vector.
    """
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    try:
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)
        
        # Extract MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        # Compute statistics over time to get a fixed-size embedding
        # Mean and std of each MFCC coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Extract additional spectral features for better speaker discrimination
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # Zero crossing rate (voice characteristics)
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # Combine all features into a single embedding vector
        embedding = np.concatenate([
            mfcc_mean,      # 13 values
            mfcc_std,       # 13 values
            [sc_mean, sc_std, zcr_mean, zcr_std, sr_mean, sr_std]  # 6 values
        ])
        
        # Total: 32-dimensional embedding
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Failed to extract speaker embedding: {e}")
        # Return random embedding as fallback
        return np.random.randn(32).astype(np.float32)


def compute_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute similarity between two speaker embeddings using cosine similarity.
    
    Returns:
        Similarity score between 0 and 1 (1 = identical, 0 = very different)
    """
    # Normalize embeddings
    embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    # Compute cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(embedding1_norm, embedding2_norm)
    
    # Ensure similarity is in [0, 1] range
    return np.clip(similarity, 0, 1)


def list_parquet_files(bucket_name: str, prefix: str, logger: logging.Logger) -> List[str]:
    """List all parquet files in a GCS bucket with the given prefix."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        parquet_files = []
        logger.info(f"Listing parquet files in gs://{bucket_name}/{prefix}")
        blobs = bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            if blob.name.endswith('.parquet'):
                parquet_files.append(f"gs://{bucket_name}/{blob.name}")
                logger.debug(f"Found parquet file: {blob.name}")
        
        logger.info(f"Found {len(parquet_files)} parquet files in gs://{bucket_name}/{prefix}")
        return parquet_files
    except Exception as e:
        logger.error(f"Failed to list parquet files: {e}")
        logger.error("Make sure you have authenticated with: gcloud auth application-default login")
        raise


def download_parquet_from_gcs(gcs_path: str, logger: logging.Logger) -> str:
    """Download a parquet file from GCS to a temporary location."""
    try:
        # Parse GCS path
        if gcs_path.startswith('gs://'):
            gcs_path = gcs_path[5:]
        
        parts = gcs_path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ''
        
        # Create temporary file path
        temp_dir = Path("/tmp/parquet_downloads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / Path(blob_name).name
        
        # Check if file already exists (caching)
        if temp_path.exists():
            logger.debug(f"Using cached file: {temp_path}")
            return str(temp_path)
        
        # Download using Google Cloud Storage client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.debug(f"Downloading {gcs_path} to {temp_path}")
        blob.download_to_filename(str(temp_path))
        
        return str(temp_path)
    except Exception as e:
        logger.error(f"Failed to download {gcs_path}: {e}")
        logger.error("If you're getting SSL errors on macOS, try running:")
        logger.error("  python fix_ssl_macos.py")
        logger.error("Or set environment variable:")
        logger.error("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json")
        raise


def extract_segment_timestamps(audio_path: str, sample_rate: int, 
                              use_vad: bool = True,
                              pause_threshold: float = 0.5,
                              word_alignments: List[Dict] = None,
                              target_duration: float = 30.0,
                              logger: logging.Logger = None) -> List[Tuple[float, float]]:
    """Extract segment timestamps without loading the full segments into memory.
    
    Can use either word timestamps (if provided) or VAD-based segmentation.
    When word_alignments is provided, segments are created by detecting pauses
    between words that exceed pause_threshold.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        use_vad: Whether to use VAD (ignored if word_alignments provided)
        pause_threshold: Minimum pause duration to split segments (seconds)
        word_alignments: List of word timestamp dicts with 'start', 'end', 'text' keys
        target_duration: Target accumulated duration (seconds)
        logger: Logger instance
        
    Returns:
        Tuple of (segments, embedding) where:
        - segments: List of (start_time, end_time) tuples in seconds
        - embedding: Speaker embedding as list of floats
    """
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
        
    try:
        # Load audio temporarily just for segmentation
        audio, orig_sr = load_audio_from_path(audio_path, logger)
        
        # Resample if needed
        if orig_sr != sample_rate:
            audio = resample_audio(audio, orig_sr, sample_rate)
            
        # Get segment sample indices
        target_accumulated = target_duration * 2.0  # Process 2x the target conversation duration
        
        # If word alignments are provided, use them instead of VAD
        if word_alignments is not None and len(word_alignments) > 0:
            logger.info(f"Using word timestamps for segmentation ({len(word_alignments)} words)")
            # Get speech segments from word timestamps
            speech_segments = detect_segments_from_word_timestamps(
                word_alignments,
                sample_rate,
                pause_threshold=pause_threshold,
                logger=logger
            )
            
            if not speech_segments:
                # Fall back to random segmentation
                segments = []
                audio_duration = len(audio) / sample_rate
                if audio_duration >= 1.0:
                    current_pos = 0
                    while current_pos < len(audio):
                        segment_duration = random.uniform(1.0, 5.0)
                        segment_samples = int(segment_duration * sample_rate)
                        end_pos = min(current_pos + segment_samples, len(audio))
                        if (end_pos - current_pos) / sample_rate >= 1.0:
                            segments.append((current_pos / sample_rate, end_pos / sample_rate))
                        current_pos = end_pos + int(random.uniform(0.1, 0.5) * sample_rate)
            else:
                # Convert speech segments to timestamps
                segments = []
                accumulated_duration = 0.0
                for start, end in speech_segments:
                    if target_accumulated and accumulated_duration >= target_accumulated:
                        break
                    segment_duration = (end - start) / sample_rate
                    if segment_duration >= 1.0:
                        if segment_duration > 5.0:
                            # Split long segments
                            num_splits = int(segment_duration / 5.0) + 1
                            split_duration = segment_duration / num_splits
                            for i in range(num_splits):
                                split_start = start / sample_rate + i * split_duration
                                split_end = start / sample_rate + (i + 1) * split_duration
                                if split_end - split_start >= 1.0:
                                    segments.append((split_start, split_end))
                                    accumulated_duration += split_end - split_start
                                    if target_accumulated and accumulated_duration >= target_accumulated:
                                        break
                        else:
                            segments.append((start / sample_rate, end / sample_rate))
                            accumulated_duration += segment_duration
        elif use_vad:
            # Get speech segments
            speech_segments = detect_speech_segments(audio, sample_rate, logger=logger)
            
            if not speech_segments:
                # Fall back to random segmentation
                segments = []
                audio_duration = len(audio) / sample_rate
                if audio_duration >= 1.0:
                    current_pos = 0
                    while current_pos < len(audio):
                        segment_duration = random.uniform(1.0, 5.0)
                        segment_samples = int(segment_duration * sample_rate)
                        end_pos = min(current_pos + segment_samples, len(audio))
                        if (end_pos - current_pos) / sample_rate >= 1.0:
                            segments.append((current_pos / sample_rate, end_pos / sample_rate))
                        current_pos = end_pos + int(random.uniform(0.1, 0.5) * sample_rate)
            else:
                # Convert speech segments to timestamps
                segments = []
                accumulated_duration = 0.0
                for start, end in speech_segments:
                    if target_accumulated and accumulated_duration >= target_accumulated:
                        break
                    segment_duration = (end - start) / sample_rate
                    if segment_duration >= 1.0:
                        if segment_duration > 5.0:
                            # Split long segments
                            num_splits = int(segment_duration / 5.0) + 1
                            split_duration = segment_duration / num_splits
                            for i in range(num_splits):
                                split_start = start / sample_rate + i * split_duration
                                split_end = start / sample_rate + (i + 1) * split_duration
                                if split_end - split_start >= 1.0:
                                    segments.append((split_start, split_end))
                                    accumulated_duration += split_end - split_start
                                    if target_accumulated and accumulated_duration >= target_accumulated:
                                        break
                        else:
                            segments.append((start / sample_rate, end / sample_rate))
                            accumulated_duration += segment_duration
        else:
            # Random segmentation
            segments = []
            audio_duration = len(audio) / sample_rate
            accumulated_duration = 0.0
            current_pos = 0
            
            while current_pos < len(audio) and accumulated_duration < target_accumulated:
                segment_duration = random.uniform(1.0, 5.0)
                segment_samples = int(segment_duration * sample_rate)
                end_pos = min(current_pos + segment_samples, len(audio))
                
                if (end_pos - current_pos) / sample_rate >= 1.0:
                    segments.append((current_pos / sample_rate, end_pos / sample_rate))
                    accumulated_duration += (end_pos - current_pos) / sample_rate
                
                current_pos = end_pos + int(random.uniform(0.1, 0.5) * sample_rate)
        
        # Extract embedding from first 30 seconds
        embedding_audio = audio[:int(30 * sample_rate)]
        embedding = extract_speaker_embedding(embedding_audio, sample_rate, logger)
        
        # Clean up audio from memory
        del audio
        gc.collect()
        
        return segments, embedding.tolist()  # Convert to list for JSON serialization
        
    except Exception as e:
        logger.warning(f"Failed to extract segments from {audio_path}: {e}")
        return [], np.random.randn(32).tolist()


def load_single_speaker_dataset(parquet_files: List[str], 
                               max_samples: int = 1000,
                               target_sample_rate: int = 8000,
                               use_vad: bool = True,
                               pause_threshold: float = 0.5,
                               logger: logging.Logger = None) -> datasets.Dataset:
    """Load samples from parquet files and extract segment timestamps using HuggingFace datasets.
    
    This version uses map functions to process audio files lazily and only stores
    segment timestamps instead of full audio data.
    """
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    logger.info(f"Found {len(parquet_files)} parquet files available")
    logger.info(f"Will load files until we collect {max_samples} single-speaker samples")
    
    all_samples = []
    total_samples_loaded = 0
    files_processed = 0
    
    # Shuffle files for more diverse sampling
    shuffled_files = parquet_files.copy()
    random.shuffle(shuffled_files)
    
    pbar = tqdm(total=max_samples, desc="Loading samples", unit="samples", leave=True)
    
    for file_idx, parquet_file in enumerate(shuffled_files):
        try:
            # Check if we already have enough samples
            if total_samples_loaded >= max_samples:
                logger.info(f"Reached target of {max_samples} samples after loading {files_processed} files")
                break
            
            # Download parquet file from GCS
            local_parquet_path = download_parquet_from_gcs(parquet_file, logger)
            
            # Load single parquet file
            logger.debug(f"Loading parquet from {local_parquet_path}")
            dataset = datasets.load_dataset(
                'parquet',
                data_files=local_parquet_path,
                split='train',
                streaming=False
            )
            
            # Filter for single speaker samples
            single_speaker_data = dataset.filter(
                lambda x: x['num_speakers'] == 1,
                desc=f"Filtering file {files_processed + 1}"
            )
            
            if len(single_speaker_data) > 0:
                # Calculate how many samples we still need
                samples_needed = max_samples - total_samples_loaded
                
                # Only take what we need from this file
                if len(single_speaker_data) > samples_needed:
                    single_speaker_data = single_speaker_data.select(range(samples_needed))
                    logger.debug(f"Taking only {samples_needed} samples from {parquet_file}")
                
                # Process samples to extract segment timestamps
                def process_sample(example):
                    # Extract segment timestamps and speaker embedding
                    segments, embedding = extract_segment_timestamps(
                        example['local_path'], 
                        target_sample_rate,
                        use_vad=use_vad,
                        pause_threshold=pause_threshold,
                        word_alignments=example.get('word_speech_alignment', None),
                        logger=logger
                    )
                    
                    return {
                        'audio_path': example['local_path'],
                        'source_parquet': parquet_file,
                        'speaker_id': example['local_path'],  # Use full path as unique ID
                        'segment_timestamps': segments,
                        'speaker_embedding': embedding,
                        'num_segments': len(segments),
                        'original_duration': example.get('duration', 0),
                        'sample_rate': target_sample_rate
                    }
                
                # Process samples with map function
                processed_data = single_speaker_data.map(
                    process_sample,
                    desc="Extracting segment timestamps",
                    remove_columns=single_speaker_data.column_names  # Remove original columns
                )
                
                # Filter out samples with no segments
                processed_data = processed_data.filter(
                    lambda x: x['num_segments'] > 0
                )
                
                # Convert to list for accumulation
                samples_from_file = processed_data.to_list()
                all_samples.extend(samples_from_file)
                
                samples_count = len(samples_from_file)
                total_samples_loaded += samples_count
                logger.debug(f"Processed {samples_count} samples with segments from {parquet_file}")
                
                # Update progress bar with new samples loaded
                pbar.update(samples_count)
                pbar.set_postfix({
                    "files": files_processed + 1
                })
            
            files_processed += 1
            
            # Clean up temporary file
            try:
                if os.path.exists(local_parquet_path):
                    os.remove(local_parquet_path)
                    logger.debug(f"Cleaned up temporary file: {local_parquet_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
            
        except Exception as e:
            logger.warning(f"Failed to process {parquet_file}: {e}")
            # Clean up temporary file on error
            try:
                if 'local_parquet_path' in locals() and os.path.exists(local_parquet_path):
                    os.remove(local_parquet_path)
            except Exception:
                pass
            continue
    
    pbar.close()
    
    if not all_samples:
        raise ValueError("No single-speaker samples with valid segments found in the parquet files")
    
    # Create final dataset from accumulated samples
    logger.info(f"Creating final dataset from {total_samples_loaded} samples")
    final_dataset = datasets.Dataset.from_list(all_samples)
    
    logger.info(f"Loaded {len(final_dataset)} single-speaker samples with segment timestamps")
    return final_dataset


def load_audio_segment_from_path(audio_path: str, start_time: float, end_time: float, 
                                target_sample_rate: int, logger: logging.Logger, 
                                bucket_name: str = None) -> np.ndarray:
    """Load a specific segment of audio from a file.
    
    Args:
        audio_path: Path to audio file (GCS or local)
        start_time: Start time in seconds
        end_time: End time in seconds
        target_sample_rate: Target sample rate for output
        logger: Logger instance
        bucket_name: GCS bucket name if applicable
        
    Returns:
        Audio segment as numpy array
    """
    try:
        # First load the full audio
        audio, sample_rate = load_audio_from_path(audio_path, logger, bucket_name)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            audio = resample_audio(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
            
        # Extract segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        segment = audio[start_sample:end_sample]
        
        # Clean up full audio
        del audio
        gc.collect()
        
        return segment
        
    except Exception as e:
        logger.error(f"Failed to load audio segment from {audio_path} [{start_time:.2f}-{end_time:.2f}s]: {e}")
        raise


def load_audio_from_path(audio_path: str, logger: logging.Logger, bucket_name: str = None) -> Tuple[np.ndarray, int]:
    """Load audio file from either GCS or local path."""
    try:
        # Check if it's a GCS path
        if audio_path.startswith('gs://'):
            # Parse GCS path
            gcs_path = audio_path[5:]
            parts = gcs_path.split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''
            
            # Create cache directory structure that mirrors the GCS structure
            temp_dir = Path("/tmp/audio_downloads") / bucket_name
            temp_path = temp_dir / blob_name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file is already cached
            if temp_path.exists():
                logger.info(f"Using cached audio file for GCS: gs://{bucket_name}/{blob_name} -> {temp_path}")
                audio_file = str(temp_path)
                cleanup_temp = False
            else:
                # Download from GCS
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                logger.info(f"Downloading from GCS: gs://{bucket_name}/{blob_name} to {temp_path}")
                blob.download_to_filename(str(temp_path))
                
                # Load audio from temp file
                audio_file = str(temp_path)
                cleanup_temp = False  # Don't cleanup cached files
        elif audio_path.startswith('/mnt/disks/podcast/'):
            # This is a mount path from the processing environment
            # Convert to GCS path and download
            # Mount path: /mnt/disks/podcast/audio_files_processed/ja/batch_id/file_id.wav
            # GCS path:   gs://audio_datasets_emuni/podcast/audio_files_processed/ja/batch_id/file_id.wav
            # The /mnt/disks/podcast/ maps to gs://audio_datasets_emuni/podcast/
            
            # Extract the relative path after the mount point
            mount_prefix = '/mnt/disks/podcast/'
            relative_path_from_mount = audio_path[len(mount_prefix):]
            
            # In GCS, this goes under podcast/
            relative_path = f"podcast/{relative_path_from_mount}"
            
            # Use audio_datasets_emuni bucket
            bucket_name = 'audio_datasets_emuni'
            
            # Create temp directory structure that mirrors the GCS structure for caching
            temp_dir = Path("/tmp/audio_downloads") / bucket_name
            temp_path = temp_dir / relative_path
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file is already cached
            if temp_path.exists():
                logger.info(f"Using cached audio file for mount path: {audio_path} -> {temp_path}")
            else:
                logger.info(f"Converting mount path to GCS: {audio_path} -> gs://{bucket_name}/{relative_path}")
                
                # Download from GCS
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(relative_path)
                    
                    # Try to download the file
                    logger.info(f"Downloading from GCS: gs://{bucket_name}/{relative_path} to {temp_path}")
                    try:
                        blob.download_to_filename(str(temp_path))
                        logger.debug("Successfully downloaded audio file")
                    except Exception as download_error:
                        if temp_path.exists():
                            temp_path.unlink()  # Remove partial download
                        raise FileNotFoundError(f"File not found in GCS: gs://{bucket_name}/{relative_path}") from download_error
                except Exception as e:
                    logger.error(f"Failed to access GCS file: {e}")
                    logger.error("Make sure you have Google Cloud credentials configured")
                    logger.error("Run: gcloud auth application-default login")
                    raise
            
            # Load audio from temp file
            audio_file = str(temp_path)
            cleanup_temp = False  # Don't cleanup cached files
        else:
            # Local path - use directly
            audio_file = audio_path
            cleanup_temp = False
            
            # Check if file exists
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Convert to numpy
        audio = waveform.squeeze().numpy()
        
        # Clean up temp file if needed
        if cleanup_temp and os.path.exists(audio_file):
            os.remove(audio_file)
        
        return audio, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio from {audio_path}: {e}")
        raise


def detect_segments_from_word_timestamps(word_alignments: List[Dict], 
                                        sample_rate: int,
                                        pause_threshold: float = 0.5,
                                        min_segment_duration: float = 1.0,
                                        max_segment_duration: float = 10.0,
                                        logger: logging.Logger = None) -> List[Tuple[int, int]]:
    """Detect speech segments from word timestamps by finding pauses.
    
    Args:
        word_alignments: List of word alignment dicts with 'start', 'end', 'text' keys
        sample_rate: Sample rate of the audio
        pause_threshold: Minimum pause duration in seconds to split segments
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds
        logger: Logger instance
        
    Returns:
        List of (start_sample, end_sample) tuples
    """
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
        
    if not word_alignments:
        logger.warning("No word alignments provided")
        return []
        
    # Sort word alignments by start time
    sorted_words = sorted(word_alignments, key=lambda x: x.get('start', 0))
    
    segments = []
    current_segment_start = None
    current_segment_end = None
    
    for i, word in enumerate(sorted_words):
        word_start = word.get('start', 0)
        word_end = word.get('end', word_start)
        
        # Skip invalid word timestamps
        if word_start < 0 or word_end < word_start:
            continue
            
        if current_segment_start is None:
            # Start first segment
            current_segment_start = word_start
            current_segment_end = word_end
        else:
            # Check pause between previous word and current word
            pause_duration = word_start - current_segment_end
            segment_duration = current_segment_end - current_segment_start
            
            # Split if pause is too long OR segment is too long
            if pause_duration >= pause_threshold or segment_duration >= max_segment_duration:
                # Save current segment if it's long enough
                if segment_duration >= min_segment_duration:
                    start_sample = int(current_segment_start * sample_rate)
                    end_sample = int(current_segment_end * sample_rate)
                    segments.append((start_sample, end_sample))
                
                # Start new segment
                current_segment_start = word_start
                current_segment_end = word_end
            else:
                # Extend current segment
                current_segment_end = word_end
    
    # Add final segment
    if current_segment_start is not None:
        segment_duration = current_segment_end - current_segment_start
        if segment_duration >= min_segment_duration:
            start_sample = int(current_segment_start * sample_rate)
            end_sample = int(current_segment_end * sample_rate)
            segments.append((start_sample, end_sample))
    
    logger.debug(f"Detected {len(segments)} segments from {len(word_alignments)} word timestamps")
    return segments


def detect_speech_segments(audio: np.ndarray, sample_rate: int, 
                          frame_duration_ms: int = 30,
                          padding_duration_ms: int = 300,
                          logger: logging.Logger = None) -> List[Tuple[int, int]]:
    """Detect speech segments using WebRTC VAD."""
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    # Initialize VAD
    vad = webrtcvad.Vad(2)  # Aggressiveness level 2
    
    # Convert audio to 16-bit PCM if needed
    if audio.dtype != np.int16:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio
    
    # Resample to 16kHz if needed (VAD requirement)
    if sample_rate != 16000:
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio_16k = signal.resample(audio_int16, num_samples).astype(np.int16)
        vad_sample_rate = 16000
    else:
        audio_16k = audio_int16
        vad_sample_rate = sample_rate
    
    # Frame parameters
    frame_length = int(vad_sample_rate * frame_duration_ms / 1000)
    
    # Detect speech frames
    speech_frames = []
    for i in range(0, len(audio_16k) - frame_length, frame_length):
        frame = audio_16k[i:i + frame_length].tobytes()
        if vad.is_speech(frame, vad_sample_rate):
            speech_frames.append(i)
    
    # Convert frames to segments with padding
    segments = []
    if speech_frames:
        # Group consecutive frames into segments
        padding_frames = int(padding_duration_ms / frame_duration_ms)
        
        segment_start = speech_frames[0]
        prev_frame = speech_frames[0]
        
        for frame_idx in speech_frames[1:]:
            if frame_idx - prev_frame > padding_frames * frame_length:
                # End current segment
                segment_end = prev_frame + frame_length
                # Convert back to original sample rate
                start_sample = int(segment_start * sample_rate / vad_sample_rate)
                end_sample = int(segment_end * sample_rate / vad_sample_rate)
                segments.append((start_sample, end_sample))
                
                # Start new segment
                segment_start = frame_idx
            
            prev_frame = frame_idx
        
        # Add final segment
        segment_end = prev_frame + frame_length
        start_sample = int(segment_start * sample_rate / vad_sample_rate)
        end_sample = int(segment_end * sample_rate / vad_sample_rate)
        segments.append((start_sample, end_sample))
    
    logger.debug(f"Detected {len(segments)} speech segments")
    return segments


def split_audio_randomly(audio: np.ndarray, sample_rate: int,
                        min_segment_duration: float = 1.0,
                        max_segment_duration: float = 5.0,
                        logger: logging.Logger = None,
                        target_accumulated_duration: float = None) -> List[np.ndarray]:
    """Split audio into random segments of varying duration."""
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    segments = []
    audio_duration = len(audio) / sample_rate
    accumulated_duration = 0.0
    
    # Skip if audio is too short
    if audio_duration < min_segment_duration:
        logger.debug(f"Audio too short ({audio_duration:.2f}s), skipping")
        return []
    
    current_pos = 0
    
    while current_pos < len(audio):
        # Check if we've accumulated enough segments
        if target_accumulated_duration and accumulated_duration >= target_accumulated_duration:
            logger.debug(f"Reached target accumulated duration of {target_accumulated_duration}s")
            break
            
        # Random segment duration between min and max
        segment_duration = random.uniform(min_segment_duration, max_segment_duration)
        segment_samples = int(segment_duration * sample_rate)
        
        # Get segment
        end_pos = min(current_pos + segment_samples, len(audio))
        segment = audio[current_pos:end_pos]
        
        # Check if remaining segment is long enough
        if (end_pos - current_pos) / sample_rate >= min_segment_duration:
            segments.append(segment)
            accumulated_duration += (end_pos - current_pos) / sample_rate
        
        # Move to next position with small random gap
        gap_duration = random.uniform(0.1, 0.5)  # Small gap between segments
        current_pos = end_pos + int(gap_duration * sample_rate)
    
    # Sort segments by duration (prioritize shorter segments)
    segments.sort(key=lambda seg: len(seg))
    
    logger.debug(f"Created {len(segments)} random segments (accumulated: {accumulated_duration:.1f}s)")
    return segments


def split_audio_into_segments(audio: np.ndarray, sample_rate: int,
                            min_segment_duration: float = 1.0,
                            max_segment_duration: float = 5.0,
                            logger: logging.Logger = None,
                            use_vad: bool = True,
                            target_accumulated_duration: float = None) -> List[np.ndarray]:
    """Split audio into segments based on speech detection and duration constraints.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds
        logger: Logger instance
        use_vad: If True, use VAD for segmentation. If False, use random segmentation.
        target_accumulated_duration: Stop processing after accumulating this much audio (seconds)
    """
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    if use_vad:
        # Detect speech segments using VAD
        speech_segments = detect_speech_segments(audio, sample_rate, logger=logger)
        
        if not speech_segments:
            logger.warning("No speech segments detected with VAD, falling back to random segmentation")
            return split_audio_randomly(audio, sample_rate, min_segment_duration, max_segment_duration, logger, target_accumulated_duration)
    else:
        # Use random segmentation
        return split_audio_randomly(audio, sample_rate, min_segment_duration, max_segment_duration, logger, target_accumulated_duration)
    
    # Split segments based on duration constraints
    final_segments = []
    accumulated_duration = 0.0
    
    for start, end in speech_segments:
        # Check if we've accumulated enough segments
        if target_accumulated_duration and accumulated_duration >= target_accumulated_duration:
            logger.info(f"Reached target accumulated duration of {target_accumulated_duration}s, stopping segmentation")
            break
            
        segment_duration = (end - start) / sample_rate
        
        if segment_duration < min_segment_duration:
            # Skip too short segments
            continue
        elif segment_duration > max_segment_duration:
            # Split long segments
            segment = audio[start:end]
            num_splits = int(segment_duration / max_segment_duration) + 1
            split_size = len(segment) // num_splits
            
            for i in range(num_splits):
                if target_accumulated_duration and accumulated_duration >= target_accumulated_duration:
                    break
                    
                split_start = i * split_size
                split_end = (i + 1) * split_size if i < num_splits - 1 else len(segment)
                if split_end - split_start > min_segment_duration * sample_rate:
                    final_segments.append(segment[split_start:split_end])
                    accumulated_duration += (split_end - split_start) / sample_rate
        else:
            # Keep segment as is
            final_segments.append(audio[start:end])
            accumulated_duration += segment_duration
    
    # Sort segments by duration (prioritize shorter segments)
    final_segments.sort(key=lambda seg: len(seg))
    
    logger.info(f"Split audio into {len(final_segments)} segments (accumulated duration: {accumulated_duration:.1f}s)")
    logger.info(f"Segment durations: min={min(len(s)/sample_rate for s in final_segments):.2f}s, max={max(len(s)/sample_rate for s in final_segments):.2f}s")
    return final_segments


def create_conversation_from_timestamps(speaker1_info: Dict, speaker2_info: Dict,
                                       speaker1_timestamps: List[Tuple[float, float]],
                                       speaker2_timestamps: List[Tuple[float, float]],
                                       sample_rate: int,
                                       output_paths: Dict[str, str],
                                       target_duration: float = 30.0,
                                       logger: logging.Logger = None) -> None:
    """Create a conversation by batch-loading segments to minimize file loading.
    
    Args:
        speaker1_info: Dict with 'audio_path' and 'sample_rate' for speaker 1
        speaker2_info: Dict with 'audio_path' and 'sample_rate' for speaker 2
        speaker1_timestamps: List of (start_time, end_time) tuples for speaker 1 segments
        speaker2_timestamps: List of (start_time, end_time) tuples for speaker 2 segments
        sample_rate: Target sample rate
        target_duration: Target duration in seconds
        output_paths: Dict with 'mix', 's1', 's2' paths for output files
        logger: Logger instance
    """
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    # Maximum segment duration to avoid overly long monologues
    max_segment_duration = 8.0  # seconds
    
    # Function to batch load segments from audio files
    def batch_load_segments(audio_path: str, timestamps: List[Tuple[float, float]], 
                           target_sample_rate: int) -> Dict[Tuple[float, float], np.ndarray]:
        """Load all segments from a single audio file at once."""
        logger.info(f"Batch loading {len(timestamps)} segments from {Path(audio_path).name}")
        
        # Load the audio file once
        audio, orig_sr = load_audio_from_path(audio_path, logger)
        
        # Resample if needed
        if orig_sr != target_sample_rate:
            audio = resample_audio(audio, orig_sr, target_sample_rate)
        
        # Extract all segments
        segments = {}
        for start_time, end_time in timestamps:
            start_sample = int(start_time * target_sample_rate)
            end_sample = int(end_time * target_sample_rate)
            
            # Ensure we don't go out of bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            # Store segment with timestamp as key
            segments[(start_time, end_time)] = audio[start_sample:end_sample].copy()
        
        # Free the full audio from memory
        del audio
        gc.collect()
        
        logger.debug(f"Extracted {len(segments)} segments, freed full audio from memory")
        return segments
    
    # Process timestamps to crop long segments
    def crop_timestamps_if_needed(timestamps: List[Tuple[float, float]], max_duration: float) -> List[Tuple[float, float]]:
        """Crop timestamps if segments exceed max duration."""
        cropped_timestamps = []
        for start, end in timestamps:
            duration = end - start
            if duration <= max_duration:
                cropped_timestamps.append((start, end))
            else:
                # Split into multiple segments
                num_splits = int(duration / max_duration) + 1
                split_duration = duration / num_splits
                for i in range(num_splits):
                    split_start = start + i * split_duration
                    split_end = start + (i + 1) * split_duration if i < num_splits - 1 else end
                    if split_end - split_start >= 1.0:  # Only keep segments >= 1 second
                        cropped_timestamps.append((split_start, split_end))
        return cropped_timestamps
    
    # Crop timestamps
    speaker1_timestamps = crop_timestamps_if_needed(speaker1_timestamps, max_segment_duration)
    speaker2_timestamps = crop_timestamps_if_needed(speaker2_timestamps, max_segment_duration)
    
    # Sort by duration to prioritize shorter ones  
    speaker1_timestamps = sorted(speaker1_timestamps, key=lambda t: t[1] - t[0])
    speaker2_timestamps = sorted(speaker2_timestamps, key=lambda t: t[1] - t[0])
    
    logger.info(f"After cropping: {len(speaker1_timestamps)} timestamps for speaker1, {len(speaker2_timestamps)} timestamps for speaker2")
    
    # Create interjection timestamps by trimming existing timestamps
    def create_interjection_timestamps(timestamps: List[Tuple[float, float]], max_interjection_duration=1.0):
        """Create short interjection timestamps by trimming existing timestamps."""
        interjections = []
        for start, end in timestamps:
            duration = end - start
            if duration > 0.3:  # Only use segments longer than 0.3s
                # Create interjection of 0.3-1.0 seconds
                interjection_duration = min(duration, random.uniform(0.3, max_interjection_duration))
                
                # Take from different parts of the segment for variety
                if duration > interjection_duration * 2:
                    # Can take from beginning, middle, or end
                    position = random.choice(['start', 'middle', 'end'])
                    if position == 'start':
                        interjections.append((start, start + interjection_duration))
                    elif position == 'middle':
                        mid_start = start + (duration - interjection_duration) / 2
                        interjections.append((mid_start, mid_start + interjection_duration))
                    else:  # end
                        interjections.append((end - interjection_duration, end))
                else:
                    # Just take from beginning
                    interjections.append((start, start + interjection_duration))
        
        # Shuffle interjections for variety
        random.shuffle(interjections)
        return interjections
    
    # Create interjection pools
    speaker1_short = create_interjection_timestamps(speaker1_timestamps)
    speaker2_short = create_interjection_timestamps(speaker2_timestamps)
    
    logger.info(f"Created {len(speaker1_short)} interjections for speaker1, {len(speaker1_timestamps)} regular timestamps")
    logger.info(f"Created {len(speaker2_short)} interjections for speaker2, {len(speaker2_timestamps)} regular timestamps")
    
    # Copy and shuffle timestamps for regular segments
    speaker1_regular = speaker1_timestamps.copy()
    speaker2_regular = speaker2_timestamps.copy()
    random.shuffle(speaker1_regular)
    random.shuffle(speaker2_regular)
    
    # Collect all timestamps for batch loading
    logger.info("Collecting all timestamps for batch loading...")
    
    # Group timestamps by audio file
    timestamps_by_file = {}
    
    # Add speaker1 timestamps
    if speaker1_info['audio_path'] not in timestamps_by_file:
        timestamps_by_file[speaker1_info['audio_path']] = set()
    timestamps_by_file[speaker1_info['audio_path']].update(speaker1_regular)
    timestamps_by_file[speaker1_info['audio_path']].update(speaker1_short)
    
    # Add speaker2 timestamps
    if speaker2_info['audio_path'] not in timestamps_by_file:
        timestamps_by_file[speaker2_info['audio_path']] = set()
    timestamps_by_file[speaker2_info['audio_path']].update(speaker2_regular)
    timestamps_by_file[speaker2_info['audio_path']].update(speaker2_short)
    
    # Batch load all segments
    logger.info(f"Batch loading segments from {len(timestamps_by_file)} unique audio files...")
    all_segments = {}
    total_timestamps = sum(len(ts) for ts in timestamps_by_file.values())
    
    for audio_path, timestamps in timestamps_by_file.items():
        try:
            # Convert set to list for batch loading
            segments = batch_load_segments(audio_path, list(timestamps), sample_rate)
            
            # Store segments with audio_path prefix to handle same timestamps from different files
            for timestamp, segment in segments.items():
                all_segments[(audio_path, timestamp)] = segment
        except Exception as e:
            logger.error(f"Failed to batch load segments from {audio_path}: {e}")
            # Continue with other files
    
    logger.info(f"Successfully pre-loaded {len(all_segments)} segments total from {total_timestamps} timestamps")
    
    # Calculate efficiency
    if len(timestamps_by_file) > 0:
        efficiency_ratio = total_timestamps / len(timestamps_by_file)
        logger.info(f"Efficiency: Loading {len(timestamps_by_file)} files instead of {total_timestamps} ({efficiency_ratio:.1f}x reduction)")
    
    # Check if we have enough segments
    if len(all_segments) == 0:
        raise ValueError("Failed to pre-load any segments. Cannot create conversation.")
    
    # Initialize audio arrays
    total_samples = int(target_duration * sample_rate)
    speaker1_audio = np.zeros(total_samples, dtype=np.float32)
    speaker2_audio = np.zeros(total_samples, dtype=np.float32)
    
    # Track last end position for each speaker to prevent self-overlap
    speaker1_last_end = 0
    speaker2_last_end = 0
    
    # Track timeline for cross-speaker interactions
    timeline_position = 0
    
    # Indices for segments
    s1_idx = 0
    s2_idx = 0
    s1_short_idx = 0
    s2_short_idx = 0
    
    # Alternate between speakers
    current_speaker = 1 if random.random() < 0.5 else 2
    
    # Keep track of segments placed for logging
    segments_placed = []
    
    # Continue placing segments until we fill the duration
    while timeline_position < total_samples:
        # Get next timestamp for current speaker
        timestamp = None
        speaker_info = None
        if current_speaker == 1 and s1_idx < len(speaker1_regular):
            timestamp = speaker1_regular[s1_idx]
            speaker_info = speaker1_info
            s1_idx += 1
        elif current_speaker == 2 and s2_idx < len(speaker2_regular):
            timestamp = speaker2_regular[s2_idx]
            speaker_info = speaker2_info
            s2_idx += 1
        else:
            # Switch speaker and try again
            current_speaker = 2 if current_speaker == 1 else 1
            if current_speaker == 1 and s1_idx < len(speaker1_regular):
                timestamp = speaker1_regular[s1_idx]
                speaker_info = speaker1_info
                s1_idx += 1
            elif current_speaker == 2 and s2_idx < len(speaker2_regular):
                timestamp = speaker2_regular[s2_idx]
                speaker_info = speaker2_info
                s2_idx += 1
            else:
                # Reset indices if we've used all segments
                if s1_idx >= len(speaker1_regular) and s2_idx >= len(speaker2_regular):
                    logger.info("Resetting segment indices to fill remaining audio")
                    s1_idx = 0
                    s2_idx = 0
                    random.shuffle(speaker1_regular)
                    random.shuffle(speaker2_regular)
                continue
        
        if timestamp is None:
            continue
            
        # Get pre-loaded segment
        start_time, end_time = timestamp
        segment_key = (speaker_info['audio_path'], (start_time, end_time))
        
        if segment_key not in all_segments:
            logger.warning(f"Segment not found in pre-loaded segments: [{start_time:.2f}-{end_time:.2f}s]")
            continue
            
        segment = all_segments[segment_key]
        
        # Calculate where to place this segment
        # For the current speaker, ensure no self-overlap
        if current_speaker == 1:
            speaker_last_end = speaker1_last_end
        else:
            speaker_last_end = speaker2_last_end
            
        # Determine start position based on timeline and avoiding self-overlap
        if len(segments_placed) > 0:
            # 90% chance of overlap with other speaker, 10% chance of pause
            if random.random() < 0.9:
                # Overlap: 0-3 seconds
                overlap_duration = random.uniform(0, 3.0)
                overlap_samples = int(overlap_duration * sample_rate)
                
                # Start position based on timeline with overlap
                start_position = max(speaker_last_end, timeline_position - overlap_samples)
            else:
                # Pause: 0-1 seconds after last segment
                pause_duration = random.uniform(0, 1.0)
                pause_samples = int(pause_duration * sample_rate)
                
                # Start position with pause
                start_position = max(speaker_last_end, timeline_position + pause_samples)
        else:
            # First segment
            start_position = 0
            
        # Make sure we don't go past the end
        if start_position >= total_samples:
            break
            
        # Calculate end position
        segment_length = len(segment)
        end_position = min(start_position + segment_length, total_samples)
        actual_length = end_position - start_position
        
        if actual_length <= 0:
            continue
            
        # Apply fade-out effect with 10% probability
        segment_to_place = segment[:actual_length].copy()
        if random.random() < 0.1:  # 10% chance
            # Random fade duration between 0.2 and 0.8 seconds
            fade_duration = random.uniform(0.2, 0.8)
            fade_samples = int(fade_duration * sample_rate)
            
            # Don't fade more than half the segment
            fade_samples = min(fade_samples, len(segment_to_place) // 2)
            
            if fade_samples > 0:
                # Create exponential fade curve
                fade_curve = np.exp(-np.linspace(0, 5, fade_samples))
                
                # Apply fade to the end of the segment
                segment_to_place[-fade_samples:] *= fade_curve
                
                logger.debug(f"Applied {fade_duration:.2f}s exponential fade to segment ending at {end_position/sample_rate:.2f}s")
        
        # Place the segment in the appropriate speaker track
        if current_speaker == 1:
            speaker1_audio[start_position:end_position] = segment_to_place
            speaker1_last_end = end_position
        else:
            speaker2_audio[start_position:end_position] = segment_to_place
            speaker2_last_end = end_position
            
        # Update timeline position
        timeline_position = end_position
        
        # Record segment placement
        segments_placed.append({
            'speaker': current_speaker,
            'start': start_position,
            'end': end_position,
            'duration': actual_length / sample_rate
        })
        
        # Try to add interjection based on segment duration
        segment_duration = actual_length / sample_rate
        
        # Interjection probability scales with duration:
        # 1s segment: ~20% chance
        # 2s segment: ~40% chance  
        # 3s+ segment: ~60% chance
        interjection_prob = min(0.6, segment_duration * 0.2)
        
        if segment_duration > 1.0 and random.random() < interjection_prob:
            # Add interjection from other speaker
            other_speaker = 2 if current_speaker == 1 else 1
            
            # Get short segment timestamp for interjection
            interjection_timestamp = None
            interjection_info = None
            if other_speaker == 1 and s1_short_idx < len(speaker1_short):
                interjection_timestamp = speaker1_short[s1_short_idx]
                interjection_info = speaker1_info
                s1_short_idx += 1
            elif other_speaker == 2 and s2_short_idx < len(speaker2_short):
                interjection_timestamp = speaker2_short[s2_short_idx]
                interjection_info = speaker2_info
                s2_short_idx += 1
                
            if interjection_timestamp is not None:
                # Get pre-loaded interjection
                interjection_key = (interjection_info['audio_path'], interjection_timestamp)
                
                if interjection_key in all_segments:
                    interjection = all_segments[interjection_key]
                else:
                    logger.warning("Interjection not found in pre-loaded segments")
                    interjection = None
                    
            if interjection is not None:
                # Place interjection in the middle of the main segment
                # But ensure it doesn't overlap with other audio from the same speaker
                interjection_offset = random.uniform(0.3, 0.7) * actual_length
                interjection_start = int(start_position + interjection_offset)
                
                # Get the last end position for the interjecting speaker
                if other_speaker == 1:
                    other_last_end = speaker1_last_end
                else:
                    other_last_end = speaker2_last_end
                    
                # Make sure interjection starts after any existing audio from that speaker
                interjection_start = max(interjection_start, other_last_end)
                
                interjection_end = min(interjection_start + len(interjection), total_samples)
                interjection_length = interjection_end - interjection_start
                
                if interjection_length > 0:
                    # Apply fade-out to interjections with 10% probability
                    interjection_to_place = interjection[:interjection_length].copy()
                    if random.random() < 0.1:  # 10% chance
                        fade_duration = random.uniform(0.1, 0.3)  # Shorter fades for interjections
                        fade_samples = int(fade_duration * sample_rate)
                        fade_samples = min(fade_samples, len(interjection_to_place) // 2)
                        
                        if fade_samples > 0:
                            fade_curve = np.exp(-np.linspace(0, 5, fade_samples))
                            interjection_to_place[-fade_samples:] *= fade_curve
                    
                    if other_speaker == 1:
                        speaker1_audio[interjection_start:interjection_end] = interjection_to_place
                        speaker1_last_end = interjection_end
                    else:
                        speaker2_audio[interjection_start:interjection_end] = interjection_to_place
                        speaker2_last_end = interjection_end
                    
                    segments_placed.append({
                        'speaker': other_speaker,
                        'start': interjection_start,
                        'end': interjection_end,
                        'duration': interjection_length / sample_rate,
                        'type': 'interjection'
                    })
        
        # Switch speakers for next turn
        current_speaker = 2 if current_speaker == 1 else 1
    
    # Check if we have gaps at the end and fill them
    last_filled_sample = 0
    for i in range(total_samples - 1, -1, -1):
        if np.abs(speaker1_audio[i]) > 0.001 or np.abs(speaker2_audio[i]) > 0.001:
            last_filled_sample = i
            break
    
    # If there's a significant gap at the end, fill it
    gap_at_end = total_samples - last_filled_sample
    if gap_at_end > sample_rate:  # More than 1 second gap
        logger.info(f"Filling {gap_at_end/sample_rate:.1f}s gap at the end")
        
        # Fill with remaining segments, respecting speaker boundaries
        fill_position = max(speaker1_last_end, speaker2_last_end)
        
        # Alternate between speakers to fill the gap
        fill_speaker = 1 if random.random() < 0.5 else 2
        
        while fill_position < total_samples - sample_rate * 0.5:  # Leave small silence at very end
            # Get next timestamp
            timestamp = None
            speaker_info = None
            if fill_speaker == 1:
                if s1_idx < len(speaker1_regular):
                    timestamp = speaker1_regular[s1_idx]
                    speaker_info = speaker1_info
                    s1_idx += 1
                elif s1_short_idx < len(speaker1_short):
                    timestamp = speaker1_short[s1_short_idx]
                    speaker_info = speaker1_info
                    s1_short_idx += 1
            else:
                if s2_idx < len(speaker2_regular):
                    timestamp = speaker2_regular[s2_idx]
                    speaker_info = speaker2_info
                    s2_idx += 1
                elif s2_short_idx < len(speaker2_short):
                    timestamp = speaker2_short[s2_short_idx]
                    speaker_info = speaker2_info
                    s2_short_idx += 1
                    
            if timestamp is None:
                # Try other speaker
                fill_speaker = 2 if fill_speaker == 1 else 1
                continue
                
            # Get pre-loaded fill segment
            segment_key = (speaker_info['audio_path'], timestamp)
            
            if segment_key not in all_segments:
                logger.warning("Fill segment not found in pre-loaded segments")
                continue
                
            segment = all_segments[segment_key]
                
            # Place segment
            if fill_speaker == 1:
                start_pos = max(fill_position, speaker1_last_end)
            else:
                start_pos = max(fill_position, speaker2_last_end)
                
            end_pos = min(start_pos + len(segment), total_samples)
            length = end_pos - start_pos
            
            if length > sample_rate * 0.5:  # At least 0.5s
                if fill_speaker == 1:
                    speaker1_audio[start_pos:end_pos] = segment[:length]
                    speaker1_last_end = end_pos
                else:
                    speaker2_audio[start_pos:end_pos] = segment[:length]
                    speaker2_last_end = end_pos
                    
                fill_position = end_pos + int(random.uniform(0, 0.5) * sample_rate)
                
            # Switch speakers
            fill_speaker = 2 if fill_speaker == 1 else 1
    
    # Normalize individual speakers to similar volume levels
    # Calculate RMS (root mean square) for each speaker
    speaker1_rms = np.sqrt(np.mean(speaker1_audio[speaker1_audio != 0]**2)) if np.any(speaker1_audio != 0) else 0
    speaker2_rms = np.sqrt(np.mean(speaker2_audio[speaker2_audio != 0]**2)) if np.any(speaker2_audio != 0) else 0
    
    # Normalize both speakers to a similar RMS level (0.1 is a reasonable target)
    target_rms = 0.1
    if speaker1_rms > 0:
        speaker1_audio = speaker1_audio * (target_rms / speaker1_rms)
    if speaker2_rms > 0:
        speaker2_audio = speaker2_audio * (target_rms / speaker2_rms)
    
    # Create mixed audio
    mixed_audio = speaker1_audio + speaker2_audio
    
    # Normalize to prevent clipping
    max_val = max(np.abs(speaker1_audio).max(), np.abs(speaker2_audio).max(), np.abs(mixed_audio).max())
    if max_val > 0.95:
        scale_factor = 0.95 / max_val
        mixed_audio = mixed_audio * scale_factor
        speaker1_audio = speaker1_audio * scale_factor
        speaker2_audio = speaker2_audio * scale_factor
    
    # Log final audio statistics
    final_silence = 0
    for i in range(len(mixed_audio) - 1, -1, -1):
        if np.abs(mixed_audio[i]) > 0.001:
            break
        final_silence += 1
    
    if final_silence > sample_rate * 0.5:  # More than 0.5 seconds
        logger.debug(f"Final silence duration: {final_silence/sample_rate:.2f}s")
    
    logger.info(f"Conversation created: {len(segments_placed)} segments placed")
    logger.debug(f"Speaker1 last end: {speaker1_last_end:.2f}s, Speaker2 last end: {speaker2_last_end:.2f}s")
    
    # Save audio files directly to disk
    try:
        sf.write(output_paths['mix'], mixed_audio, sample_rate)
        sf.write(output_paths['s1'], speaker1_audio, sample_rate)
        sf.write(output_paths['s2'], speaker2_audio, sample_rate)
        
        logger.info(f"Saved conversation to: {output_paths['mix']}")
        
        # Calculate and return duration for metadata
        duration = len(mixed_audio) / sample_rate
        return duration
        
    except Exception as e:
        logger.error(f"Failed to save audio files: {e}")
        raise
    finally:
        # Clean up memory
        del mixed_audio
        del speaker1_audio  
        del speaker2_audio
        # Clean up pre-loaded segments
        all_segments.clear()
        del all_segments
        gc.collect()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    # Use scipy for resampling
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


def generate_conversations_from_real_audio(output_dir: str,
                                         bucket_name: str,
                                         prefix: str,
                                         num_conversations: int,
                                         target_sample_rate: int = 8000,
                                         target_duration: float = 30.0,
                                         max_samples_to_load: int = 1000,
                                         streaming: bool = False,
                                         use_vad: bool = True,
                                         hard_sample_percentage: float = 0.3,
                                         pause_threshold: float = 0.5) -> None:
    """Generate conversation dataset from real single-speaker audio files."""
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("Starting conversation generation from real audio")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GCS bucket: {bucket_name}")
    logger.info(f"Prefix: {prefix}")
    logger.info(f"Number of conversations to generate: {num_conversations}")
    logger.info(f"Target sample rate: {target_sample_rate} Hz")
    logger.info(f"Target duration: {target_duration} seconds")
    logger.info(f"Streaming mode: {'enabled' if streaming else 'disabled'}")
    logger.info(f"VAD segmentation: {'enabled' if use_vad else 'disabled (using random segmentation)'}")
    logger.info(f"Hard sample percentage: {hard_sample_percentage * 100:.0f}% (similar speakers)")
    logger.info(f"Pause threshold: {pause_threshold}s (for word timestamp-based segmentation)")
    logger.info(f"Early stopping: Will process up to {target_duration * 2:.0f}s of audio per file")
    
    # List parquet files
    parquet_files = list_parquet_files(bucket_name, prefix, logger)
    
    if not parquet_files:
        logger.error("No parquet files found")
        return
    
    # Load single-speaker samples with segment timestamps
    logger.info("Loading single-speaker samples from parquet files...")
    samples_dataset = load_single_speaker_dataset(
        parquet_files, 
        max_samples_to_load, 
        target_sample_rate,
        use_vad,
        pause_threshold,
        logger
    )
    
    # Create output directories
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    mix_dir = output_path / "mix"
    s1_dir = output_path / "s1" 
    s2_dir = output_path / "s2"
    
    mix_dir.mkdir(exist_ok=True)
    s1_dir.mkdir(exist_ok=True)
    s2_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created output directories: {mix_dir}, {s1_dir}, {s2_dir}")
    
    # Process dataset to filter by number of segments
    logger.info("Filtering samples by number of segments...")
    
    min_segments_per_speaker = 5  # Minimum segments per speaker
    
    # Filter samples with sufficient segments
    valid_samples = samples_dataset.filter(
        lambda x: x['num_segments'] >= min_segments_per_speaker,
        desc="Filtering samples with sufficient segments"
    )
    
    logger.info(f"Found {len(valid_samples)} samples with at least {min_segments_per_speaker} segments")
    
    if len(valid_samples) < 2:
        logger.error(f"Not enough audio files with sufficient segments. Found {len(valid_samples)}, need at least 2")
        logger.error("Each audio file must have at least 5 speech segments to be used")
        return
    
    # Extract embeddings for similarity computation
    embeddings_by_speaker = {}
    for sample in valid_samples:
        speaker_id = sample['speaker_id']
        embeddings_by_speaker[speaker_id] = np.array(sample['speaker_embedding'])
    
    # Check total available segments
    total_segments = sum(sample['num_segments'] for sample in valid_samples)
    if total_segments < num_conversations * 4:
        logger.error(f"Not enough segments to create requested number of conversations. Have {total_segments}, need at least {num_conversations * 4}")
        return
    
    # Log speaker statistics
    for i, sample in enumerate(valid_samples):
        if i >= 5:  # Show first 5 speakers
            logger.info(f"  ... and {len(valid_samples) - 5} more files")
            break
        speaker_name = Path(sample['speaker_id']).name if '/' in sample['speaker_id'] else sample['speaker_id']
        logger.info(f"  File '{speaker_name}': {sample['num_segments']} segments")
    
    # Generate conversations
    all_data = []
    successful_generations = 0
    failed_generations = 0
    
    # Convert to list for easier sampling
    speaker_list = [s['speaker_id'] for s in valid_samples]
    valid_samples_dict = {s['speaker_id']: s for s in valid_samples}
    
    # Pre-compute similarity matrix for efficient speaker pair selection
    logger.info("Computing speaker similarity matrix...")
    similarity_matrix = {}
    for i, speaker1 in enumerate(speaker_list):
        for j, speaker2 in enumerate(speaker_list):
            if i < j:  # Only compute upper triangle
                similarity = compute_embedding_similarity(
                    embeddings_by_speaker[speaker1],
                    embeddings_by_speaker[speaker2]
                )
                similarity_matrix[(speaker1, speaker2)] = similarity
                similarity_matrix[(speaker2, speaker1)] = similarity  # Symmetric
    
    # Track used speaker pairs to ensure diversity
    used_pairs = set()
    hard_samples_count = 0
    
    for i in tqdm(range(num_conversations), desc="Generating conversations"):
        try:
            # Determine if this should be a hard sample
            is_hard_sample = (i / num_conversations) < hard_sample_percentage
            
            # Select two different speakers for this conversation
            if len(speaker_list) == 2:
                # If we only have 2 speakers, use them
                speaker1_id, speaker2_id = speaker_list
            else:
                # Select speakers based on whether this is a hard sample
                if is_hard_sample:
                    # Find similar speaker pairs (hard samples)
                    # Get all possible pairs sorted by similarity
                    all_pairs = [(s1, s2, sim) for (s1, s2), sim in similarity_matrix.items() 
                                if s1 < s2]  # Avoid duplicates
                    all_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity (high to low)
                    
                    # Select from top 25% most similar pairs that haven't been used recently
                    similar_threshold = int(len(all_pairs) * 0.25)
                    candidate_pairs = all_pairs[:similar_threshold]
                    
                    # Try to find an unused or least recently used pair
                    selected_pair = None
                    for s1, s2, sim in candidate_pairs:
                        pair_key = tuple(sorted([s1, s2]))
                        if pair_key not in used_pairs:
                            selected_pair = (s1, s2)
                            break
                    
                    # If all similar pairs have been used, select the least recently used one
                    if selected_pair is None and candidate_pairs:
                        s1, s2, _ = random.choice(candidate_pairs)
                        selected_pair = (s1, s2)
                    
                    if selected_pair:
                        speaker1_id, speaker2_id = selected_pair
                        hard_samples_count += 1
                        logger.debug(f"Hard sample {hard_samples_count}: similarity = {similarity_matrix[(speaker1_id, speaker2_id)]:.3f}")
                    else:
                        # Fallback to random selection
                        speaker1_id, speaker2_id = random.sample(speaker_list, 2)
                else:
                    # Easy sample: select dissimilar speakers
                    # Get all possible pairs sorted by similarity
                    all_pairs = [(s1, s2, sim) for (s1, s2), sim in similarity_matrix.items() 
                                if s1 < s2]  # Avoid duplicates
                    all_pairs.sort(key=lambda x: x[2])  # Sort by similarity (low to high)
                    
                    # Select from bottom 50% least similar pairs
                    dissimilar_threshold = int(len(all_pairs) * 0.5)
                    candidate_pairs = all_pairs[:dissimilar_threshold]
                    
                    # Try to find an unused or least recently used pair
                    selected_pair = None
                    for s1, s2, sim in candidate_pairs:
                        pair_key = tuple(sorted([s1, s2]))
                        if pair_key not in used_pairs:
                            selected_pair = (s1, s2)
                            break
                    
                    # If all dissimilar pairs have been used, select randomly
                    if selected_pair is None and candidate_pairs:
                        s1, s2, _ = random.choice(candidate_pairs)
                        selected_pair = (s1, s2)
                    
                    if selected_pair:
                        speaker1_id, speaker2_id = selected_pair
                    else:
                        # Fallback to random selection
                        speaker1_id, speaker2_id = random.sample(speaker_list, 2)
            
            # Track used pairs (keep only last N pairs to allow reuse after some time)
            pair_key = tuple(sorted([speaker1_id, speaker2_id]))
            used_pairs.add(pair_key)
            if len(used_pairs) > max(20, num_conversations // 5):  # Keep track of recent pairs
                # Remove oldest pair (convert to list, remove first, convert back)
                used_pairs = set(list(used_pairs)[1:])
            
            # Get speaker data
            speaker1_data = valid_samples_dict[speaker1_id]
            speaker2_data = valid_samples_dict[speaker2_id]
            
            speaker1_timestamps = speaker1_data['segment_timestamps']
            speaker2_timestamps = speaker2_data['segment_timestamps']
            
            # Select random timestamps for this conversation from each speaker
            num_segments_per_speaker = min(10, min(len(speaker1_timestamps), len(speaker2_timestamps)) // 2)
            num_segments_per_speaker = max(2, num_segments_per_speaker)  # At least 2 segments per speaker
            
            s1_timestamps = random.sample(speaker1_timestamps, num_segments_per_speaker)
            s2_timestamps = random.sample(speaker2_timestamps, num_segments_per_speaker)
            
            # Extract just the filename for cleaner logging
            speaker1_name = Path(speaker1_id).name if '/' in speaker1_id else speaker1_id
            speaker2_name = Path(speaker2_id).name if '/' in speaker2_id else speaker2_id
            logger.info(f"Conversation {i}: {speaker1_name} vs {speaker2_name}")
            logger.debug(f"Using {len(s1_timestamps)} timestamps from speaker1, {len(s2_timestamps)} timestamps from speaker2")
            
            # Validate timestamps
            if not s1_timestamps or not s2_timestamps:
                logger.error(f"No timestamps available for conversation {i}")
                failed_generations += 1
                continue
                
            # Create conversation output paths
            conversation_id = f"conversation_{i:04d}"
            output_paths = {
                'mix': str((mix_dir / f"{conversation_id}_mix.wav").absolute()),
                's1': str((s1_dir / f"{conversation_id}_s1.wav").absolute()),
                's2': str((s2_dir / f"{conversation_id}_s2.wav").absolute())
            }
            
            # Create conversation (loading segments on-demand and saving to disk)
            duration = create_conversation_from_timestamps(
                speaker1_data, speaker2_data,
                s1_timestamps, s2_timestamps,
                target_sample_rate,
                output_paths,
                target_duration=target_duration,
                logger=logger
            )
            
            logger.debug(f"Generated conversation {i:04d} (duration: {duration:.2f}s)")
            
            # Add to data list
            all_data.append({
                "mix_path": output_paths['mix'],
                "s1_path": output_paths['s1'],
                "s2_path": output_paths['s2'],
                "duration": duration
            })
            
            successful_generations += 1
            
        except Exception as e:
            logger.error(f"Error generating conversation {i}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            failed_generations += 1
            continue
    
    logger.info(f"Generation complete. Successful: {successful_generations}, Failed: {failed_generations}")
    
    if successful_generations == 0:
        logger.error("No conversations were successfully generated!")
        return
    
    logger.info(f"Hard samples (similar speakers): {hard_samples_count} ({hard_samples_count/successful_generations*100:.1f}%)")
    logger.info(f"Easy samples (dissimilar speakers): {successful_generations - hard_samples_count} ({(successful_generations - hard_samples_count)/successful_generations*100:.1f}%)")
    
    # Create train/valid/test splits
    random.shuffle(all_data)
    n_train = int(0.8 * len(all_data))
    n_valid = int(0.1 * len(all_data))
    
    train_data = all_data[:n_train]
    valid_data = all_data[n_train:n_train+n_valid]
    test_data = all_data[n_train+n_valid:]
    
    # Write SCP files
    def write_scp(data: List[Dict], filename: str):
        scp_path = output_path / filename
        with open(scp_path, 'w') as f:
            for item in data:
                f.write(f"{item['mix_path']} {item['s1_path']} {item['s2_path']}\n")
        logger.info(f"Wrote {len(data)} entries to {scp_path}")
    
    write_scp(train_data, "train.scp")
    write_scp(valid_data, "valid.scp")
    write_scp(test_data, "test.scp")
    
    logger.info(f"Created SCP files - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Save metadata
    metadata = {
        "dataset_info": {
            "total_conversations": successful_generations,
            "sample_rate": target_sample_rate,
            "target_duration_sec": target_duration,
            "source_bucket": bucket_name,
            "source_prefix": prefix
        },
        "splits": {
            "train": len(train_data),
            "valid": len(valid_data),
            "test": len(test_data)
        },
        "conversations": all_data
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up HuggingFace datasets to free memory
    logger.info("Cleaning up datasets and freeing memory...")
    del samples_dataset
    del valid_samples
    del embeddings_by_speaker
    del similarity_matrix
    gc.collect()
    
    logger.info("Dataset generation completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Generate conversation dataset from real audio")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for generated data")
    parser.add_argument("--bucket_name", type=str, default="audio_datasets_emuni",
                      help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default="parquet_processed/ja",
                      help="Prefix for parquet files in GCS")
    parser.add_argument("--num_conversations", type=int, default=100,
                      help="Number of conversations to generate")
    parser.add_argument("--sample_rate", type=int, default=8000,
                      help="Target sample rate (Hz)")
    parser.add_argument("--target_duration", type=float, default=30.0,
                      help="Target duration for each conversation in seconds")
    parser.add_argument("--max_samples", type=int, default=1000,
                      help="Maximum number of single-speaker samples to load")
    parser.add_argument("--streaming", action="store_true",
                      help="Use streaming mode for memory-efficient processing")
    parser.add_argument("--no-vad", action="store_true",
                      help="Disable VAD and use random segmentation instead")
    parser.add_argument("--hard_sample_percentage", type=float, default=0.3,
                      help="Percentage of conversations that should be hard samples (similar speakers, 0.0-1.0)")
    parser.add_argument("--pause_threshold", type=float, default=0.5,
                      help="Minimum pause duration in seconds to split segments when using word timestamps")
    
    args = parser.parse_args()
    
    generate_conversations_from_real_audio(
        args.output_dir,
        args.bucket_name,
        args.prefix,
        args.num_conversations,
        args.sample_rate,
        args.target_duration,
        args.max_samples,
        args.streaming,
        use_vad=not args.no_vad,
        hard_sample_percentage=args.hard_sample_percentage,
        pause_threshold=args.pause_threshold
    )


if __name__ == "__main__":
    main()