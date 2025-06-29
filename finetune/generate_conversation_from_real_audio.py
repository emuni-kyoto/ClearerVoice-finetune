#!/usr/bin/env python3
"""Generate conversation datasets from real single-speaker audio files.

This script:
1. Loads preprocessed parquet files from GCP containing single-speaker audio
2. Efficiently loads files one-by-one until sufficient data is collected
3. Handles audio paths: GCS URLs (gs://...), mount paths (/mnt/disks/...), and local paths
4. Selects samples with only one speaker
5. Splits audio from different speakers into segments based on pauses or randomly
6. Creates synthetic conversations by overlapping segments from different speakers
7. Saves in ClearerVoice format for finetuning

Usage:
    python generate_conversation_from_real_audio.py \
        --output_dir ./real_conversation_data \
        --bucket_name audio_datasets_emuni \
        --prefix parquet_processed/ja \
        --num_conversations 100 \
        --sample_rate 8000 \
        --max_samples 1000

Optional flags:
    --streaming: Enable streaming mode for processing (experimental)
    --max_samples: Maximum number of single-speaker samples to load (default: 1000)
    --no-vad: Disable VAD and use random segmentation (useful when VAD fails)

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
import numpy as np
import soundfile as sf
import torchaudio
import webrtcvad
from google.cloud import storage
from scipy import signal
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


def load_single_speaker_dataset(parquet_files: List[str], 
                               max_samples: int = 1000,
                               logger: logging.Logger = None) -> datasets.Dataset:
    """Load samples from parquet files that contain only single speakers using HuggingFace datasets.
    
    Loads files one by one until we have sufficient samples, avoiding unnecessary downloads.
    """
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    logger.info(f"Found {len(parquet_files)} parquet files available")
    logger.info(f"Will load files until we collect {max_samples} single-speaker samples")
    logger.info("Note: May not need to load all files if target is reached early")
    
    all_samples = []
    total_samples_loaded = 0
    files_processed = 0
    
    # Shuffle files for more diverse sampling
    shuffled_files = parquet_files.copy()
    random.shuffle(shuffled_files)
    
    # Process files one by one
    # Use manual progress bar since we don't know exactly how many files we'll need
    pbar = tqdm(desc="Loading parquet files", unit="files")
    
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
                streaming=False  # Load individual file into memory
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
                
                # Add source file info
                def add_source_info(example):
                    example['source_parquet'] = parquet_file
                    # Extract a more meaningful speaker ID from the file path
                    example['speaker_id'] = Path(parquet_file).stem
                    return example
                
                single_speaker_data = single_speaker_data.map(
                    add_source_info,
                    desc="Adding source info"
                )
                
                # Convert to list of dicts for accumulation
                samples_from_file = list(single_speaker_data)
                all_samples.extend(samples_from_file)
                
                samples_count = len(samples_from_file)
                total_samples_loaded += samples_count
                logger.debug(f"Found {samples_count} single-speaker samples in {parquet_file}")
            
            files_processed += 1
            pbar.update(1)
            pbar.set_postfix({
                "loaded": f"{total_samples_loaded}/{max_samples} samples",
                "files": files_processed,
                "available": f"{len(shuffled_files)} total"
            })
            
            # Clean up temporary file
            try:
                if 'local_parquet_path' in locals() and os.path.exists(local_parquet_path):
                    os.remove(local_parquet_path)
                    logger.debug(f"Cleaned up temporary file: {local_parquet_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
            
        except Exception as e:
            logger.warning(f"Failed to load {parquet_file}: {e}")
            pbar.update(1)
            # Clean up temporary file on error
            try:
                if 'local_parquet_path' in locals() and os.path.exists(local_parquet_path):
                    os.remove(local_parquet_path)
            except Exception:
                pass
            continue
    
    pbar.close()
    
    if not all_samples:
        raise ValueError("No single-speaker samples found in the parquet files")
    
    # Create final dataset from accumulated samples
    logger.info(f"Creating final dataset from {total_samples_loaded} samples")
    final_dataset = datasets.Dataset.from_list(all_samples)
    
    logger.info(f"Loaded {len(final_dataset)} single-speaker samples from {files_processed} files")
    return final_dataset


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
            
            # Download to temporary file
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            temp_path = f"/tmp/{Path(blob_name).name}"
            blob.download_to_filename(temp_path)
            
            # Load audio from temp file
            audio_file = temp_path
            cleanup_temp = True
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
            
            logger.info(f"Converting mount path to GCS: {audio_path} -> gs://{bucket_name}/{relative_path}")
            
            # Download from GCS
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(relative_path)
                
                # Create temp directory structure that mirrors the GCS structure for caching
                temp_dir = Path("/tmp/audio_downloads") / bucket_name
                temp_path = temp_dir / relative_path
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if file is already cached
                if temp_path.exists():
                    logger.debug(f"Using cached audio file: {temp_path}")
                else:
                    # Try to download the file
                    logger.debug(f"Downloading from GCS: gs://{bucket_name}/{relative_path} to {temp_path}")
                    try:
                        blob.download_to_filename(str(temp_path))
                        logger.debug(f"Successfully downloaded audio file")
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


def create_conversation_from_segments(speaker1_segments: List[np.ndarray],
                                    speaker2_segments: List[np.ndarray],
                                    sample_rate: int,
                                    overlap_ratio: float = 0.2,  # Kept for backward compatibility, but not used
                                    target_duration: float = 30.0,
                                    logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a conversation by interleaving segments from two speakers.
    
    Features:
    - Natural turn-taking with random overlaps (0-3 seconds) between speakers
    - Occasional pauses between turns (0-1 seconds)
    - Short interjections during longer segments
    - Ensures NO overlapping within each speaker's track
    - Fills audio completely without silence at the end
    """
    
    if logger is None:
        logger = logging.getLogger('real_conversation_generator')
    
    # Maximum segment duration to avoid overly long monologues
    max_segment_duration = 8.0  # seconds
    
    # Sort segments by duration to prioritize shorter ones
    speaker1_segments = sorted(speaker1_segments, key=lambda seg: len(seg))
    speaker2_segments = sorted(speaker2_segments, key=lambda seg: len(seg))
    
    # Preprocess segments: crop if too long
    def crop_segment_if_needed(segment: np.ndarray, max_duration: float) -> List[np.ndarray]:
        """Crop segment if it exceeds max duration, return list of segments."""
        segment_duration = len(segment) / sample_rate
        if segment_duration <= max_duration:
            return [segment]
        
        # Split into multiple segments
        max_samples = int(max_duration * sample_rate)
        segments = []
        for i in range(0, len(segment), max_samples):
            end = min(i + max_samples, len(segment))
            if (end - i) / sample_rate >= 1.0:  # Only keep segments >= 1 second
                segments.append(segment[i:end])
        return segments
    
    # Crop all segments
    speaker1_segments_cropped = []
    for seg in speaker1_segments:
        cropped = crop_segment_if_needed(seg, max_segment_duration)
        if len(cropped) > 1:
            logger.debug(f"Cropped speaker1 segment from {len(seg)/sample_rate:.2f}s into {len(cropped)} segments")
        speaker1_segments_cropped.extend(cropped)
    
    speaker2_segments_cropped = []
    for seg in speaker2_segments:
        cropped = crop_segment_if_needed(seg, max_segment_duration)
        if len(cropped) > 1:
            logger.debug(f"Cropped speaker2 segment from {len(seg)/sample_rate:.2f}s into {len(cropped)} segments")
        speaker2_segments_cropped.extend(cropped)
    
    speaker1_segments = speaker1_segments_cropped
    speaker2_segments = speaker2_segments_cropped
    
    logger.info(f"After cropping: {len(speaker1_segments)} segments for speaker1, {len(speaker2_segments)} segments for speaker2")
    
    # All segments can be used for main utterances
    speaker1_regular = speaker1_segments.copy()
    speaker2_regular = speaker2_segments.copy()
    
    # Create interjection pools by trimming existing segments
    def create_interjection_segments(segments, sample_rate, max_interjection_duration=1.0):
        """Create short interjection segments by trimming existing segments."""
        interjections = []
        for seg in segments:
            duration = len(seg) / sample_rate
            if duration > 0.3:  # Only use segments longer than 0.3s
                # Create interjection of 0.3-1.0 seconds
                interjection_duration = min(duration, random.uniform(0.3, max_interjection_duration))
                interjection_samples = int(interjection_duration * sample_rate)
                
                # Take from different parts of the segment for variety
                if duration > interjection_duration * 2:
                    # Can take from beginning, middle, or end
                    position = random.choice(['start', 'middle', 'end'])
                    if position == 'start':
                        interjection = seg[:interjection_samples]
                    elif position == 'middle':
                        mid_start = (len(seg) - interjection_samples) // 2
                        interjection = seg[mid_start:mid_start + interjection_samples]
                    else:  # end
                        interjection = seg[-interjection_samples:]
                else:
                    # Just take from beginning
                    interjection = seg[:interjection_samples]
                    
                interjections.append(interjection)
        
        # Shuffle interjections for variety
        random.shuffle(interjections)
        return interjections
    
    # Create interjection pools
    speaker1_short = create_interjection_segments(speaker1_segments, sample_rate)
    speaker2_short = create_interjection_segments(speaker2_segments, sample_rate)
    
    logger.info(f"Created {len(speaker1_short)} interjections for speaker1, {len(speaker1_regular)} regular segments")
    logger.info(f"Created {len(speaker2_short)} interjections for speaker2, {len(speaker2_regular)} regular segments")
    
    # Shuffle regular segments
    random.shuffle(speaker1_regular)
    random.shuffle(speaker2_regular)
    
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
        # Get next segment for current speaker
        segment = None
        if current_speaker == 1 and s1_idx < len(speaker1_regular):
            segment = speaker1_regular[s1_idx]
            s1_idx += 1
        elif current_speaker == 2 and s2_idx < len(speaker2_regular):
            segment = speaker2_regular[s2_idx]
            s2_idx += 1
        else:
            # Switch speaker and try again
            current_speaker = 2 if current_speaker == 1 else 1
            if current_speaker == 1 and s1_idx < len(speaker1_regular):
                segment = speaker1_regular[s1_idx]
                s1_idx += 1
            elif current_speaker == 2 and s2_idx < len(speaker2_regular):
                segment = speaker2_regular[s2_idx]
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
        
        if segment is None:
            continue
        
        # Calculate where to place this segment
        # For the current speaker, ensure no self-overlap
        if current_speaker == 1:
            speaker_last_end = speaker1_last_end
        else:
            speaker_last_end = speaker2_last_end
            
        # Determine start position based on timeline and avoiding self-overlap
        if len(segments_placed) > 0:
            last_segment = segments_placed[-1]
            
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
        end_position = min(start_position + len(segment), total_samples)
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
            
            # Get short segment for interjection
            interjection = None
            if other_speaker == 1 and s1_short_idx < len(speaker1_short):
                interjection = speaker1_short[s1_short_idx]
                s1_short_idx += 1
            elif other_speaker == 2 and s2_short_idx < len(speaker2_short):
                interjection = speaker2_short[s2_short_idx]
                s2_short_idx += 1
                
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
            # Get next segment
            segment = None
            if fill_speaker == 1:
                if s1_idx < len(speaker1_regular):
                    segment = speaker1_regular[s1_idx]
                    s1_idx += 1
                elif s1_short_idx < len(speaker1_short):
                    segment = speaker1_short[s1_short_idx]
                    s1_short_idx += 1
            else:
                if s2_idx < len(speaker2_regular):
                    segment = speaker2_regular[s2_idx]
                    s2_idx += 1
                elif s2_short_idx < len(speaker2_short):
                    segment = speaker2_short[s2_short_idx]
                    s2_short_idx += 1
                    
            if segment is None:
                # Try other speaker
                fill_speaker = 2 if fill_speaker == 1 else 1
                continue
                
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
    
    return mixed_audio, speaker1_audio, speaker2_audio


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
                                         use_vad: bool = True) -> None:
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
    logger.info(f"Early stopping: Will process up to {target_duration * 2:.0f}s of audio per file")
    
    # List parquet files
    parquet_files = list_parquet_files(bucket_name, prefix, logger)
    
    if not parquet_files:
        logger.error("No parquet files found")
        return
    
    # Load single-speaker samples
    logger.info("Loading single-speaker samples from parquet files...")
    samples_dataset = load_single_speaker_dataset(parquet_files, max_samples_to_load, logger)
    
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
    
    # Process audio files and create segment pool by speaker
    logger.info("Processing audio files and creating segment pools by speaker...")
    
    # Dictionary to store segments by speaker (file source)
    segments_by_speaker = {}
    processed_files = 0
    
    # Process audio files in batches
    batch_size = 10
    min_speakers_needed = 2  # Need at least 2 different speakers
    min_segments_per_speaker = 5  # Minimum segments per speaker
    
    # Process in batches for better memory management
    total_samples = len(samples_dataset)
    pbar = tqdm(desc="Processing audio files", unit="files")
    
    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        # Get batch of samples
        end_idx = min(i + batch_size, total_samples)
        batch = samples_dataset.select(range(i, end_idx))
        
        for sample in batch:
            try:
                # Create unique speaker ID based on file path
                # Each file represents a different person, even if internal speaker_ids match
                file_path = sample.get('local_path', '')
                if file_path:
                    # Use the full file path as unique identifier
                    # e.g., /mnt/disks/podcast/audio_files_processed/ja/batch_id/file_id.wav
                    # This ensures each file is treated as a different speaker
                    speaker_id = file_path
                else:
                    # Fallback to using processed file counter
                    speaker_id = f'file_{processed_files}'
                
                # Load audio
                logger.debug(f"Loading audio for speaker {speaker_id} from: {sample['local_path']}")
                audio, sample_rate = load_audio_from_path(sample['local_path'], logger, bucket_name=bucket_name)
                
                # Resample to target rate
                if sample_rate != target_sample_rate:
                    audio = resample_audio(audio, sample_rate, target_sample_rate)
                
                # Split into segments - only process enough audio for ~2x target conversation duration
                # This avoids processing very long audio files unnecessarily
                target_accumulated = target_duration * 2.0  # Process 2x the target conversation duration
                segments = split_audio_into_segments(audio, target_sample_rate, logger=logger, use_vad=use_vad, 
                                                   target_accumulated_duration=target_accumulated)
                
                if segments:
                    if speaker_id not in segments_by_speaker:
                        segments_by_speaker[speaker_id] = []
                    segments_by_speaker[speaker_id].extend(segments)
                    processed_files += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "processed": processed_files, 
                    "speakers": len(segments_by_speaker),
                    "segments": sum(len(segs) for segs in segments_by_speaker.values()),
                    "total_files": total_samples
                })
                
            except Exception as e:
                logger.warning(f"Failed to process audio file: {e}")
                pbar.update(1)
                continue
        
        # Clean up memory after each batch
        gc.collect()
        
        # Check if we have enough speakers with enough segments
        valid_speakers = [spk for spk, segs in segments_by_speaker.items() 
                         if len(segs) >= min_segments_per_speaker]
        
        if len(valid_speakers) >= min_speakers_needed:
            # Check if we have enough total segments for conversations
            total_segments = sum(len(segments_by_speaker[spk]) for spk in valid_speakers)
            if total_segments >= num_conversations * 10:  # ~10 segments per conversation minimum
                pbar.set_description("Processing audio files [COMPLETE - target reached]")
                logger.info(f"✓ Collected enough data: {len(valid_speakers)} audio files, {total_segments} total segments")
                logger.info(f"  Processed {i + batch_size} out of {total_samples} files")
                break
    
    pbar.close()
    
    # Filter out speakers with too few segments
    valid_speakers = {spk: segs for spk, segs in segments_by_speaker.items() 
                     if len(segs) >= min_segments_per_speaker}
    
    logger.info(f"Processed {processed_files} audio files, found {len(valid_speakers)} files with sufficient segments")
    
    # Log speaker statistics
    for speaker_id, segments in list(valid_speakers.items())[:5]:  # Show first 5 speakers
        speaker_name = Path(speaker_id).name if '/' in speaker_id else speaker_id
        logger.info(f"  File '{speaker_name}': {len(segments)} segments")
    if len(valid_speakers) > 5:
        logger.info(f"  ... and {len(valid_speakers) - 5} more files")
    
    if len(valid_speakers) < 2:
        logger.error(f"Not enough audio files with sufficient segments. Found {len(valid_speakers)}, need at least 2")
        logger.error("Each audio file must have at least 5 speech segments to be used")
        return
    
    total_segments = sum(len(segs) for segs in valid_speakers.values())
    if total_segments < num_conversations * 4:  # Need at least 4 segments per conversation
        logger.error(f"Not enough segments to create requested number of conversations. Have {total_segments}, need at least {num_conversations * 4}")
        return
    
    # Generate conversations
    all_data = []
    successful_generations = 0
    failed_generations = 0
    
    # Convert speaker dict to list for easier sampling
    speaker_list = list(valid_speakers.keys())
    
    for i in tqdm(range(num_conversations), desc="Generating conversations"):
        try:
            # Select two different speakers for this conversation
            if len(speaker_list) == 2:
                # If we only have 2 speakers, use them
                speaker1_id, speaker2_id = speaker_list
            else:
                # Randomly select 2 different speakers
                speaker1_id, speaker2_id = random.sample(speaker_list, 2)
            
            speaker1_segments = valid_speakers[speaker1_id]
            speaker2_segments = valid_speakers[speaker2_id]
            
            # Select random segments for this conversation from each speaker
            num_segments_per_speaker = min(10, min(len(speaker1_segments), len(speaker2_segments)) // 2)
            num_segments_per_speaker = max(2, num_segments_per_speaker)  # At least 2 segments per speaker
            
            s1_segments = random.sample(speaker1_segments, num_segments_per_speaker)
            s2_segments = random.sample(speaker2_segments, num_segments_per_speaker)
            
            # Extract just the filename for cleaner logging
            speaker1_name = Path(speaker1_id).name if '/' in speaker1_id else speaker1_id
            speaker2_name = Path(speaker2_id).name if '/' in speaker2_id else speaker2_id
            logger.info(f"Conversation {i}: {speaker1_name} vs {speaker2_name}")
            
            # Create conversation
            mixed, speaker1, speaker2 = create_conversation_from_segments(
                s1_segments, s2_segments, target_sample_rate,
                overlap_ratio=0.2, target_duration=target_duration, logger=logger
            )
            
            # Save audio files
            conversation_id = f"conversation_{i:04d}"
            mix_file = (mix_dir / f"{conversation_id}_mix.wav").absolute()
            s1_file = (s1_dir / f"{conversation_id}_s1.wav").absolute()
            s2_file = (s2_dir / f"{conversation_id}_s2.wav").absolute()
            
            sf.write(str(mix_file), mixed, target_sample_rate)
            sf.write(str(s1_file), speaker1, target_sample_rate)
            sf.write(str(s2_file), speaker2, target_sample_rate)
            
            duration = len(mixed) / target_sample_rate
            logger.debug(f"Saved conversation {i:04d} (duration: {duration:.2f}s)")
            
            # Add to data list
            all_data.append({
                "mix_path": str(mix_file),
                "s1_path": str(s1_file),
                "s2_path": str(s2_file),
                "duration": duration
            })
            
            successful_generations += 1
            
        except Exception as e:
            logger.error(f"Error generating conversation {i}: {e}")
            failed_generations += 1
            continue
    
    logger.info(f"Generation complete. Successful: {successful_generations}, Failed: {failed_generations}")
    
    if successful_generations == 0:
        logger.error("No conversations were successfully generated!")
        return
    
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
        use_vad=not args.no_vad
    )


if __name__ == "__main__":
    main()