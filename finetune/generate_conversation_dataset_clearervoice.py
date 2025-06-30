#!/usr/bin/env python3
"""Generate a dataset of overlapping conversations for ClearerVoice finetuning.

This script:
1. Uses Gemini API with structured output to generate realistic conversation transcripts
2. Uses Google Cloud Text-to-Speech API to synthesize speech for each turn
3. Mixes the speech with random overlaps to simulate natural conversations
4. Applies random noise augmentation to the mixed audio using audiomentations
5. Saves the mixed audio and individual speaker tracks in ClearerVoice format
6. Creates SCP files with absolute paths for training/validation/testing

Usage:
    python generate_conversation_dataset_clearervoice.py --output_dir ./conversation_data --num_conversations 100
"""

import argparse
import json
import logging
import os
import random
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    PitchShift,
    Shift,
    TimeStretch,
)
from google import genai
from google.cloud import texttospeech
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm


# Setup logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "generation.log"
    
    # Create logger
    logger = logging.getLogger('conversation_generator')
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


# Data structures for conversation generation
class ConversationTurn(BaseModel):
    speaker: str
    text: str
    emotion: str  # e.g., "cheerful", "serious", "excited", "calm"
    
class Conversation(BaseModel):
    topic: str
    turns: List[ConversationTurn]
    context: str  # e.g., "casual", "business", "academic"


@dataclass
class AudioSegment:
    """Represents an audio segment with timing information."""
    audio: np.ndarray
    start_time: float
    end_time: float
    speaker: str
    text: str


def save_wave_file(filename: str, pcm: bytes, channels: int = 1, 
                   rate: int = 24000, sample_width: int = 2) -> None:
    """Save PCM data as WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def generate_conversation(topic: str = None, language: str = "en", 
                         target_duration_sec: int = 30, logger: logging.Logger = None) -> Conversation:
    """Generate a conversation using Gemini API with structured output.
    
    Args:
        topic: Conversation topic (if None, randomly selected)
        language: Language code ('en' or 'ja')
        target_duration_sec: Target duration in seconds (default: 30)
        logger: Logger instance
    """
    
    if logger is None:
        logger = logging.getLogger('conversation_generator')
    
    # Calculate approximate number of turns based on target duration
    # Assuming each turn is about 2-3 seconds on average
    min_turns = target_duration_sec // 3
    max_turns = target_duration_sec // 2
    
    # List of conversation topics based on language
    if language == "ja":
        topics = [
            "週末の旅行計画",
            "新しいプロジェクトについての話し合い",
            "夕食に何を作るか決める",
            "最近見た映画について",
            "誕生日パーティーの計画",
            "気候変動について議論",
            "新しい技術について話す",
            "家のリフォーム計画",
            "好きな本について話す",
            "スポーツについて話す"
        ]
    else:
        topics = [
            "planning a weekend trip",
            "discussing a new project at work",
            "deciding what to cook for dinner",
            "talking about a recent movie",
            "planning a birthday party",
            "discussing climate change",
            "talking about new technology",
            "planning home renovations",
            "discussing favorite books",
            "talking about sports"
        ]
    
    if topic is None:
        topic = random.choice(topics)
    
    logger.info(f"Generating conversation about: {topic} (target: {target_duration_sec}s)")
    
    # Language-specific prompts
    if language == "ja":
        prompt = f"""日本語で自然な2人の会話を生成してください。
                トピック: {topic}
                
                会話の条件:
                - 合計{min_turns}-{max_turns}回のターン（約{target_duration_sec}秒分）
                - 割り込みや重なり合う会話パターンを含む
                - カジュアルで自然な日本語を使用
                - 様々な感情（楽しい、真剣、興奮、落ち着いている、困惑、驚き）を含む
                - 話者は'Speaker1'と'Speaker2'を使用
                - 各ターンは1-3文程度
                - コンテキストは"casual"、"business"、"academic"のいずれかを設定
                - リアルな間や考える時間を表現する"""
    else:
        prompt = f"""Generate a realistic 2-person conversation about: {topic}
                
                The conversation should:
                - Have {min_turns}-{max_turns} natural turns total (approximately {target_duration_sec} seconds of dialogue)
                - Include interruptions and overlapping speech patterns
                - Use casual, natural language with conversational fillers
                - Have varied emotions (cheerful, serious, excited, calm, confused, surprised)
                - Use 'Speaker1' and 'Speaker2' for the speakers
                - Each turn should be 1-3 sentences
                - Set context as one of: "casual", "business", "academic"
                - Include realistic pauses and thinking moments"""
    
    try:
        # Generate conversation using structured output
        gemini_client = genai.Client()
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Conversation,
            }
        )
        
        conversation = response.parsed
        logger.info(f"Generated conversation with {len(conversation.turns)} turns, context: {conversation.context}")
        return conversation
        
    except Exception as e:
        logger.error(f"Failed to generate conversation: {e}")
        raise


def synthesize_speech(tts_client: texttospeech.TextToSpeechClient, text: str, 
                     speaker_name: str = "Speaker1", emotion: str = "neutral",
                     language: str = "en", logger: logging.Logger = None) -> bytes:
    """Synthesize speech using Google Cloud Text-to-Speech API with diverse voices."""
    
    if logger is None:
        logger = logging.getLogger('conversation_generator')
    
    # Voice options for different languages
    voice_options = {
        "en": {
            # English female voices
            "female": ["en-US-Wavenet-C", "en-US-Wavenet-E", "en-US-Wavenet-F", "en-US-Wavenet-G", "en-US-Wavenet-H"],
            # English male voices  
            "male": ["en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-D", "en-US-Wavenet-I", "en-US-Wavenet-J"]
        },
        "ja": {
            # Japanese female voices
            "female": [
                "ja-JP-Wavenet-A", "ja-JP-Wavenet-B", "ja-JP-Neural2-B"
            ],
            # Japanese male voices
            "male": [
                "ja-JP-Wavenet-C", "ja-JP-Wavenet-D", "ja-JP-Neural2-C", "ja-JP-Neural2-D"
            ]
        }
    }
    
    # Assign consistent voice per speaker name (deterministic based on name)
    import hashlib
    name_hash = int(hashlib.md5(speaker_name.encode()).hexdigest()[:8], 16)
    
    # Determine gender based on speaker name
    if speaker_name == "Speaker1":
        voice_list = voice_options.get(language, voice_options["en"])["female"]
    else:  # Speaker2
        voice_list = voice_options.get(language, voice_options["en"])["male"]
    
    # Select voice based on speaker name (consistent voice per speaker)
    voice_index = name_hash % len(voice_list)
    selected_voice = voice_list[voice_index]
    
    logger.debug(f"Synthesizing speech for {speaker_name} using voice {selected_voice}: '{text[:50]}...'")
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request with specific voice name
    language_code = "ja-JP" if language == "ja" else "en-US"
    voice = texttospeech.VoiceSelectionParams(
        name=selected_voice,
        language_code=language_code
    )
    
    # Select the type of audio file you want returned (PCM for further processing)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000
    )
    
    try:
        # Perform the text-to-speech request
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        logger.debug(f"Successfully synthesized {len(response.audio_content)} bytes for {speaker_name}")
        return response.audio_content
        
    except Exception as e:
        logger.error(f"Failed to synthesize speech for {speaker_name}: {e}")
        raise


def pcm_to_numpy(pcm_data: bytes, sample_rate: int = 24000) -> np.ndarray:
    """Convert PCM bytes to numpy array."""
    # Assuming 16-bit PCM
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    # Convert to float32 normalized to [-1, 1]
    audio = audio.astype(np.float32) / 32768.0
    return audio


def create_noise_presets(sample_rate: int = 8000) -> List[Compose]:
    """Create multiple noise augmentation presets for variety."""
    
    # Preset 1: Light augmentation (minimal noise)
    light_preset = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.7),
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
        Shift(min_shift=-0.05, max_shift=0.05, p=0.3),
    ])
    
    # Preset 2: Medium augmentation (moderate noise)
    medium_preset = Compose([
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.8),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
        Shift(min_shift=-0.1, max_shift=0.1, p=0.4),
    ])
    
    # Preset 3: Heavy augmentation (noisy environment)
    heavy_preset = Compose([
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.03, p=0.9),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
        Shift(min_shift=-0.15, max_shift=0.15, p=0.5),
    ])
    
    # Preset 4: Variable noise (random intensity)
    variable_preset = Compose([
        AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.025, p=0.6),
        TimeStretch(min_rate=0.92, max_rate=1.08, p=0.3),
        Shift(min_shift=-0.08, max_shift=0.08, p=0.4),
    ])
    
    # Preset 5: Clean (no augmentation for control)
    clean_preset = Compose([])
    
    return [light_preset, medium_preset, heavy_preset, variable_preset, clean_preset]


def mix_conversation_with_overlap(segments: List[AudioSegment], 
                                overlap_ratio: float = 0.3,
                                sample_rate: int = 24000,
                                volume_variation: float = 0.5,
                                logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix conversation segments with random overlaps and random speaker volumes.
    
    Args:
        segments: List of audio segments with timing info
        overlap_ratio: Maximum overlap as ratio of segment duration (0.0-1.0)
        sample_rate: Audio sample rate
        volume_variation: Maximum volume variation factor (0.0-1.0)
                         0.0 = no variation, 1.0 = can vary from 0 to 2x original volume
        logger: Logger instance
        
    Returns:
        Tuple of (mixed_audio, speaker1_audio, speaker2_audio)
    """
    
    if logger is None:
        logger = logging.getLogger('conversation_generator')
    
    logger.debug(f"Mixing {len(segments)} segments with overlap_ratio={overlap_ratio}, volume_variation={volume_variation}")
    
    # Generate random volume scaling for each speaker (consistent per speaker, but random per conversation)
    speaker_volumes = {}
    for segment in segments:
        if segment.speaker not in speaker_volumes:
            # Random volume scaling: 1.0 +/- volume_variation
            # This ensures volume stays positive and varies around 1.0
            volume_scale = 1.0 + random.uniform(-volume_variation, volume_variation)
            speaker_volumes[segment.speaker] = max(0.1, volume_scale)  # Ensure minimum volume
    
    logger.debug(f"Speaker volume scales: {speaker_volumes}")
    
    # Calculate total duration needed
    current_time = 0.0
    segment_times = []
    
    for i, segment in enumerate(segments):
        duration = len(segment.audio) / sample_rate
        
        if i > 0:
            # Add random overlap with previous segment
            max_overlap = min(duration * overlap_ratio, 
                            (segment_times[-1][1] - segment_times[-1][0]) * overlap_ratio)
            overlap = random.uniform(0, max_overlap)
            start_time = segment_times[-1][1] - overlap
        else:
            start_time = 0.0
            
        end_time = start_time + duration
        segment_times.append((start_time, end_time))
        current_time = max(current_time, end_time)
        
        logger.debug(f"Segment {i} ({segment.speaker}): {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
    
    # Create output arrays
    total_samples = int(current_time * sample_rate)
    mixed_audio = np.zeros(total_samples, dtype=np.float32)
    speaker1_audio = np.zeros(total_samples, dtype=np.float32)
    speaker2_audio = np.zeros(total_samples, dtype=np.float32)
    
    logger.debug(f"Total duration: {current_time:.2f}s ({total_samples} samples)")
    
    # Mix segments
    for segment, (start_time, end_time) in zip(segments, segment_times):
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(segment.audio)
        
        # Ensure we don't exceed array bounds
        if end_sample > total_samples:
            segment.audio = segment.audio[:total_samples - start_sample]
            end_sample = total_samples
        
        # Apply volume scaling for mixed audio only
        volume_scale = speaker_volumes[segment.speaker]
        scaled_audio = segment.audio * volume_scale
        
        # Add scaled audio to mixed track, original audio to individual tracks
        mixed_audio[start_sample:end_sample] += scaled_audio
        
        if segment.speaker == "Speaker1":
            speaker1_audio[start_sample:end_sample] = segment.audio  # Original volume
        else:
            speaker2_audio[start_sample:end_sample] = segment.audio  # Original volume
    
    # Normalize mixed audio to prevent clipping
    max_val = np.abs(mixed_audio).max()
    if max_val > 0:
        mixed_audio = mixed_audio / max_val * 0.95
        logger.debug(f"Normalized mixed audio by factor {1.0 / max_val * 0.95:.3f}")
    
    return mixed_audio, speaker1_audio, speaker2_audio


def generate_conversation_dataset(output_dir: str, num_conversations: int,
                                sample_rate: int = 8000, language: str = "en", 
                                volume_variation: float = 0.5,
                                target_duration_sec: int = 30) -> None:
    """Generate a complete conversation dataset for ClearerVoice training.
    
    Args:
        output_dir: Directory to save generated data
        num_conversations: Number of conversations to generate
        sample_rate: Target sample rate in Hz
        language: Language code ('en' for English, 'ja' for Japanese)
        volume_variation: Maximum volume variation factor (0.0-1.0)
        target_duration_sec: Target duration for each conversation in seconds
    """
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("Starting conversation dataset generation for ClearerVoice")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of conversations: {num_conversations}")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Language: {language}")
    logger.info(f"Volume variation: {volume_variation}")
    logger.info(f"Target duration: {target_duration_sec} seconds per conversation")
    
    # Initialize API clients
    try:
        tts_client = texttospeech.TextToSpeechClient()
        logger.info("Successfully initialized Text-to-Speech client")
    except Exception as e:
        logger.error(f"Failed to initialize TTS client: {e}")
        raise
    
    # Create noise augmentation presets
    noise_presets = create_noise_presets(sample_rate)
    logger.info(f"Created {len(noise_presets)} noise augmentation presets")
    
    # Create output directories with absolute paths
    output_path = Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    mix_dir = output_path / "mix"
    s1_dir = output_path / "s1"
    s2_dir = output_path / "s2"
    
    mix_dir.mkdir(exist_ok=True)
    s1_dir.mkdir(exist_ok=True)
    s2_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created output directories: {mix_dir}, {s1_dir}, {s2_dir}")
    
    # Lists to store file paths for SCP files
    all_data = []
    
    # Generate conversations
    successful_generations = 0
    failed_generations = 0
    
    for i in tqdm(range(num_conversations), desc=f"Generating {language} conversations"):
        try:
            logger.info(f"Generating conversation {i+1}/{num_conversations}")
            
            # Generate conversation text using structured output
            conversation = generate_conversation(language=language, 
                                               target_duration_sec=target_duration_sec,
                                               logger=logger)
            
            # Synthesize speech for each turn
            segments = []
            for j, turn in enumerate(conversation.turns):
                logger.debug(f"Synthesizing turn {j+1}/{len(conversation.turns)} for {turn.speaker}")
                
                # Synthesize at 24kHz (TTS default), then resample if needed
                audio_data = synthesize_speech(tts_client, turn.text, 
                                             turn.speaker, turn.emotion, language, logger)
                audio = pcm_to_numpy(audio_data, 24000)
                
                # Resample if needed
                if sample_rate != 24000:
                    logger.debug(f"Resampling from 24kHz to {sample_rate}Hz")
                    # Simple resampling - for production use scipy.signal.resample
                    resample_ratio = sample_rate / 24000
                    indices = np.arange(0, len(audio), 1/resample_ratio).astype(int)
                    indices = indices[indices < len(audio)]
                    audio = audio[indices]
                
                segments.append(AudioSegment(
                    audio=audio,
                    start_time=0,  # Will be set by mixer
                    end_time=0,    # Will be set by mixer
                    speaker=turn.speaker,
                    text=turn.text
                ))
            
            logger.debug(f"Synthesized {len(segments)} audio segments")
            
            # Mix with overlaps and random volume variation
            mixed, speaker1, speaker2 = mix_conversation_with_overlap(
                segments, overlap_ratio=0.3, sample_rate=sample_rate, 
                volume_variation=volume_variation, logger=logger
            )
            
            # Apply random noise augmentation to mixed audio only
            # Keep individual speaker tracks clean for training
            selected_preset = random.choice(noise_presets)
            preset_name = ["light", "medium", "heavy", "variable", "clean"][noise_presets.index(selected_preset)]
            logger.debug(f"Applying {preset_name} noise augmentation")
            
            augmented_mixed = selected_preset(samples=mixed, sample_rate=sample_rate)
            
            # Save audio files with absolute paths
            conversation_id = f"conversation_{i:04d}"
            mix_file = (mix_dir / f"{conversation_id}_mix.wav").absolute()
            s1_file = (s1_dir / f"{conversation_id}_s1.wav").absolute()
            s2_file = (s2_dir / f"{conversation_id}_s2.wav").absolute()
            
            # Save augmented mixed audio and clean individual speaker tracks
            sf.write(str(mix_file), augmented_mixed, sample_rate)
            sf.write(str(s1_file), speaker1, sample_rate)
            sf.write(str(s2_file), speaker2, sample_rate)
            
            duration = len(augmented_mixed) / sample_rate
            logger.info(f"Saved conversation {i:04d} (duration: {duration:.2f}s, topic: {conversation.topic})")
            
            # Verify files exist
            if not mix_file.exists() or not s1_file.exists() or not s2_file.exists():
                logger.error(f"Failed to save files for conversation {i:04d}")
                failed_generations += 1
                continue
            
            # Add to data list for SCP files with absolute paths
            all_data.append({
                "mix_path": str(mix_file),
                "s1_path": str(s1_file),
                "s2_path": str(s2_file),
                "duration": duration,
                "topic": conversation.topic,
                "language": language
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
    random.shuffle(all_data)  # Shuffle for random split
    n_train = int(0.8 * len(all_data))
    n_valid = int(0.1 * len(all_data))
    
    train_data = all_data[:n_train]
    valid_data = all_data[n_train:n_train+n_valid]
    test_data = all_data[n_train+n_valid:]
    
    # Write SCP files with absolute paths
    def write_scp(data: List[Dict], filename: str):
        scp_path = output_path / filename
        with open(scp_path, 'w') as f:
            for item in data:
                # Format: mixture_path speaker1_path speaker2_path
                f.write(f"{item['mix_path']} {item['s1_path']} {item['s2_path']}\n")
        logger.info(f"Wrote {len(data)} entries to {scp_path}")
        
        # Verify the file
        if scp_path.exists():
            with open(scp_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    logger.info(f"First line of {filename}: {first_line}")
    
    write_scp(train_data, "train.scp")
    write_scp(valid_data, "valid.scp")
    write_scp(test_data, "test.scp")
    
    logger.info(f"Created SCP files - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Save metadata as JSON for reference
    metadata = {
        "dataset_info": {
            "total_conversations": successful_generations,
            "sample_rate": sample_rate,
            "language": language,
            "target_duration_sec": target_duration_sec,
            "volume_variation": volume_variation
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
    parser = argparse.ArgumentParser(description="Generate conversation dataset for ClearerVoice")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for generated data")
    parser.add_argument("--num_conversations", type=int, default=100,
                      help="Number of conversations to generate")
    parser.add_argument("--sample_rate", type=int, default=8000,
                      help="Target sample rate (Hz)")
    parser.add_argument("--language", type=str, default="ja",
                      choices=["en", "ja"],
                      help="Language for conversations ('en' for English, 'ja' for Japanese)")
    parser.add_argument("--volume_variation", type=float, default=0.5,
                      help="Maximum volume variation factor (0.0-1.0, default: 0.5)")
    parser.add_argument("--target_duration", type=int, default=30,
                      help="Target duration for each conversation in seconds (default: 30)")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Set root logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    generate_conversation_dataset(
        args.output_dir,
        args.num_conversations,
        args.sample_rate,
        args.language,
        args.volume_variation,
        args.target_duration
    )


if __name__ == "__main__":
    main()