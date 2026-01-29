#!/usr/bin/env python3
"""
Prepare NeMo manifest from local audio files and transcripts.

This script converts CSV/TSV transcript files into NeMo-compatible JSONL manifests.
It supports automatic duration detection, audio resampling, and train/val splitting.

Usage:
    python scripts/prepare_local_data.py \
        --input /path/to/transcripts.csv \
        --audio_dir /path/to/audio \
        --output_dir ./data \
        --val_split 0.1
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare NeMo manifest from local audio files and transcripts"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV/TSV file with audio paths and transcripts"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Base directory for audio files (if paths in CSV are relative)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output manifests"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds (default: 20.0)"
    )
    parser.add_argument(
        "--normalize_text",
        action="store_true",
        help="Apply text normalization (lowercase, remove punctuation)"
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default=None,
        help="Column name for audio path (auto-detected if not specified)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Column name for transcript (auto-detected if not specified)"
    )
    parser.add_argument(
        "--duration_column",
        type=str,
        default=None,
        help="Column name for duration (calculated from audio if not specified)"
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample audio to 16kHz mono (requires librosa)"
    )
    parser.add_argument(
        "--resample_output_dir",
        type=str,
        default=None,
        help="Directory to save resampled audio (default: {output_dir}/resampled)"
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=None,
        help="CSV/TSV delimiter (auto-detected if not specified)"
    )
    parser.add_argument(
        "--language_tag",
        type=str,
        default=None,
        help="Language tag to append to transcripts (e.g., 'hi-IN', 'en-US'). "
             "Parakeet multilingual outputs format: 'text <lang-tag>'"
    )
    return parser.parse_args()


def detect_delimiter(filepath: str) -> str:
    """Auto-detect CSV/TSV delimiter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    if '\t' in first_line:
        return '\t'
    elif ',' in first_line:
        return ','
    elif '|' in first_line:
        return '|'
    else:
        return ','


def detect_column(df: pd.DataFrame, candidates: list, column_type: str) -> str:
    """Auto-detect column name from candidates."""
    columns_lower = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    raise ValueError(
        f"Could not auto-detect {column_type} column. "
        f"Available columns: {list(df.columns)}. "
        f"Please specify using --{column_type.replace(' ', '_')}_column"
    )


def normalize_text(text: str) -> str:
    """Apply text normalization."""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation (keep spaces and alphanumeric)
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"Warning: Could not read {audio_path}: {e}")
        return None


def resample_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000
) -> Tuple[bool, Optional[float]]:
    """Resample audio to target sample rate and mono."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for resampling. Install with: pip install librosa")

    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save resampled audio
        sf.write(output_path, y, target_sr)

        duration = len(y) / target_sr
        return True, duration
    except Exception as e:
        print(f"Warning: Could not resample {input_path}: {e}")
        return False, None


def validate_audio(audio_path: str, min_duration: float, max_duration: float) -> Tuple[bool, Optional[float], str]:
    """Validate audio file and return (is_valid, duration, error_message)."""
    if not os.path.exists(audio_path):
        return False, None, f"File not found: {audio_path}"

    duration = get_audio_duration(audio_path)
    if duration is None:
        return False, None, f"Could not read audio: {audio_path}"

    if duration < min_duration:
        return False, duration, f"Duration {duration:.2f}s < min {min_duration}s"

    if duration > max_duration:
        return False, duration, f"Duration {duration:.2f}s > max {max_duration}s"

    return True, duration, ""


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect delimiter
    delimiter = args.delimiter or detect_delimiter(args.input)
    print(f"Using delimiter: {repr(delimiter)}")

    # Load transcript file
    print(f"Loading transcript file: {args.input}")
    df = pd.read_csv(args.input, delimiter=delimiter, dtype=str)
    df = df.fillna('')
    print(f"Loaded {len(df)} entries")

    # Auto-detect columns
    audio_column = args.audio_column or detect_column(
        df,
        ['audio_filepath', 'audio_path', 'audio', 'path', 'file', 'filename', 'wav'],
        'audio'
    )
    text_column = args.text_column or detect_column(
        df,
        ['text', 'transcript', 'sentence', 'transcription', 'label'],
        'text'
    )
    duration_column = args.duration_column
    if duration_column and duration_column not in df.columns:
        print(f"Warning: Duration column '{duration_column}' not found, will calculate from audio")
        duration_column = None

    print(f"Audio column: {audio_column}")
    print(f"Text column: {text_column}")
    print(f"Duration column: {duration_column or '(calculated from audio)'}")
    if args.language_tag:
        print(f"Language tag: <{args.language_tag}> (will be appended to all transcripts)")

    # Setup resampling
    resample_dir = None
    if args.resample:
        if not LIBROSA_AVAILABLE:
            print("Error: librosa is required for resampling. Install with: pip install librosa")
            sys.exit(1)
        resample_dir = args.resample_output_dir or os.path.join(args.output_dir, "resampled")
        os.makedirs(resample_dir, exist_ok=True)
        print(f"Resampling audio to: {resample_dir}")

    # Process entries
    valid_entries = []
    skipped = {"not_found": 0, "read_error": 0, "too_short": 0, "too_long": 0, "empty_text": 0}

    print("\nProcessing audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[audio_column]
        text = row[text_column].strip()

        # Skip empty transcripts
        if not text:
            skipped["empty_text"] += 1
            continue

        # Resolve audio path
        if args.audio_dir and not os.path.isabs(audio_path):
            audio_path = os.path.join(args.audio_dir, audio_path)

        audio_path = os.path.abspath(audio_path)

        # Resample if requested
        if args.resample:
            # Create output path preserving relative structure
            if args.audio_dir:
                rel_path = os.path.relpath(audio_path, args.audio_dir)
            else:
                rel_path = os.path.basename(audio_path)

            # Ensure .wav extension
            rel_path = os.path.splitext(rel_path)[0] + '.wav'
            resampled_path = os.path.join(resample_dir, rel_path)

            success, duration = resample_audio(audio_path, resampled_path)
            if not success:
                skipped["read_error"] += 1
                continue

            audio_path = resampled_path
        else:
            # Validate and get duration
            is_valid, duration, error = validate_audio(
                audio_path, args.min_duration, args.max_duration
            )

            if not is_valid:
                if "not found" in error.lower():
                    skipped["not_found"] += 1
                elif "could not read" in error.lower():
                    skipped["read_error"] += 1
                elif "< min" in error:
                    skipped["too_short"] += 1
                elif "> max" in error:
                    skipped["too_long"] += 1
                continue

        # Use provided duration if available
        if duration_column and row[duration_column]:
            try:
                duration = float(row[duration_column])
            except ValueError:
                pass  # Use calculated duration

        # Check duration bounds
        if duration < args.min_duration:
            skipped["too_short"] += 1
            continue
        if duration > args.max_duration:
            skipped["too_long"] += 1
            continue

        # Normalize text if requested
        if args.normalize_text:
            text = normalize_text(text)
            if not text:
                skipped["empty_text"] += 1
                continue

        # Append language tag if specified (Parakeet multilingual format)
        if args.language_tag:
            text = f"{text} <{args.language_tag}>"

        valid_entries.append({
            "audio_filepath": audio_path,
            "text": text,
            "duration": round(duration, 3)
        })

    print(f"\nValid entries: {len(valid_entries)}")
    print(f"Skipped entries:")
    for reason, count in skipped.items():
        if count > 0:
            print(f"  - {reason}: {count}")

    if len(valid_entries) == 0:
        print("Error: No valid entries found!")
        sys.exit(1)

    # Shuffle and split
    import random
    random.seed(args.seed)
    random.shuffle(valid_entries)

    val_size = int(len(valid_entries) * args.val_split)
    train_entries = valid_entries[val_size:]
    val_entries = valid_entries[:val_size]

    print(f"\nTrain entries: {len(train_entries)}")
    print(f"Validation entries: {len(val_entries)}")

    # Calculate statistics
    train_duration = sum(e["duration"] for e in train_entries)
    val_duration = sum(e["duration"] for e in val_entries)
    print(f"\nTotal training duration: {train_duration/3600:.2f} hours")
    print(f"Total validation duration: {val_duration/3600:.2f} hours")

    # Write manifests
    train_manifest_path = os.path.join(args.output_dir, "train_manifest.json")
    val_manifest_path = os.path.join(args.output_dir, "val_manifest.json")

    with open(train_manifest_path, 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(val_manifest_path, 'w', encoding='utf-8') as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nManifests saved to:")
    print(f"  Train: {train_manifest_path}")
    print(f"  Val: {val_manifest_path}")

    # Print example usage
    print(f"\nExample finetune command:")
    print(f"python finetune.py \\")
    print(f"  --config configs/finetune_local.yaml \\")
    print(f"  --model /path/to/parakeet-rnnt-1.1b-multilingual.nemo \\")
    print(f"  --output_dir ./experiments/my_finetune \\")
    print(f"  local_data_cfg.train_manifest={train_manifest_path} \\")
    print(f"  local_data_cfg.val_manifest={val_manifest_path}")


if __name__ == "__main__":
    main()
