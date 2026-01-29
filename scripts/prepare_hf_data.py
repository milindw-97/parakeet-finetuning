#!/usr/bin/env python3
"""
Prepare tokenizer and manifests from HuggingFace dataset.

This script can:
1. Generate NeMo manifests from HuggingFace datasets
2. Train a custom SentencePiece BPE tokenizer (for new languages or specialized vocabulary)
3. Extract vocabulary for analysis

Usage:
    # Generate manifests only (using pretrained tokenizer)
    python scripts/prepare_hf_data.py \
        --dataset mozilla-foundation/common_voice_16_1 \
        --subset hi \
        --output_dir ./data/hindi

    # Train custom tokenizer (recommended for 50+ hours of new language data)
    python scripts/prepare_hf_data.py \
        --dataset mozilla-foundation/common_voice_16_1 \
        --subset hi \
        --output_dir ./data/hindi \
        --train_tokenizer \
        --vocab_size 1024
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare tokenizer and manifests from HuggingFace dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'mozilla-foundation/common_voice_16_1')"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/configuration (e.g., 'hi' for Hindi)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of training split (default: 'train')"
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="validation",
        help="Name of validation split (default: 'validation')"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=None,
        help="Name of test split (optional)"
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Column containing audio data (default: 'audio')"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="sentence",
        help="Column containing transcription (default: 'sentence')"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process per split (for testing)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for dataset loading"
    )

    # Tokenizer training options
    parser.add_argument(
        "--train_tokenizer",
        action="store_true",
        help="Train a custom SentencePiece BPE tokenizer"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1024,
        help="Vocabulary size for tokenizer (default: 1024)"
    )
    parser.add_argument(
        "--tokenizer_coverage",
        type=float,
        default=0.9999,
        help="Character coverage for tokenizer (default: 0.9999)"
    )

    # Manifest generation options
    parser.add_argument(
        "--generate_manifests",
        action="store_true",
        default=True,
        help="Generate NeMo manifests (default: True)"
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save audio files to disk (required for manifest generation)"
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        default="wav",
        choices=["wav", "flac"],
        help="Audio format for saved files (default: wav)"
    )
    parser.add_argument(
        "--language_tag",
        type=str,
        default=None,
        help="Language tag to append to transcripts (e.g., 'hi-IN', 'en-US'). "
             "Parakeet multilingual outputs format: 'text <lang-tag>'"
    )

    return parser.parse_args()


def load_dataset_split(dataset_name, subset, split, cache_dir, trust_remote_code, max_samples=None):
    """Load a dataset split from HuggingFace."""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            dataset_name,
            subset,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds
    except Exception as e:
        print(f"Warning: Could not load split '{split}': {e}")
        return None


def extract_text_from_dataset(dataset, text_column: str):
    """Extract all text from dataset for tokenizer training."""
    texts = []
    for example in tqdm(dataset, desc="Extracting text"):
        text = example.get(text_column, "")
        if text and isinstance(text, str):
            texts.append(text.strip())
    return texts


def train_sentencepiece_tokenizer(
    texts: list,
    output_dir: str,
    vocab_size: int = 1024,
    coverage: float = 0.9999,
    model_type: str = "bpe"
):
    """Train a SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Error: sentencepiece is required for tokenizer training.")
        print("Install with: pip install sentencepiece")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Write texts to temporary file
    text_file = os.path.join(output_dir, "train_text.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    # Train tokenizer
    model_prefix = os.path.join(output_dir, "tokenizer")

    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<blank>"]
    )

    print(f"Tokenizer saved to: {model_prefix}.model")
    print(f"Vocabulary saved to: {model_prefix}.vocab")

    # Clean up text file
    os.remove(text_file)

    return model_prefix + ".model"


def save_audio_and_create_manifest(
    dataset,
    output_dir: str,
    split_name: str,
    audio_column: str,
    text_column: str,
    audio_format: str = "wav",
    min_duration: float = 0.1,
    max_duration: float = 20.0,
    language_tag: Optional[str] = None
):
    """Save audio files and create NeMo manifest."""
    import soundfile as sf
    import numpy as np

    audio_dir = os.path.join(output_dir, "audio", split_name)
    os.makedirs(audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, f"{split_name}_manifest.json")
    manifest_entries = []

    skipped = {"no_audio": 0, "no_text": 0, "too_short": 0, "too_long": 0, "error": 0}

    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        try:
            # Get audio data
            audio_data = example.get(audio_column)
            if audio_data is None:
                skipped["no_audio"] += 1
                continue

            # Handle different audio formats from HuggingFace
            if isinstance(audio_data, dict):
                array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                skipped["no_audio"] += 1
                continue

            if array is None:
                skipped["no_audio"] += 1
                continue

            # Get text
            text = example.get(text_column, "")
            if not text or not isinstance(text, str):
                skipped["no_text"] += 1
                continue
            text = text.strip()
            if not text:
                skipped["no_text"] += 1
                continue

            # Convert to numpy array if needed
            if not isinstance(array, np.ndarray):
                array = np.array(array)

            # Calculate duration
            duration = len(array) / sample_rate

            # Filter by duration
            if duration < min_duration:
                skipped["too_short"] += 1
                continue
            if duration > max_duration:
                skipped["too_long"] += 1
                continue

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    import librosa
                    array = librosa.resample(array, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                    duration = len(array) / sample_rate
                except ImportError:
                    # Use scipy if librosa not available
                    from scipy import signal
                    num_samples = int(len(array) * 16000 / sample_rate)
                    array = signal.resample(array, num_samples)
                    sample_rate = 16000
                    duration = len(array) / sample_rate

            # Save audio file
            audio_filename = f"{split_name}_{idx:08d}.{audio_format}"
            audio_path = os.path.join(audio_dir, audio_filename)
            sf.write(audio_path, array, sample_rate)

            # Append language tag if specified (Parakeet multilingual format)
            if language_tag:
                text = f"{text} <{language_tag}>"

            # Add to manifest
            manifest_entries.append({
                "audio_filepath": os.path.abspath(audio_path),
                "text": text,
                "duration": round(duration, 3)
            })

        except Exception as e:
            print(f"Warning: Error processing example {idx}: {e}")
            skipped["error"] += 1
            continue

    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n{split_name} split:")
    print(f"  Valid entries: {len(manifest_entries)}")
    print(f"  Skipped: {sum(skipped.values())}")
    for reason, count in skipped.items():
        if count > 0:
            print(f"    - {reason}: {count}")

    total_duration = sum(e["duration"] for e in manifest_entries)
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    print(f"  Manifest: {manifest_path}")

    return manifest_path, len(manifest_entries)


def main():
    args = parse_args()

    # Import datasets
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library is required. Install with: pip install datasets")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset splits
    print(f"Loading dataset: {args.dataset}")
    if args.subset:
        print(f"Subset: {args.subset}")

    train_ds = load_dataset_split(
        args.dataset, args.subset, args.train_split,
        args.cache_dir, args.trust_remote_code, args.max_samples
    )

    val_ds = load_dataset_split(
        args.dataset, args.subset, args.val_split,
        args.cache_dir, args.trust_remote_code, args.max_samples
    )

    test_ds = None
    if args.test_split:
        test_ds = load_dataset_split(
            args.dataset, args.subset, args.test_split,
            args.cache_dir, args.trust_remote_code, args.max_samples
        )

    if train_ds is None:
        print("Error: Could not load training data")
        sys.exit(1)

    print(f"\nLoaded splits:")
    print(f"  Train: {len(train_ds)} examples")
    if val_ds:
        print(f"  Validation: {len(val_ds)} examples")
    if test_ds:
        print(f"  Test: {len(test_ds)} examples")

    # Train tokenizer if requested
    if args.train_tokenizer:
        print("\n" + "="*50)
        print("Training tokenizer...")
        print("="*50)

        # Extract text from training data
        all_texts = extract_text_from_dataset(train_ds, args.text_column)

        if len(all_texts) == 0:
            print("Error: No text found in training data")
            sys.exit(1)

        print(f"Extracted {len(all_texts)} text samples")

        # Train tokenizer
        tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
        tokenizer_path = train_sentencepiece_tokenizer(
            all_texts,
            tokenizer_dir,
            vocab_size=args.vocab_size,
            coverage=args.tokenizer_coverage
        )

        print(f"\nTo use custom tokenizer, add to your config:")
        print(f"  model.tokenizer.update_tokenizer: true")
        print(f"  model.tokenizer.dir: {tokenizer_dir}")

    # Generate manifests if requested
    if args.generate_manifests and args.save_audio:
        print("\n" + "="*50)
        print("Generating manifests...")
        print("="*50)

        manifests = {}

        if args.language_tag:
            print(f"Language tag: <{args.language_tag}> (will be appended to all transcripts)")

        # Process training data
        train_manifest, train_count = save_audio_and_create_manifest(
            train_ds, args.output_dir, "train",
            args.audio_column, args.text_column,
            args.audio_format, args.min_duration, args.max_duration,
            args.language_tag
        )
        manifests["train"] = train_manifest

        # Process validation data
        if val_ds:
            val_manifest, val_count = save_audio_and_create_manifest(
                val_ds, args.output_dir, "val",
                args.audio_column, args.text_column,
                args.audio_format, args.min_duration, args.max_duration,
                args.language_tag
            )
            manifests["val"] = val_manifest

        # Process test data
        if test_ds:
            test_manifest, test_count = save_audio_and_create_manifest(
                test_ds, args.output_dir, "test",
                args.audio_column, args.text_column,
                args.audio_format, args.min_duration, args.max_duration,
                args.language_tag
            )
            manifests["test"] = test_manifest

        print("\n" + "="*50)
        print("Manifest generation complete!")
        print("="*50)

        print("\nExample finetune command:")
        print(f"python finetune.py \\")
        print(f"  --config configs/finetune_local.yaml \\")
        print(f"  --model /path/to/parakeet-rnnt-1.1b-multilingual.nemo \\")
        print(f"  --output_dir ./experiments/my_finetune \\")
        print(f"  local_data_cfg.train_manifest={manifests['train']} \\")
        if "val" in manifests:
            print(f"  local_data_cfg.val_manifest={manifests['val']}")

    elif args.generate_manifests and not args.save_audio:
        print("\nNote: To generate manifests, use --save_audio flag to save audio files to disk.")
        print("Without this, the finetuning script will handle HuggingFace data directly.")


if __name__ == "__main__":
    main()
