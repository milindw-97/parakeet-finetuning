#!/usr/bin/env python3
"""
Inference script for finetuned Parakeet RNNT model.

This script supports:
- Transcribing single audio files
- Transcribing a directory of audio files
- Evaluating WER on a test manifest
- Batch inference for efficiency

Usage:
    # Transcribe single file
    python inference.py \
        --model ./experiments/finetune/final_model.nemo \
        --audio /path/to/audio.wav

    # Transcribe directory
    python inference.py \
        --model ./experiments/finetune/final_model.nemo \
        --audio_dir /path/to/audio_folder \
        --output results.json

    # Evaluate on test manifest
    python inference.py \
        --model ./experiments/finetune/final_model.nemo \
        --manifest /path/to/test_manifest.json \
        --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with finetuned Parakeet RNNT model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to finetuned .nemo model"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio",
        type=str,
        help="Path to single audio file"
    )
    input_group.add_argument(
        "--audio_dir",
        type=str,
        help="Path to directory of audio files"
    )
    input_group.add_argument(
        "--manifest",
        type=str,
        help="Path to NeMo manifest for evaluation"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--audio_extensions",
        type=str,
        nargs="+",
        default=[".wav", ".flac", ".mp3", ".ogg"],
        help="Audio file extensions to process (default: .wav .flac .mp3 .ogg)"
    )

    return parser.parse_args()


def load_model(model_path: str, device: str):
    """Load NeMo ASR model."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    print(f"Loading model from: {model_path}")
    model = EncDecRNNTBPEModel.restore_from(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    return model


def transcribe_files(model, audio_files: List[str], batch_size: int) -> List[str]:
    """Transcribe a list of audio files."""
    transcriptions = model.transcribe(
        audio_files,
        batch_size=batch_size,
        return_hypotheses=False
    )

    # Handle different return formats
    if isinstance(transcriptions, tuple):
        transcriptions = transcriptions[0]

    return transcriptions


def calculate_wer(references: List[str], hypotheses: List[str]) -> dict:
    """Calculate Word Error Rate and related metrics."""
    try:
        from jiwer import wer, mer, wil, wip, compute_measures
    except ImportError:
        print("Warning: jiwer not installed. Install with: pip install jiwer")
        return {"wer": None, "error": "jiwer not installed"}

    # Filter out empty references
    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]

    if not valid_pairs:
        return {"wer": None, "error": "No valid reference texts"}

    refs, hyps = zip(*valid_pairs)
    refs = list(refs)
    hyps = list(hyps)

    measures = compute_measures(refs, hyps)

    return {
        "wer": measures["wer"],
        "mer": measures["mer"],  # Match Error Rate
        "wil": measures["wil"],  # Word Information Lost
        "wip": measures["wip"],  # Word Information Preserved
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "num_reference_words": sum(len(r.split()) for r in refs),
        "num_samples": len(refs)
    }


def transcribe_single_file(model, audio_path: str) -> str:
    """Transcribe a single audio file."""
    transcriptions = transcribe_files(model, [audio_path], batch_size=1)
    return transcriptions[0] if transcriptions else ""


def transcribe_directory(model, audio_dir: str, extensions: List[str], batch_size: int) -> dict:
    """Transcribe all audio files in a directory."""
    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(audio_dir).glob(f"**/*{ext}"))
        audio_files.extend(Path(audio_dir).glob(f"**/*{ext.upper()}"))

    audio_files = sorted([str(f) for f in audio_files])

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return {}

    print(f"Found {len(audio_files)} audio files")

    # Transcribe in batches
    transcriptions = transcribe_files(model, audio_files, batch_size)

    # Create results
    results = {}
    for audio_path, transcription in zip(audio_files, transcriptions):
        results[audio_path] = transcription
        print(f"{os.path.basename(audio_path)}: {transcription}")

    return results


def evaluate_manifest(model, manifest_path: str, batch_size: int) -> dict:
    """Evaluate model on a test manifest and calculate WER."""
    # Load manifest
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            entries.append(entry)

    if not entries:
        print(f"No entries found in manifest: {manifest_path}")
        return {}

    print(f"Evaluating on {len(entries)} samples...")

    # Extract audio files and references
    audio_files = [e["audio_filepath"] for e in entries]
    references = [e["text"] for e in entries]

    # Transcribe
    hypotheses = transcribe_files(model, audio_files, batch_size)

    # Calculate WER
    metrics = calculate_wer(references, hypotheses)

    # Create detailed results
    results = {
        "metrics": metrics,
        "samples": []
    }

    for entry, ref, hyp in zip(entries, references, hypotheses):
        results["samples"].append({
            "audio_filepath": entry["audio_filepath"],
            "reference": ref,
            "hypothesis": hyp,
            "duration": entry.get("duration")
        })

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    if metrics.get("wer") is not None:
        print(f"WER: {metrics['wer']*100:.2f}%")
        print(f"MER: {metrics['mer']*100:.2f}%")
        print(f"WIL: {metrics['wil']*100:.2f}%")
        print(f"Substitutions: {metrics['substitutions']}")
        print(f"Deletions: {metrics['deletions']}")
        print(f"Insertions: {metrics['insertions']}")
        print(f"Total reference words: {metrics['num_reference_words']}")
        print(f"Total samples: {metrics['num_samples']}")
    else:
        print(f"Could not calculate WER: {metrics.get('error')}")

    print("="*60)

    # Print some examples
    print("\nSample predictions:")
    for i, sample in enumerate(results["samples"][:5]):
        print(f"\n[{i+1}] {os.path.basename(sample['audio_filepath'])}")
        print(f"  REF: {sample['reference']}")
        print(f"  HYP: {sample['hypothesis']}")

    return results


def main():
    args = parse_args()

    # Load model
    model = load_model(args.model, args.device)

    results = {}

    # Process based on input type
    if args.audio:
        # Single file
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}")
            sys.exit(1)

        transcription = transcribe_single_file(model, args.audio)
        print(f"\nTranscription: {transcription}")

        results = {
            "audio_filepath": args.audio,
            "transcription": transcription
        }

    elif args.audio_dir:
        # Directory of files
        if not os.path.isdir(args.audio_dir):
            print(f"Error: Directory not found: {args.audio_dir}")
            sys.exit(1)

        results = transcribe_directory(
            model, args.audio_dir, args.audio_extensions, args.batch_size
        )

    elif args.manifest:
        # Evaluate on manifest
        if not os.path.exists(args.manifest):
            print(f"Error: Manifest not found: {args.manifest}")
            sys.exit(1)

        results = evaluate_manifest(model, args.manifest, args.batch_size)

    # Save results if output specified
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
