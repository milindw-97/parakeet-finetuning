#!/usr/bin/env python3
"""
Finetune Parakeet RNNT 1.1B model using NeMo framework.

This script supports:
- HuggingFace datasets (streaming or downloaded)
- Local audio files with NeMo manifests
- Continuing training from previously finetuned models
- Multi-GPU training via DDP

Usage:
    # Finetune with HuggingFace dataset
    python finetune.py \
        --config configs/finetune_huggingface.yaml \
        --model /path/to/parakeet-rnnt-1.1b-multilingual.nemo \
        --output_dir ./experiments/hindi_finetune \
        hf_data_cfg.path=mozilla-foundation/common_voice_16_1 \
        hf_data_cfg.name=hi

    # Finetune with local data
    python finetune.py \
        --config configs/finetune_local.yaml \
        --model /path/to/parakeet-rnnt-1.1b-multilingual.nemo \
        --output_dir ./experiments/local_finetune \
        local_data_cfg.train_manifest=./data/train_manifest.json \
        local_data_cfg.val_manifest=./data/val_manifest.json

    # Continue training from checkpoint
    python finetune.py \
        --config configs/finetune_local.yaml \
        --model ./experiments/previous/checkpoints/best_model.nemo \
        --output_dir ./experiments/continued_finetune
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Parakeet RNNT model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base .nemo model or previously finetuned model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (optional)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config without training"
    )
    parser.add_argument(
        "--debug_config",
        action="store_true",
        help="Print detailed config loading debug info"
    )
    parser.add_argument(
        "--skip_data_prep",
        action="store_true",
        help="Skip data preparation (use existing manifests from previous run)"
    )

    # Allow arbitrary config overrides via Hydra-style arguments
    args, overrides = parser.parse_known_args()
    return args, overrides


def apply_config_overrides(cfg: DictConfig, overrides: list, verbose: bool = False) -> DictConfig:
    """Apply command-line config overrides in Hydra style."""
    if verbose and overrides:
        print(f"\nApplying {len(overrides)} config overrides:")

    for override in overrides:
        if '=' not in override:
            print(f"Warning: Ignoring invalid override (missing '='): {override}")
            continue

        key, value = override.split('=', 1)
        original_value = value

        # Handle nested keys (e.g., hf_data_cfg.path)
        keys = key.split('.')

        # Convert value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.lower() == 'null' or value.lower() == 'none':
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

        if verbose:
            print(f"  {key} = {value!r} (from '{original_value}')")

        # Navigate to the correct config node and set value
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    return cfg


def prepare_hf_manifests(cfg: DictConfig, output_dir: str):
    """Prepare NeMo manifests from HuggingFace dataset."""
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np
    import json
    from tqdm import tqdm

    hf_cfg = cfg.hf_data_cfg

    print(f"Loading HuggingFace dataset: {hf_cfg.path}")
    if hf_cfg.get('name'):
        print(f"Subset: {hf_cfg.name}")
    if hf_cfg.get('language_tag'):
        print(f"Language tag: <{hf_cfg.language_tag}> (will be appended to all transcripts)")

    # Load dataset with error handling for script-based datasets
    load_kwargs = {
        "path": hf_cfg.path,
        "name": hf_cfg.get('name'),
        "streaming": hf_cfg.get('streaming', False),
        "cache_dir": hf_cfg.get('cache_dir'),
        "trust_remote_code": hf_cfg.get('trust_remote_code', True),  # Default to True
    }

    try:
        train_ds = load_dataset(**load_kwargs, split=hf_cfg.train_split)
        val_ds = load_dataset(**load_kwargs, split=hf_cfg.val_split)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print("\nError: This dataset uses a custom script which is not supported in datasets>=3.0")
            print("Solutions:")
            print("  1. Downgrade: pip install 'datasets<3.0'")
            print("  2. Use a dataset without custom scripts")
            print("  3. Download the data manually and use local data config instead")
            sys.exit(1)
        raise

    # Apply sample limits if specified
    max_train = hf_cfg.get('max_train_samples')
    max_val = hf_cfg.get('max_val_samples')

    if max_train and not hf_cfg.get('streaming', False):
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_val and not hf_cfg.get('streaming', False):
        val_ds = val_ds.select(range(min(max_val, len(val_ds))))

    # Prepare output directories
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(os.path.join(audio_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "val"), exist_ok=True)

    def process_split(dataset, split_name, max_samples=None):
        """Process a dataset split and create manifest."""
        manifest_path = os.path.join(output_dir, f"{split_name}_manifest.json")
        entries = []

        audio_col = hf_cfg.audio_column
        text_col = hf_cfg.text_column
        min_dur = cfg.model.train_ds.get('min_duration', 0.1)
        max_dur = cfg.model.train_ds.get('max_duration', 20.0)
        lang_tag = hf_cfg.get('language_tag')

        # Track skip reasons for debugging
        skip_reasons = {
            "no_audio_col": 0,
            "audio_not_dict": 0,
            "no_array": 0,
            "no_text": 0,
            "empty_text": 0,
            "too_short": 0,
            "too_long": 0,
            "error": 0,
        }

        # Debug: show first example structure
        first_example = next(iter(dataset))
        print(f"\n[DEBUG] Dataset columns: {list(first_example.keys())}")
        print(f"[DEBUG] Looking for audio_col='{audio_col}', text_col='{text_col}'")
        if audio_col in first_example:
            audio_sample = first_example[audio_col]
            print(f"[DEBUG] Audio column type: {type(audio_sample)}")
            if isinstance(audio_sample, dict):
                print(f"[DEBUG] Audio dict keys: {list(audio_sample.keys())}")
        else:
            print(f"[DEBUG] WARNING: audio_col '{audio_col}' not found in dataset!")
        if text_col in first_example:
            print(f"[DEBUG] Text sample: {first_example[text_col][:100] if first_example[text_col] else 'None'}...")
        else:
            print(f"[DEBUG] WARNING: text_col '{text_col}' not found in dataset!")

        iterator = enumerate(dataset)
        if not hf_cfg.get('streaming', False):
            total = min(max_samples, len(dataset)) if max_samples else len(dataset)
            iterator = enumerate(tqdm(dataset, total=total, desc=f"Processing {split_name}"))

        for idx, example in iterator:
            if max_samples and idx >= max_samples:
                break

            try:
                # Get audio
                audio_data = example.get(audio_col)
                if audio_data is None:
                    skip_reasons["no_audio_col"] += 1
                    continue

                if isinstance(audio_data, dict):
                    array = audio_data.get("array")
                    sample_rate = audio_data.get("sampling_rate", 16000)
                else:
                    skip_reasons["audio_not_dict"] += 1
                    continue

                if array is None:
                    skip_reasons["no_array"] += 1
                    continue

                # Get text
                text = example.get(text_col, "")
                if not text or not isinstance(text, str):
                    skip_reasons["no_text"] += 1
                    continue
                text = text.strip()
                if not text:
                    skip_reasons["empty_text"] += 1
                    continue

                # Convert to numpy
                if not isinstance(array, np.ndarray):
                    array = np.array(array)

                # Calculate duration
                duration = len(array) / sample_rate

                # Filter by duration
                if duration < min_dur:
                    skip_reasons["too_short"] += 1
                    continue
                if duration > max_dur:
                    skip_reasons["too_long"] += 1
                    continue

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    try:
                        import librosa
                        array = librosa.resample(array, orig_sr=sample_rate, target_sr=16000)
                    except ImportError:
                        from scipy import signal
                        num_samples = int(len(array) * 16000 / sample_rate)
                        array = signal.resample(array, num_samples)
                    sample_rate = 16000
                    duration = len(array) / sample_rate

                # Save audio
                audio_path = os.path.join(audio_dir, split_name, f"{idx:08d}.wav")
                sf.write(audio_path, array, sample_rate)

                # Append language tag if specified (Parakeet multilingual format)
                if lang_tag:
                    text = f"{text} <{lang_tag}>"

                entries.append({
                    "audio_filepath": os.path.abspath(audio_path),
                    "text": text,
                    "duration": round(duration, 3)
                })

            except Exception as e:
                skip_reasons["error"] += 1
                if skip_reasons["error"] <= 3:  # Only print first 3 errors
                    print(f"Warning: Error processing {split_name} example {idx}: {e}")
                continue

        # Write manifest
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        total_duration = sum(e["duration"] for e in entries) if entries else 0
        print(f"\n{split_name}: {len(entries)} examples, {total_duration/3600:.2f} hours")

        # Print skip summary if any were skipped
        total_skipped = sum(skip_reasons.values())
        if total_skipped > 0:
            print(f"  Skipped {total_skipped} samples:")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"    - {reason}: {count}")

        return manifest_path

    print("\nPreparing training data...")
    train_manifest = process_split(train_ds, "train", max_train)

    print("\nPreparing validation data...")
    val_manifest = process_split(val_ds, "val", max_val)

    return train_manifest, val_manifest


def setup_trainer(cfg: DictConfig, num_gpus: int):
    """Setup PyTorch Lightning trainer."""
    from nemo.utils.exp_manager import exp_manager

    # Try to use NeMo's trainer if available (for compatibility)
    try:
        from nemo.lightning import NeMoLogger, Trainer as NeMoTrainer
        use_nemo_trainer = True
    except ImportError:
        use_nemo_trainer = False

    trainer_cfg = cfg.trainer

    # Update GPU settings
    if num_gpus > 1:
        trainer_cfg.devices = num_gpus
        trainer_cfg.strategy = "ddp"
    else:
        trainer_cfg.devices = 1
        trainer_cfg.strategy = "auto"

    # Force logger and checkpointing off - exp_manager handles these
    trainer_cfg.logger = False
    trainer_cfg.enable_checkpointing = False

    # Create trainer - use NeMo's import to ensure compatibility
    from nemo.core.config import hydra_runner
    from pytorch_lightning import Trainer

    trainer = Trainer(**OmegaConf.to_container(trainer_cfg))

    return trainer


def load_model(model_path: str, cfg: DictConfig):
    """Load model from .nemo file."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    print(f"Loading model from: {model_path}")
    model = EncDecRNNTBPEModel.restore_from(model_path)

    return model


def setup_model_for_finetuning(model, cfg: DictConfig):
    """Configure model for finetuning."""
    # Update data loaders
    if cfg.model.train_ds.manifest_filepath and cfg.model.train_ds.manifest_filepath != "__hf_generated__":
        model.setup_training_data(cfg.model.train_ds)

    if cfg.model.validation_ds.manifest_filepath and cfg.model.validation_ds.manifest_filepath != "__hf_generated__":
        model.setup_validation_data(cfg.model.validation_ds)

    # Update optimizer
    if cfg.model.get('optim'):
        model.setup_optimization(cfg.model.optim)

    # Update spec augment if specified
    if cfg.model.get('spec_augment'):
        spec_aug_cfg = cfg.model.spec_augment
        if hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
            model.spec_augmentation.freq_masks = spec_aug_cfg.get('freq_masks', 2)
            model.spec_augmentation.freq_width = spec_aug_cfg.get('freq_width', 27)
            model.spec_augmentation.time_masks = spec_aug_cfg.get('time_masks', 10)
            model.spec_augmentation.time_width = spec_aug_cfg.get('time_width', 0.05)

    return model


def load_config_with_inheritance(config_path: str, verbose: bool = False) -> DictConfig:
    """Load config file and merge with base configs specified in 'defaults'."""
    cfg = OmegaConf.load(config_path)
    config_dir = os.path.dirname(config_path)

    if verbose:
        print(f"\n[DEBUG] Loaded config from: {config_path}")

    # Handle defaults/inheritance (Hydra-style)
    defaults = OmegaConf.select(cfg, 'defaults')
    if defaults is not None:
        # Convert to list if needed
        if hasattr(defaults, '__iter__'):
            defaults_list = list(defaults)
        else:
            defaults_list = [defaults]

        if verbose:
            print(f"[DEBUG] Found defaults: {defaults_list}")

        # Load and merge base configs first
        merged_cfg = OmegaConf.create({})
        for default in defaults_list:
            # Handle both string format and dict format
            if isinstance(default, str):
                base_name = default
            elif hasattr(default, 'keys'):
                # Format like: - base: finetune_base
                base_name = list(default.values())[0]
            else:
                continue

            base_path = os.path.join(config_dir, f"{base_name}.yaml")
            if os.path.exists(base_path):
                if verbose:
                    print(f"[DEBUG] Loading base config: {base_path}")
                base_cfg = OmegaConf.load(base_path)
                merged_cfg = OmegaConf.merge(merged_cfg, base_cfg)
            else:
                print(f"Warning: Base config not found: {base_path}")

        # Remove defaults from current config before merging
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict.pop('defaults', None)
        cfg = OmegaConf.create(cfg_dict)

        if verbose:
            print(f"[DEBUG] Config after removing 'defaults' key:")
            print(OmegaConf.to_yaml(cfg))

        # Merge: base configs first, then current config overlays
        cfg = OmegaConf.merge(merged_cfg, cfg)

        if verbose:
            print(f"[DEBUG] Config after merging with base:")
            print(OmegaConf.to_yaml(cfg))

    return cfg


def main():
    args, overrides = parse_args()

    # Load configuration with inheritance
    print(f"Loading configuration from: {args.config}")
    cfg = load_config_with_inheritance(args.config, verbose=args.debug_config)

    # Apply command-line overrides
    cfg = apply_config_overrides(cfg, overrides, verbose=args.debug_config)

    # Set output directory
    cfg.exp_manager.exp_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Set model path
    cfg.model.init_from_nemo_model = args.model

    # Set resume checkpoint if specified
    if args.resume:
        cfg.model.resume_from_checkpoint = args.resume

    # Print final config
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")

    if args.dry_run:
        print("Dry run complete. Configuration is valid.")
        return

    # Import NeMo (delayed import for faster --help)
    try:
        import nemo.collections.asr as nemo_asr
        from nemo.utils.exp_manager import exp_manager
    except ImportError:
        print("Error: NeMo is not installed. Install with: pip install nemo_toolkit[asr]")
        sys.exit(1)

    # Prepare data based on source
    data_source = cfg.get('data_source', 'local')

    if data_source == 'huggingface':
        data_dir = os.path.join(args.output_dir, "hf_data")
        train_manifest = os.path.join(data_dir, "train_manifest.json")
        val_manifest = os.path.join(data_dir, "val_manifest.json")

        # Check if we can skip data prep
        if args.skip_data_prep:
            if os.path.exists(train_manifest) and os.path.exists(val_manifest):
                print(f"\n--skip_data_prep: Using existing manifests:")
                print(f"  Train: {train_manifest}")
                print(f"  Val: {val_manifest}")
            else:
                print("Error: --skip_data_prep specified but manifests not found:")
                print(f"  Train: {train_manifest} (exists: {os.path.exists(train_manifest)})")
                print(f"  Val: {val_manifest} (exists: {os.path.exists(val_manifest)})")
                sys.exit(1)
        else:
            print("\nPreparing HuggingFace dataset...")
            os.makedirs(data_dir, exist_ok=True)
            train_manifest, val_manifest = prepare_hf_manifests(cfg, data_dir)

        # Update config with generated manifests
        cfg.model.train_ds.manifest_filepath = train_manifest
        cfg.model.validation_ds.manifest_filepath = val_manifest

    elif data_source == 'local':
        # Validate manifest paths
        train_manifest = cfg.model.train_ds.manifest_filepath
        val_manifest = cfg.model.validation_ds.manifest_filepath

        if not train_manifest or train_manifest == "__hf_generated__":
            train_manifest = cfg.get('local_data_cfg', {}).get('train_manifest')

        if not val_manifest or val_manifest == "__hf_generated__":
            val_manifest = cfg.get('local_data_cfg', {}).get('val_manifest')

        if not train_manifest:
            print("Error: Training manifest not specified.")
            print("Use: local_data_cfg.train_manifest=/path/to/manifest.json")
            sys.exit(1)

        if not os.path.exists(train_manifest):
            print(f"Error: Training manifest not found: {train_manifest}")
            sys.exit(1)

        if val_manifest and not os.path.exists(val_manifest):
            print(f"Warning: Validation manifest not found: {val_manifest}")

        cfg.model.train_ds.manifest_filepath = train_manifest
        cfg.model.validation_ds.manifest_filepath = val_manifest

        print(f"Training manifest: {train_manifest}")
        print(f"Validation manifest: {val_manifest}")

    # Setup trainer
    print("\nSetting up trainer...")
    trainer = setup_trainer(cfg, args.num_gpus)

    # Setup experiment manager
    # Convert to plain dict to avoid OmegaConf interpolation issues
    exp_manager_cfg = {
        "exp_dir": cfg.exp_manager.get("exp_dir"),
        "name": cfg.exp_manager.get("name", "parakeet_finetune"),
        "create_tensorboard_logger": cfg.exp_manager.get("create_tensorboard_logger", True),
        "create_wandb_logger": cfg.exp_manager.get("create_wandb_logger", False),
        "create_checkpoint_callback": cfg.exp_manager.get("create_checkpoint_callback", True),
        "checkpoint_callback_params": OmegaConf.to_container(cfg.exp_manager.checkpoint_callback_params) if "checkpoint_callback_params" in cfg.exp_manager else {},
        "resume_if_exists": cfg.exp_manager.get("resume_if_exists", True),
        "resume_ignore_no_checkpoint": cfg.exp_manager.get("resume_ignore_no_checkpoint", True),
    }
    exp_manager(trainer, exp_manager_cfg)

    # Load model
    print("\nLoading model...")
    model = load_model(args.model, cfg)

    # Connect model to trainer (required for NeMo + PyTorch Lightning 2.x)
    model.set_trainer(trainer)

    # Configure model for finetuning
    print("Configuring model for finetuning...")
    model = setup_model_for_finetuning(model, cfg)

    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    # Fix PTL/NeMo LightningModule mismatch by patching the type check
    # Patch in both locations where it might be imported
    from nemo.core import ModelPT

    def patched_maybe_unwrap(model):
        """Patched version that accepts NeMo models."""
        if isinstance(model, ModelPT):
            return model
        # Original logic for torch.compile models
        import torch._dynamo
        if isinstance(model, torch._dynamo.OptimizedModule):
            return model._orig_mod
        from pytorch_lightning import LightningModule
        if not isinstance(model, LightningModule):
            raise TypeError(
                f"`model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `{type(model).__name__}`"
            )
        return model

    # Patch in both modules
    import pytorch_lightning.utilities.compile as pl_compile
    import pytorch_lightning.trainer.trainer as pl_trainer
    pl_compile._maybe_unwrap_optimized = patched_maybe_unwrap
    pl_trainer._maybe_unwrap_optimized = patched_maybe_unwrap
    print("[INFO] Patched PTL model type check for NeMo compatibility")

    # Resume from checkpoint if specified
    ckpt_path = args.resume or cfg.model.get('resume_from_checkpoint')

    trainer.fit(model, ckpt_path=ckpt_path)

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.nemo")
    model.save_to(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
