# Parakeet RNNT 1.1B Multilingual Finetuning

A simple, configurable project for finetuning NVIDIA's `parakeet-rnnt-1.1b-multilingual` model using the NeMo framework.

## Features

- Support for **HuggingFace datasets** (Common Voice, LibriSpeech, FLEURS, etc.)
- Support for **local audio files** with CSV/TSV transcripts
- **Resume training** from checkpoints or previously finetuned models
- **Multi-GPU training** via DDP
- Optimized for **high-VRAM GPUs** (80-96GB like A100, RTX PRO 6000)
- Easy configuration via YAML files with command-line overrides

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download the Base Model

Download the Parakeet RNNT 1.1B multilingual model from NVIDIA NGC:

```bash
# Using NGC CLI
ngc registry model download-version "nvidia/nemo/parakeet-rnnt-1.1b-multilingual:1.0"

# Or download directly from HuggingFace
# https://huggingface.co/nvidia/parakeet-rnnt-1.1b-multilingual
```

## Quick Start

### Option 1: Finetune with HuggingFace Dataset

```bash
python finetune.py \
    --config configs/finetune_huggingface.yaml \
    --model /path/to/parakeet-rnnt-1.1b-multilingual.nemo \
    --output_dir ./experiments/hindi_cv \
    hf_data_cfg.path=mozilla-foundation/common_voice_16_1 \
    hf_data_cfg.name=hi
```

### Option 2: Finetune with Local Data

1. **Prepare your data** - Create a CSV/TSV file with audio paths and transcripts:

```csv
audio_path,text
/data/audio/001.wav,यह एक परीक्षण है
/data/audio/002.wav,नमस्ते दुनिया
```

2. **Generate NeMo manifest**:

```bash
python scripts/prepare_local_data.py \
    --input /path/to/transcripts.csv \
    --audio_dir /path/to/audio \
    --output_dir ./data \
    --val_split 0.1
```

3. **Run finetuning**:

```bash
python finetune.py \
    --config configs/finetune_local.yaml \
    --model /path/to/parakeet-rnnt-1.1b-multilingual.nemo \
    --output_dir ./experiments/my_finetune \
    local_data_cfg.train_manifest=./data/train_manifest.json \
    local_data_cfg.val_manifest=./data/val_manifest.json
```

## Inference

### Transcribe a Single File

```bash
python inference.py \
    --model ./experiments/my_finetune/final_model.nemo \
    --audio /path/to/audio.wav
```

### Transcribe a Directory

```bash
python inference.py \
    --model ./experiments/my_finetune/final_model.nemo \
    --audio_dir /path/to/audio_folder \
    --output results.json
```

### Evaluate on Test Set

```bash
python inference.py \
    --model ./experiments/my_finetune/final_model.nemo \
    --manifest /path/to/test_manifest.json \
    --output eval_results.json
```

## Configuration

### Base Configuration (`configs/finetune_base.yaml`)

Key parameters optimized for 80-96GB VRAM:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 48 | Training batch size |
| `precision` | bf16-mixed | Mixed precision training |
| `max_epochs` | 50 | Maximum training epochs |
| `lr` | 1e-4 | Learning rate |
| `warmup_steps` | 2000 | LR warmup steps |
| `max_duration` | 20.0 | Max audio duration (seconds) |

### Override Parameters via Command Line

Use Hydra-style overrides:

```bash
python finetune.py \
    --config configs/finetune_local.yaml \
    --model /path/to/model.nemo \
    --output_dir ./experiments/test \
    trainer.max_epochs=100 \
    model.train_ds.batch_size=32 \
    model.optim.lr=5e-5
```

## Resume Training

### From a Checkpoint

```bash
python finetune.py \
    --config configs/finetune_local.yaml \
    --model /path/to/original_model.nemo \
    --output_dir ./experiments/resumed \
    --resume ./experiments/previous/checkpoints/epoch=10-val_wer=0.1234.ckpt
```

### From a Finetuned Model

To continue finetuning on additional data:

```bash
python finetune.py \
    --config configs/finetune_local.yaml \
    --model ./experiments/v1/final_model.nemo \
    --output_dir ./experiments/v2 \
    local_data_cfg.train_manifest=./data/additional_train.json \
    local_data_cfg.val_manifest=./data/additional_val.json
```

## Data Preparation

### Local Data Script

The `scripts/prepare_local_data.py` script supports various input formats:

```bash
python scripts/prepare_local_data.py \
    --input transcripts.csv \
    --audio_dir /path/to/audio \
    --output_dir ./data \
    --val_split 0.1 \
    --language_tag hi-IN \    # Append language tag for Parakeet multilingual
    --normalize_text \        # Optional: lowercase + remove punctuation
    --resample \              # Optional: resample to 16kHz mono
    --min_duration 0.5 \
    --max_duration 15.0
```

**Supported CSV columns** (auto-detected):
- Audio: `audio_filepath`, `audio_path`, `audio`, `path`, `file`, `filename`, `wav`
- Text: `text`, `transcript`, `sentence`, `transcription`, `label`
- Duration: `duration` (optional, calculated from audio if missing)

### HuggingFace Data Script

For large datasets, pre-download and convert to local manifests:

```bash
python scripts/prepare_hf_data.py \
    --dataset mozilla-foundation/common_voice_16_1 \
    --subset hi \
    --output_dir ./data/common_voice_hindi \
    --save_audio \
    --train_tokenizer \      # Optional: train custom tokenizer
    --vocab_size 1024
```

## Language Tags (Parakeet Multilingual)

Parakeet multilingual model outputs transcriptions with language tags in the format: `"transcription <lang-tag>"` (e.g., `"hello world <en-US>"`).

**Important**: Your training data should include these tags to maintain model behavior.

### Using Language Tags

For local data:
```bash
python scripts/prepare_local_data.py \
    --input transcripts.csv \
    --audio_dir /path/to/audio \
    --output_dir ./data \
    --language_tag hi-IN
```

For HuggingFace data (via config):
```yaml
hf_data_cfg:
  language_tag: "hi-IN"  # Appended to all transcripts
```

Or via command line:
```bash
python finetune.py \
    --config configs/finetune_huggingface.yaml \
    --model /path/to/model.nemo \
    --output_dir ./experiments/hindi \
    hf_data_cfg.language_tag=hi-IN
```

### Common Language Tags

| Language | Tag |
|----------|-----|
| English (US) | `en-US` |
| Hindi (India) | `hi-IN` |
| German (Germany) | `de-DE` |
| French (France) | `fr-FR` |
| Spanish (Spain) | `es-ES` |
| Mandarin (China) | `zh-CN` |

If your data already contains language tags, set `language_tag: null` or omit the `--language_tag` flag.

## Multi-GPU Training

```bash
# Using PyTorch DDP
python finetune.py \
    --config configs/finetune_local.yaml \
    --model /path/to/model.nemo \
    --output_dir ./experiments/multi_gpu \
    --num_gpus 4
```

Or use `torchrun`:

```bash
torchrun --nproc_per_node=4 finetune.py \
    --config configs/finetune_local.yaml \
    --model /path/to/model.nemo \
    --output_dir ./experiments/multi_gpu
```

## Custom Tokenizer

For new languages or specialized vocabulary (recommended for 50+ hours of data):

```bash
# Train tokenizer from HuggingFace data
python scripts/prepare_hf_data.py \
    --dataset mozilla-foundation/common_voice_16_1 \
    --subset hi \
    --output_dir ./data/hindi \
    --train_tokenizer \
    --vocab_size 1024

# Use in finetuning
python finetune.py \
    --config configs/finetune_local.yaml \
    --model /path/to/model.nemo \
    --output_dir ./experiments/hindi \
    model.tokenizer.update_tokenizer=true \
    model.tokenizer.dir=./data/hindi/tokenizer
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `model.train_ds.batch_size=24`
2. Reduce max duration: `model.train_ds.max_duration=15`
3. Increase gradient accumulation: `trainer.accumulate_grad_batches=4`

### Slow Training

1. Increase num_workers: `model.train_ds.num_workers=16`
2. Enable pin_memory (default): `model.train_ds.pin_memory=true`
3. Use bf16 precision (default for Ampere+): `trainer.precision=bf16-mixed`

### Poor WER

1. Increase epochs: `trainer.max_epochs=100`
2. Lower learning rate: `model.optim.lr=5e-5`
3. Increase warmup: `model.optim.sched.warmup_steps=5000`
4. Check data quality - ensure transcripts match audio

## Project Structure

```
parakeet-finetuning/
├── configs/
│   ├── finetune_base.yaml          # Shared training settings
│   ├── finetune_huggingface.yaml   # HuggingFace dataset config
│   └── finetune_local.yaml         # Local data config
├── scripts/
│   ├── prepare_local_data.py       # CSV/TSV to NeMo manifest
│   └── prepare_hf_data.py          # HuggingFace data preparation
├── finetune.py                     # Main training script
├── inference.py                    # Inference and evaluation
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## IMPORTANT: NeMo Bug Fix:
```bash
# Find the file on your server                                                                                                                                                                          
NEMO_FILE=$(python3 -c "import nemo; import os; print(os.path.join(os.path.dirname(nemo.__file__), 'collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py')")
# Apply the fix (change 6 underscores to 5)
sed -i 's/capture_status, _, graph, _, _, _ = cu_call/capture_status, _, graph, _, _ = cu_call/' "$NEMO_FILE"
```

## License

This project is provided for educational and research purposes. The Parakeet model is subject to NVIDIA's license terms.
