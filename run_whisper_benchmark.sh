#!/usr/bin/env bash
# Benchmark all 11 OpenAI Whisper model sizes across LibriSpeech (dev/test clean & other)
# and long-form continuous audio (54.3-min Speckled Band) on 8x Cloud TPU v5e (TorchTPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA="${DATA:-${DATA_DIR:-$HOME/data}}"
if [ -z "${BIN:-}" ]; then
  if [ -x "$HOME/miniconda3/envs/TorchTPU/bin/python" ]; then
    BIN="$HOME/miniconda3/envs/TorchTPU/bin"
  else
    BIN="$(dirname "$(command -v python3 || command -v python)")"
  fi
fi

export HF_HOME="${HF_HOME:-$DATA/hf_cache}"
export TMPDIR="${TMPDIR:-$DATA/tmp}"
export TORCH_TPU_TIER2_COMPILATION_CACHE=1
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:-$DATA/torchtpu_cache}"
export TORCH_TPU_DEVICES_PER_HOST="${TORCH_TPU_DEVICES_PER_HOST:-8}"
mkdir -p "$HF_HOME" "$TMPDIR" "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" "$DATA/whisper_bench"

echo "[$(date +%H:%M:%S)] Stage 1/2: Pre-caching OpenAI Whisper checkpoints to $HF_HOME..."
"$BIN/python" - <<'PY'
import os, time
from huggingface_hub import snapshot_download

models = [
    "openai/whisper-tiny.en",
    "openai/whisper-tiny",
    "openai/whisper-base.en",
    "openai/whisper-base",
    "openai/whisper-small.en",
    "openai/whisper-small",
    "openai/whisper-medium.en",
    "openai/whisper-medium",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
]
for m in models:
    t0 = time.time()
    snapshot_download(
        repo_id=m,
        allow_patterns=["*.json", "*.txt", "*.tiktoken", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"],
    )
    print(f"[{time.strftime('%H:%M:%S')}] cached {m} ({time.time()-t0:.1f}s)", flush=True)
PY

echo "[$(date +%H:%M:%S)] Stage 2/2: Launching 8-chip TorchTPU Whisper benchmark..."
eval "$("$BIN/python" -m torch.tpu.distributed | sed 's/^/export /')"

"$BIN/torchrun" --nproc_per_node="${WORLD_SIZE:-8}" benchmark_whisper_tpu.py \
  --librispeech-dir "$DATA/hf_librispeech" \
  --longform-audio "$DATA/speckled_band/audio_16k.wav" \
  --longform-ref "$DATA/speckled_band/transcript.txt" \
  --out-json "$DATA/whisper_bench/whisper_tpu_benchmark.json" \
  --tmp-dir "$DATA/tmp/whisper_bench" \
  "$@"
