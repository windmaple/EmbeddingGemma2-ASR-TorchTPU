#!/usr/bin/env bash
# End-to-end reproduction pipeline for v3 (ReaderB(k=6), pretrained EmbeddingGemma 2 layers 0..5, 34.5M params)
# using cached soft tokens from Stage 1-3 of run_full_pipeline.sh on an 8-chip Cloud TPU VM (TorchTPU).
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA="${DATA:-$HOME/data}"
if [ -z "${BIN:-}" ]; then
  if [ -d "$HOME/miniconda3/envs/TorchTPU/bin" ]; then
    BIN="$HOME/miniconda3/envs/TorchTPU/bin"
  else
    BIN="$(dirname "$(command -v python3)")"
  fi
fi

V3_DIR="$DATA/runs/v3"
V3_READER="$V3_DIR/reader_dir"

export TORCH_TPU_TIER2_COMPILATION_CACHE=1
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$DATA/torchtpu_cache"
export HF_HOME="$DATA/hf_cache"
export TMPDIR="$DATA/tmp"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" "$V3_DIR" "$V3_READER" "$REPO/runs"
ln -sfn "$V3_DIR" "$REPO/runs/v3"
ln -sfn "$REPO/vocab.txt" "$V3_READER/vocab.txt"
ln -sfn "$REPO/word_table.npy" "$V3_READER/word_table.npy"
cd "$REPO"

run_8chip() {
  (
    eval "$("$BIN/python" -m torch.tpu.distributed | sed 's/^/export /')"
    export TORCH_TPU_DEVICES_PER_HOST="${TORCH_TPU_DEVICES_PER_HOST:-8}"
    "$BIN/torchrun" --nproc_per_node="$WORLD_SIZE" "$@"
  )
}

echo "=================================================================="
echo "[Stage 1/4] 8-Chip TorchTPU DDP Training: ReaderB(k=6) on Full 960h LibriSpeech (v3)"
echo "=================================================================="
run_8chip train.py \
  --arch b --k 6 --compile \
  --model-dir "$DATA/embeddinggemma-2" \
  --cache "$DATA/soft_tokens_cache" \
  --vocab "$REPO/vocab.txt" \
  --table "$REPO/word_table.npy" \
  --run-dir "$V3_DIR" \
  --train-split train-clean-100,train-clean-360,train-other-500 \
  --dev-split dev-clean \
  --epochs 12 --schedule-epochs 12 \
  --batch-size 32 --chunk 1024 \
  --lr 3e-4 --head-lr 8e-4 \
  --warmup 500 --patience 3 --log-every 500 --precision high

echo "=================================================================="
echo "[Stage 2/4] Export v3 ReaderB(k=6) & Evaluate Greedy WER (No LM) on All 4 Splits"
echo "=================================================================="
run_8chip eval_wer.py \
  --from-run "$V3_DIR" \
  --version v3 \
  --reader-dir "$V3_READER" \
  --cache "$DATA/soft_tokens_cache" \
  --lexicon "$DATA/librispeech-lexicon.txt" \
  --out-dir "$V3_DIR"

LM_PATH=""
if [ -f "$DATA/lm/4-gram.bin" ]; then
  LM_PATH="$DATA/lm/4-gram.bin"
elif [ -f "$DATA/lm/4-gram.arpa.gz" ]; then
  LM_PATH="$DATA/lm/4-gram.arpa.gz"
fi

if [ -n "$LM_PATH" ]; then
  echo "=================================================================="
  echo "[Stage 3/4] Evaluate v3 ReaderB(k=6) + 200k-Word CTC Beam + 4-Gram KenLM on All 4 Splits"
  echo "=================================================================="
  run_8chip beam_decode.py \
    --reader-dir "$V3_READER" \
    --cache "$DATA/soft_tokens_cache" \
    --lm "$LM_PATH" \
    --lexicon "$DATA/librispeech-lexicon.txt" \
    --out-dir "$V3_DIR" \
    --top-k 64 --beam-size 32 --tune-dev
fi

echo "=================================================================="
echo "[Stage 4/4] End-to-End 54.3-Min Speckled Band Benchmarks on v3 ReaderB(k=6)"
echo "=================================================================="
"$BIN/python" transcribe.py \
  "$DATA/speckled_band/audio_16k.wav" \
  --reader-dir "$V3_READER" \
  --model-dir "$DATA/embeddinggemma-2" \
  --reference "$DATA/speckled_band/transcript.txt" \
  --warmup \
  --out-json "$V3_DIR/speckled_band_1chip.json"

run_8chip transcribe.py \
  "$DATA/speckled_band/audio_16k.wav" \
  --reader-dir "$V3_READER" \
  --model-dir "$DATA/embeddinggemma-2" \
  --reference "$DATA/speckled_band/transcript.txt" \
  --warmup \
  --out-json "$V3_DIR/speckled_band_8chip.json"

if [ -n "$LM_PATH" ]; then
  read -r ALPHA BETA < <(V3_DIR_ENV="$V3_DIR" "$BIN/python" -c 'import json, os; d=json.load(open(os.path.join(os.environ["V3_DIR_ENV"], "wer_beam_4gram.json"))); print(d["alpha"], d["beta"])')
  run_8chip transcribe.py \
    "$DATA/speckled_band/audio_16k.wav" \
    --reader-dir "$V3_READER" \
    --model-dir "$DATA/embeddinggemma-2" \
    --reference "$DATA/speckled_band/transcript.txt" \
    --lm "$LM_PATH" \
    --beam-size 32 --lm-alpha "$ALPHA" --lm-beta "$BETA" \
    --warmup \
    --out-json "$V3_DIR/speckled_band_8chip_beam4gram.json"
fi

rm -rf "$DATA/tmp"/*
echo "=================================================================="
echo "ALL V3 960H READERB(K=6) STAGES COMPLETE!"
echo "=================================================================="
