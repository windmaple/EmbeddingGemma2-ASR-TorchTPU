#!/usr/bin/env bash
# End-to-end reproduction pipeline for v2 (ReaderA, 6-layer 8-head RoPE Transformer, 19.6M params)
# across the full 960-hour LibriSpeech corpus on an 8-chip Cloud TPU VM (TorchTPU).
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

export TORCH_TPU_TIER2_COMPILATION_CACHE=1
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$DATA/torchtpu_cache"
export HF_HOME="$DATA/hf_cache"
export TMPDIR="$DATA/tmp"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" "$DATA/runs/v2" "$DATA/soft_tokens_cache" "$REPO/runs"
ln -sfn "$DATA/runs/v2" "$REPO/runs/v2"
cd "$REPO"

run_8chip() {
  (
    eval "$("$BIN/python" -m torch.tpu.distributed | sed 's/^/export /')"
    export TORCH_TPU_DEVICES_PER_HOST="${TORCH_TPU_DEVICES_PER_HOST:-8}"
    "$BIN/torchrun" --nproc_per_node="$WORLD_SIZE" "$@"
  )
}

echo "=================================================================="
echo "[Stage 1/6] Download EmbeddingGemma 2, LibriSpeech 960h, Lexicon & 4-Gram LM -> $DATA"
echo "=================================================================="
"$BIN/python" download_assets.py --data-dir "$DATA" --max-workers 16

echo "=================================================================="
echo "[Stage 2/6] Build 200,000-Word Embedding Table (vocab.txt + word_table.npy)"
echo "=================================================================="
if [ ! -f "$REPO/word_table.npy" ] || [ ! -f "$REPO/vocab.txt" ]; then
  run_8chip make_table.py \
    --lexicon "$DATA/librispeech-lexicon.txt" \
    --model-dir "$DATA/embeddinggemma-2" \
    --out-dir "$REPO" \
    --report "$REPO/make_table_report.json"
fi

echo "=================================================================="
echo "[Stage 3/6] Cache EmbeddingGemma 2 Soft Tokens for All 7 Splits (8x TPU)"
echo "=================================================================="
run_8chip cache_soft_tokens.py \
  --data-root "$DATA/hf_librispeech" \
  --model-dir "$DATA/embeddinggemma-2" \
  --out "$DATA/soft_tokens_cache" \
  --splits dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500 \
  --workers 6

"$BIN/python" data.py \
  --cache "$DATA/soft_tokens_cache" \
  --vocab "$REPO/vocab.txt" \
  --splits train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other

echo "=================================================================="
echo "[Stage 4/6] 8-Chip TorchTPU DDP Training on Full 960h LibriSpeech (v2 ReaderA)"
echo "=================================================================="
run_8chip train.py \
  --arch a --layers 6 \
  --cache "$DATA/soft_tokens_cache" \
  --vocab "$REPO/vocab.txt" \
  --table "$REPO/word_table.npy" \
  --run-dir "$DATA/runs/v2" \
  --train-split train-clean-100,train-clean-360,train-other-500 \
  --dev-split dev-clean \
  --epochs 12 --schedule-epochs 12 \
  --batch-size 32 --chunk 1024 --lr 4e-4 \
  --warmup 500 --patience 2 --log-every 500 --precision high

echo "=================================================================="
echo "[Stage 5/6] Export v2 Reader & Evaluate Greedy + 4-Gram Beam WER on All 4 Splits"
echo "=================================================================="
run_8chip eval_wer.py \
  --from-run "$DATA/runs/v2" \
  --version v2 \
  --reader-dir "$REPO" \
  --cache "$DATA/soft_tokens_cache" \
  --lexicon "$DATA/librispeech-lexicon.txt" \
  --out-dir "$DATA/runs/v2"

LM_PATH=""
if [ -f "$DATA/lm/4-gram.bin" ]; then
  LM_PATH="$DATA/lm/4-gram.bin"
elif [ -f "$DATA/lm/4-gram.arpa.gz" ]; then
  LM_PATH="$DATA/lm/4-gram.arpa.gz"
fi

if [ -n "$LM_PATH" ]; then
  run_8chip beam_decode.py \
    --reader-dir "$REPO" \
    --cache "$DATA/soft_tokens_cache" \
    --lm "$LM_PATH" \
    --lexicon "$DATA/librispeech-lexicon.txt" \
    --out-dir "$DATA/runs/v2" \
    --top-k 64 --beam-size 32 --tune-dev
fi

echo "=================================================================="
echo "[Stage 6/6] End-to-End Short & 54.3-Min Long-Form Benchmarks (1x & 8x TPU)"
echo "=================================================================="
DATA_ENV="$DATA" "$BIN/python" -c '
import glob, io, os, pyarrow.parquet as pq, soundfile as sf
data_dir = os.environ["DATA_ENV"]
pf = sorted(glob.glob(os.path.join(data_dir, "hf_librispeech/clean/test/*.parquet")))[0]
row = next(pq.ParquetFile(pf).iter_batches(batch_size=1, columns=["id", "text", "audio"])).to_pylist()[0]
wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
sf.write(os.path.join(data_dir, "runs/v2/sample_test_clean.wav"), wav, sr)
with open(os.path.join(data_dir, "runs/v2/sample_test_clean.txt"), "w") as f:
    f.write(row["text"].lower() + "\n")
print("Extracted sample utterance:", row["id"], "->", row["text"].lower())
'

"$BIN/python" transcribe.py \
  "$DATA/runs/v2/sample_test_clean.wav" \
  --reader-dir "$REPO" \
  --model-dir "$DATA/embeddinggemma-2" \
  --reference "$DATA/runs/v2/sample_test_clean.txt" \
  --warmup \
  --out-json "$DATA/runs/v2/sample_utt_1chip.json"

"$BIN/python" transcribe.py \
  "$DATA/speckled_band/audio_16k.wav" \
  --reader-dir "$REPO" \
  --model-dir "$DATA/embeddinggemma-2" \
  --reference "$DATA/speckled_band/transcript.txt" \
  --warmup \
  --out-json "$DATA/runs/v2/speckled_band_1chip.json"

run_8chip transcribe.py \
  "$DATA/speckled_band/audio_16k.wav" \
  --reader-dir "$REPO" \
  --model-dir "$DATA/embeddinggemma-2" \
  --reference "$DATA/speckled_band/transcript.txt" \
  --warmup \
  --out-json "$DATA/runs/v2/speckled_band_8chip.json"

if [ -n "$LM_PATH" ]; then
  run_8chip transcribe.py \
    "$DATA/speckled_band/audio_16k.wav" \
    --reader-dir "$REPO" \
    --model-dir "$DATA/embeddinggemma-2" \
    --reference "$DATA/speckled_band/transcript.txt" \
    --lm "$LM_PATH" \
    --beam-size 32 --lm-alpha 0.55 --lm-beta 2.0 \
    --warmup \
    --out-json "$DATA/runs/v2/speckled_band_8chip_beam4gram.json"
fi

rm -rf "$DATA/tmp"/*
echo "=================================================================="
echo "ALL V2 960H STAGES COMPLETE!"
echo "=================================================================="
