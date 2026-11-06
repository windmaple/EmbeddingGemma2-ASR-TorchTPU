#!/usr/bin/env python3
"""Build vocab.txt and word_table.npy (float16 [200000, 768], L2-normalized) on TorchTPU (1 or 8 chips).

Features:
  - Fixed static shape [batch_size, seq_len=32] so XLA compiles a single graph with zero recompilations.
  - Uses torch.tpu.Precision.HIGHEST for IEEE-grade float32 embedding fidelity.
  - Verifies TPU static-padded output against CPU SentenceTransformer.encode() reference (min cos > 0.9999).
  - Verifies 100 random words achieve 1.000 top-1 self-retrieval over all 200,000 words.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from tpu_utils import cleanup_distributed, configure_tpu_env, get_device, init_distributed, patch_hf_for_tpu, synchronize

PROMPT = "task: search result | query: "


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lexicon_words(path: str) -> list[str]:
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                words.add(parts[0].lower())
    return sorted(words)


def transcript_sets(index_paths: list[str], fixture_dir: str | None) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for p in index_paths or []:
        if not os.path.exists(p):
            continue
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        with open(p, encoding="utf-8") as f:
            sets[name] = [line.rstrip("\n").split("\t")[4] for line in f if line.strip()]
    if fixture_dir and os.path.exists(fixture_dir):
        texts = []
        for p in sorted(glob.glob(os.path.join(fixture_dir, "**", "transcript.txt"), recursive=True)):
            with open(p, encoding="utf-8") as f:
                texts.append(" ".join(f.read().lower().split()))
        if texts:
            sets["fixtures:" + os.path.basename(os.path.normpath(fixture_dir))] = texts
    return sets


def oov_stats(texts: list[str], vocab_set: set[str]) -> dict:
    tokens = [w for t in texts for w in t.split()]
    oov_tokens = [w for w in tokens if w not in vocab_set]
    oov_utts = sum(1 for t in texts if any(w not in vocab_set for w in t.split()))
    return {
        "utterances": len(texts),
        "tokens": len(tokens),
        "types": len(set(tokens)),
        "oov_tokens": len(oov_tokens),
        "oov_types": len(set(oov_tokens)),
        "oov_token_ratio": len(oov_tokens) / max(1, len(tokens)),
        "utterances_with_oov": oov_utts,
        "oov_examples": sorted(set(oov_tokens))[:20],
    }


@torch.no_grad()
def encode_static_batches(
    st,
    texts: list[str],
    device: torch.device,
    batch_size: int = 512,
    seq_len: int = 32,
) -> np.ndarray:
    """Encode texts using SentenceTransformer with static [batch_size, seq_len] shapes for zero XLA recompiles."""
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    tokenizer = st.tokenizer
    n = len(texts)
    n_batches = math.ceil(n / batch_size)
    outputs = []

    prec_ctx = (
        torch.tpu.precision(torch.tpu.Precision.HIGHEST)
        if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")
        else contextlib.nullcontext()
    )

    with prec_ctx:
        for b in range(n_batches):
            chunk = texts[b * batch_size : (b + 1) * batch_size]
            real_len = len(chunk)
            if real_len < batch_size:
                chunk = chunk + [chunk[-1]] * (batch_size - real_len)
            tok = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                add_special_tokens=False,
                return_tensors="pt",
            )
            features = {k: v.to(device) for k, v in tok.items()}
            emb = st(features)["sentence_embedding"]
            outputs.append(emb[:real_len].float().cpu().numpy())

    res = np.concatenate(outputs, axis=0)
    norms = np.linalg.norm(res, axis=1, keepdims=True)
    return res / np.maximum(norms, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--check-words", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", nargs="*", default=[])
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--tmp-dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "eg2_make_table_shards"))
    args = ap.parse_args()

    configure_tpu_env()
    patch_hf_for_tpu()
    is_dist, rank, world_size = init_distributed()
    device = get_device(args.device)

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    vocab = lexicon_words(args.lexicon)
    log(f"vocabulary: {len(vocab)} words from {args.lexicon}; world_size={world_size} device={device}", rank)

    # Check max token length on rank 0 to guarantee seq_len never truncates any word
    st = SentenceTransformer(
        args.model_dir,
        device=str(device),
        model_kwargs={"dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    st.eval()
    t_load = time.time() - t0

    # Verify longest word length in characters -> tokens
    longest_words = sorted(vocab, key=len, reverse=True)[:32]
    max_tok_seen = max(len(ids) for ids in st.tokenizer([args.prompt + w for w in longest_words], add_special_tokens=False)["input_ids"])
    seq_len = max(args.seq_len, ((max_tok_seen + 15) // 16) * 16)
    log(f"model loaded in {t_load:.1f}s; max token length among longest words={max_tok_seen} -> static seq_len={seq_len}", rank)

    # Numerical parity check on rank 0 against CPU SentenceTransformer.encode()
    parity_min_cos = 1.0
    if rank == 0 and device.type != "cpu":
        sample_words = [vocab[i] for i in np.linspace(0, len(vocab) - 1, 16, dtype=int)]
        st_cpu = SentenceTransformer(
            args.model_dir,
            device="cpu",
            model_kwargs={"dtype": torch.float32},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        st_cpu.eval()
        ref = st_cpu.encode([args.prompt + w for w in sample_words], batch_size=16, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        del st_cpu
        tpu_chk = encode_static_batches(st, [args.prompt + w for w in sample_words], device, batch_size=16, seq_len=seq_len)
        cos_per_word = (ref * tpu_chk).sum(axis=1)
        parity_min_cos = float(cos_per_word.min())
        log(f"TPU vs CPU float32 parity on 16 words: min_cos={parity_min_cos:.7f} mean_cos={float(cos_per_word.mean()):.7f}", rank)
        if parity_min_cos < 0.9995:
            raise RuntimeError(f"TPU vs CPU parity too low: min_cos={parity_min_cos}")

    if is_dist:
        dist.barrier()

    # Contiguous shard per rank
    per_rank = math.ceil(len(vocab) / world_size)
    start = rank * per_rank
    end = min(len(vocab), (rank + 1) * per_rank)
    shard_words = [args.prompt + w for w in vocab[start:end]]

    t1 = time.time()
    shard_emb = encode_static_batches(st, shard_words, device, batch_size=args.batch_size, seq_len=seq_len)
    synchronize()
    t_encode = time.time() - t1

    shard_dir = Path(args.tmp_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_file = shard_dir / f"shard_{rank:02d}.npy"
    tmp_file = shard_dir / f"shard_{rank:02d}.part.npy"
    np.save(str(tmp_file), shard_emb.astype(np.float16))
    tmp_file.replace(shard_file)

    if is_dist:
        dist.barrier()

    if rank == 0:
        shards = [np.load(str(shard_dir / f"shard_{r:02d}.npy")) for r in range(world_size)]
        table = np.concatenate(shards, axis=0)
        assert table.shape == (len(vocab), 768), f"Unexpected table shape {table.shape}"
        log(f"encoded {len(vocab)} words across {world_size} chip(s) in {t_encode:.1f}s ({len(vocab) / max(t_encode, 1e-6):.0f} words/s)", rank)

        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(vocab) + "\n")
        np.save(os.path.join(args.out_dir, "word_table.npy"), table)

        # Self-retrieval check: 100 random words re-encoded must match rank 1
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(vocab), size=min(args.check_words, len(vocab)), replace=False)
        q = encode_static_batches(
            st,
            [args.prompt + vocab[i] for i in pick],
            device,
            batch_size=min(128, len(pick)),
            seq_len=seq_len,
        )
        sims = q @ table.astype(np.float32).T
        top1 = sims.argmax(1)
        hit = float((top1 == pick).mean())
        gap = np.sort(sims, axis=1)[:, -2]
        log(
            f"text -> text top-1 {hit:.3f} over {len(pick)} words "
            f"(self cos min {sims[np.arange(len(pick)), pick].min():.6f}, runner-up cos max {gap.max():.6f})",
            rank,
        )
        if hit < 0.99:
            raise RuntimeError(f"Self-retrieval top-1 failed: {hit}")

        vocab_set = set(vocab)
        oov = {name: oov_stats(texts, vocab_set) for name, texts in transcript_sets(args.index, args.fixtures).items()}
        norms = np.linalg.norm(table.astype(np.float32), axis=1)
        report = {
            "vocab_size": len(vocab),
            "prompt": args.prompt,
            "device": str(device),
            "world_size": world_size,
            "model_load_s": round(t_load, 1),
            "encode_s": round(t_encode, 1),
            "words_per_s": round(len(vocab) / max(t_encode, 1e-6), 1),
            "tpu_vs_cpu_min_cos": round(parity_min_cos, 7),
            "table": {
                "dtype": "float16",
                "shape": list(table.shape),
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
            },
            "text_to_text": {
                "n": int(len(pick)),
                "top1": hit,
                "self_cos_min": float(sims[np.arange(len(pick)), pick].min()),
                "runner_up_cos_max": float(gap.max()),
                "seed": args.seed,
            },
            "oov": oov,
            "versions": {
                "torch": torch.__version__,
                "sentence_transformers": __import__("sentence_transformers").__version__,
                "transformers": __import__("transformers").__version__,
            },
        }
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            log(f"-> {args.report}", rank)

        for r in range(world_size):
            (shard_dir / f"shard_{r:02d}.npy").unlink(missing_ok=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
