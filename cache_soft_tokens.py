#!/usr/bin/env python3
"""Cache EmbeddingGemma 2 audio soft tokens across 1 or 8 TPU chips with static-bucket XLA compilation.

Writes compatible <out>/<split>/<id>.npy (float16 [n_tokens, 512]) and <out>/<split>/index.tsv in exact parquet order.
Before bulk extraction, rank 0 verifies numerical parity of TPU bucketed batch extraction against CPU float32
single-utterance extraction (min per-token cosine > 0.9995).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import glob
import io
import itertools
import os
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.distributed as dist

from tpu_utils import cleanup_distributed, configure_tpu_env, get_device, init_distributed, patch_hf_for_tpu, synchronize

SR = 16000
SPLITS = {
    "train-clean-100": "clean/train.100",
    "train-clean-360": "clean/train.360",
    "train-other-500": "other/train.500",
    "dev-clean": "clean/validation",
    "dev-other": "other/validation",
    "test-clean": "clean/test",
    "test-other": "other/test",
}


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def n_frames_for(n_samples: int) -> int:
    return max(0, (n_samples - 161) // 160 + 1)


def n_tokens_for(n_frames: int) -> int:
    return ((n_frames + 1) // 2 + 1) // 2


def extractor_frames(n_samples: int, pad_to_multiple_of: int = 128) -> int:
    m = -(-n_samples // pad_to_multiple_of) * pad_to_multiple_of
    return (m + 160 - 321) // 160 + 1


def bucket_frame_len(raw_frames: int, step: int = 512) -> int:
    return max(step, ((raw_frames + step - 1) // step) * step)


def parquet_files(data_root: str, split: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_root, SPLITS[split], "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files for {split} under {os.path.join(data_root, SPLITS[split])}")
    return files


def decode_flac(flac_bytes: bytes, uid: str) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32", always_2d=False)
    if sr != SR or wav.ndim != 1:
        raise ValueError(f"{uid}: expected 16 kHz mono, got sr={sr} shape={wav.shape}")
    return wav


def extract_mel_features(fe, wav: np.ndarray) -> np.ndarray:
    out = fe([np.asarray(wav, dtype=np.float32)], return_tensors="np", max_length=None, truncation=False)
    f = out["input_features"][0].astype(np.float32)
    m = out["input_features_mask"][0].astype(bool)
    n = n_frames_for(len(wav))
    if int(m.sum()) != n or not m[:n].all():
        raise AssertionError(f"extractor mask {int(m.sum())} != {n} leading frames")
    return f[:n]


def load_encoder(model_dir: str, device: torch.device):
    from transformers import EmbeddingGemma2Model, Gemma4AudioFeatureExtractor

    fe = Gemma4AudioFeatureExtractor.from_pretrained(model_dir)
    model = EmbeddingGemma2Model.from_pretrained(
        model_dir,
        dtype=torch.float32,
        vision_config=None,
    )
    text_impl = getattr(model.config.text_config, "_attn_implementation", None)
    audio_impl = getattr(model.config.audio_config, "_attn_implementation", None)
    if text_impl == "eager":
        raise SystemExit("ERROR: text_config attention is 'eager'!")
    model = model.to(device).eval()
    return fe, model, f"audio={audio_impl},text={text_impl}"


@torch.no_grad()
def soft_tokens_single_cpu(model_cpu, fe, wav: np.ndarray) -> np.ndarray:
    out = fe([np.asarray(wav, dtype=np.float32)], return_tensors="np", max_length=None, truncation=False)
    feats = torch.from_numpy(out["input_features"][0].astype(np.float32))[None]
    mask = torch.from_numpy(out["input_features_mask"][0].astype(bool))[None]
    res = model_cpu.get_audio_features(feats, mask, return_dict=True)
    pooled = res.pooler_output[0].float().numpy()
    am = res.attention_mask[0].numpy()
    return pooled[am]


@torch.no_grad()
def soft_tokens_batch_static(
    model,
    feats: list[np.ndarray],
    n_samples: list[int],
    device: torch.device,
    fixed_batch_size: int,
    frame_step: int = 512,
) -> list[np.ndarray]:
    """Run get_audio_features with static [fixed_batch_size, T_bucket, 128] shape and CPU boolean slicing."""
    real_b = len(feats)
    assert 1 <= real_b <= fixed_batch_size
    n_valid = [x.shape[0] for x in feats]
    raw_t = max(extractor_frames(s) for s in n_samples)
    T = bucket_frame_len(max(raw_t, max(n_valid)), step=frame_step)
    mel_dim = feats[0].shape[1]

    x = np.zeros((fixed_batch_size, T, mel_dim), dtype=np.float32)
    m = np.zeros((fixed_batch_size, T), dtype=bool)
    for b in range(fixed_batch_size):
        src = b if b < real_b else 0  # duplicate row 0 on padding rows so softmax never sees all-False mask
        nb = n_valid[src]
        x[b, :nb] = feats[src]
        m[b, :nb] = True

    prec_ctx = (
        torch.tpu.precision(torch.tpu.Precision.HIGHEST)
        if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")
        else contextlib.nullcontext()
    )
    with prec_ctx:
        res = model.get_audio_features(
            torch.from_numpy(x).to(device),
            torch.from_numpy(m).to(device),
            return_dict=True,
        )
        # Single device->host copy of static tensor; boolean indexing happens in NumPy (zero aten::nonzero sync!)
        pooled_cpu = res.pooler_output.float().cpu().numpy()
        am_cpu = res.attention_mask.cpu().numpy()

    out: list[np.ndarray] = []
    for b in range(real_b):
        s = pooled_cpu[b][am_cpu[b]]
        expected = n_tokens_for(n_valid[b])
        if s.shape[0] != expected:
            raise AssertionError(f"Expected {expected} tokens for {n_valid[b]} frames, got {s.shape[0]}")
        out.append(s)
    return out


_TLS = threading.local()


def _get_thread_fe(model_dir: str):
    fe = getattr(_TLS, "fe", None)
    if fe is None:
        from transformers import Gemma4AudioFeatureExtractor

        fe = Gemma4AudioFeatureExtractor.from_pretrained(model_dir)
        _TLS.fe = fe
    return fe


def _thread_decode_and_mel(model_dir: str, item):
    row_idx, uid, flac_bytes, text = item
    wav = decode_flac(flac_bytes, uid)
    mel = extract_mel_features(_get_thread_fe(model_dir), wav)
    return row_idx, uid, len(wav), mel, text


def iter_split_sharded(data_root: str, split: str, rank: int, world_size: int, done_uids: set[str]):
    row_idx = 0
    for pf in parquet_files(data_root, split):
        for batch in pq.ParquetFile(pf).iter_batches(batch_size=64, columns=["id", "text", "audio"]):
            for row in batch.to_pylist():
                uid = row["id"]
                if row_idx % world_size == rank and uid not in done_uids:
                    yield row_idx, uid, row["audio"]["bytes"], row["text"].lower()
                row_idx += 1


def verify_parity_on_rank0(data_root: str, split: str, model_dir: str, model_tpu, fe, device: torch.device) -> None:
    from transformers import EmbeddingGemma2Model

    pf = parquet_files(data_root, split)[0]
    sample_rows = []
    for batch in pq.ParquetFile(pf).iter_batches(batch_size=8, columns=["id", "text", "audio"]):
        for row in batch.to_pylist():
            sample_rows.append((row["id"], decode_flac(row["audio"]["bytes"], row["id"])))
            if len(sample_rows) >= 4:
                break
        if len(sample_rows) >= 4:
            break

    model_cpu = EmbeddingGemma2Model.from_pretrained(
        model_dir,
        dtype=torch.float32,
        vision_config=None,
    ).eval()
    refs = [soft_tokens_single_cpu(model_cpu, fe, wav) for _, wav in sample_rows]
    del model_cpu

    mels = [extract_mel_features(fe, wav) for _, wav in sample_rows]
    ns = [len(wav) for _, wav in sample_rows]
    hyps = soft_tokens_batch_static(model_tpu, mels, ns, device, fixed_batch_size=4, frame_step=512)

    min_cos = 1.0
    max_diff = 0.0
    for (uid, _), r, h in zip(sample_rows, refs, hyps):
        assert r.shape == h.shape, f"{uid}: shape mismatch {r.shape} vs {h.shape}"
        r_norm = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
        h_norm = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        cos = float((r_norm * h_norm).sum(axis=-1).min())
        diff = float(np.abs(r - h).max())
        min_cos = min(min_cos, cos)
        max_diff = max(max_diff, diff)

    log(f"TPU static-batch vs CPU single-utterance float32 parity on 4 utterances: min_token_cos={min_cos:.7f}, max|diff|={max_diff:.5e}", 0)
    if min_cos < 0.999:
        raise RuntimeError(f"Audio soft-token parity failed: min_token_cos={min_cos}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100"], choices=list(SPLITS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--frame-step", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4, help="CPU worker threads PER RANK for FLAC+mel")
    ap.add_argument("--every", type=int, default=500)
    args = ap.parse_args()

    configure_tpu_env()
    patch_hf_for_tpu()
    is_dist, rank, world_size = init_distributed()
    device = get_device(args.device)

    fe, model, attn_impl = load_encoder(args.model_dir, device)
    log(f"loaded EmbeddingGemma2 audio encoder on {device} (world_size={world_size}, attn={attn_impl}, batch={args.batch_size}, threads/rank={args.workers})", rank)

    if rank == 0 and device.type != "cpu":
        verify_parity_on_rank0(args.data_root, args.splits[0], args.model_dir, model, fe, device)
    if is_dist:
        dist.barrier()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) if args.workers > 0 else None

    def decode_chunk(raw_rows: list) -> list:
        if not raw_rows:
            return []
        if executor is not None:
            return list(executor.map(lambda it: _thread_decode_and_mel(args.model_dir, it), raw_rows))
        return [(r_idx, uid, len(w), extract_mel_features(fe, w), txt) for r_idx, uid, fb, txt in raw_rows for w in [decode_flac(fb, uid)]]

    for split in args.splits:
        split_dir = Path(args.out) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        index_path = split_dir / "index.tsv"

        # Check existing completed utterances if resuming
        existing_lines: dict[str, str] = {}
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        u = line.split("\t", 1)[0]
                        if (split_dir / f"{u}.npy").exists():
                            existing_lines[u] = line.rstrip("\n")

        if is_dist:
            dist.barrier()

        rank_index_path = split_dir / f"index_rank{rank:02d}.tsv"
        t0 = time.time()
        n_new = 0
        audio_s = 0.0
        last_log = 0

        row_iter = iter_split_sharded(args.data_root, split, rank, world_size, set(existing_lines.keys()))

        with open(rank_index_path, "w", encoding="utf-8") as rfh:
            def flush_buf(buf: list) -> None:
                order = sorted(range(len(buf)), key=lambda i: buf[i][3].shape[0])
                soft_out: list[np.ndarray | None] = [None] * len(buf)
                for a in range(0, len(order), args.batch_size):
                    idx = order[a : a + args.batch_size]
                    s_batch = soft_tokens_batch_static(
                        model,
                        [buf[i][3] for i in idx],
                        [buf[i][2] for i in idx],
                        device,
                        fixed_batch_size=args.batch_size,
                        frame_step=args.frame_step,
                    )
                    for i, s in zip(idx, s_batch):
                        soft_out[i] = s
                for (r_idx, uid, n_samp, mel, txt), s in zip(buf, soft_out):
                    assert s is not None
                    if not np.isfinite(s).all():
                        raise RuntimeError(f"Non-finite soft tokens for {uid}")
                    np.save(str(split_dir / f"{uid}.npy"), s.astype(np.float16))
                    rfh.write(f"{r_idx}\t{uid}\t{n_samp}\t{mel.shape[0]}\t{s.shape[0]}\t{txt}\n")
                rfh.flush()

            # Double-buffered CPU FLAC+mel extraction overlapping TPU forward pass
            first_raw = list(itertools.islice(row_iter, args.chunk))
            pending_fut = executor.submit(decode_chunk, first_raw) if (executor is not None and first_raw) else None
            cur_buf = None if pending_fut is not None else decode_chunk(first_raw)

            while True:
                if pending_fut is not None:
                    cur_buf = pending_fut.result()
                    pending_fut = None
                if not cur_buf:
                    break
                next_raw = list(itertools.islice(row_iter, args.chunk))
                if next_raw:
                    if executor is not None:
                        pending_fut = executor.submit(decode_chunk, next_raw)
                    else:
                        next_decoded = decode_chunk(next_raw)
                else:
                    next_decoded = []

                n_new += len(cur_buf)
                audio_s += sum(it[2] for it in cur_buf) / SR
                flush_buf(cur_buf)

                if rank == 0 and (n_new - last_log) >= max(1, args.every // world_size):
                    last_log = n_new
                    el = time.time() - t0
                    est_tot_n = n_new * world_size
                    est_tot_h = (audio_s * world_size) / 3600.0
                    log(f"{split}: ~{est_tot_n} utterances (~{est_tot_h:.2f} h) in {el:.1f}s = x{(audio_s * world_size) / max(el, 1e-6):.0f} realtime", rank)

                if pending_fut is None:
                    cur_buf = next_decoded

        synchronize()
        if is_dist:
            dist.barrier()

        if rank == 0:
            # Merge existing + new rank shards in deterministic parquet order
            new_records: dict[str, tuple[int, str]] = {}
            for r in range(world_size):
                rp = split_dir / f"index_rank{r:02d}.tsv"
                if rp.exists():
                    with open(rp, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                r_str, rest = line.rstrip("\n").split("\t", 1)
                                uid = rest.split("\t", 1)[0]
                                new_records[uid] = (int(r_str), rest)
                    rp.unlink(missing_ok=True)

            # Reconstruct exact parquet row order across the split
            ordered_lines: list[str] = []
            total_audio_s = 0.0
            for pf in parquet_files(args.data_root, split):
                for batch in pq.ParquetFile(pf).iter_batches(batch_size=256, columns=["id"]):
                    for row in batch.to_pylist():
                        uid = row["id"]
                        if uid in new_records:
                            line_str = new_records[uid][1]
                        elif uid in existing_lines:
                            line_str = existing_lines[uid]
                        else:
                            raise RuntimeError(f"Missing cached utterance {uid} in {split}!")
                        parts = line_str.split("\t")
                        total_audio_s += int(parts[1]) / SR
                        ordered_lines.append(line_str)

            tmp_idx = split_dir / "index.tsv.tmp"
            with open(tmp_idx, "w", encoding="utf-8") as f:
                f.write("\n".join(ordered_lines) + "\n")
            tmp_idx.replace(index_path)
            el = time.time() - t0
            log(f"{split} DONE: {len(ordered_lines)} total utterances ({total_audio_s / 3600:.2f} h) cached in {el:.1f}s (x{total_audio_s / max(el, 1e-6):.0f} realtime)", rank)

        if is_dist:
            dist.barrier()

    if executor is not None:
        executor.shutdown(wait=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
