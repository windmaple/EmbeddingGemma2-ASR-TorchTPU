#!/usr/bin/env python3
"""Benchmark all official OpenAI Whisper model sizes on TorchTPU (8x Cloud TPU v5e).

Features:
  - Zero-recompile static-shape KV-cached greedy decoder (`[n_layers, B, n_heads, max_cache_len, head_dim]`)
    with device-tensor position indexing (`index_select` + masked cache writes) and block-16 early EOS exit.
  - Verifies exact output parity against Hugging Face `WhisperForConditionalGeneration.generate()` on CPU.
  - Evaluates all 4 LibriSpeech evaluation splits (`dev-clean`, `dev-other`, `test-clean`, `test-other`)
    plus long-form continuous audio (*The Adventure of the Speckled Band*, 54.29 min) across all 8 TPU chips.
  - Reports both Whisper `EnglishTextNormalizer` WER and Raw WER alongside wall-clock inference speed (`x` realtime).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import glob
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.distributed as dist

from beam_decode import load_whisper_normalizer, score_whisper_norm
from eval_wer import score as score_raw_librispeech
from tpu_utils import cleanup_distributed, configure_tpu_env, get_device, init_distributed, synchronize
from transcribe import load_audio_16k, normalize_for_wer

SR = 16000
MAX_SAMPLES_30S = 30 * SR  # 480,000 samples

# (repo_id, batch_size_per_chip, n_mels)
DEFAULT_MODELS: List[Tuple[str, int, int]] = [
    ("openai/whisper-tiny.en", 32, 80),
    ("openai/whisper-tiny", 32, 80),
    ("openai/whisper-base.en", 32, 80),
    ("openai/whisper-base", 32, 80),
    ("openai/whisper-small.en", 32, 80),
    ("openai/whisper-small", 32, 80),
    ("openai/whisper-medium.en", 16, 80),
    ("openai/whisper-medium", 16, 80),
    ("openai/whisper-large-v3-turbo", 16, 128),
    ("openai/whisper-large-v2", 8, 80),
    ("openai/whisper-large-v3", 8, 128),
]

SPLIT_PARQUET_SUBDIR = {
    "dev-clean": "clean/validation",
    "dev-other": "other/validation",
    "test-clean": "clean/test",
    "test-other": "other/test",
}


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def precision_context(device: torch.device, mode: str):
    if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision"):
        mapping = {
            "default": torch.tpu.Precision.DEFAULT,
            "high": torch.tpu.Precision.HIGH,
            "highest": torch.tpu.Precision.HIGHEST,
        }
        return torch.tpu.precision(mapping[mode])
    return contextlib.nullcontext()


def find_silence_cut(wav: np.ndarray, lo_sample: int, hi_sample: int, win: int = 1600, hop: int = 400) -> int:
    """Find the lowest-RMS 100ms frame inside [lo_sample, hi_sample] so chunks never split mid-word."""
    lo = max(0, min(lo_sample, len(wav) - win))
    hi = max(lo + hop, min(hi_sample, len(wav) - win))
    if hi <= lo + hop:
        return min(len(wav), hi_sample)
    best_pos = (lo + hi) // 2
    best_energy = float("inf")
    for pos in range(lo, hi, hop):
        seg = wav[pos : pos + win]
        e = float(np.dot(seg, seg))
        if e < best_energy:
            best_energy = e
            best_pos = pos + win // 2
    return best_pos


def split_utterance_30s(wav: np.ndarray) -> List[np.ndarray]:
    """Keep <=30s utterances intact; split >30s utterances cleanly at an interior silence valley."""
    if len(wav) <= MAX_SAMPLES_30S:
        return [wav]
    chunks: List[np.ndarray] = []
    rem = wav
    while len(rem) > MAX_SAMPLES_30S:
        if len(rem) <= int(36.5 * SR):
            lo = max(len(rem) - MAX_SAMPLES_30S + SR, int(12.0 * SR))
            hi = min(MAX_SAMPLES_30S - SR, int(24.0 * SR))
        else:
            lo = int(24.0 * SR)
            hi = MAX_SAMPLES_30S
        cut = find_silence_cut(rem, lo, hi)
        chunks.append(rem[:cut])
        rem = rem[cut:]
    if len(rem) > 0:
        rms = float(np.sqrt(np.mean(rem * rem))) if len(rem) > 0 else 0.0
        if len(rem) >= int(0.4 * SR) or rms > 5e-4:
            chunks.append(rem)
    return chunks


def split_longform_30s(wav: np.ndarray) -> List[np.ndarray]:
    """Split continuous multi-minute audio into <=30s chunks aligned to local silence valleys in [24s, 30s]."""
    chunks: List[np.ndarray] = []
    pos = 0
    n = len(wav)
    while pos < n:
        rem = n - pos
        if rem <= MAX_SAMPLES_30S:
            tail = wav[pos:]
            rms = float(np.sqrt(np.mean(tail * tail))) if len(tail) > 0 else 0.0
            if len(tail) >= int(0.5 * SR) and rms > 5e-4:
                chunks.append(tail)
            break
        cut_rel = find_silence_cut(wav[pos : pos + MAX_SAMPLES_30S], int(24.0 * SR), MAX_SAMPLES_30S)
        chunks.append(wav[pos : pos + cut_rel])
        pos += cut_rel
    return chunks


def load_librispeech_split_sharded(
    librispeech_dir: str,
    split_name: str,
    rank: int,
    world_size: int,
) -> Tuple[List[str], List[str], float, List[Tuple[str, int, np.ndarray]], np.ndarray, str]:
    """Load split metadata on all ranks and decode FLAC audio shards for `rank::world_size` sorted by size."""
    subdir = SPLIT_PARQUET_SUBDIR[split_name]
    files = sorted(glob.glob(os.path.join(librispeech_dir, subdir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {os.path.join(librispeech_dir, subdir)}")

    ids: List[str] = []
    texts: List[str] = []
    byte_lens: List[int] = []
    raw_bytes: List[bytes] = []

    for fp in files:
        tbl = pq.read_table(fp, columns=["id", "audio", "text"])
        col_id = tbl["id"].to_pylist()
        col_tx = tbl["text"].to_pylist()
        col_au = tbl["audio"].to_pylist()
        for u, t, a in zip(col_id, col_tx, col_au):
            b = a["bytes"]
            ids.append(str(u))
            texts.append(str(t))
            byte_lens.append(len(b))
            raw_bytes.append(b)

    # Sort globally by compressed byte size so `rank::world_size` gives balanced workloads & tight length buckets
    order = sorted(range(len(ids)), key=lambda i: byte_lens[i])
    local_order = order[rank::world_size]

    # Decode shortest utterance on rank 0 for fast CPU parity check
    sample_idx = order[min(8, len(order) - 1)]
    sample_wav, _ = sf.read(io.BytesIO(raw_bytes[sample_idx]), dtype="float32", always_2d=False)
    if sample_wav.ndim > 1:
        sample_wav = sample_wav.mean(axis=1)
    sample_wav = np.asarray(sample_wav, dtype=np.float32)
    sample_text = texts[sample_idx]

    local_chunks: List[Tuple[str, int, np.ndarray]] = []
    local_samples = 0
    for idx in local_order:
        w, sr = sf.read(io.BytesIO(raw_bytes[idx]), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        assert sr == SR, f"Expected {SR} Hz, got {sr}"
        w = np.asarray(w, dtype=np.float32)
        local_samples += len(w)
        parts = split_utterance_30s(w)
        for p_idx, part in enumerate(parts):
            local_chunks.append((ids[idx], p_idx, part))

    # Sort local chunks by exact waveform sample count so batches exit early on short utterances
    local_chunks.sort(key=lambda item: len(item[2]))
    return ids, texts, local_samples / SR, local_chunks, sample_wav, sample_text


def extract_mel_cache(
    n_mels: int,
    chunks: Sequence[Tuple[str, int, np.ndarray]],
) -> Tuple[List[Tuple[str, int, np.ndarray]], float]:
    """Extract float16 [n_mels, 3000] Whisper log-Mel features using official WhisperFeatureExtractor."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=n_mels, sampling_rate=SR)
    t0 = time.time()
    out: List[Tuple[str, int, np.ndarray]] = []
    bs = 64
    for a in range(0, len(chunks), bs):
        sub = chunks[a : a + bs]
        wavs = [c[2] for c in sub]
        feats = fe(wavs, sampling_rate=SR, return_tensors="np").input_features  # [b, n_mels, 3000] float32
        feats_f16 = feats.astype(np.float16)
        for j, (uid, pidx, _) in enumerate(sub):
            out.append((uid, pidx, feats_f16[j]))
    elapsed = time.time() - t0
    return out, elapsed


class WhisperDecoderStep(torch.nn.Module):
    """Single static-shape autoregressive Whisper decoder step over preallocated KV caches."""

    def __init__(self, hf_model, max_cache_len: int = 256):
        super().__init__()
        self.dec = hf_model.model.decoder
        self.proj_out = hf_model.proj_out
        self.n_layers = hf_model.config.decoder_layers
        self.n_heads = hf_model.config.decoder_attention_heads
        self.d_model = hf_model.config.d_model
        self.head_dim = self.d_model // self.n_heads
        self.max_cache_len = max_cache_len
        self.register_buffer("arange_cache", torch.arange(max_cache_len, dtype=torch.long), persistent=False)

    def forward(
        self,
        tokens_b1: torch.Tensor,
        pos_t: torch.Tensor,
        self_k: torch.Tensor,
        self_v: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = tokens_b1.shape[0]
        dtype = self_k.dtype
        pos_emb = self.dec.embed_positions.weight.index_select(0, pos_t).unsqueeze(0)
        h = self.dec.embed_tokens(tokens_b1) + pos_emb

        w_mask = (self.arange_cache == pos_t).view(1, 1, self.max_cache_len, 1).to(dtype)
        c_mask = torch.where(
            (self.arange_cache <= pos_t).view(1, 1, 1, self.max_cache_len),
            torch.zeros((), dtype=dtype, device=tokens_b1.device),
            torch.full((), -1e9, dtype=dtype, device=tokens_b1.device),
        )

        new_k_list = []
        new_v_list = []
        for li, layer in enumerate(self.dec.layers):
            # 1. Self-attention
            res = h
            h_norm = layer.self_attn_layer_norm(h)
            sa = layer.self_attn
            q = (sa.q_proj(h_norm) * sa.scaling).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
            ks = sa.k_proj(h_norm).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
            vs = sa.v_proj(h_norm).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

            kl = self_k[li] + ks * w_mask
            vl = self_v[li] + vs * w_mask
            new_k_list.append(kl)
            new_v_list.append(vl)

            aw = torch.matmul(q, kl.transpose(-1, -2)) + c_mask
            ap = torch.softmax(aw.float(), dim=-1).to(dtype)
            ao = torch.matmul(ap, vl).transpose(1, 2).reshape(B, 1, self.d_model)
            h = res + sa.out_proj(ao)

            # 2. Cross-attention (against precomputed encoder K, V)
            res = h
            h_norm = layer.encoder_attn_layer_norm(h)
            ca = layer.encoder_attn
            qc = (ca.q_proj(h_norm) * ca.scaling).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
            cw = torch.matmul(qc, cross_k[li].transpose(-1, -2))
            cp = torch.softmax(cw.float(), dim=-1).to(dtype)
            co = torch.matmul(cp, cross_v[li]).transpose(1, 2).reshape(B, 1, self.d_model)
            h = res + ca.out_proj(co)

            # 3. FFN
            res = h
            h_norm = layer.final_layer_norm(h)
            h = res + layer.fc2(layer.activation_fn(layer.fc1(h_norm)))

        h = self.dec.layer_norm(h)
        logits = self.proj_out(h[:, 0, :])
        return logits, torch.stack(new_k_list, dim=0), torch.stack(new_v_list, dim=0)


class WhisperStaticRunner:
    """Static-shape encoder + KV-cached greedy decoder wrapper for TorchTPU."""

    def __init__(
        self,
        hf_model,
        processor,
        device: torch.device,
        max_cache_len: int = 256,
        use_compile: bool = True,
    ):
        self.processor = processor
        self.device = device
        self.max_cache_len = max_cache_len
        self.cfg = hf_model.config
        self.n_layers = self.cfg.decoder_layers
        self.n_heads = self.cfg.decoder_attention_heads
        self.d_model = self.cfg.d_model
        self.head_dim = self.d_model // self.n_heads

        gcfg = hf_model.generation_config
        self.is_multilingual = bool(getattr(gcfg, "is_multilingual", False))
        self.eos_id = int(gcfg.eos_token_id)
        start_id = int(gcfg.decoder_start_token_id)
        no_ts_id = gcfg.no_timestamps_token_id

        if self.is_multilingual:
            forced = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
            self.prompt_ids = [start_id] + [int(tid) for _, tid in forced]
        else:
            self.prompt_ids = [start_id, int(no_ts_id)]

        vocab_size = self.cfg.vocab_size
        suppress_mask = torch.zeros((vocab_size,), dtype=torch.float32, device=device)
        if getattr(gcfg, "suppress_tokens", None):
            idx = torch.tensor(list(gcfg.suppress_tokens), dtype=torch.long, device=device)
            suppress_mask.index_fill_(0, idx, -1e9)
        if no_ts_id is not None:
            suppress_mask[int(no_ts_id) :] = -1e9

        begin_suppress_mask = suppress_mask.clone()
        if getattr(gcfg, "begin_suppress_tokens", None):
            bidx = torch.tensor(list(gcfg.begin_suppress_tokens), dtype=torch.long, device=device)
            begin_suppress_mask.index_fill_(0, bidx, -1e9)

        self.suppress_mask = suppress_mask
        self.begin_suppress_mask = begin_suppress_mask

        self.hf_model = hf_model.eval().to(device)
        self.step_mod = WhisperDecoderStep(self.hf_model, max_cache_len=max_cache_len).eval().to(device)

        if use_compile and device.type == "tpu":
            self.enc_fn = torch.compile(self.hf_model.model.encoder, backend="tpu", dynamic=False)
            self.step_fn = torch.compile(self.step_mod, backend="tpu", dynamic=False)
        else:
            self.enc_fn = self.hf_model.model.encoder
            self.step_fn = self.step_mod

    @torch.no_grad()
    def decode_batch(self, input_features: torch.Tensor) -> torch.Tensor:
        B = input_features.shape[0]
        dtype = input_features.dtype
        enc_out = self.enc_fn(input_features).last_hidden_state
        src_len = enc_out.shape[1]

        ck = torch.stack(
            [
                layer.encoder_attn.k_proj(enc_out).view(B, src_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
                for layer in self.hf_model.model.decoder.layers
            ],
            dim=0,
        )
        cv = torch.stack(
            [
                layer.encoder_attn.v_proj(enc_out).view(B, src_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
                for layer in self.hf_model.model.decoder.layers
            ],
            dim=0,
        )
        sk = torch.zeros((self.n_layers, B, self.n_heads, self.max_cache_len, self.head_dim), dtype=dtype, device=self.device)
        sv = torch.zeros((self.n_layers, B, self.n_heads, self.max_cache_len, self.head_dim), dtype=dtype, device=self.device)

        p_len = len(self.prompt_ids)
        logits = None
        for idx_p, pt in enumerate(self.prompt_ids):
            tok_in = torch.full((B, 1), pt, dtype=torch.long, device=self.device)
            pos_t = torch.tensor([idx_p], dtype=torch.long, device=self.device)
            logits, sk, sv = self.step_fn(tok_in, pos_t, sk, sv, ck, cv)

        next_tok = (logits + self.begin_suppress_mask.unsqueeze(0)).argmax(dim=-1)
        finished = next_tok == self.eos_id
        out_tokens = [next_tok]
        eos_vec = torch.full((B,), self.eos_id, dtype=torch.long, device=self.device)

        max_new = self.max_cache_len - p_len
        for s in range(1, max_new):
            pos_t = torch.tensor([p_len + s - 1], dtype=torch.long, device=self.device)
            logits, sk, sv = self.step_fn(next_tok.unsqueeze(1), pos_t, sk, sv, ck, cv)
            cand = (logits + self.suppress_mask.unsqueeze(0)).argmax(dim=-1)
            next_tok = torch.where(finished, eos_vec, cand)
            finished = finished | (next_tok == self.eos_id)
            out_tokens.append(next_tok)
            if s % 16 == 0 and bool(finished.all().item()):
                break

        return torch.stack(out_tokens, dim=1)

    def decode_mel_items(
        self,
        mel_items: Sequence[Tuple[str, int, np.ndarray]],
        batch_size: int,
        precision_mode: str = "high",
    ) -> List[Tuple[str, int, str]]:
        if not mel_items:
            return []
        n_mels = mel_items[0][2].shape[0]
        results: List[Tuple[str, int, str]] = []
        with precision_context(self.device, precision_mode):
            for a in range(0, len(mel_items), batch_size):
                sub = mel_items[a : a + batch_size]
                real_b = len(sub)
                buf = np.zeros((batch_size, n_mels, 3000), dtype=np.float32)
                for b in range(batch_size):
                    src = b if b < real_b else 0
                    buf[b] = sub[src][2].astype(np.float32)
                feats = torch.from_numpy(buf).to(self.device)
                tok_ids = self.decode_batch(feats)[:real_b].cpu()
                texts = self.processor.batch_decode(tok_ids, skip_special_tokens=True)
                for b in range(real_b):
                    results.append((sub[b][0], sub[b][1], texts[b].strip()))
        return results


def gather_chunk_hyps(
    local_out: List[Tuple[str, int, str]],
    tag: str,
    rank: int,
    world_size: int,
    is_dist: bool,
    tmp_dir: Path,
) -> Dict[str, str]:
    if not is_dist:
        by_id: Dict[str, List[Tuple[int, str]]] = {}
        for uid, pidx, txt in local_out:
            by_id.setdefault(uid, []).append((pidx, txt))
        return {
            uid: " ".join(t for _, t in sorted(parts, key=lambda x: x[0]) if t).strip()
            for uid, parts in by_id.items()
        }

    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_p = tmp_dir / f"whisper_{tag}_rank{rank:02d}.json"
    with open(part_p, "w", encoding="utf-8") as f:
        json.dump(local_out, f)
    dist.barrier()

    merged: Dict[str, str] = {}
    if rank == 0:
        by_id: Dict[str, List[Tuple[int, str]]] = {}
        for r in range(world_size):
            rp = tmp_dir / f"whisper_{tag}_rank{r:02d}.json"
            with open(rp, encoding="utf-8") as f:
                rows = json.load(f)
            for uid, pidx, txt in rows:
                by_id.setdefault(uid, []).append((int(pidx), str(txt)))
            rp.unlink(missing_ok=True)
        merged = {
            uid: " ".join(t for _, t in sorted(parts, key=lambda x: x[0]) if t).strip()
            for uid, parts in by_id.items()
        }
    dist.barrier()
    return merged


def main() -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Optional subset of HF model IDs to evaluate")
    ap.add_argument("--librispeech-dir", default=os.path.expanduser("~/data/hf_librispeech"))
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other", "test-clean", "test-other"])
    ap.add_argument("--longform-audio", default=os.path.expanduser("~/data/speckled_band/audio_16k.wav"))
    ap.add_argument("--longform-ref", default=os.path.expanduser("~/data/speckled_band/transcript.txt"))
    ap.add_argument("--out-json", default=os.path.expanduser("~/data/whisper_bench/whisper_tpu_benchmark.json"))
    ap.add_argument("--tmp-dir", default=os.path.expanduser("~/data/tmp/whisper_bench"))
    ap.add_argument("--max-cache-len", type=int, default=256)
    ap.add_argument("--precision", choices=["default", "high", "highest"], default="high")
    ap.add_argument("--no-compile", action="store_true", help="Disable torch.compile on TPU encoder/step")
    ap.add_argument("--resume", action="store_true", help="Skip models already present in --out-json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/data/hf_cache"))
    torch._dynamo.config.cache_size_limit = 64
    configure_tpu_env()
    is_dist, rank, world_size = init_distributed()
    device = get_device(args.device)
    tmp_dir = Path(args.tmp_dir)

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        wnorm = load_whisper_normalizer()
    else:
        wnorm = None

    if args.models:
        spec_map = {m: (m, b, n) for m, b, n in DEFAULT_MODELS}
        model_specs = [spec_map.get(m, (m, 16, 128 if "v3" in m else 80)) for m in args.models]
    else:
        model_specs = list(DEFAULT_MODELS)

    # 1. Load & shard audio across ranks once
    log(f"Loading & sharding LibriSpeech splits ({args.splits}) across {world_size}x {device}...", rank)
    split_meta: Dict[str, dict] = {}
    local_split_wavs: Dict[str, List[Tuple[str, int, np.ndarray]]] = {}
    sample_wav: Optional[np.ndarray] = None

    for sp in args.splits:
        ids, texts, local_sec, local_chunks, swav, _ = load_librispeech_split_sharded(
            args.librispeech_dir, sp, rank, world_size
        )
        if sample_wav is None:
            sample_wav = swav
        sec_tensor = torch.tensor([local_sec], dtype=torch.float64)
        if is_dist:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            sp_p = tmp_dir / f"dur_{sp}_r{rank:02d}.txt"
            sp_p.write_text(f"{local_sec:.6f}")
            dist.barrier()
            if rank == 0:
                total_sec = sum(float((tmp_dir / f"dur_{sp}_r{r:02d}.txt").read_text()) for r in range(world_size))
                for r in range(world_size):
                    (tmp_dir / f"dur_{sp}_r{r:02d}.txt").unlink(missing_ok=True)
            else:
                total_sec = local_sec
            dist.barrier()
        else:
            total_sec = float(sec_tensor.item())
        split_meta[sp] = {"ids": ids, "texts": texts, "audio_seconds": round(total_sec, 2), "audio_hours": round(total_sec / 3600.0, 3)}
        local_split_wavs[sp] = local_chunks
        log(f"  {sp}: {len(ids)} utts ({total_sec/3600.0:.2f} h), rank0_chunks={len(local_chunks)}", rank)

    # Load longform audio (Speckled Band)
    has_longform = bool(args.longform_audio and os.path.exists(args.longform_audio))
    longform_sec = 0.0
    longform_n_chunks = 0
    local_lf_wavs: List[Tuple[str, int, np.ndarray]] = []
    longform_ref_raw = ""
    if has_longform:
        lf_wav = load_audio_16k(args.longform_audio)
        longform_sec = len(lf_wav) / SR
        lf_chunks = split_longform_30s(lf_wav)
        longform_n_chunks = len(lf_chunks)
        for c_idx in range(rank, len(lf_chunks), world_size):
            local_lf_wavs.append(("longform", c_idx, lf_chunks[c_idx]))
        if rank == 0 and args.longform_ref and os.path.exists(args.longform_ref):
            with open(args.longform_ref, encoding="utf-8") as f:
                longform_ref_raw = f.read()
        log(f"  longform ({os.path.basename(args.longform_audio)}): {longform_sec/60.0:.2f} min -> {longform_n_chunks} silence-aligned chunks", rank)

    # 2. Pre-extract 80-mel and 128-mel features across all 8 processes and record exact CPU feature time
    needed_mels = sorted({nm for _, _, nm in model_specs})
    mel_caches: Dict[int, Dict[str, List[Tuple[str, int, np.ndarray]]]] = {}
    feat_times: Dict[int, Dict[str, float]] = {}

    for nm in needed_mels:
        mel_caches[nm] = {}
        feat_times[nm] = {}
        for sp in args.splits:
            if is_dist:
                dist.barrier()
            t0 = time.time()
            items_mel, _ = extract_mel_cache(nm, local_split_wavs[sp])
            if is_dist:
                dist.barrier()
            ft = round(time.time() - t0, 3)
            mel_caches[nm][sp] = items_mel
            feat_times[nm][sp] = ft
            log(f"  extracted {nm}-bin Mel features for {sp} in {ft:.2f}s across {world_size} ranks", rank)
        if has_longform:
            if is_dist:
                dist.barrier()
            t0 = time.time()
            lf_mel, _ = extract_mel_cache(nm, local_lf_wavs)
            if is_dist:
                dist.barrier()
            ft = round(time.time() - t0, 3)
            mel_caches[nm]["longform"] = lf_mel
            feat_times[nm]["longform"] = ft
            log(f"  extracted {nm}-bin Mel features for longform in {ft:.2f}s across {world_size} ranks", rank)

    # Free raw float32 waveform shards now that float16 mel spectrograms are cached
    del local_split_wavs
    del local_lf_wavs
    gc.collect()

    # Load existing partial results if resuming
    all_results: Dict[str, dict] = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, encoding="utf-8") as f:
                prev = json.load(f)
            all_results = prev.get("models", {})
        except Exception:
            all_results = {}
    if is_dist:
        dist.barrier()

    use_compile = not args.no_compile

    # 3. Benchmark each Whisper model size
    for model_id, batch_size, n_mels in model_specs:
        if args.resume and model_id in all_results:
            log(f"Skipping already completed {model_id} (--resume)", rank)
            continue
        log(f"============================================================", rank)
        log(f"Benchmarking {model_id} (batch/chip={batch_size}, global_batch={batch_size*world_size}, mels={n_mels})", rank)

        # Ensure Hub files are cached on rank 0 before all 8 ranks load from local disk
        if rank == 0:
            try:
                _ = WhisperProcessor.from_pretrained(model_id, local_files_only=True)
                cpu_model = WhisperForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32, local_files_only=True
                ).eval()
            except Exception:
                _ = WhisperProcessor.from_pretrained(model_id)
                cpu_model = WhisperForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32
                ).eval()
        if is_dist:
            dist.barrier()

        processor = WhisperProcessor.from_pretrained(model_id, local_files_only=True)
        if rank != 0:
            cpu_model = WhisperForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float32, local_files_only=True
            ).eval()

        unique_params = sum(p.numel() for p in cpu_model.parameters())
        enc_params = sum(p.numel() for p in cpu_model.model.encoder.parameters())
        dec_params = sum(p.numel() for p in cpu_model.model.decoder.parameters())

        # Quick CPU vs TPU exact-output parity check on rank 0 before moving weights to TPU
        cpu_ref_text = ""
        sample_feat_np = None
        if rank == 0 and sample_wav is not None:
            s_feat = processor([sample_wav], sampling_rate=SR, return_tensors="pt").input_features
            sample_feat_np = s_feat.numpy()
            if bool(getattr(cpu_model.generation_config, "is_multilingual", False)):
                c_ids = cpu_model.generate(s_feat, language="en", task="transcribe", return_timestamps=False, do_sample=False, num_beams=1)
            else:
                c_ids = cpu_model.generate(s_feat, return_timestamps=False, do_sample=False, num_beams=1)
            cpu_ref_text = processor.batch_decode(c_ids, skip_special_tokens=True)[0].strip()

        runner = WhisperStaticRunner(
            cpu_model,
            processor,
            device=device,
            max_cache_len=args.max_cache_len,
            use_compile=use_compile,
        )
        del cpu_model
        gc.collect()

        # Warmup / XLA compile static batch on all ranks
        if is_dist:
            dist.barrier()
        t_w0 = time.time()
        dummy = np.zeros((batch_size, n_mels, 3000), dtype=np.float32)
        if rank == 0 and sample_feat_np is not None:
            dummy[0] = sample_feat_np[0]
        with precision_context(device, args.precision):
            warm_ids = runner.decode_batch(torch.from_numpy(dummy).to(device))
        synchronize()
        if is_dist:
            dist.barrier()
        warmup_s = round(time.time() - t_w0, 2)

        parity_match = None
        if rank == 0 and sample_feat_np is not None:
            tpu_sample_text = processor.batch_decode(warm_ids[:1].cpu(), skip_special_tokens=True)[0].strip()
            parity_match = (tpu_sample_text == cpu_ref_text)
            log(
                f"  warmup/compile={warmup_s:.2f}s | params={unique_params:,} | CPU vs TPU parity={parity_match} ({tpu_sample_text!r})",
                rank,
            )

        model_split_res: Dict[str, dict] = {}
        for sp in args.splits:
            if is_dist:
                dist.barrier()
            t_s0 = time.time()
            local_out = runner.decode_mel_items(mel_caches[n_mels][sp], batch_size=batch_size, precision_mode=args.precision)
            synchronize()
            hyps_map = gather_chunk_hyps(local_out, f"{model_id.split('/')[-1]}_{sp}", rank, world_size, is_dist, tmp_dir)
            infer_s = round(time.time() - t_s0, 2)

            if rank == 0:
                refs = split_meta[sp]["texts"]
                ids = split_meta[sp]["ids"]
                hyp_list = [hyps_map[u] for u in ids]
                _, raw_s = score_raw_librispeech(refs, hyp_list)
                wn_wer = round(score_whisper_norm(refs, hyp_list, wnorm), 5)
                ft_s = feat_times[n_mels][sp]
                tot_s = round(ft_s + infer_s, 2)
                aud_s = split_meta[sp]["audio_seconds"]
                raw_s["wer_whisper_norm"] = wn_wer
                raw_s["feature_s"] = ft_s
                raw_s["tpu_infer_s"] = infer_s
                raw_s["total_s"] = tot_s
                raw_s["audio_seconds"] = aud_s
                raw_s["x_realtime_e2e"] = round(aud_s / max(tot_s, 1e-6), 1)
                raw_s["x_realtime_tpu"] = round(aud_s / max(infer_s, 1e-6), 1)
                model_split_res[sp] = raw_s
                log(
                    f"  {sp:10s}: Whisper-norm WER={100*wn_wer:5.2f}% | Raw WER={100*raw_s['wer']:5.2f}% "
                    f"| TPU={infer_s:5.2f}s + Mel={ft_s:.2f}s = {tot_s:5.2f}s ({raw_s['x_realtime_e2e']:.1f}x RT)",
                    rank,
                )

        lf_res: Optional[dict] = None
        if has_longform:
            if is_dist:
                dist.barrier()
            t_l0 = time.time()
            local_lf_out = runner.decode_mel_items(
                mel_caches[n_mels]["longform"], batch_size=batch_size, precision_mode=args.precision
            )
            synchronize()
            lf_map = gather_chunk_hyps(local_lf_out, f"{model_id.split('/')[-1]}_lf", rank, world_size, is_dist, tmp_dir)
            lf_infer_s = round(time.time() - t_l0, 2)

            if rank == 0:
                lf_transcript = lf_map.get("longform", "")
                ft_s = feat_times[n_mels]["longform"]
                tot_s = round(ft_s + lf_infer_s, 2)
                lf_res = {
                    "audio_seconds": round(longform_sec, 2),
                    "audio_minutes": round(longform_sec / 60.0, 2),
                    "chunks": longform_n_chunks,
                    "feature_s": ft_s,
                    "tpu_infer_s": lf_infer_s,
                    "total_s": tot_s,
                    "x_realtime_e2e": round(longform_sec / max(tot_s, 1e-6), 1),
                    "x_realtime_tpu": round(longform_sec / max(lf_infer_s, 1e-6), 1),
                    "word_count": len(lf_transcript.split()),
                }
                if longform_ref_raw:
                    import jiwer

                    ref_n = normalize_for_wer(longform_ref_raw)
                    hyp_n = normalize_for_wer(lf_transcript)
                    o_lf = jiwer.process_words(ref_n, hyp_n)
                    lf_res["wer_raw"] = round(float(o_lf.wer), 5)
                    lf_res["wer_whisper_norm"] = round(score_whisper_norm([longform_ref_raw], [lf_transcript], wnorm), 5)
                    lf_res["substitutions"] = int(o_lf.substitutions)
                    lf_res["deletions"] = int(o_lf.deletions)
                    lf_res["insertions"] = int(o_lf.insertions)
                    lf_res["ref_words"] = int(o_lf.substitutions + o_lf.deletions + o_lf.hits)
                    log(
                        f"  speckled  : Whisper-norm WER={100*lf_res['wer_whisper_norm']:5.2f}% | Raw WER={100*lf_res['wer_raw']:5.2f}% "
                        f"| TPU={lf_infer_s:5.2f}s + Mel={ft_s:.2f}s = {tot_s:5.2f}s ({lf_res['x_realtime_e2e']:.1f}x RT)",
                        rank,
                    )

        if rank == 0:
            all_results[model_id] = {
                "model_id": model_id,
                "short_name": model_id.split("/")[-1],
                "params": int(unique_params),
                "encoder_params": int(enc_params),
                "decoder_params": int(dec_params),
                "encoder_layers": int(runner.cfg.encoder_layers),
                "decoder_layers": int(runner.cfg.decoder_layers),
                "d_model": int(runner.d_model),
                "n_mels": int(n_mels),
                "is_multilingual": runner.is_multilingual,
                "batch_size_per_chip": batch_size,
                "global_batch_size": batch_size * world_size,
                "world_size": world_size,
                "precision": args.precision,
                "compiled": use_compile,
                "warmup_compile_s": warmup_s,
                "cpu_vs_tpu_sample_parity": parity_match,
                "splits": model_split_res,
                "speckled_band": lf_res,
            }
            payload = {
                "benchmark": "OpenAI Whisper All Sizes on TorchTPU (8x Cloud TPU v5e)",
                "device": str(device),
                "world_size": world_size,
                "precision": args.precision,
                "max_cache_len": args.max_cache_len,
                "compiled": use_compile,
                "models": all_results,
            }
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            log(f"  saved incremental benchmark report -> {args.out_json}", rank)

        del runner
        torch._dynamo.reset()
        gc.collect()
        if is_dist:
            dist.barrier()

    log("ALL WHISPER BENCHMARKS COMPLETE!", rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
