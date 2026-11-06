#!/usr/bin/env python3
"""End-to-end Speech Recognition with EmbeddingGemma 2 + Reader on TorchTPU.

Supports arbitrary-length audio (.wav, .flac, .mp3) from 1 second to multi-hour recordings:
  1. Loads audio -> mono 16 kHz float32 (via soundfile + soxr, or ffmpeg fallback).
  2. Extracts EmbeddingGemma 2 audio soft tokens [T_total, 512] on TPU using static-shape 30s causal-overlap windows
     (750 tokens/window, 96 left-overlap tokens, 1 right-extra token dropped => exact continuous-equivalent stream).
  3. Runs bidirectional RoPE Reader + 200,000-word cosine-similarity CTC decoding in static 768-token windows
     with 64-token left/right overlap, stitches frame-level argmax IDs seamlessly, and collapses CTC blanks/repeats.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from cache_soft_tokens import bucket_frame_len, extract_mel_features, load_encoder, soft_tokens_batch_static
from decode import load_reader_dir
from model import collapse, greedy_labels_static, topk_logprobs_static
from tpu_utils import cleanup_distributed, configure_tpu_env, get_device, init_distributed, patch_hf_for_tpu, synchronize

SR = 16000
SAMPLES_PER_TOKEN = 640  # 40 ms @ 16 kHz = 25 Hz soft tokens


def load_audio_16k(path: str) -> np.ndarray:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != SR:
            import soxr

            mono = soxr.resample(mono, sr, SR, quality="HQ")
        return np.asarray(mono, dtype=np.float32)
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            subprocess.run(
                [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "wav", tmp.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            wav, _ = sf.read(tmp.name, dtype="float32", always_2d=False)
            return np.asarray(wav, dtype=np.float32)


def slice_audio_windows(
    wav: np.ndarray,
    win_tokens: int = 750,
    left_overlap: int = 96,
    right_extra: int = 1,
) -> List[Tuple[np.ndarray, int, int]]:
    """Slice waveform into 30s windows returning (wav_slice, trim_left_tokens, trim_right_tokens).

    Stitched slices cover every soft token continuously without boundary artifacts from causal audio encoding.
    """
    win_samples = win_tokens * SAMPLES_PER_TOKEN
    if len(wav) <= win_samples:
        return [(wav, 0, 0)]

    windows: List[Tuple[np.ndarray, int, int]] = []
    # Window 0 starts at sample 0, drops right_extra tokens -> keeps (win_tokens - right_extra) tokens
    next_valid_tok = win_tokens - right_extra
    windows.append((wav[:win_samples], 0, right_extra))

    step_tokens = win_tokens - left_overlap - right_extra
    assert step_tokens > 0
    while True:
        start_tok = next_valid_tok - left_overlap
        start_sample = start_tok * SAMPLES_PER_TOKEN
        end_sample = start_sample + win_samples
        if end_sample >= len(wav):
            windows.append((wav[start_sample:], left_overlap, 0))
            break
        windows.append((wav[start_sample:end_sample], left_overlap, right_extra))
        next_valid_tok += step_tokens

    return windows


@torch.no_grad()
def extract_longform_soft_tokens(
    wav: np.ndarray,
    fe,
    encoder_model,
    device: torch.device,
    batch_size: int = 16,
    rank: int = 0,
    world_size: int = 1,
    tmp_dir: Optional[str] = None,
) -> np.ndarray:
    if tmp_dir is None:
        tmp_dir = os.path.expanduser("~/data/tmp/transcribe_shards")
    windows = slice_audio_windows(wav)
    local_indices = list(range(len(windows)))[rank::world_size]

    mels = [extract_mel_features(fe, windows[i][0]) for i in local_indices]
    ns = [len(windows[i][0]) for i in local_indices]

    local_pieces: dict[int, np.ndarray] = {}
    fixed_bs = min(batch_size, max(1, len(local_indices)))
    f_step = 3072 if len(windows) > 1 else 512
    for a in range(0, len(local_indices), batch_size):
        sub_mels = mels[a : a + batch_size]
        sub_ns = ns[a : a + batch_size]
        sub_ids = local_indices[a : a + batch_size]
        soft_list = soft_tokens_batch_static(
            encoder_model,
            sub_mels,
            sub_ns,
            device,
            fixed_batch_size=fixed_bs,
            frame_step=f_step,
        )
        for win_idx, s in zip(sub_ids, soft_list):
            _, trim_l, trim_r = windows[win_idx]
            end_idx = s.shape[0] - trim_r if trim_r > 0 else s.shape[0]
            local_pieces[win_idx] = s[trim_l:end_idx]

    synchronize()
    if world_size > 1:
        import torch.distributed as dist

        td = Path(tmp_dir)
        td.mkdir(parents=True, exist_ok=True)
        np.savez(str(td / f"soft_rank{rank:02d}.npz"), **{str(k): v for k, v in local_pieces.items()})
        dist.barrier()
        if rank == 0:
            all_pieces: dict[int, np.ndarray] = {}
            for r in range(world_size):
                p = td / f"soft_rank{r:02d}.npz"
                with np.load(str(p)) as z:
                    for k in z.files:
                        all_pieces[int(k)] = z[k]
                p.unlink(missing_ok=True)
            ordered = [all_pieces[i] for i in range(len(windows))]
            stitched = np.concatenate(ordered, axis=0)
        else:
            stitched = np.zeros((0, 512), dtype=np.float32)
        dist.barrier()
        return stitched
    else:
        return np.concatenate([local_pieces[i] for i in range(len(windows))], axis=0)


@torch.no_grad()
def decode_soft_tokens_longform(
    soft: np.ndarray,
    reader_model,
    prep_table,
    vocab: List[str],
    device: torch.device,
    win_size: int = 750,
    overlap: int = 64,
    pad_bucket: int = 768,
    batch_size: int = 32,
    lm=None,
    lm_vocab: Optional[List[str]] = None,
    beam_size: int = 32,
    lm_alpha: float = 0.5,
    lm_beta: float = 1.0,
    top_k: int = 64,
) -> str:
    T_total = soft.shape[0]
    if T_total == 0:
        return ""

    # Slice into bidirectional overlapping windows of <=750 tokens (matching LibriSpeech training length)
    slices: List[Tuple[np.ndarray, int, int]] = []
    if T_total <= win_size:
        slices.append((soft, 0, T_total))
    else:
        step = win_size - 2 * overlap
        pos = 0
        while pos < T_total:
            end = min(T_total, pos + win_size)
            trim_l = 0 if pos == 0 else overlap
            trim_r = 0 if end == T_total else overlap
            slices.append((soft[pos:end], trim_l, (end - pos) - trim_r))
            if end == T_total:
                break
            pos += step

    frame_labels: List[int] = []
    top_lp_parts: List[np.ndarray] = []
    top_id_parts: List[np.ndarray] = []
    blank_lp_parts: List[np.ndarray] = []

    eff_bs = min(batch_size, len(slices))
    prec_ctx = (
        torch.tpu.precision(torch.tpu.Precision.HIGHEST)
        if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")
        else contextlib.nullcontext()
    )
    with prec_ctx:
        for a in range(0, len(slices), eff_bs):
            chunk = slices[a : a + eff_bs]
            real_b = len(chunk)
            x = torch.zeros((eff_bs, pad_bucket, soft.shape[1]), dtype=torch.float32)
            lens = [1] * eff_bs
            for b in range(eff_bs):
                src = b if b < real_b else 0
                arr = chunk[src][0]
                x[b, : arr.shape[0]] = torch.from_numpy(arr)
                lens[b] = arr.shape[0]
            h, blank = reader_model(x.to(device), torch.tensor(lens, dtype=torch.long, device=device))
            if lm is None:
                lbl = greedy_labels_static(h, blank, prep_table, reader_model.scale, chunk=1024).cpu().numpy()
                for b in range(real_b):
                    _, keep_s, keep_e = chunk[b]
                    frame_labels.extend(lbl[b, keep_s:keep_e].tolist())
            else:
                tlp_d, tid_d, blp_d = topk_logprobs_static(
                    h, blank, prep_table, reader_model.scale, top_k=top_k, chunk=1024
                )
                tlp_np = tlp_d.cpu().numpy()
                tid_np = tid_d.cpu().numpy()
                blp_np = blp_d.cpu().numpy()
                for b in range(real_b):
                    _, keep_s, keep_e = chunk[b]
                    top_lp_parts.append(tlp_np[b, keep_s:keep_e])
                    top_id_parts.append(tid_np[b, keep_s:keep_e])
                    blank_lp_parts.append(blp_np[b, keep_s:keep_e])

    if lm is None:
        word_ids = collapse(frame_labels, prep_table.n_vocab)
        return " ".join(vocab[i] for i in word_ids)

    from beam_decode import ctc_prefix_beam_search

    stitched_tlp = np.concatenate(top_lp_parts, axis=0)
    stitched_tid = np.concatenate(top_id_parts, axis=0)
    stitched_blp = np.concatenate(blank_lp_parts, axis=0)
    return ctc_prefix_beam_search(
        stitched_tlp,
        stitched_tid,
        stitched_blp,
        lm,
        lm_vocab,
        vocab,
        beam_size=beam_size,
        alpha=lm_alpha,
        beta=lm_beta,
        score_eos=(T_total <= win_size),
    )


def normalize_for_wer(text: str) -> str:
    import re

    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)
    return " ".join(t.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to .wav / .flac / .mp3 file")
    ap.add_argument("--reader-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--model-dir", default=os.path.expanduser("~/data/embeddinggemma-2"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reference", default=None, help="Optional reference transcript file to compute WER")
    ap.add_argument("--lm", default=None, help="Optional KenLM .bin/.arpa path for CTC prefix beam decoding")
    ap.add_argument("--beam-size", type=int, default=32)
    ap.add_argument("--lm-alpha", type=float, default=0.55)
    ap.add_argument("--lm-beta", type=float, default=2.0)
    ap.add_argument("--warmup", action="store_true", help="Run a warm-up pass before timing")
    ap.add_argument("--out-json", default=None, help="Optional path to save JSON transcription & speed report")
    args = ap.parse_args()

    configure_tpu_env()
    patch_hf_for_tpu()
    is_dist, rank, world_size = init_distributed()
    device = get_device(args.device)

    fe, encoder_model, _ = load_encoder(args.model_dir, device)
    reader_model, prep_table, vocab, _ = load_reader_dir(args.reader_dir, device)
    lm, lm_vocab = None, None
    if rank == 0 and args.lm:
        import kenlm
        from beam_decode import build_lm_vocab

        lm = kenlm.Model(args.lm)
        lm_vocab = build_lm_vocab(lm, vocab)

    wav = load_audio_16k(args.audio)
    audio_sec = len(wav) / SR

    if args.warmup:
        _w_soft = extract_longform_soft_tokens(wav, fe, encoder_model, device, batch_size=16, rank=rank, world_size=world_size)
        if rank == 0:
            _ = decode_soft_tokens_longform(
                _w_soft, reader_model, prep_table, vocab, device,
                lm=lm, lm_vocab=lm_vocab, beam_size=args.beam_size, lm_alpha=args.lm_alpha, lm_beta=args.lm_beta,
            )
        synchronize()
        if is_dist:
            import torch.distributed as dist

            dist.barrier()

    t0 = time.time()
    soft = extract_longform_soft_tokens(wav, fe, encoder_model, device, batch_size=16, rank=rank, world_size=world_size)
    t_enc = time.time() - t0

    if rank == 0:
        t1 = time.time()
        transcript = decode_soft_tokens_longform(
            soft, reader_model, prep_table, vocab, device,
            lm=lm, lm_vocab=lm_vocab, beam_size=args.beam_size, lm_alpha=args.lm_alpha, lm_beta=args.lm_beta,
        )
        synchronize()
        t_dec = time.time() - t1
        t_total = t_enc + t_dec
        x_rt = audio_sec / max(t_total, 1e-6)

        res = {
            "audio_path": os.path.abspath(args.audio),
            "audio_seconds": round(audio_sec, 2),
            "audio_minutes": round(audio_sec / 60.0, 2),
            "soft_tokens": int(soft.shape[0]),
            "device": str(device),
            "world_size": world_size,
            "decode_mode": f"beam={args.beam_size} + {os.path.basename(args.lm)} (alpha={args.lm_alpha}, beta={args.lm_beta})" if args.lm else "greedy",
            "encode_seconds": round(t_enc, 3),
            "decode_seconds": round(t_dec, 3),
            "total_seconds": round(t_total, 3),
            "x_realtime": round(x_rt, 1),
            "word_count": len(transcript.split()),
            "transcript_preview": transcript[:500],
        }

        if args.reference and os.path.exists(args.reference):
            import jiwer

            with open(args.reference, encoding="utf-8") as f:
                ref_raw = f.read()
            ref_norm = normalize_for_wer(ref_raw)
            hyp_norm = normalize_for_wer(transcript)
            o = jiwer.process_words(ref_norm, hyp_norm)
            res["wer"] = round(float(o.wer), 5)
            res["substitutions"] = int(o.substitutions)
            res["deletions"] = int(o.deletions)
            res["insertions"] = int(o.insertions)
            res["ref_words"] = int(o.substitutions + o.deletions + o.hits)
            try:
                from beam_decode import load_whisper_normalizer, score_whisper_norm

                wnorm = load_whisper_normalizer()
                res["wer_whisper_norm"] = round(score_whisper_norm([ref_raw], [transcript], wnorm), 5)
            except Exception:
                pass

        print(f"=== Transcription ({audio_sec/60:.2f} min audio in {t_total:.2f}s = {x_rt:.1f}x realtime on {world_size}x {device}) ===")
        if "wer" in res:
            wn_s = f", Whisper-norm: {100 * res['wer_whisper_norm']:.2f}%" if "wer_whisper_norm" in res else ""
            print(f"Indicative WER vs reference: {100 * res['wer']:.2f}%{wn_s} ({res['ref_words']} ref words)")
        print(transcript[:1200] + (" ..." if len(transcript) > 1200 else ""))

        if args.out_json:
            res["transcript"] = transcript
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved report -> {args.out_json}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
