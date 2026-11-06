#!/usr/bin/env python3
"""200k-word CTC Prefix Beam Search + KenLM n-gram rescorer on TorchTPU + multi-core CPU.

Architecture:
  1. Static-shape TPU extraction (`topk_logprobs_static` in `Precision.HIGHEST`) computes exact
     200,001-way log-normalized blank log-probs `[B, T]` and top-K (`K=64`) word log-probs/IDs `[B, T, K]`.
  2. Exact word-level CTC prefix beam search with an O(1) integer prefix trie `(parent_id, word_id) -> node_id`
     and KenLM state memoization scores candidate word extensions in parallel across CPU workers.
  3. Supports dev-set `(alpha, beta)` tuning, full 4-split LibriSpeech evaluation (raw WER + Whisper
     `EnglishTextNormalizer` WER + homophone analysis), and continuous 1-hour+ long-form decoding.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import glob
import heapq
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NEG_INF = -1e30
LN10 = math.log(10.0)
ENGLISH_JSON_URL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/normalizers/english.json"

# Worker-global state for zero-copy fork/spawn multiprocessing (kept 100% torch/libtpu-free!)
_WORKER_LM = None
_WORKER_LM_VOCAB: Optional[List[str]] = None
_WORKER_VOCAB: Optional[List[str]] = None


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _logaddexp(a: float, b: float) -> float:
    if a <= -1e20:
        return b
    if b <= -1e20:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def build_lm_vocab(lm, vocab: Sequence[str]) -> List[str]:
    """Auto-detect whether the KenLM binary uses UPPERCASE (OpenSLR LibriSpeech) or lowercase words."""
    n_upper = sum(1 for w in vocab[:200] if (w.upper() in lm))
    n_lower = sum(1 for w in vocab[:200] if (w.lower() in lm))
    use_upper = n_upper >= n_lower
    return [w.upper() if use_upper else w for w in vocab]


def ctc_prefix_beam_search(
    top_lp: np.ndarray,
    top_ids: np.ndarray,
    blank_lp: np.ndarray,
    lm,
    lm_vocab: Sequence[str],
    vocab: Sequence[str],
    beam_size: int = 32,
    alpha: float = 0.5,
    beta: float = 0.5,
    token_prune: float = 12.0,
    score_eos: bool = True,
) -> str:
    """Exact word-level CTC prefix beam search with O(1) prefix trie and KenLM state caching."""
    import kenlm

    T, K = top_lp.shape
    if T == 0:
        return ""

    init_state = kenlm.State()
    lm.BeginSentenceWrite(init_state)

    node_parent: List[int] = [-1]
    node_word: List[int] = [-1]
    node_len: List[int] = [0]
    node_lm: List[float] = [0.0]
    node_state: List[ object ] = [init_state]
    children: Dict[Tuple[int, int], int] = {}

    # nid -> (p_blank, p_non_blank)
    beams: Dict[int, Tuple[float, float]] = {0: (0.0, NEG_INF)}

    for t in range(T):
        b_lp = float(blank_lp[t])
        row_lp = top_lp[t]
        row_ids = top_ids[t]
        first_lp = float(row_lp[0])
        best_lp = b_lp if b_lp > first_lp else first_lp
        cutoff = best_lp - token_prune

        # Fast path: pure blank frame where no word is within `token_prune` nats of blank
        if b_lp > -1e-4 and first_lp < cutoff:
            floor_lp = float(row_lp[-1])
            for nid, (p_b, p_nb) in list(beams.items()):
                p_tot = _logaddexp(p_b, p_nb)
                nb_pb = p_tot + b_lp
                nb_pnb = (p_nb + floor_lp) if (nid != 0 and p_nb > -1e20) else NEG_INF
                beams[nid] = (nb_pb, nb_pnb)
            continue

        cands: List[Tuple[int, float]] = []
        for j in range(K):
            v = float(row_lp[j])
            if v < cutoff:
                break
            cands.append((int(row_ids[j]), v))

        row_lookup = dict(zip(row_ids.tolist(), row_lp.tolist()))
        floor_lp = float(row_lp[-1])
        next_beams: Dict[int, Tuple[float, float]] = {}

        for nid, (p_b, p_nb) in beams.items():
            p_tot = _logaddexp(p_b, p_nb)
            nb_pb = p_tot + b_lp
            w_last = node_word[nid]
            nb_pnb = (p_nb + row_lookup.get(w_last, floor_lp)) if (w_last >= 0 and p_nb > -1e20) else NEG_INF

            prev = next_beams.get(nid)
            if prev is None:
                next_beams[nid] = (nb_pb, nb_pnb)
            else:
                next_beams[nid] = (_logaddexp(prev[0], nb_pb), _logaddexp(prev[1], nb_pnb))

            for w, w_lp in cands:
                # Repeating the same word as a *new* token (`w_last <b> w_last`) requires blank at t-1 (`p_b`);
                # transitioning to a different word `w != w_last` can come from either `p_b` or `p_nb` (`p_tot`).
                ext_pnb = (p_b + w_lp) if (w == w_last) else (p_tot + w_lp)
                if ext_pnb <= -1e20:
                    continue
                key = (nid, w)
                cid = children.get(key)
                if cid is None:
                    cid = len(node_parent)
                    out_st = kenlm.State()
                    step_lm = lm.BaseScore(node_state[nid], lm_vocab[w], out_st) * LN10
                    node_parent.append(nid)
                    node_word.append(w)
                    node_len.append(node_len[nid] + 1)
                    node_lm.append(node_lm[nid] + step_lm)
                    node_state.append(out_st)
                    children[key] = cid

                c_prev = next_beams.get(cid)
                if c_prev is None:
                    next_beams[cid] = (NEG_INF, ext_pnb)
                else:
                    next_beams[cid] = (c_prev[0], _logaddexp(c_prev[1], ext_pnb))

        if len(next_beams) > beam_size:
            top_items = heapq.nlargest(
                beam_size,
                next_beams.items(),
                key=lambda kv: _logaddexp(kv[1][0], kv[1][1]) + alpha * node_lm[kv[0]] + beta * node_len[kv[0]],
            )
            beams = dict(top_items)
        else:
            beams = next_beams

    best_nid = 0
    best_score = NEG_INF
    eos_st = kenlm.State()
    for nid, (p_b, p_nb) in beams.items():
        ac = _logaddexp(p_b, p_nb)
        lm_tot = node_lm[nid]
        if score_eos:
            lm_tot += lm.BaseScore(node_state[nid], "</s>", eos_st) * LN10
        tot = ac + alpha * lm_tot + beta * node_len[nid]
        if tot > best_score:
            best_score = tot
            best_nid = nid

    words_rev: List[str] = []
    cur = best_nid
    while cur > 0:
        words_rev.append(vocab[node_word[cur]])
        cur = node_parent[cur]
    words_rev.reverse()
    return " ".join(words_rev)


def _init_worker(lm_path: str, vocab: List[str]) -> None:
    import kenlm

    global _WORKER_LM, _WORKER_LM_VOCAB, _WORKER_VOCAB
    _WORKER_LM = kenlm.Model(lm_path)
    _WORKER_VOCAB = vocab
    _WORKER_LM_VOCAB = build_lm_vocab(_WORKER_LM, vocab)


def _decode_chunk_worker(args) -> Dict[str, str]:
    chunk_items, beam_size, alpha, beta, token_prune, score_eos = args
    out: Dict[str, str] = {}
    for utt_id, top_lp, top_ids, blank_lp in chunk_items:
        out[utt_id] = ctc_prefix_beam_search(
            top_lp,
            top_ids,
            blank_lp,
            _WORKER_LM,
            _WORKER_LM_VOCAB,
            _WORKER_VOCAB,
            beam_size=beam_size,
            alpha=alpha,
            beta=beta,
            token_prune=token_prune,
            score_eos=score_eos,
        )
    return out


def extract_split_topk(
    model,
    split,
    prep_table,
    device,
    top_k: int = 64,
    batch_size: int = 32,
    items: Optional[Sequence[int]] = None,
    pad_step: int = 256,
    chunk: int = 1024,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run Reader + top-K 200k-word log-probs on TPU with static shape bucketing."""
    import torch
    import data as D
    from model import topk_logprobs_static

    items_list = list(range(len(split))) if items is None else list(items)
    order = sorted(items_list, key=lambda i: int(split.n_tokens[i]))
    prec_ctx = (
        torch.tpu.precision(torch.tpu.Precision.HIGHEST)
        if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")
        else contextlib.nullcontext()
    )

    extracted: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with torch.no_grad(), prec_ctx:
        for a in range(0, len(order), batch_size):
            idx = order[a : a + batch_size]
            real_b = len(idx)
            xs = [split.tokens(i) for i in idx]
            real_lens = [int(x.shape[0]) for x in xs]
            T = D.choose_bucket(max(real_lens), step=pad_step)
            x = torch.zeros((batch_size, T, xs[0].shape[1]), dtype=torch.float32)
            lens_pad = [1] * batch_size
            for b in range(batch_size):
                src = b if b < real_b else 0
                arr = np.asarray(xs[src], dtype=np.float32)
                x[b, : arr.shape[0]] = torch.from_numpy(arr)
                lens_pad[b] = real_lens[src]

            lengths = torch.tensor(lens_pad, dtype=torch.long, device=device)
            h, blank = model(x.to(device), lengths)
            top_lp_dev, top_ids_dev, blank_lp_dev = topk_logprobs_static(
                h, blank, prep_table, model.scale, top_k=top_k, chunk=chunk
            )
            top_lp_np = top_lp_dev.cpu().numpy()
            top_ids_np = top_ids_dev.cpu().numpy()
            blank_lp_np = blank_lp_dev.cpu().numpy()
            for b in range(real_b):
                n = real_lens[b]
                utt_id = split.ids[idx[b]]
                extracted[utt_id] = (
                    top_lp_np[b, :n].astype(np.float16),
                    top_ids_np[b, :n].astype(np.int32),
                    blank_lp_np[b, :n].astype(np.float32),
                )
    return extracted


def load_whisper_normalizer(cache_dir: Optional[str] = None):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    resolved = os.path.expanduser(cache_dir or "~/data/lm")
    os.makedirs(resolved, exist_ok=True)
    p = os.path.join(resolved, "whisper_english.json")
    if not os.path.exists(p):
        urllib.request.urlretrieve(ENGLISH_JSON_URL, p)
    with open(p, encoding="utf-8") as f:
        spelling = json.load(f)
    return EnglishTextNormalizer(spelling)


def score_whisper_norm(refs: List[str], hyps: List[str], wnorm) -> float:
    import jiwer

    r_n = [wnorm(r) for r in refs]
    h_n = [wnorm(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(r_n, h_n) if r.strip()]
    return float(jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]))


def decode_parallel(
    pool: concurrent.futures.ProcessPoolExecutor,
    utterances: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    beam_size: int,
    alpha: float,
    beta: float,
    token_prune: float = 12.0,
    score_eos: bool = True,
    n_chunks: int = 64,
) -> Dict[str, str]:
    if not utterances:
        return {}
    step = max(1, math.ceil(len(utterances) / n_chunks))
    tasks = [
        (utterances[i : i + step], beam_size, alpha, beta, token_prune, score_eos)
        for i in range(0, len(utterances), step)
    ]
    merged: Dict[str, str] = {}
    for part in pool.map(_decode_chunk_worker, tasks):
        merged.update(part)
    return merged


def main() -> None:
    import torch
    import torch.distributed as dist
    import data as D
    from decode import load_reader_dir
    from eval_wer import norm, pronunciations, score, substitution_pairs
    from tpu_utils import cleanup_distributed, configure_tpu_env, get_device, init_distributed, synchronize

    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lm", default=os.path.expanduser("~/data/lm/4-gram.bin"))
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "dev-other", "test-clean", "test-other"])
    ap.add_argument("--lexicon", default=os.path.expanduser("~/data/librispeech-lexicon.txt"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--beam-size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--tune-dev", action="store_true", help="Sweep (alpha, beta) on dev-clean + dev-other before test")
    ap.add_argument("--token-prune", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pad-to", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tmp-dir", default=os.path.expanduser("~/data/tmp/beam_eval"))
    args = ap.parse_args()

    configure_tpu_env()
    is_dist, rank, world_size = init_distributed()
    device = get_device(args.device)

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    if is_dist:
        dist.barrier()

    model, prep_table, vocab, cfg = load_reader_dir(args.reader_dir, device)
    widx = {w: i for i, w in enumerate(vocab)}

    # Stage 1: Distributed 8-chip TPU top-K acoustic extraction across all splits
    split_objs: Dict[str, D.Split] = {}
    split_utts: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = {}
    tpu_extract_s: Dict[str, float] = {}

    for name in args.splits:
        sp = D.Split(args.cache, name, widx)
        split_objs[name] = sp
        t0 = time.time()
        local_items = list(range(len(sp)))[rank::world_size]
        local_ext = extract_split_topk(
            model,
            sp,
            prep_table,
            device,
            top_k=args.top_k,
            batch_size=args.batch_size,
            items=local_items,
            pad_step=args.pad_to,
        )
        synchronize()
        if is_dist:
            td = Path(args.tmp_dir)
            td.mkdir(parents=True, exist_ok=True)
            part_p = td / f"topk_{name}_rank{rank:02d}.npz"
            flat_dict = {}
            for k_idx, (uid, (tlp, tid, blp)) in enumerate(local_ext.items()):
                flat_dict[f"id_{k_idx}"] = np.array(uid)
                flat_dict[f"tlp_{k_idx}"] = tlp
                flat_dict[f"tid_{k_idx}"] = tid
                flat_dict[f"blp_{k_idx}"] = blp
            flat_dict["count"] = np.array(len(local_ext), dtype=np.int32)
            np.savez(str(part_p), **flat_dict)
            dist.barrier()
            if rank == 0:
                merged_ext: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for r in range(world_size):
                    rp = td / f"topk_{name}_rank{r:02d}.npz"
                    with np.load(str(rp), allow_pickle=False) as z:
                        cnt = int(z["count"])
                        for k_idx in range(cnt):
                            uid = str(z[f"id_{k_idx}"])
                            merged_ext[uid] = (
                                z[f"tlp_{k_idx}"].astype(np.float32),
                                z[f"tid_{k_idx}"],
                                z[f"blp_{k_idx}"],
                            )
                    rp.unlink(missing_ok=True)
                split_utts[name] = [(u, *merged_ext[u]) for u in sp.ids]
            dist.barrier()
        else:
            split_utts[name] = [
                (u, local_ext[u][0].astype(np.float32), local_ext[u][1], local_ext[u][2]) for u in sp.ids
            ]
        tpu_extract_s[name] = round(time.time() - t0, 2)
        log(f"TPU extracted top-{args.top_k} log-probs for {name} ({len(sp)} utts) in {tpu_extract_s[name]:.2f}s", rank)

    # Stage 2: Rank 0 runs multi-core KenLM CTC beam decoding (and optional dev hyperparameter sweep)
    if rank == 0:
        import multiprocessing as mp

        wnorm = load_whisper_normalizer()
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.lm, vocab),
        ) as pool:
            # Warm up worker pool
            if split_utts:
                first_name = next(iter(split_utts))
                _ = decode_parallel(pool, split_utts[first_name][: min(64, len(split_utts[first_name]))], 8, 0.5, 0.5, args.token_prune)

            best_alpha, best_beta = args.alpha, args.beta
            tuning_history = []
            if args.tune_dev and "dev-clean" in split_utts and "dev-other" in split_utts:
                log("Sweeping (alpha, beta) on dev-clean + dev-other (tune_beam=16)...")
                alphas = [0.25, 0.40, 0.55, 0.70]
                betas = [0.0, 0.5, 1.0, 1.5, 2.0]
                best_avg = float("inf")
                for al in alphas:
                    for be in betas:
                        t_sw = time.time()
                        hc = decode_parallel(pool, split_utts["dev-clean"], 16, al, be, args.token_prune, n_chunks=args.workers * 2)
                        ho = decode_parallel(pool, split_utts["dev-other"], 16, al, be, args.token_prune, n_chunks=args.workers * 2)
                        _, sc = score(split_objs["dev-clean"].texts, [hc[u] for u in split_objs["dev-clean"].ids])
                        _, so = score(split_objs["dev-other"].texts, [ho[u] for u in split_objs["dev-other"].ids])
                        # Weight dev-clean and dev-other equally by normalized relative / average WER
                        avg_wer = 0.5 * (sc["wer"] + so["wer"])
                        rec = {
                            "alpha": al,
                            "beta": be,
                            "dev_clean_wer": round(sc["wer"], 5),
                            "dev_other_wer": round(so["wer"], 5),
                            "avg_dev_wer": round(avg_wer, 5),
                            "sec": round(time.time() - t_sw, 2),
                        }
                        tuning_history.append(rec)
                        marker = ""
                        if avg_wer < best_avg:
                            best_avg = avg_wer
                            best_alpha, best_beta = al, be
                            marker = "  <-- BEST"
                        log(
                            f"  alpha={al:.2f} beta={be:.2f}: dev-clean={100*sc['wer']:.2f}% "
                            f"dev-other={100*so['wer']:.2f}% avg={100*avg_wer:.2f}% ({rec['sec']:.1f}s){marker}"
                        )
                log(f"Selected dev-optimal LM weights: alpha={best_alpha}, beta={best_beta}")

            results = {}
            objects = {}
            for name in args.splits:
                sp = split_objs[name]
                t_dec = time.time()
                hyps = decode_parallel(
                    pool,
                    split_utts[name],
                    args.beam_size,
                    best_alpha,
                    best_beta,
                    args.token_prune,
                    n_chunks=args.workers * 2,
                )
                beam_s = round(time.time() - t_dec, 2)
                hyp_list = [hyps[u] for u in sp.ids]
                o, s = score(sp.texts, hyp_list)
                s["wer_whisper_norm"] = round(score_whisper_norm(sp.texts, hyp_list, wnorm), 5)
                s["tpu_extract_s"] = tpu_extract_s[name]
                s["beam_decode_s"] = beam_s
                s["total_decode_s"] = round(tpu_extract_s[name] + beam_s, 2)
                s["utterances_with_oov"] = len(sp) - len(sp.in_vocab())
                results[name] = s
                objects[name] = (o, sp)

                with open(os.path.join(args.out_dir, f"hyp_beam_{name}.tsv"), "w", encoding="utf-8") as f:
                    for u, ref in zip(sp.ids, sp.texts):
                        f.write(f"{u}\t{norm(ref)}\t{norm(hyps[u])}\n")

                log(
                    f"{name} (beam={args.beam_size}, alpha={best_alpha}, beta={best_beta}): "
                    f"raw WER={100*s['wer']:.2f}%, Whisper-norm WER={100*s['wer_whisper_norm']:.2f}% "
                    f"(S={s['substitutions']} D={s['deletions']} I={s['insertions']} / {s['ref_words']}) "
                    f"[TPU {s['tpu_extract_s']:.2f}s + CPU beam {beam_s:.2f}s = {s['total_decode_s']:.2f}s]"
                )

        analysis = {}
        if args.lexicon and os.path.exists(args.lexicon):
            pron = pronunciations(args.lexicon)
            homo = lambda a, b: bool(pron.get(a, set()) & pron.get(b, set()))  # noqa: E731
            for name in ("test-clean", "test-other"):
                if name not in objects:
                    continue
                pairs = substitution_pairs(objects[name][0])
                total = sum(pairs.values())
                top = pairs.most_common(20)
                analysis[f"substitutions_{name}"] = {
                    "total": total,
                    "distinct_pairs": len(pairs),
                    "top20": [{"ref": a, "hyp": b, "count": c, "homophone": homo(a, b)} for (a, b), c in top],
                    "top20_homophone_pairs": sum(1 for (a, b), _ in top if homo(a, b)),
                    "homophone_share_of_all": sum(c for (a, b), c in pairs.items() if homo(a, b)) / max(1, total),
                }

        report = {
            "decode": f"200k-word CTC prefix beam search + 4-gram KenLM ({os.path.basename(args.lm)})",
            "beam_size": args.beam_size,
            "top_k": args.top_k,
            "alpha": best_alpha,
            "beta": best_beta,
            "token_prune": args.token_prune,
            "reader": cfg["architecture"],
            "scale": cfg["scale"],
            "vocab": len(vocab),
            "tuning_history": tuning_history,
            "results": results,
            "analysis": analysis,
        }
        out_json = os.path.join(args.out_dir, "wer_beam_4gram.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(f"Saved beam + 4-gram evaluation -> {out_json}")

    if is_dist:
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
