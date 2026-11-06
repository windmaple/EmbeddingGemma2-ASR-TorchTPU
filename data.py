#!/usr/bin/env python3
"""Data loader for TorchTPU DDP training & evaluation: cached soft tokens + SPMD-aligned static-bucket batches."""
from __future__ import annotations

import argparse
import os
import queue
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_BUCKETS = (256, 512, 768, 1024)


def load_vocab(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def read_index(cache_dir: str, split: str) -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    with open(os.path.join(cache_dir, split, "index.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                uid, _, _, n_tok, text = line.rstrip("\n").split("\t")
                rows.append((uid, int(n_tok), " ".join(text.split())))
    return rows


def choose_bucket(max_len: int, buckets: Sequence[int] = DEFAULT_BUCKETS, step: int = 256) -> int:
    for b in buckets:
        if max_len <= b:
            return b
    return ((max_len + step - 1) // step) * step


class Split:
    """Cached soft-token split (<cache>/<split>/<id>.npy + index.tsv)."""

    def __init__(self, cache_dir: str, split: str, word_index: Dict[str, int]):
        self.cache_dir, self.name = cache_dir, split
        self.parts = split.split(",")
        rows: List[Tuple[str, int, str]] = []
        self.dirs: List[str] = []
        for part in self.parts:
            part_rows = read_index(cache_dir, part)
            rows += part_rows
            self.dirs += [part] * len(part_rows)
        self.ids = [r[0] for r in rows]
        self.n_tokens = np.array([r[1] for r in rows], dtype=np.int64)
        self.texts = [r[2] for r in rows]
        self.word_ids: List[Optional[np.ndarray]] = []
        for text in self.texts:
            ids = [word_index.get(w) for w in text.split()]
            self.word_ids.append(None if any(i is None for i in ids) else np.array(ids, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.ids)

    def in_vocab(self) -> List[int]:
        return [i for i, w in enumerate(self.word_ids) if w is not None]

    def tokens(self, i: int) -> np.ndarray:
        return np.load(os.path.join(self.cache_dir, self.dirs[i], f"{self.ids[i]}.npy")).astype(np.float32)

    def part_stats(self) -> dict:
        out = {}
        for part in self.parts:
            idx = [i for i, d in enumerate(self.dirs) if d == part]
            keep = [i for i in idx if self.word_ids[i] is not None]
            out[part] = {
                "utterances": len(idx),
                "hours": round(float(self.n_tokens[idx].sum()) / 25 / 3600, 3),
                "tokens": int(self.n_tokens[idx].sum()),
                "in_vocab_utterances": len(keep),
                "in_vocab_hours": round(float(self.n_tokens[keep].sum()) / 25 / 3600, 3),
                "oov_left_out": len(idx) - len(keep),
            }
        return out


def ctc_feasible(n_tokens: int, word_ids: np.ndarray) -> bool:
    repeats = int((word_ids[1:] == word_ids[:-1]).sum()) if len(word_ids) > 1 else 0
    return int(n_tokens) >= len(word_ids) + repeats


def sorted_batches(lengths: np.ndarray, items: Sequence[int], batch_size: int) -> List[List[int]]:
    order = np.asarray(items)[np.argsort(np.asarray(lengths)[items], kind="stable")]
    return [order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def bucket_batches_ddp(
    lengths: np.ndarray,
    items: Sequence[int],
    global_batch_size: int,
    rank: int,
    world_size: int,
    rng: np.random.Generator,
    jitter: float = 0.05,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> List[Tuple[List[int], List[float], int]]:
    """Build SPMD-identical length-bucketed batches across all DDP ranks.

    Returns list of `(local_idx, local_mask, t_bucket)` where:
      - `len(local_idx) == global_batch_size // world_size` on every step and every rank.
      - `t_bucket` is identical across all `world_size` ranks on every step!
      - `local_mask` has 1.0 for real utterances and 0.0 for padded dummy utterances in the final batch.
    """
    assert global_batch_size % world_size == 0, f"global_batch_size {global_batch_size} must be divisible by {world_size}"
    lens_sub = np.asarray(lengths, dtype=np.float64)[items]
    key = lens_sub * (1.0 + rng.uniform(-jitter, jitter, size=len(items)))
    order = np.asarray(items)[np.argsort(key, kind="stable")].tolist()

    raw_batches: List[Tuple[List[int], List[float], int]] = []
    for a in range(0, len(order), global_batch_size):
        g_idx = order[a : a + global_batch_size]
        real_n = len(g_idx)
        g_mask = [1.0] * real_n
        if real_n < global_batch_size:
            pad_n = global_batch_size - real_n
            g_idx = g_idx + [g_idx[-1]] * pad_n
            g_mask = g_mask + [0.0] * pad_n
        max_len = int(max(lengths[i] for i in g_idx))
        t_bucket = choose_bucket(max_len, buckets=buckets)
        # Interleave across ranks (r::world_size) so every chip gets balanced work
        local_idx = g_idx[rank::world_size]
        local_mask = g_mask[rank::world_size]
        raw_batches.append((local_idx, local_mask, t_bucket))

    perm = rng.permutation(len(raw_batches)).tolist()
    return [raw_batches[p] for p in perm]


def collate_static(
    split: Split,
    idx: Sequence[int],
    utt_mask: Optional[Sequence[float]] = None,
    t_bucket: Optional[int] = None,
    with_targets: bool = True,
) -> dict:
    xs = [split.tokens(i) for i in idx]
    lens_py = [int(x.shape[0]) for x in xs]
    lengths = torch.tensor(lens_py, dtype=torch.long)
    T = t_bucket if t_bucket is not None else choose_bucket(max(lens_py))
    x = torch.zeros((len(xs), T, xs[0].shape[1]), dtype=torch.float32)
    for b, t in enumerate(xs):
        x[b, : t.shape[0]] = torch.from_numpy(t)
    out = {
        "x": x,
        "lengths": lengths,
        "lengths_py": lens_py,
        "idx": list(idx),
        "t_bucket": T,
        "utt_mask": torch.tensor(utt_mask if utt_mask is not None else [1.0] * len(xs), dtype=torch.float32),
    }
    if with_targets:
        out["words"] = [torch.from_numpy(split.word_ids[i]) for i in idx]
    return out


class Prefetcher:
    """Background batch prefetcher so ~/data .npy reads overlap TPU MXU execution."""

    def __init__(self, split: Split, batch_specs: Sequence[Tuple[List[int], List[float], int]], with_targets: bool = True, depth: int = 16):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self._run, args=(split, batch_specs, with_targets), daemon=True)
        self.t.start()

    def _run(self, split: Split, batch_specs, with_targets: bool) -> None:
        try:
            for idx, mask, t_bucket in batch_specs:
                self.q.put(collate_static(split, idx, utt_mask=mask, t_bucket=t_bucket, with_targets=with_targets))
        except BaseException as e:  # noqa: BLE001
            self.q.put(e)
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--splits", nargs="+", default=["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"])
    args = ap.parse_args()
    vocab = load_vocab(args.vocab)
    word_index = {w: i for i, w in enumerate(vocab)}
    print(f"vocabulary {len(vocab)} words")
    for name in args.splits:
        s = Split(args.cache, name, word_index)
        keep = s.in_vocab()
        infeasible = sum(1 for i in keep if not ctc_feasible(int(s.n_tokens[i]), s.word_ids[i]))
        print(
            f"{name}: {len(s)} utterances, {int(s.n_tokens.sum())} tokens ({s.n_tokens.sum() / 25 / 3600:.2f} h), "
            f"in vocabulary {len(keep)} ({int(s.n_tokens[keep].sum()) / 25 / 3600:.2f} h), "
            f"CTC-infeasible {infeasible}, max tokens {int(s.n_tokens.max())}"
        )


if __name__ == "__main__":
    main()
