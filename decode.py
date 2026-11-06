#!/usr/bin/env python3
"""Static-shape greedy CTC decoding on TorchTPU: soft tokens [T, 512] -> word transcript."""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from data import choose_bucket
from model import PreparedTable, Reader, collapse, greedy_labels_static
from tpu_utils import configure_tpu_env, get_device


def load_prep_table(path: str, device: torch.device, align: int = 128) -> PreparedTable:
    t = torch.from_numpy(np.load(path).astype(np.float32)).to(device)
    t = F.normalize(t, dim=-1, eps=1e-12)
    return PreparedTable(t, align=align)


def load_reader_dir(reader_dir: str, device: torch.device):
    with open(os.path.join(reader_dir, "reader_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(reader_dir, cfg["vocab"]), encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f if line.strip()]
    arch = cfg["architecture"]
    model = Reader(arch["name"], len(vocab), k=arch.get("k", 4), layers=arch["layers"] if arch["name"] == "a" else 6)
    sd = torch.load(os.path.join(reader_dir, "reader.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    prep_table = load_prep_table(os.path.join(reader_dir, cfg["table"]["file"]), device)
    return model, prep_table, vocab, cfg


@torch.no_grad()
def transcribe_batch(
    model: Reader,
    xs: Sequence[np.ndarray],
    prep_table: PreparedTable,
    vocab: Sequence[str],
    device: torch.device,
    fixed_batch_size: Optional[int] = None,
    pad_step: int = 256,
    chunk: int = 1024,
) -> List[str]:
    """Static-shape greedy decoding without boolean-mask device indexing."""
    if not xs:
        return []
    real_b = len(xs)
    b_pad = fixed_batch_size if (fixed_batch_size is not None and fixed_batch_size >= real_b) else real_b
    real_lens = [int(x.shape[0]) for x in xs]
    T = choose_bucket(max(real_lens), step=pad_step) if pad_step > 0 else max(real_lens)

    x = torch.zeros((b_pad, T, xs[0].shape[1]), dtype=torch.float32)
    lens_pad = [1] * b_pad
    for b in range(b_pad):
        src = b if b < real_b else 0
        arr = np.asarray(xs[src], dtype=np.float32)
        x[b, : arr.shape[0]] = torch.from_numpy(arr)
        lens_pad[b] = real_lens[src]

    lengths = torch.tensor(lens_pad, dtype=torch.long, device=device)
    x_dev = x.to(device)

    prec_ctx = (
        torch.tpu.precision(torch.tpu.Precision.HIGHEST)
        if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")
        else contextlib.nullcontext()
    )
    with prec_ctx:
        h, blank = model(x_dev, lengths)
        labels_dev = greedy_labels_static(h, blank, prep_table, model.scale, chunk=chunk)

    labels_cpu = labels_dev.cpu().numpy()
    out: List[str] = []
    for b in range(real_b):
        n = real_lens[b]
        words = collapse(labels_cpu[b, :n].tolist(), prep_table.n_vocab)
        out.append(" ".join(vocab[i] for i in words))
    return out


@torch.no_grad()
def transcribe_split(
    model: Reader,
    split,
    prep_table: PreparedTable,
    vocab: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    items: Optional[Sequence[int]] = None,
    pad_step: int = 256,
) -> Dict[str, str]:
    items_list = list(range(len(split))) if items is None else list(items)
    order = sorted(items_list, key=lambda i: int(split.n_tokens[i]))
    hyps: Dict[str, str] = {}
    for a in range(0, len(order), batch_size):
        idx = order[a : a + batch_size]
        texts = transcribe_batch(
            model,
            [split.tokens(i) for i in idx],
            prep_table,
            vocab,
            device,
            fixed_batch_size=batch_size,
            pad_step=pad_step,
        )
        for i, t in zip(idx, texts):
            hyps[split.ids[i]] = t
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-dir", required=True)
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("npy", nargs="*")
    args = ap.parse_args()

    configure_tpu_env()
    device = get_device(args.device)
    model, prep_table, vocab, _ = load_reader_dir(args.reader_dir, device)

    paths = list(args.npy)
    if args.fixtures:
        paths += sorted(glob.glob(os.path.join(args.fixtures, "**", "soft_tokens.npy"), recursive=True))
    for p in paths:
        arr = np.load(p).astype(np.float32)
        text = transcribe_batch(model, [arr], prep_table, vocab, device, fixed_batch_size=1, pad_step=256)[0]
        print(f"{p}\t{text}")


if __name__ == "__main__":
    main()
