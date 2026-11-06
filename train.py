#!/usr/bin/env python3
"""8-chip TorchTPU DDP trainer for EmbeddingGemma 2 Speech Reader (200k-word vector-similarity CTC).

TorchTPU best practices built in:
  - SPMD-identical static shape bucketing across all 8 ranks on every step (B_local x T_bucket x 512).
  - Contiguous flattened all_reduce(SUM) across all parameters over ICI.
  - Zero per-step host syncs (.item() only called every --log-every steps).
  - Precision.HIGH (TF32+ 21-bit mantissa) for training MXU matmuls, Precision.HIGHEST for dev WER decoding.
  - Sharded 8-chip parallel dev WER evaluation after every epoch.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import data as D
from decode import load_prep_table, transcribe_split
from model import Reader, _LogSumExpCosStatic, describe, word_ctc_loss
from tpu_utils import cleanup_distributed, configure_tpu_env, get_device, init_distributed, synchronize

DEFAULT_LR = {"a": 3e-4, "b": 1e-4, "c": 3e-4}


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_wer(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    o = jiwer.process_words(refs, hyps)
    n_ref = o.substitutions + o.deletions + o.hits
    return {
        "wer": float(o.wer),
        "substitutions": int(o.substitutions),
        "deletions": int(o.deletions),
        "insertions": int(o.insertions),
        "ref_words": int(n_ref),
    }


def max_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


def get_precision_ctx(device: torch.device, mode: str):
    if device.type != "tpu" or not (hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")):
        return contextlib.nullcontext()
    mapping = {
        "default": torch.tpu.Precision.DEFAULT,
        "high": torch.tpu.Precision.HIGH,
        "highest": torch.tpu.Precision.HIGHEST,
    }
    return torch.tpu.precision(mapping[mode])


@torch.no_grad()
def broadcast_model(model: torch.nn.Module, src: int = 0) -> None:
    for p in model.parameters():
        dist.broadcast(p.data, src=src)
    for b in model.buffers():
        dist.broadcast(b.data, src=src)


@torch.no_grad()
def all_reduce_and_clip_grads_flat(params: list[torch.nn.Parameter], world_size: int, max_norm: float) -> None:
    """Single fused ICI all_reduce(SUM) + global L2 norm clip over a 128-aligned flat gradient buffer."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads], dim=0)
    rem = (-flat.numel()) % 128
    flat_padded = torch.cat([flat, flat.new_zeros(rem)], dim=0) if rem > 0 else flat
    if world_size > 1:
        dist.all_reduce(flat_padded, op=dist.ReduceOp.SUM)
        flat_padded.mul_(1.0 / world_size)
    if max_norm > 0.0:
        total_norm = torch.linalg.vector_norm(flat_padded)
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        flat_padded.mul_(clip_coef)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat_padded[offset : offset + n].view_as(g))
        offset += n


@torch.no_grad()
def evaluate_distributed(
    model: Reader,
    dev_split: D.Split,
    prep_table,
    vocab: list[str],
    device: torch.device,
    rank: int,
    world_size: int,
    tmp_dir: str,
    batch_size: int = 32,
) -> tuple[dict, dict[str, str]]:
    model.eval()
    t0 = time.time()
    local_items = list(range(len(dev_split)))[rank::world_size]
    local_hyps = transcribe_split(
        model,
        dev_split,
        prep_table,
        vocab,
        device,
        batch_size=batch_size,
        items=local_items,
        pad_step=256,
    )
    synchronize()
    if world_size > 1:
        td = Path(tmp_dir)
        td.mkdir(parents=True, exist_ok=True)
        part_p = td / f"dev_hyps_rank{rank:02d}.json"
        with open(part_p, "w", encoding="utf-8") as f:
            json.dump(local_hyps, f)
        dist.barrier()
        if rank == 0:
            merged: dict[str, str] = {}
            for r in range(world_size):
                rp = td / f"dev_hyps_rank{r:02d}.json"
                with open(rp, encoding="utf-8") as f:
                    merged.update(json.load(f))
                rp.unlink(missing_ok=True)
            s = score_wer(dev_split.texts, [merged[u] for u in dev_split.ids])
            s["eval_s"] = round(time.time() - t0, 2)
        else:
            merged = {}
            s = {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0, "eval_s": 0.0}
        # Broadcast WER scalar to all ranks so early-stopping stays strictly in lockstep
        wer_t = torch.tensor([s["wer"]], dtype=torch.float32, device=device)
        dist.broadcast(wer_t, src=0)
        s["wer"] = float(wer_t.cpu().item())
        dist.barrier()
    else:
        merged = local_hyps
        s = score_wer(dev_split.texts, [merged[u] for u in dev_split.ids])
        s["eval_s"] = round(time.time() - t0, 2)
    model.train()
    return s, merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["a", "b", "c"], default="a")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--schedule-epochs", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None, help="Optional separate LR for blank + log_scale head")
    ap.add_argument("--batch-size", type=int, default=32, help="Global batch size across all TPU chips")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--precision", choices=["default", "high", "highest"], default="high")
    ap.add_argument("--compile", action="store_true", help="Compile Reader forward/backward with torch.compile(backend='tpu')")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--train-split", default="train-clean-100")
    ap.add_argument("--dev-split", default="dev-clean")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--time-steps", type=int, default=0, help="Warmup 4 steps (1 per bucket) then benchmark N steps")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-dev", type=int, default=0)
    ap.add_argument("--tmp-dir", default=os.path.expanduser("~/data/tmp/train_eval"))
    args = ap.parse_args()

    configure_tpu_env()
    warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf Tensor")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    is_dist, rank, world_size = init_distributed()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.batch_size % world_size != 0:
        raise SystemExit(f"--batch-size ({args.batch_size}) must be divisible by world_size ({world_size})")
    local_bs = args.batch_size // world_size

    if rank == 0:
        os.makedirs(args.run_dir, exist_ok=True)
    if is_dist:
        dist.barrier()

    vocab = D.load_vocab(args.vocab)
    widx = {w: i for i, w in enumerate(vocab)}
    prep_table = load_prep_table(args.table, device, align=128)
    if prep_table.n_vocab != len(vocab):
        raise SystemExit(f"table rows {prep_table.n_vocab} != vocabulary {len(vocab)}")

    train_split = D.Split(args.cache, args.train_split, widx)
    in_vocab = train_split.in_vocab()
    items = [i for i in in_vocab if D.ctc_feasible(int(train_split.n_tokens[i]), train_split.word_ids[i])]
    n_infeasible = len(in_vocab) - len(items)
    if args.limit_train:
        items = items[: args.limit_train]

    log(
        f"{args.train_split}: {len(train_split)} utterances ({train_split.n_tokens.sum() / 25 / 3600:.2f} h); "
        f"left out {len(train_split) - len(in_vocab)} OOV and {n_infeasible} CTC-infeasible; "
        f"training on {len(items)} ({train_split.n_tokens[items].sum() / 25 / 3600:.2f} h, {int(train_split.n_tokens[items].sum())} tokens)",
        rank,
    )

    dev_split = D.Split(args.cache, args.dev_split, widx)
    if args.limit_dev:
        dev_split.ids = dev_split.ids[: args.limit_dev]
        dev_split.texts = dev_split.texts[: args.limit_dev]
        dev_split.n_tokens = dev_split.n_tokens[: args.limit_dev]
        dev_split.word_ids = dev_split.word_ids[: args.limit_dev]
    log(
        f"{args.dev_split}: {len(dev_split)} utterances ({dev_split.n_tokens.sum() / 25 / 3600:.2f} h), "
        f"{len(dev_split) - len(dev_split.in_vocab())} OOV utterances kept (counted in WER)",
        rank,
    )

    model = Reader(args.arch, len(vocab), k=args.k, layers=args.layers)
    if args.arch == "b" and not args.resume:
        if not args.model_dir:
            raise SystemExit("--model-dir required for arch b")
        model.body.load_eg2(os.path.join(args.model_dir, "model.safetensors"))
    model = model.to(device)

    all_params = list(model.parameters())
    n_params = sum(p.numel() for p in all_params)
    lr = args.lr or DEFAULT_LR[args.arch]
    head_lr = args.head_lr or lr
    body_params = list(model.body.parameters())
    head_params = list(model.blank.parameters()) + [model.log_scale]
    groups = [
        {"params": [p for p in body_params if p.ndim >= 2], "lr": lr, "weight_decay": args.weight_decay},
        {"params": [p for p in body_params if p.ndim < 2], "lr": lr, "weight_decay": 0.0},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr, foreach=True)
    steps_per_epoch = math.ceil(len(items) / args.batch_size)
    total_steps = max(args.warmup + 1, int(round(args.schedule_epochs * steps_per_epoch)))

    def lr_lambda(s: int) -> float:
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, total_steps - args.warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step, start_epoch, history = 0, 0, []

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, start_epoch, history = ck["step"], ck["epoch"], ck["history"]
        log(f"resumed {args.resume}: epoch {start_epoch}, step {step}", rank)
    else:
        # Initialize blank.bias on rank 0 to median word log-normalizer of a median-length batch (~0.5 blank prob)
        if rank == 0:
            with torch.no_grad(), get_precision_ctx(device, "highest"):
                sb = D.sorted_batches(train_split.n_tokens, items, min(16, len(items)))
                mid_idx = sb[len(sb) // 2]
                b0 = D.collate_static(train_split, mid_idx, with_targets=False)
                h0, _ = model(b0["x"].to(device), b0["lengths"].to(device))
                B0, T0, D0 = h0.shape
                lse0 = _LogSumExpCosStatic.apply(
                    h0.reshape(B0 * T0, D0),
                    prep_table.table_padded,
                    prep_table.table_T,
                    prep_table.col_mask,
                    model.scale,
                    args.chunk,
                ).reshape(B0, T0).cpu()
                valid_vals = torch.cat([lse0[b, : b0["lengths_py"][b]] for b in range(B0)])
                med_val = float(valid_vals.median().item())
                model.blank.weight.zero_()
                model.blank.bias.fill_(med_val)
            log(f"blank bias initialized to {med_val:.3f} (log N = {math.log(len(vocab)):.3f})", rank)
        if is_dist:
            broadcast_model(model, src=0)
            dist.barrier()

    env_info = {
        "device": str(device),
        "world_size": world_size,
        "global_batch_size": args.batch_size,
        "local_batch_size": local_bs,
        "precision": args.precision,
        "eval_precision": "highest",
        "torch": torch.__version__,
    }
    depth_desc = f"k={args.k}" if args.arch == "b" else f"{args.layers} layers"
    log(
        f"arch {args.arch} ({describe(args.arch, args.k, args.layers)['kind']}, {depth_desc}), "
        f"{n_params:,} params, lr={lr:g} (head_lr={head_lr:g}), global_bs={args.batch_size} ({local_bs}/chip x {world_size} chips), "
        f"{steps_per_epoch} steps/epoch, schedule={total_steps} steps, precision={args.precision}",
        rank,
    )

    def save_ckpt(path: str, ep: int) -> None:
        if rank != 0:
            return
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": step,
                "epoch": ep,
                "history": history,
                "args": vars(args),
                "arch": args.arch,
                "k": args.k,
                "layers": args.layers,
                "n_vocab": len(vocab),
            },
            path,
        )

    def write_metrics(extra: dict | None = None) -> None:
        if rank != 0:
            return
        best = min(history, key=lambda h: h["dev"]["wer"]) if history else None
        m = {
            "arch": describe(args.arch, args.k, args.layers),
            "params": n_params,
            "lr": lr,
            "head_lr": head_lr,
            "batch_size": args.batch_size,
            "local_batch_size": local_bs,
            "world_size": world_size,
            "warmup": args.warmup,
            "weight_decay": args.weight_decay,
            "clip": args.clip,
            "schedule_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "train_utterances": len(items),
            "train_tokens": int(train_split.n_tokens[items].sum()),
            "vocab": len(vocab),
            "dev_split": args.dev_split,
            "train_split": args.train_split,
            "train_splits": train_split.part_stats(),
            "ctc_infeasible_left_out": n_infeasible,
            "env": env_info,
            "history": history,
            "best_epoch": best["epoch"] if best else None,
            "best_dev_wer": best["dev"]["wer"] if best else None,
        }
        m.update(extra or {})
        with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    t_train_start = time.time()
    best_wer = min((h["dev"]["wer"] for h in history), default=float("inf"))
    bad_epochs = 0
    stop_reason = f"reached {args.epochs} epochs"

    model.train()
    fwd_model = (
        torch.compile(model, backend="tpu", dynamic=False)
        if (args.compile and device.type == "tpu")
        else model
    )
    if args.compile and device.type == "tpu":
        log("Enabled torch.compile(model, backend='tpu', dynamic=False) for static buckets", rank)

    # Pre-warm all 4 static T_buckets (256, 512, 768, 1024) on step 0 if --time-steps or --compile is set
    if args.time_steps or (args.compile and device.type == "tpu"):
        log("Pre-compiling static T_buckets (256, 512, 768, 1024) across all ranks...", rank)
        dummy_words = [torch.tensor([1, 2, 3], dtype=torch.long) for _ in range(local_bs)]
        for tb in D.DEFAULT_BUCKETS:
            with get_precision_ctx(device, args.precision):
                dx = torch.zeros((local_bs, tb, 512), device=device)
                dl = torch.full((local_bs,), tb // 2, dtype=torch.long, device=device)
                dh, dbl = fwd_model(dx, dl)
                dloss, _ = word_ctc_loss(dh, dbl, dl, dummy_words, prep_table, model.scale, chunk=args.chunk, lengths_cpu=[tb // 2] * local_bs)
                opt.zero_grad(set_to_none=True)
                dloss.backward()
                all_reduce_and_clip_grads_flat(all_params, world_size, args.clip)
        opt.zero_grad(set_to_none=True)
        synchronize()
        if is_dist:
            dist.barrier()
        log("Static bucket pre-compilation complete.", rank)

    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        batch_specs = D.bucket_batches_ddp(
            train_split.n_tokens, items, args.batch_size, rank, world_size, rng
        )
        t_ep = time.time()
        t_win = time.time()
        win_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        win_steps = 0
        ep_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        ep_steps = 0
        tok_window_local = 0
        n_bad_local = 0
        timed_steps: list[tuple[float, int]] = []

        for bi, batch in enumerate(D.Prefetcher(train_split, batch_specs, with_targets=True)):
            if args.time_steps:
                synchronize()
                t_step = time.time()

            with get_precision_ctx(device, args.precision):
                x = batch["x"].to(device)
                lengths = batch["lengths"].to(device)
                h, blank = fwd_model(x, lengths)
                loss, n_bad = word_ctc_loss(
                    h,
                    blank,
                    lengths,
                    batch["words"],
                    prep_table,
                    model.scale,
                    chunk=args.chunk,
                    utt_mask=batch["utt_mask"],
                    lengths_cpu=batch["lengths_py"],
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                all_reduce_and_clip_grads_flat(all_params, world_size, args.clip)
                opt.step()

            sched.step()
            step += 1

            loss_det = loss.detach()
            win_loss_sum = win_loss_sum + loss_det
            ep_loss_sum = ep_loss_sum + loss_det
            win_steps += 1
            ep_steps += 1
            real_tok_local = int(sum(l for l, m in zip(batch["lengths_py"], batch["utt_mask"].tolist()) if m > 0))
            tok_window_local += real_tok_local
            n_bad_local += n_bad

            if args.time_steps:
                synchronize()
                dt = time.time() - t_step
                if bi >= 2:
                    timed_steps.append((dt, real_tok_local * world_size))
                if len(timed_steps) >= args.time_steps:
                    if is_dist:
                        dist.all_reduce(win_loss_sum, op=dist.ReduceOp.SUM)
                    lv = float((win_loss_sum / (win_steps * world_size)).cpu().item())
                    sec = np.array([t for t, _ in timed_steps])
                    tok = np.array([n for _, n in timed_steps])
                    med_s = float(np.median(sec))
                    ep_est_s = med_s * steps_per_epoch
                    res = {
                        "arch": args.arch,
                        "layers": args.layers,
                        "params": n_params,
                        "env": env_info,
                        "steps_timed": len(timed_steps),
                        "mean_loss": round(lv, 4),
                        "median_step_s": round(med_s, 4),
                        "step_s": sec.round(4).tolist(),
                        "global_batch_tokens": tok.tolist(),
                        "tok_per_s": round(float(tok.sum() / sec.sum()), 1),
                        "steps_per_epoch": steps_per_epoch,
                        "epoch_estimate_s": round(ep_est_s, 1),
                    }
                    if rank == 0:
                        with open(os.path.join(args.run_dir, "timing.json"), "w", encoding="utf-8") as f:
                            json.dump(res, f, indent=2)
                    log(
                        f"TIMING ({world_size} chips, global_bs={args.batch_size}): median {med_s*1000:.1f} ms/step, "
                        f"{tok.sum()/sec.sum():.0f} tok/s, est epoch={ep_est_s:.1f}s ({ep_est_s/60:.2f} min), mean_loss={lv:.4f}",
                        rank,
                    )
                    cleanup_distributed()
                    return

            if step % args.log_every == 0:
                if is_dist:
                    dist.all_reduce(win_loss_sum, op=dist.ReduceOp.SUM)
                lv = float((win_loss_sum / (win_steps * world_size)).cpu().item())
                if not math.isfinite(lv):
                    save_ckpt(os.path.join(args.run_dir, f"nonfinite_step{step}.pt"), epoch)
                    cleanup_distributed()
                    raise SystemExit(f"Non-finite loss {lv} at step {step}!")
                el = time.time() - t_win
                global_tok = tok_window_local * world_size
                sc_val = float(model.scale.detach().cpu().item())
                bb_val = float(model.blank.bias.detach().cpu().item())
                log(
                    f"step {step} ep {epoch + 1}/{args.epochs} [{bi + 1}/{len(batch_specs)}] "
                    f"loss={lv:.4f} lr={sched.get_last_lr()[0]:.3e} "
                    f"{win_steps / max(el, 1e-6):.2f} step/s {global_tok / max(el, 1e-6):.0f} tok/s "
                    f"scale={sc_val:.2f} blank_bias={bb_val:.2f} wall={(time.time() - t_train_start)/60:.1f}m",
                    rank,
                )
                win_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                win_steps = 0
                tok_window_local = 0
                t_win = time.time()

        synchronize()
        if is_dist:
            dist.all_reduce(ep_loss_sum, op=dist.ReduceOp.SUM)
        ep_loss_mean = float((ep_loss_sum / (ep_steps * world_size)).cpu().item())
        ep_wall = time.time() - t_ep

        dev_score, hyps = evaluate_distributed(
            model, dev_split, prep_table, vocab, device, rank, world_size, args.tmp_dir, batch_size=32
        )
        sc_val = float(model.scale.detach().cpu().item())
        rec = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss_mean": round(ep_loss_mean, 5),
            "dev": dev_score,
            "wall_s": round(ep_wall, 2),
            "max_rss_gb": round(max_rss_gb(), 2),
            "lr_end": sched.get_last_lr()[0],
            "scale": round(sc_val, 3),
            "ctc_infeasible_local": n_bad_local,
        }
        history.append(rec)
        log(
            f"EPOCH {epoch + 1}/{args.epochs} DONE: train_loss={ep_loss_mean:.4f}, "
            f"{args.dev_split} WER={100 * dev_score['wer']:.2f}% "
            f"(S={dev_score['substitutions']} D={dev_score['deletions']} I={dev_score['insertions']} / {dev_score['ref_words']}), "
            f"train={ep_wall:.1f}s, eval={dev_score['eval_s']:.1f}s, scale={sc_val:.2f}",
            rank,
        )

        ck_path = os.path.join(args.run_dir, f"ckpt_epoch{epoch + 1}.pt")
        save_ckpt(ck_path, epoch + 1)
        if rank == 0:
            # Keep only latest 2 epoch checkpoints to save disk space, plus best_model.pt
            old_ck = os.path.join(args.run_dir, f"ckpt_epoch{epoch - 1}.pt")
            if epoch >= 2 and os.path.exists(old_ck):
                os.remove(old_ck)
            with open(os.path.join(args.run_dir, f"hyp_{args.dev_split}_epoch{epoch + 1}.tsv"), "w", encoding="utf-8") as f:
                for u, ref in zip(dev_split.ids, dev_split.texts):
                    f.write(f"{u}\t{ref}\t{hyps[u]}\n")

        if dev_score["wer"] < best_wer:
            best_wer = dev_score["wer"]
            bad_epochs = 0
            if rank == 0:
                torch.save(model.state_dict(), os.path.join(args.run_dir, "best_model.pt"))
            log(f"  -> NEW BEST {args.dev_split} WER: {100 * best_wer:.2f}% (saved best_model.pt)", rank)
        else:
            bad_epochs += 1

        write_metrics()
        if bad_epochs >= args.patience:
            stop_reason = f"early stop: no better {args.dev_split} WER for {bad_epochs} epochs"
            break

    total_wall = time.time() - t_train_start
    write_metrics({"stop_reason": stop_reason, "wall_s_total": round(total_wall, 1)})
    log(f"TRAINING COMPLETE ({stop_reason}): best {args.dev_split} WER = {100 * best_wer:.2f}% in {total_wall / 60:.1f} min", rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
