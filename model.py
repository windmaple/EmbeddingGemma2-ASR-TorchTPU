#!/usr/bin/env python3
"""TorchTPU-optimized Reader model and 200k-word vector-similarity CTC objective.

Architecture matches the Embedding Gemma Transcribe / eg2-transcribe specification:
  reader(soft_tokens [B, T, 512]) -> v [B, T, 768]
  h = v / ||v||_2  (768-d unit vector in EmbeddingGemma 2 text embedding space)
  blank_logit = Linear(768 -> 1)(v)
  word_logits[w] = exp(log_scale) * cos(h_t, word_table[w])  for w in 0..N-1
  blank_logit at class index N

TorchTPU optimizations:
  1. Word table is pre-padded to a 128-aligned row count (e.g. 200,000 -> 200,064) with
     pre-transposed contiguous `table_T` [768, N_pad] and additive mask `-1e9` on padded columns.
  2. Static-shape `_LogSumExpCos` runs over all `B * T` bucketed tokens in fixed `chunk` (1024) blocks
     without boolean-mask indexing (`h[valid]`), eliminating `aten::nonzero` host syncs and XLA recompiles.
  3. Gathered CTC columns are padded to a fixed 128-aligned class width (`C_bucket = 128`).
  4. Loss stays on device (`0-D` tensor) with zero `.item()` syncs inside training steps.
"""
from __future__ import annotations

import argparse
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN, D_OUT = 512, 768


def key_additive_mask(lengths: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
    """[B, 1, 1, T] additive attention mask: 0.0 for valid keys (t < lengths), -1e9 for padding."""
    valid = torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]
    zero = torch.zeros((), device=lengths.device, dtype=dtype)
    neg = torch.full((), -1e9, device=lengths.device, dtype=dtype)
    return torch.where(valid, zero, neg)[:, None, None, :]


def rope(T: int, dim: int, base: float, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    f = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------- (a) Bidirectional RoPE Transformer Reader

class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.scale = self.head_dim ** -0.5
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ffn)
        self.ff2 = nn.Linear(ffn, d)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # Explicit matmul + softmax attention avoids SDPA boolean-mask lowering quirks across backends
        # while fusing cleanly on TPU MXU/VPU (T <= 1024).
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        a = torch.matmul(probs, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class ReaderA(nn.Module):
    def __init__(self, d: int = 512, layers: int = 6, heads: int = 8, ffn: int = 2048, rope_base: float = 10000.0):
        super().__init__()
        self.heads, self.rope_base = heads, rope_base
        self.ln_in = nn.LayerNorm(D_IN)
        self.inp = nn.Linear(D_IN, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, D_OUT)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        # Clamp lengths >= 1 so dummy batch-padding rows never produce all -inf softmax rows
        safe_lengths = lengths.clamp(min=1)
        attn_bias = key_additive_mask(safe_lengths, T, x.dtype)
        cos, sin = rope(T, self.inp.out_features // self.heads, self.rope_base, x.device, x.dtype)
        h = self.inp(self.ln_in(x))
        for blk in self.blocks:
            h = blk(h, attn_bias, cos, sin)
        return self.out(self.ln_out(h))


# ---------------------------------------------------------------- (b) EmbeddingGemma 2 text layers 0..k-1

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.type_as(x)


class GemmaLayer(nn.Module):
    def __init__(self, d: int = 512, heads: int = 4, kv_heads: int = 2, head_dim: int = 256,
                 ffn: int = 2048, d_ple: int = 512, eps: float = 1e-6, rope_base: float = 10000.0):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.rope_base = heads, kv_heads, head_dim, rope_base
        self.input_layernorm = RMSNorm(d, eps)
        self.q_proj = nn.Linear(d, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, d, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.v_norm = RMSNorm(head_dim, eps, scale=False)
        self.post_attention_layernorm = RMSNorm(d, eps)
        self.pre_feedforward_layernorm = RMSNorm(d, eps)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        self.post_feedforward_layernorm = RMSNorm(d, eps)
        self.per_layer_input_gate = nn.Linear(d, d_ple, bias=False)
        self.per_layer_projection = nn.Linear(d_ple, d, bias=False)
        self.post_per_layer_input_norm = RMSNorm(d, eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: torch.Tensor, ple: torch.Tensor, attn_bias: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(h).view(B, T, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(B, T, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_norm(self.v_proj(h).view(B, T, self.kv_heads, self.head_dim)).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) + attn_bias
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        a = self.o_proj(torch.matmul(probs, v).transpose(1, 2).reshape(B, T, -1))
        x = x + self.post_attention_layernorm(a)
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h))
        x = x + self.post_feedforward_layernorm(h)
        g = self.per_layer_projection(F.gelu(self.per_layer_input_gate(x), approximate="tanh") * ple)
        x = x + self.post_per_layer_input_norm(g)
        return x * self.layer_scalar


class ReaderB(nn.Module):
    """EmbeddingGemma 2 text layers 0..k-1 (supports both local layers and every-6th global layer 5, 11, 17, 23)."""

    def __init__(self, k: int = 4, d: int = 512, d_ple: int = 512, eps: float = 1e-6,
                 rope_base: float = 10000.0, head_dim: int = 256, full_rope_base: float = 1000000.0):
        super().__init__()
        self.k, self.d, self.d_ple, self.rope_base, self.head_dim = k, d, d_ple, rope_base, head_dim
        self.full_rope_base = full_rope_base
        self.per_layer_model_projection = nn.Linear(d, k * d_ple, bias=False)
        self.per_layer_projection_norm = RMSNorm(d_ple, eps)
        layers = []
        for i in range(k):
            if i % 6 == 5:
                # EmbeddingGemma 2 global full_attention layers (5, 11, 17, 23): head_dim=512, kv_heads=1, rope=1e6
                layers.append(GemmaLayer(d=d, heads=4, kv_heads=1, head_dim=512, d_ple=d_ple, eps=eps, rope_base=full_rope_base))
            else:
                layers.append(GemmaLayer(d=d, heads=4, kv_heads=2, head_dim=head_dim, d_ple=d_ple, eps=eps, rope_base=rope_base))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(d, eps)
        self.embedding_projection = nn.Linear(d, D_OUT, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        ple = self.per_layer_model_projection(x) * (self.d ** -0.5)
        ple = self.per_layer_projection_norm(ple.view(B, T, self.k, self.d_ple))
        safe_lengths = lengths.clamp(min=1)
        attn_bias = key_additive_mask(safe_lengths, T, x.dtype)
        rope_cache: dict[Tuple[int, float], Tuple[torch.Tensor, torch.Tensor]] = {}
        h = x
        for i, layer in enumerate(self.layers):
            key = (layer.head_dim, layer.rope_base)
            if key not in rope_cache:
                rope_cache[key] = rope(T, layer.head_dim, layer.rope_base, x.device, x.dtype)
            cos, sin = rope_cache[key]
            h = layer(h, ple[:, :, i], attn_bias, cos, sin)
        return self.embedding_projection(self.norm(h))

    @torch.no_grad()
    def load_eg2(self, safetensors_path: str) -> None:
        from safetensors import safe_open
        p = "language_model."
        with safe_open(safetensors_path, "pt") as f:
            get = lambda name: f.get_tensor(p + name).float()  # noqa: E731
            self.per_layer_model_projection.weight.copy_(get("ple.per_layer_model_projection.weight")[:self.k * self.d_ple])
            self.per_layer_projection_norm.weight.copy_(get("ple.per_layer_projection_norm.weight"))
            self.norm.weight.copy_(get("norm.weight"))
            self.embedding_projection.weight.copy_(get("embedding_projection.weight"))
            names = {
                "input_layernorm": "input_layernorm", "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
                "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj", "q_norm": "self_attn.q_norm",
                "k_norm": "self_attn.k_norm", "post_attention_layernorm": "post_attention_layernorm",
                "pre_feedforward_layernorm": "pre_feedforward_layernorm", "gate_proj": "mlp.gate_proj",
                "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj",
                "post_feedforward_layernorm": "post_feedforward_layernorm",
                "per_layer_input_gate": "ple_block.per_layer_input_gate",
                "per_layer_projection": "ple_block.per_layer_projection",
                "post_per_layer_input_norm": "ple_block.post_per_layer_input_norm",
            }
            for i, layer in enumerate(self.layers):
                for mine, theirs in names.items():
                    getattr(layer, mine).weight.copy_(get(f"layers.{i}.{theirs}.weight"))
                layer.layer_scalar.copy_(get(f"layers.{i}.layer_scalar"))


# ---------------------------------------------------------------- (c) Per-token MLP

class ReaderC(nn.Module):
    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(D_IN), nn.Linear(D_IN, hidden), nn.GELU(), nn.Linear(hidden, D_OUT))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------- Reader wrapper

class Reader(nn.Module):
    def __init__(self, arch: str, n_vocab: int, k: int = 4, layers: int = 6):
        super().__init__()
        self.arch = arch
        self.body = {"a": lambda: ReaderA(layers=layers), "b": lambda: ReaderB(k=k), "c": lambda: ReaderC()}[arch]()
        self.blank = nn.Linear(D_OUT, 1)
        nn.init.constant_(self.blank.bias, math.log(max(2, n_vocab)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> unit vectors [B, T, 768], blank logits [B, T]."""
        v = self.body(x, lengths)
        return F.normalize(v, dim=-1, eps=1e-12), self.blank(v).squeeze(-1)


def describe(arch: str, k: int = 4, layers: int = 6) -> dict:
    if arch == "a":
        return {
            "name": "a", "kind": "transformer encoder from scratch", "layers": layers, "d_model": 512,
            "heads": 8, "ffn": 2048, "activation": "gelu", "norm": "pre-LayerNorm",
            "positions": "rotary (base 10000)", "input": "LayerNorm -> Linear 512->512",
            "output": "LayerNorm -> Linear 512->768",
        }
    if arch == "b":
        has_global = k >= 6
        return {
            "name": "b", "kind": "EmbeddingGemma 2 text model cut to layers 0..k-1, fine-tuned", "k": k,
            "hidden": 512, "heads": 4,
            "kv_heads": "2 (local) / 1 (global layer 5,11,..)" if has_global else 2,
            "head_dim": "256 (local) / 512 (global layer 5,11,..)" if has_global else 256,
            "ffn": 2048,
            "positions": "rotary (base 10000 local, 1000000 global), 0..T-1" if has_global else "rotary (base 10000), 0..T-1",
            "attention": "bidirectional, no sliding window",
        }
    return {"name": "c", "kind": "per-token MLP (no context)",
            "layers": "LayerNorm -> Linear 512->1024 -> GELU -> Linear 1024->768"}


# ---------------------------------------------------------------- 128-tile-aligned Word Table & Static CTC

class PreparedTable:
    """Holds 128-row-aligned word embedding table + contiguous transpose + column mask on accelerator."""

    def __init__(self, table: torch.Tensor, align: int = 128):
        # table: [N, D] L2-normalized rows
        self.n_vocab = int(table.shape[0])
        self.dim = int(table.shape[1])
        self.device = table.device
        self.dtype = table.dtype
        rem = (-self.n_vocab) % align
        self.n_pad = self.n_vocab + rem
        if rem > 0:
            pad = torch.zeros((rem, self.dim), device=table.device, dtype=table.dtype)
            self.table_padded = torch.cat([table, pad], dim=0).contiguous()
        else:
            self.table_padded = table.contiguous()
        self.table_T = self.table_padded.transpose(0, 1).contiguous()
        col_mask = torch.zeros((1, self.n_pad), device=table.device, dtype=table.dtype)
        if rem > 0:
            col_mask[:, self.n_vocab:] = -1e9
        self.col_mask = col_mask


def pad_rows(x: torch.Tensor, multiple: int) -> torch.Tensor:
    extra = (-x.shape[0]) % multiple
    return torch.cat([x, x.new_zeros(extra, *x.shape[1:])], dim=0) if extra else x


class _LogSumExpCosStatic(torch.autograd.Function):
    """Chunked logsumexp over 128-aligned PreparedTable with zero dynamic masks or NaN edge cases."""

    @staticmethod
    def forward(ctx, h_flat: torch.Tensor, table_padded: torch.Tensor, table_T: torch.Tensor,
                col_mask: torch.Tensor, scale: torch.Tensor, chunk: int):
        M = h_flat.shape[0]
        hp = pad_rows(h_flat, chunk)
        P = hp.shape[0]
        lse_chunks = []
        ecos_chunks = []
        probs = []
        for a in range(0, P, chunk):
            raw_cos = torch.mm(hp[a:a + chunk], table_T)
            logit = raw_cos * scale + col_mask
            z = torch.logsumexp(logit, dim=1)
            p = torch.exp(logit - z[:, None])
            lse_chunks.append(z)
            ecos_chunks.append((p * raw_cos).sum(dim=1))
            probs.append(p)
        lse = torch.cat(lse_chunks, dim=0)[:M]
        ecos = torch.cat(ecos_chunks, dim=0)[:M]
        ctx.probs = probs
        ctx.M = M
        ctx.chunk = chunk
        ctx.save_for_backward(table_padded, scale, ecos)
        return lse

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        table_padded, scale, ecos = ctx.saved_tensors
        gp = pad_rows(g, ctx.chunk)
        grad_chunks = []
        for idx, p in enumerate(ctx.probs):
            a = idx * ctx.chunk
            g_slice = gp[a:a + ctx.chunk]
            grad_chunks.append(torch.mm(p, table_padded) * (scale * g_slice)[:, None])
        grad_h = torch.cat(grad_chunks, dim=0)[:ctx.M]
        ctx.probs = None
        grad_scale = (g * ecos).sum()
        return grad_h, None, None, None, grad_scale, None


class _GatheredCTC(torch.autograd.Function):
    """Runs exact CTC dynamic programming on gathered [B, T, C_bucket] utterance classes on CPU."""

    @staticmethod
    def forward(ctx, lp: torch.Tensor, targets: torch.Tensor, in_lens: torch.Tensor,
                tgt_lens: torch.Tensor, feasible: torch.Tensor):
        x = lp.detach().to("cpu", dtype=torch.float32 if lp.dtype != torch.float64 else torch.float64)
        x = x.transpose(0, 1).contiguous().requires_grad_(True)
        with torch.enable_grad():
            nll = F.ctc_loss(x, targets, in_lens, tgt_lens, blank=0, reduction="none", zero_infinity=True)
            nll.sum().backward()
        valid = (torch.arange(x.shape[0])[:, None] < in_lens[None, :]) & feasible[None, :]
        grad = x.grad - x.detach().exp() * valid[..., None]
        ctx.save_for_backward(grad.transpose(0, 1).to(device=lp.device, dtype=lp.dtype))
        return nll.to(device=lp.device, dtype=lp.dtype)

    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (grad,) = ctx.saved_tensors
        return grad * gout[:, None, None], None, None, None, None


def word_ctc_loss(
    h: torch.Tensor,
    blank: torch.Tensor,
    lengths: torch.Tensor,
    words: Sequence[torch.Tensor],
    prep_table: PreparedTable,
    scale: torch.Tensor,
    chunk: int = 1024,
    c_bucket: int = 128,
    utt_mask: Optional[torch.Tensor] = None,
    lengths_cpu: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, int]:
    """Static-shape 200k-word CTC loss.

    Args:
      h: [B, T, 768] L2-normalized vectors on accelerator.
      blank: [B, T] blank logits on accelerator.
      lengths: [B] int64 valid frame counts on accelerator.
      words: list of B CPU 1-D int64 word-id tensors.
      prep_table: PreparedTable on accelerator.
      scale: 0-D tensor (`model.scale`).
      chunk: row chunk size for 200k normalizer (default 1024).
      c_bucket: static gathered class dimension including blank (default 128).
      utt_mask: optional [B] float32 mask on CPU/accelerator (1.0 real utt, 0.0 padded dummy utt).
      lengths_cpu: optional pre-known Python list of int lengths to avoid `.cpu()` sync!
    """
    B, T, D = h.shape
    dev = h.device
    lens_py = lengths_cpu if lengths_cpu is not None else lengths.cpu().tolist()

    # 1. Static normalizer over all B * T frames (zero boolean-indexing host syncs!)
    lse_words = _LogSumExpCosStatic.apply(
        h.reshape(B * T, D), prep_table.table_padded, prep_table.table_T, prep_table.col_mask, scale, chunk
    ).reshape(B, T)
    # Use logsumexp(stack(...)) since aten::logaddexp.out is not lowered in torch_tpu 0.2.0
    logz_bt = torch.logsumexp(torch.stack([lse_words, blank], dim=-1), dim=-1)

    # 2. Gather unique words per utterance on CPU, padded to static c_bucket (e.g. 128)
    uniq, inv = zip(*[torch.unique(w, return_inverse=True) for w in words])
    max_u = max(len(u) for u in uniq)
    c_tot = max(c_bucket, ((max_u + 1 + 127) // 128) * 128)  # blank + unique words, 128-aligned
    c_words = c_tot - 1
    L = max(1, max(len(w) for w in words))

    ids_cpu = torch.zeros((B, c_words), dtype=torch.long)
    cls_mask_cpu = torch.full((B, 1, c_tot), -1e4, dtype=h.dtype)
    targets = torch.zeros((B, L), dtype=torch.long)
    tgt_lens = torch.zeros((B,), dtype=torch.long)
    in_lens = torch.tensor(lens_py, dtype=torch.long)
    feasible = torch.zeros((B,), dtype=torch.bool)

    for b in range(B):
        nu = len(uniq[b])
        nw = len(words[b])
        if nu > 0:
            ids_cpu[b, :nu] = uniq[b]
        cls_mask_cpu[b, 0, :nu + 1] = 0.0
        if nw > 0:
            targets[b, :nw] = inv[b] + 1
            tgt_lens[b] = nw
            w = words[b]
            rep = int((w[1:] == w[:-1]).sum()) if nw > 1 else 0
            feasible[b] = (lens_py[b] >= nw + rep)
        else:
            tgt_lens[b] = 1
            feasible[b] = False

    if utt_mask is not None:
        utt_mask_cpu = utt_mask.cpu().bool() if isinstance(utt_mask, torch.Tensor) else torch.tensor(utt_mask, dtype=torch.bool)
        feasible = feasible & utt_mask_cpu

    ids_dev = ids_cpu.to(dev)
    cls_mask_dev = cls_mask_cpu.to(dev)
    gathered_w = prep_table.table_padded.index_select(0, ids_dev.reshape(-1)).reshape(B, c_words, D)
    z = torch.bmm(h, gathered_w.transpose(1, 2)) * scale
    lp = torch.cat([blank[..., None], z], dim=-1) - logz_bt[..., None] + cls_mask_dev

    nll = _GatheredCTC.apply(lp, targets, in_lens, tgt_lens, feasible)
    denom_tgt = tgt_lens.clamp(min=1).to(device=dev, dtype=nll.dtype)
    per_utt = nll / denom_tgt
    if utt_mask is not None:
        w_dev = utt_mask.to(device=dev, dtype=nll.dtype) if isinstance(utt_mask, torch.Tensor) else torch.tensor(utt_mask, device=dev, dtype=nll.dtype)
        loss = (per_utt * w_dev).sum() / w_dev.sum().clamp(min=1.0)
        n_bad = int((~feasible & utt_mask_cpu).sum())
    else:
        loss = per_utt.mean()
        n_bad = int((~feasible).sum())
    return loss, n_bad


@torch.no_grad()
def greedy_labels_static(
    h: torch.Tensor,
    blank: torch.Tensor,
    prep_table: PreparedTable,
    scale: torch.Tensor,
    chunk: int = 1024,
) -> torch.Tensor:
    """Static-shape greedy decoding on accelerator: h [B, T, D], blank [B, T] -> labels [B, T] (N = blank)."""
    B, T, D = h.shape
    M = B * T
    hp = pad_rows(h.reshape(M, D), chunk)
    bp = pad_rows(blank.reshape(M), chunk)
    out = []
    for a in range(0, hp.shape[0], chunk):
        scores = torch.mm(hp[a:a + chunk], prep_table.table_T) * scale + prep_table.col_mask
        v, i = scores.max(dim=1)
        is_blank = bp[a:a + chunk] > v
        out.append(torch.where(is_blank, torch.full_like(i, prep_table.n_vocab), i))
    return torch.cat(out, dim=0)[:M].reshape(B, T)


@torch.no_grad()
def topk_logprobs_static(
    h: torch.Tensor,
    blank: torch.Tensor,
    prep_table: PreparedTable,
    scale: torch.Tensor,
    top_k: int = 64,
    chunk: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static-shape top-K word log-probs + blank log-prob on accelerator.

    Returns:
      top_lp: [B, T, top_k] float32 log P(w_j | frame t)
      top_ids: [B, T, top_k] int32 word indices in 0..N-1
      blank_lp: [B, T] float32 log P(blank | frame t)
    """
    B, T, D = h.shape
    M = B * T
    hp = pad_rows(h.reshape(M, D), chunk)
    bp = pad_rows(blank.reshape(M), chunk)
    top_lp_chunks = []
    top_id_chunks = []
    blank_lp_chunks = []
    max_id = prep_table.n_vocab - 1
    for a in range(0, hp.shape[0], chunk):
        logits = torch.mm(hp[a:a + chunk], prep_table.table_T) * scale + prep_table.col_mask
        lse_words = torch.logsumexp(logits, dim=1)
        b_slice = bp[a:a + chunk]
        log_z = torch.logsumexp(torch.stack([lse_words, b_slice], dim=-1), dim=-1)
        vals, ids = torch.topk(logits, k=top_k, dim=1)
        top_lp_chunks.append(vals - log_z[:, None])
        top_id_chunks.append(ids.clamp(max=max_id).to(torch.int32))
        blank_lp_chunks.append(b_slice - log_z)
    top_lp = torch.cat(top_lp_chunks, dim=0)[:M].reshape(B, T, top_k)
    top_ids = torch.cat(top_id_chunks, dim=0)[:M].reshape(B, T, top_k)
    blank_lp = torch.cat(blank_lp_chunks, dim=0)[:M].reshape(B, T)
    return top_lp, top_ids, blank_lp


def collapse(labels: Sequence[int], blank_id: int) -> List[int]:
    out: List[int] = []
    prev: Optional[int] = None
    for c in labels:
        if c != prev and c != blank_id:
            out.append(c)
        prev = c
    return out


# ---------------------------------------------------------------- Self-tests (CPU float64 + TPU float32)

def self_test(device_str: str = "cpu") -> bool:
    """Verify static-shape gathered CTC == dense full-vocab F.ctc_loss in value & all gradients."""
    import contextlib
    torch.manual_seed(0)
    dev = torch.device(device_str)
    dtype = torch.float64 if device_str == "cpu" else torch.float32
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    ctx = (
        torch.tpu.precision(torch.tpu.Precision.HIGHEST)
        if device_str == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "precision")
        else contextlib.nullcontext()
    )

    # Test non-multiple-of-128 vocab (e.g. 300 -> padded to 384) AND padded time T=32 > max(lengths)=30
    N, D, B = 300, 768, 4
    raw_table = F.normalize(torch.randn(N, D, dtype=dtype, device=dev), dim=-1)
    prep = PreparedTable(raw_table, align=128)
    lens_py = [30, 25, 18, 3]
    lengths = torch.tensor(lens_py, dtype=torch.long, device=dev)
    T = 32  # padded bucket > max(lengths)=30!
    words = [
        torch.tensor([5, 7, 7, 9, 5, 12]),
        torch.tensor([1, 2, 3, 2, 1]),
        torch.tensor([42, 42, 42, 8]),
        torch.tensor([10, 11, 12, 13]),  # infeasible in 3 frames
    ]
    v0 = torch.randn(B, T, D, dtype=dtype, device=dev)
    b0 = torch.randn(B, T, dtype=dtype, device=dev) + 3.0
    s0 = torch.tensor(5.0, dtype=dtype, device=dev)

    results = {}
    with ctx:
        for name in ("gathered_static", "full_dense"):
            if name == "gathered_static":
                v = v0.clone().requires_grad_(True)
                bl = b0.clone().requires_grad_(True)
                s = s0.clone().requires_grad_(True)
                h = F.normalize(v, dim=-1, eps=1e-12)
                loss, n_bad = word_ctc_loss(h, bl, lengths, words, prep, s, chunk=64, c_bucket=128, lengths_cpu=lens_py)
            else:
                v = v0.cpu().clone().requires_grad_(True)
                bl = b0.cpu().clone().requires_grad_(True)
                s = s0.cpu().clone().requires_grad_(True)
                h = F.normalize(v, dim=-1, eps=1e-12)
                logits = torch.cat([s * torch.matmul(h, raw_table.cpu().T), bl[..., None]], dim=-1)
                lp = logits.log_softmax(-1).transpose(0, 1)
                tg = torch.zeros(B, max(len(w) for w in words), dtype=torch.long)
                for i, w in enumerate(words):
                    tg[i, :len(w)] = w
                loss = F.ctc_loss(
                    lp, tg, torch.tensor(lens_py), torch.tensor([len(w) for w in words]),
                    blank=N, reduction="mean", zero_infinity=True,
                )
            loss.backward()
            results[name] = (
                float(loss.detach().cpu()),
                v.grad.detach().cpu(),
                bl.grad.detach().cpu(),
                float(s.grad.detach().cpu()),
            )

    g, f = results["gathered_static"], results["full_dense"]
    diffs = {
        "loss": abs(g[0] - f[0]),
        "grad_v": float((g[1] - f[1]).abs().max()),
        "grad_blank": float((g[2] - f[2]).abs().max()),
        "grad_scale": abs(g[3] - f[3]),
    }
    ok = all(d < tol for d in diffs.values())
    print(f"[{device_str} {dtype}] loss gathered={g[0]:.8f} full={f[0]:.8f}; max|d|={diffs}; infeasible={n_bad} -> {'PASS' if ok else 'FAIL'}")

    if device_str != "cpu":
        # Verify full ReaderA forward + backward + clip_grad_norm_ + AdamW + greedy_labels_static on accelerator
        reader = Reader("a", N, layers=2).to(dev)
        opt = torch.optim.AdamW(reader.parameters(), lr=3e-4, weight_decay=0.01)
        x_in = torch.randn(B, T, D_IN, device=dev)
        h_r, bl_r = reader(x_in, lengths)
        loss_r, _ = word_ctc_loss(h_r, bl_r, lengths, words, prep, reader.scale, chunk=64, c_bucket=128, lengths_cpu=lens_py)
        opt.zero_grad(set_to_none=True)
        loss_r.backward()
        gn = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        opt.step()
        lbl = greedy_labels_static(h_r.detach(), bl_r.detach(), prep, reader.scale.detach(), chunk=64)
        print(f"[{device_str} ReaderA e2e] loss={float(loss_r.detach().cpu()):.4f} grad_norm={float(gn.detach().cpu()):.4f} labels_shape={tuple(lbl.shape)} -> PASS")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", type=int, default=6)
    args = ap.parse_args()
    if args.self_test:
        ok_cpu = self_test("cpu")
        ok_dev = self_test(args.device) if args.device != "cpu" else True
        if not (ok_cpu and ok_dev):
            raise SystemExit(1)
    for arch in ("a", "b", "c"):
        n = sum(p.numel() for p in Reader(arch, 200000, layers=args.layers).parameters())
        print(f"arch {arch}{f' ({args.layers} layers)' if arch == 'a' else ''}: {n:,} parameters")


if __name__ == "__main__":
    main()
