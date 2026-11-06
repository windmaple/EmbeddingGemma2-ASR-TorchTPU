#!/usr/bin/env python3
"""TorchTPU initialization, HuggingFace runtime patches, and distributed helpers."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple


def configure_tpu_env(cache_dir: Optional[str] = None) -> None:
    """Configure persistent Tier-2/Tier-3 XLA compilation caches before TPU initialization."""
    resolved = os.path.expanduser(
        cache_dir or os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", "~/data/torchtpu_cache")
    )
    Path(resolved).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", "1")
    os.environ.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", str(resolved))


def is_tpu_available() -> bool:
    """Detect native TorchTPU via PyTorch's stable torch.accelerator API."""
    import torch

    if not hasattr(torch, "tpu") and importlib.util.find_spec("torch_tpu") is not None:
        try:
            importlib.import_module("torch_tpu")
        except Exception:
            pass

    accel = getattr(torch, "accelerator", None)
    if accel is not None and hasattr(accel, "is_available") and hasattr(accel, "current_accelerator"):
        try:
            if accel.is_available():
                curr = accel.current_accelerator()
                return curr is not None and getattr(curr, "type", str(curr)) == "tpu"
        except Exception:
            pass
    return False


def patch_hf_for_tpu() -> list[str]:
    """Patch HuggingFace transformers and sentence-transformers for zero-sync native TPU execution."""
    patched: list[str] = []

    def _get_mod(name: str):
        try:
            return importlib.import_module(name)
        except Exception:
            return None

    st_env = _get_mod("sentence_transformers.util.environment")
    if st_env is not None and hasattr(st_env, "get_device_name"):
        orig_st = st_env.get_device_name
        if not getattr(orig_st, "_tpu_patched", False):
            def _dev_name(*a, **kw) -> str:
                if is_tpu_available():
                    return "tpu"
                return orig_st(*a, **kw)

            setattr(_dev_name, "_tpu_patched", True)
            st_env.get_device_name = _dev_name
            st_util = _get_mod("sentence_transformers.util")
            if st_util is not None and hasattr(st_util, "get_device_name"):
                st_util.get_device_name = _dev_name
            patched.append("sentence_transformers.get_device_name")

    tf_mask = _get_mod("transformers.masking_utils")
    if tf_mask is not None and hasattr(tf_mask, "_ignore_causal_mask_sdpa"):
        orig_ignore = tf_mask._ignore_causal_mask_sdpa
        if not getattr(orig_ignore, "_tpu_patched", False):
            def _no_sync_ignore(padding_mask, *a, **kw) -> bool:
                if padding_mask is not None and getattr(getattr(padding_mask, "device", None), "type", None) == "tpu":
                    return False
                return bool(orig_ignore(padding_mask, *a, **kw))

            setattr(_no_sync_ignore, "_tpu_patched", True)
            tf_mask._ignore_causal_mask_sdpa = _no_sync_ignore
            patched.append("transformers.masking_utils._ignore_causal_mask_sdpa")

    # Allow SentenceTransformer v6+ to load text/audio EmbeddingGemma 2 without requiring torchvision for video
    st_tr = _get_mod("sentence_transformers.base.modules.transformer")
    if st_tr is not None and hasattr(st_tr, "AutoProcessor"):
        orig_ap = st_tr.AutoProcessor.from_pretrained
        if not getattr(orig_ap, "_no_tv_patched", False):
            from transformers import AutoTokenizer

            def _safe_auto_processor(pretrained_model_name_or_path, *a, **kw):
                try:
                    return orig_ap(pretrained_model_name_or_path, *a, **kw)
                except ImportError as e:
                    if "Torchvision" in str(e) or "torchvision" in str(e) or "vision" in str(e).lower():
                        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *a, **kw)
                    raise

            setattr(_safe_auto_processor, "_no_tv_patched", True)
            st_tr.AutoProcessor.from_pretrained = _safe_auto_processor
            patched.append("sentence_transformers.AutoProcessor.fallback_tokenizer")

    return patched


def get_device(preferred: Optional[str] = None):
    """Return torch.device('tpu' | 'cuda' | 'mps' | 'cpu')."""
    import torch

    if preferred and preferred != "auto":
        return torch.device(preferred)
    if is_tpu_available():
        return torch.device("tpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize() -> None:
    """Synchronize current accelerator cleanly using stable PyTorch APIs."""
    import torch

    accel = getattr(torch, "accelerator", None)
    if accel is not None and hasattr(accel, "is_available") and accel.is_available():
        accel.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def init_distributed() -> Tuple[bool, int, int]:
    """Initialize torch.distributed with tpu_dist (or nccl/gloo) if launched under torchrun."""
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1

    if not dist.is_initialized():
        if is_tpu_available():
            backend = "tpu_dist"
        elif torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)

    return True, dist.get_rank(), dist.get_world_size()


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
