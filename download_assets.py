#!/usr/bin/env python3
"""Download EmbeddingGemma 2, all 7 LibriSpeech parquet splits (960h + dev + test), lexicon, and benchmark audio into ~/data."""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


EG2_REPO = "google/embeddinggemma-2"
EG2_REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"

LS_REPO = "openslr/librispeech_asr"
LS_REV = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
LS_PATTERNS_ALL = [
    "clean/train.100/*.parquet",
    "clean/train.360/*.parquet",
    "other/train.500/*.parquet",
    "clean/validation/*.parquet",
    "other/validation/*.parquet",
    "clean/test/*.parquet",
    "other/test/*.parquet",
]
EXPECTED_COUNTS = {
    "clean/train.100": 14,
    "clean/train.360": 48,
    "other/train.500": 64,
    "clean/validation": 1,
    "other/validation": 1,
    "clean/test": 1,
    "other/test": 1,
}

LEXICON_URL = "https://www.openslr.org/resources/11/librispeech-lexicon.txt"
LM_ARPA_GZ_URL = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
SPECKLED_MP3_URL = "https://archive.org/download/adventures_holmes/adventureholmes_08_doyle.mp3"
SPECKLED_TXT_URLS = [
    "https://raw.githubusercontent.com/john-rocky/eg2-transcribe/main/fixtures/long-speckled-band/transcript.txt",
    "https://raw.githubusercontent.com/john-rocky/eg2-transcribe/main/reader/fixtures/long-speckled-band/transcript.txt",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def url_download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        log(f"exists: {dst} ({dst.stat().st_size / 1e6:.2f} MB)")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dst)
    log(f"downloaded: {dst} ({dst.stat().st_size / 1e6:.2f} MB)")


def convert_to_16k_wav(src_mp3: Path, dst_wav: Path) -> None:
    if dst_wav.exists() and dst_wav.stat().st_size > 1_000_000:
        log(f"exists: {dst_wav} ({dst_wav.stat().st_size / 1e6:.2f} MB)")
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-i", str(src_mp3), "-ac", "1", "-ar", "16000", "-f", "wav", str(dst_wav)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        log(f"converted via ffmpeg: {dst_wav} ({dst_wav.stat().st_size / 1e6:.2f} MB)")
    else:
        import numpy as np
        import soundfile as sf
        import soxr

        wav, sr = sf.read(str(src_mp3), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != 16000:
            mono = soxr.resample(mono, sr, 16000, quality="HQ")
        sf.write(str(dst_wav), np.asarray(mono, dtype=np.float32), 16000, subtype="PCM_16")
        log(f"converted via soundfile+soxr: {dst_wav} ({dst_wav.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser(os.environ.get("DATA_DIR", "~/data")))
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--skip-lm", action="store_true", help="Skip downloading/building the OpenSLR 4-gram KenLM")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(data_dir / "hf_cache"))
    os.environ.setdefault("TMPDIR", str(data_dir / "tmp"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    # 1. Lexicon
    lex_path = data_dir / "librispeech-lexicon.txt"
    url_download(LEXICON_URL, lex_path)

    # 2. EmbeddingGemma 2
    eg2_dir = data_dir / "embeddinggemma-2"
    if not (eg2_dir / "model.safetensors").exists() and not list(eg2_dir.glob("model*.safetensors")):
        t0 = time.time()
        log(f"downloading {EG2_REPO} @ {EG2_REV[:8]} -> {eg2_dir} ...")
        snapshot_download(
            repo_id=EG2_REPO,
            revision=EG2_REV,
            local_dir=str(eg2_dir),
            max_workers=args.max_workers,
        )
        log(f"downloaded {EG2_REPO} in {time.time() - t0:.1f}s")
    else:
        log(f"exists: {eg2_dir}")

    # 3. Entire LibriSpeech dataset (all 7 splits: 960h train + dev + test = 130 parquet shards, ~61.8 GB)
    ls_dir = data_dir / "hf_librispeech"
    t0 = time.time()
    log(f"downloading full {LS_REPO} (960h train + dev + test) @ {LS_REV[:8]} -> {ls_dir} ...")
    snapshot_download(
        repo_id=LS_REPO,
        repo_type="dataset",
        revision=LS_REV,
        allow_patterns=LS_PATTERNS_ALL,
        local_dir=str(ls_dir),
        max_workers=args.max_workers,
    )
    log(f"downloaded {LS_REPO} splits in {time.time() - t0:.1f}s")

    total_bytes = 0
    for sub, exp_n in EXPECTED_COUNTS.items():
        pfiles = sorted(glob.glob(str(ls_dir / sub / "*.parquet")))
        if len(pfiles) != exp_n:
            raise RuntimeError(f"Expected {exp_n} parquet files in {sub}, got {len(pfiles)}")
        sub_bytes = sum(os.path.getsize(p) for p in pfiles)
        total_bytes += sub_bytes
        log(f"  verified {sub:18s}: {len(pfiles):3d} parquet shards ({sub_bytes / 1e9:.2f} GB)")
    log(f"verified all 7 LibriSpeech splits: 130 parquet shards ({total_bytes / 1e9:.2f} GB)")

    # 4. Long-form Sherlock Holmes benchmark (54.3 min)
    sb_dir = data_dir / "speckled_band"
    sb_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = sb_dir / "adventureholmes_08_doyle.mp3"
    wav_path = sb_dir / "audio_16k.wav"
    txt_path = sb_dir / "transcript.txt"
    try:
        url_download(SPECKLED_MP3_URL, mp3_path)
        convert_to_16k_wav(mp3_path, wav_path)
    except Exception as e:
        log(f"warning: speckled_band audio download/convert error: {e}")

    if not txt_path.exists():
        for u in SPECKLED_TXT_URLS:
            try:
                url_download(u, txt_path)
                break
            except Exception:
                continue

    # 5. Optional OpenSLR SLR11 4-gram language model (compiled to KenLM trie binary if build_binary is installed)
    if not args.skip_lm:
        lm_dir = data_dir / "lm"
        lm_dir.mkdir(parents=True, exist_ok=True)
        bin_path = lm_dir / "4-gram.bin"
        arpa_gz_path = lm_dir / "4-gram.arpa.gz"
        if not bin_path.exists():
            url_download(LM_ARPA_GZ_URL, arpa_gz_path)
            build_binary = shutil.which("build_binary")
            if build_binary:
                log(f"compiling {arpa_gz_path} -> {bin_path} via KenLM build_binary ...")
                subprocess.run([build_binary, "trie", str(arpa_gz_path), str(bin_path)], check=True)
                log(f"compiled KenLM binary: {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)")
            else:
                log(f"downloaded {arpa_gz_path}; install KenLM build_binary to compile {bin_path} (or pass {arpa_gz_path} directly)")
        else:
            log(f"exists: {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)")

    log("ALL DOWNLOADS COMPLETE")


if __name__ == "__main__":
    main()
