#!/usr/bin/env python3
"""Plot training curves and Whisper vs EmbeddingGemma 2 benchmark charts from `results/`."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_steps_csv(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {
        "step": [], "epoch": [], "loss": [], "lr": [],
        "step_per_s": [], "tok_per_s": [], "scale": [], "blank_bias": [],
    }
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def plot_training_curves(results_dir: Path, out_paths: list[Path]) -> None:
    apply_style()
    v1_steps = load_steps_csv(results_dir / "v1" / "steps.csv")
    v2_steps = load_steps_csv(results_dir / "v2" / "steps.csv")
    v3_steps = load_steps_csv(results_dir / "v3" / "steps.csv")

    v1_m = load_json(results_dir / "v1" / "metrics.json")
    v2_m = load_json(results_dir / "v2" / "metrics.json")
    v3_m = load_json(results_dir / "v3" / "metrics.json")
    v2_beam = load_json(results_dir / "v2" / "wer_beam_4gram.json")

    c_v2 = "#1d4ed8"   # Royal blue (ReaderA 960h)
    c_v3 = "#d97706"   # Warm amber (ReaderB k=6 960h)
    c_v1 = "#64748b"   # Slate gray (ReaderA 100h)
    c_lm = "#059669"   # Emerald (+ 4-gram LM)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.4), dpi=300)
    fig.suptitle(
        "EmbeddingGemma 2 ASR Reader Training Dynamics on 8× Cloud TPU v5e (TorchTPU)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    # Panel (a): Step-Level & Per-Epoch CTC Training Loss
    ax = axes[0, 0]
    ax.plot(v2_steps["step"] / 1e3, v2_steps["loss"], color=c_v2, alpha=0.32, lw=1.1, label="_nolegend_")
    ax.plot(v3_steps["step"] / 1e3, v3_steps["loss"], color=c_v3, alpha=0.32, lw=1.1, label="_nolegend_")
    ax.plot(v1_steps["step"] / 1e3, v1_steps["loss"], color=c_v1, alpha=0.30, lw=1.0, ls=":", label="_nolegend_")

    v2_ep_s = [h["step"] / 1e3 for h in v2_m["history"]]
    v2_ep_l = [h["train_loss_mean"] for h in v2_m["history"]]
    v3_ep_s = [h["step"] / 1e3 for h in v3_m["history"]]
    v3_ep_l = [h["train_loss_mean"] for h in v3_m["history"]]
    v1_ep_s = [h["step"] / 1e3 for h in v1_m["history"]]
    v1_ep_l = [h["train_loss_mean"] for h in v1_m["history"]]

    ax.plot(v2_ep_s, v2_ep_l, "o-", color=c_v2, lw=2.2, ms=5.0, label="v2 ReaderA (19.6M, 960h) — Epoch Mean & 500-Step")
    ax.plot(v3_ep_s, v3_ep_l, "s-", color=c_v3, lw=2.2, ms=4.8, label="v3 ReaderB k=6 (34.5M, 960h) — Epoch Mean & 500-Step")
    ax.plot(v1_ep_s, v1_ep_l, "^--", color=c_v1, lw=1.6, ms=4.0, label="v1 ReaderA (19.6M, 100h) — 20 Epochs (Overfits Loss)")

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-3, 15.0)
    ax.set_xlabel("Optimization Steps (× 1,000, global batch = 32 across 8× TPU v5e)")
    ax.set_ylabel("200,001-Way Word CTC Loss (log scale)")
    ax.set_title("(a) Training CTC Loss Convergence (Step-Level & Epoch Mean)", y=1.08)
    ax.grid(True, which="both")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ep_ticks = [e * 8.172 for e in range(2, 13, 2)]
    ax_top.set_xticks(ep_ticks)
    ax_top.set_xticklabels([f"Ep {e}" for e in range(2, 13, 2)], fontsize=8.5, color="#475569")
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["top"].set_color("#cbd5e1")
    ax_top.spines["right"].set_visible(False)

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.6)

    # Panel (b): Validation dev-clean WER Trajectory
    ax = axes[0, 1]
    v2_wer = [100.0 * h["dev"]["wer"] for h in v2_m["history"]]
    v3_wer = [100.0 * h["dev"]["wer"] for h in v3_m["history"]]
    v1_wer = [100.0 * h["dev"]["wer"] for h in v1_m["history"]]

    ax.plot(v1_ep_s, v1_wer, "^--", color=c_v1, lw=1.8, ms=4.5, label="v1 ReaderA (100h, Greedy): best 7.55% @ Ep 17")
    ax.plot(v3_ep_s, v3_wer, "s-", color=c_v3, lw=2.2, ms=5.2, label="v3 ReaderB k=6 (960h, Greedy): best 4.45% @ Ep 11")
    ax.plot(v2_ep_s, v2_wer, "o-", color=c_v2, lw=2.4, ms=5.5, label="v2 ReaderA (960h, Greedy): best 4.19% (4.14% WN) @ Ep 10")

    b2_idx = v2_m["best_epoch"] - 1
    b3_idx = v3_m["best_epoch"] - 1
    b1_idx = v1_m["best_epoch"] - 1
    ax.scatter([v2_ep_s[b2_idx]], [v2_wer[b2_idx]], s=150, marker="*", color=c_v2, zorder=6, edgecolors="black", linewidths=0.6)
    ax.scatter([v3_ep_s[b3_idx]], [v3_wer[b3_idx]], s=150, marker="*", color=c_v3, zorder=6, edgecolors="black", linewidths=0.6)
    ax.scatter([v1_ep_s[b1_idx]], [v1_wer[b1_idx]], s=120, marker="*", color=c_v1, zorder=6, edgecolors="black", linewidths=0.6)

    v2_lm_raw = 100.0 * v2_beam["results"]["dev-clean"]["wer"]
    v2_lm_wn = 100.0 * v2_beam["results"]["dev-clean"]["wer_whisper_norm"]
    ax.axhline(
        v2_lm_wn, color=c_lm, ls="-.", lw=1.6, alpha=0.85,
        label=f"v2 + 4-Gram Beam (dev-clean): {v2_lm_wn:.2f}% Whisper-norm ({v2_lm_raw:.2f}% raw)",
    )
    ax.annotate(
        "Ep 1 (960h): 7.76%\n(matches 20-ep 100h after 1 pass)",
        xy=(v2_ep_s[0], v2_wer[0]),
        xytext=(21, 11.8),
        fontsize=8.3,
        color="#1e293b",
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
    )
    ax.annotate(
        f"Best Greedy: {v2_wer[b2_idx]:.2f}%\n+ 4-Gram LM: {v2_lm_wn:.2f}% WN",
        xy=(v2_ep_s[b2_idx], v2_wer[b2_idx]),
        xytext=(58, 6.8),
        fontsize=8.5,
        fontweight="bold",
        color=c_v2,
        arrowprops=dict(arrowstyle="->", color=c_v2, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#93c5fd", lw=0.8),
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(2.8, 24.5)
    ax.set_xlabel("Optimization Steps (× 1,000)")
    ax.set_ylabel("LibriSpeech dev-clean WER (%) ↓")
    ax.set_title("(b) Validation WER Convergence (All 169 OOV Utterances Retained)", y=1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.4)

    # Panel (c): Error Type Decomposition Across 960h Epochs (v2 ReaderA)
    ax = axes[1, 0]
    epochs_960 = np.array([h["epoch"] for h in v2_m["history"]])
    ref_w = v2_m["history"][0]["dev"]["ref_words"]
    sub_rate = np.array([100.0 * h["dev"]["substitutions"] / ref_w for h in v2_m["history"]])
    del_rate = np.array([100.0 * h["dev"]["deletions"] / ref_w for h in v2_m["history"]])
    ins_rate = np.array([100.0 * h["dev"]["insertions"] / ref_w for h in v2_m["history"]])

    ax.plot(epochs_960, sub_rate, "o-", color="#2563eb", lw=2.2, ms=5.0, label="Substitutions (%) — Greedy")
    ax.plot(epochs_960, del_rate, "s-", color="#dc2626", lw=1.9, ms=4.5, label="Deletions (%) — Greedy")
    ax.plot(epochs_960, ins_rate, "^-", color="#7c3aed", lw=1.9, ms=4.5, label="Insertions (%) — Greedy")

    lm_res = v2_beam["results"]["dev-clean"]
    lm_sub = 100.0 * lm_res["substitutions"] / lm_res["ref_words"]
    lm_del = 100.0 * lm_res["deletions"] / lm_res["ref_words"]
    lm_ins = 100.0 * lm_res["insertions"] / lm_res["ref_words"]
    ax.scatter([12.8], [lm_sub], s=95, marker="D", color="#2563eb", edgecolors="black", zorder=6)
    ax.scatter([12.8], [lm_del], s=80, marker="D", color="#dc2626", edgecolors="black", zorder=6)
    ax.scatter([12.8], [lm_ins], s=80, marker="D", color="#7c3aed", edgecolors="black", zorder=6)
    ax.axvline(12.4, color="#94a3b8", ls=":", lw=1.2)
    ax.annotate(
        f"+4-Gram\nSub: {lm_sub:.2f}%\n(−0.52 pp)",
        xy=(12.8, lm_sub),
        xytext=(10.2, 4.35),
        fontsize=8.3,
        color="#1e3a8a",
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#bfdbfe", lw=0.8),
    )

    ax.set_xticks(list(range(1, 13)) + [12.8])
    ax.set_xticklabels([str(e) for e in range(1, 13)] + ["+LM"])
    ax.set_xlim(0.6, 13.4)
    ax.set_ylim(0.0, 6.8)
    ax.set_xlabel("Training Epoch on 960h LibriSpeech (v2 ReaderA) + Best w/ 4-Gram LM")
    ax.set_ylabel("Error Rate Component (% of 54,402 ref words)")
    ax.set_title("(c) dev-clean Error Decomposition: Substitutions Dominate & Drop w/ LM")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.6)

    # Panel (d): Learned Cosine Temperature Scale exp(log_scale) & LR
    ax = axes[1, 1]
    ax.plot(v2_steps["step"] / 1e3, v2_steps["scale"], color=c_v2, lw=2.2, label="v2 ReaderA Scale exp(log_scale): 20 → 174.6")
    ax.plot(v3_steps["step"] / 1e3, v3_steps["scale"], color=c_v3, lw=2.2, ls="--", label="v3 ReaderB Scale exp(log_scale): 20 → 174.3")
    ax.plot(v1_steps["step"] / 1e3, v1_steps["scale"], color=c_v1, lw=1.8, ls=":", label="v1 ReaderA (100h) Scale: 20 → 109.1")

    ax.set_xlim(0, 100)
    ax.set_ylim(15, 190)
    ax.set_xlabel("Optimization Steps (× 1,000)")
    ax.set_ylabel("Learned Cosine Logit Scale exp(log_scale)")
    ax.set_title("(d) Learned Cosine Logit Scale & Warmup-Cosine LR Schedule")
    ax.grid(True)

    ax_lr = ax.twinx()
    ax_lr.plot(v2_steps["step"] / 1e3, v2_steps["lr"] * 1e4, color="#0f766e", lw=1.5, alpha=0.55, ls="-.", label="v2 LR (×10⁻⁴)")
    ax_lr.plot(v3_steps["step"] / 1e3, v3_steps["lr"] * 1e4, color="#be123c", lw=1.5, alpha=0.50, ls="-.", label="v3 Backbone LR (×10⁻⁴)")
    ax_lr.set_ylabel("Learning Rate (× 10⁻⁴)", color="#0f766e")
    ax_lr.set_ylim(0, 4.6)
    ax_lr.spines["top"].set_visible(False)
    ax_lr.spines["right"].set_visible(True)
    ax_lr.spines["right"].set_color("#94a3b8")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)

    plt.tight_layout(rect=[0, 0, 1, 0.962])
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
        print(f"Saved training curves -> {p}")
    plt.close(fig)


def plot_whisper_comparison(results_dir: Path, whisper_json: Path, out_paths: list[Path]) -> None:
    apply_style()
    wdata = load_json(whisper_json)["models"]
    v2_g = load_json(results_dir / "v2" / "wer.json")["results"]
    v2_b = load_json(results_dir / "v2" / "wer_beam_4gram.json")["results"]
    v2_sb_g = load_json(results_dir / "v2" / "speckled_band_8chip.json")
    v2_sb_b = load_json(results_dir / "v2" / "speckled_band_8chip_beam4gram.json")

    v3_g = load_json(results_dir / "v3" / "wer.json")["results"]
    v3_b = load_json(results_dir / "v3" / "wer_beam_4gram.json")["results"]
    v3_sb_g = load_json(results_dir / "v3" / "speckled_band_8chip.json")
    v3_sb_b = load_json(results_dir / "v3" / "speckled_band_8chip_beam4gram.json")

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.6), dpi=300)
    fig.suptitle(
        "EmbeddingGemma 2 Word-CTC vs All 11 OpenAI Whisper Sizes on 8× Cloud TPU v5e",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    en_models = [k for k in wdata if k.endswith(".en")]
    multi_models = [k for k in wdata if not k.endswith(".en")]

    eg2_audio_m = 305.61
    v2_tot_m = eg2_audio_m + 19.57
    v3_tot_m = eg2_audio_m + 34.49

    def _plot_scale_panel(ax, split_name: str, title: str, y_top: float) -> None:
        if multi_models:
            mx = [wdata[k]["params"] / 1e6 for k in multi_models]
            my = [100.0 * wdata[k]["splits"][split_name]["wer_whisper_norm"] for k in multi_models]
            ax.plot(mx, my, "o--", color="#64748b", lw=1.7, ms=6.8, label="Whisper Multilingual (tiny → large-v3)")
            for k, x, y in zip(multi_models, mx, my):
                lbl = wdata[k]["short_name"].replace("whisper-", "")
                if "turbo" in lbl:
                    off = (-22, -13)
                elif lbl == "large-v3":
                    off = (0, -14)
                elif lbl == "large-v2":
                    off = (0, 8)
                elif lbl == "medium" and split_name == "test-clean":
                    off = (-24, -8)
                else:
                    off = (0, 8)
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=8.1, color="#475569")

        if en_models:
            ex = [wdata[k]["params"] / 1e6 for k in en_models]
            ey = [100.0 * wdata[k]["splits"][split_name]["wer_whisper_norm"] for k in en_models]
            ax.plot(ex, ey, "s-", color="#0284c7", lw=2.0, ms=7.0, label="Whisper English-Only (.en)")
            for k, x, y in zip(en_models, ex, ey):
                lbl = wdata[k]["short_name"].replace("whisper-", "")
                if lbl == "medium.en" and split_name == "test-clean":
                    off = (0, 8)
                elif lbl == "medium.en":
                    off = (-14, -14)
                else:
                    off = (0, -14)
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=8.2, color="#0369a1", fontweight="bold")

        g_v2 = 100.0 * v2_g[split_name]["wer_whisper_norm"]
        b_v2 = 100.0 * v2_b[split_name]["wer_whisper_norm"]
        b_v3 = 100.0 * v3_b[split_name]["wer_whisper_norm"]

        ax.scatter([19.57], [b_v2], s=135, marker="*", facecolors="none", edgecolors="#059669", linewidths=1.8, zorder=7,
                   label="EG2 v2 + 4-Gram (Trainable Reader: 19.6M)")
        ax.plot([19.57, v2_tot_m], [b_v2, b_v2], ":", color="#059669", lw=1.3, alpha=0.75)
        ax.scatter([v2_tot_m], [g_v2], s=140, marker="D", color="#1d4ed8", edgecolors="black", linewidths=0.6, zorder=7,
                   label=f"EG2 v2 Greedy (325M total): {g_v2:.2f}%")
        ax.scatter([v2_tot_m], [b_v2], s=205, marker="*", color="#059669", edgecolors="black", linewidths=0.7, zorder=8,
                   label=f"EG2 v2 + 4-Gram (325M total): {b_v2:.2f}%")
        ax.scatter([v3_tot_m], [b_v3], s=165, marker="*", color="#d97706", edgecolors="black", linewidths=0.6, zorder=7,
                   label=f"EG2 v3 + 4-Gram (340M total): {b_v3:.2f}%")

        ax.set_xscale("log")
        ax.set_xlim(15, 2200)
        ax.set_ylim(1.2, y_top)
        ax.set_xlabel("Active Audio-Encoder + Decoder Parameters (Millions, log scale)")
        ax.set_ylabel(f"LibriSpeech {split_name} WER (%, Whisper-Norm) ↓")
        ax.set_title(title)
        ax.grid(True, which="both")
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.1)

    # Panel (a): test-clean WER vs Scale
    _plot_scale_panel(axes[0, 0], "test-clean", "(a) Clean Speech Accuracy vs Model Scale (test-clean)", y_top=8.4)

    # Panel (b): test-other WER vs Scale
    _plot_scale_panel(axes[0, 1], "test-other", "(b) Challenging Speech Accuracy vs Model Scale (test-other)", y_top=18.2)

    # Panel (c): LibriSpeech test-other End-to-End Speed (x Realtime) vs WER
    ax = axes[1, 0]
    aud_to_s = wdata["openai/whisper-base.en"]["splits"]["test-other"]["audio_seconds"]  # 19229.57s (5.34h)
    eg2_enc_to_s = aud_to_s / 3120.0  # 3,120x RT soft-token extraction on 8x TPU v5e (~6.16s)

    mx_c = [wdata[k]["splits"]["test-other"]["x_realtime_e2e"] for k in multi_models]
    my_c = [100.0 * wdata[k]["splits"]["test-other"]["wer_whisper_norm"] for k in multi_models]
    ax.scatter(mx_c, my_c, s=65, marker="o", color="#64748b", label="Whisper Multilingual (8× TPU v5e)")
    for k, x, y in zip(multi_models, mx_c, my_c):
        lbl = wdata[k]["short_name"].replace("whisper-", "")
        off = (0, 7)
        if lbl == "large-v3":
            off = (-4, -13)
        elif lbl == "large-v3-turbo":
            off = (0, -14)
        elif lbl == "medium":
            off = (-16, 7)
        elif lbl == "tiny":
            off = (0, 7)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=8.0, color="#475569")

    ex_c = [wdata[k]["splits"]["test-other"]["x_realtime_e2e"] for k in en_models]
    ey_c = [100.0 * wdata[k]["splits"]["test-other"]["wer_whisper_norm"] for k in en_models]
    ax.scatter(ex_c, ey_c, s=70, marker="s", color="#0284c7", label="Whisper English-Only (8× TPU v5e)")
    for k, x, y in zip(en_models, ex_c, ey_c):
        lbl = wdata[k]["short_name"].replace("whisper-", "")
        off = (0, -13) if lbl != "medium.en" else (18, -12)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=8.1, color="#0369a1", fontweight="bold")

    v2_g_rt = aud_to_s / (eg2_enc_to_s + v2_g["test-other"]["decode_s"])
    v2_b_rt = aud_to_s / (eg2_enc_to_s + v2_b["test-other"]["total_decode_s"])
    v3_b_rt = aud_to_s / (eg2_enc_to_s + v3_b["test-other"]["total_decode_s"])

    ax.scatter([v2_g_rt], [100.0 * v2_g["test-other"]["wer_whisper_norm"]], s=145, marker="D", color="#1d4ed8",
               edgecolors="black", linewidths=0.7, zorder=8,
               label=f"EG2 v2 Greedy: {v2_g_rt:.0f}× RT, {100*v2_g['test-other']['wer_whisper_norm']:.2f}% WER")
    ax.scatter([v2_b_rt], [100.0 * v2_b["test-other"]["wer_whisper_norm"]], s=210, marker="*", color="#059669",
               edgecolors="black", linewidths=0.7, zorder=9,
               label=f"EG2 v2 + 4-Gram: {v2_b_rt:.0f}× RT, {100*v2_b['test-other']['wer_whisper_norm']:.2f}% WER")
    ax.scatter([v3_b_rt], [100.0 * v3_b["test-other"]["wer_whisper_norm"]], s=170, marker="*", color="#d97706",
               edgecolors="black", linewidths=0.6, zorder=8,
               label=f"EG2 v3 + 4-Gram: {v3_b_rt:.0f}× RT, {100*v3_b['test-other']['wer_whisper_norm']:.2f}% WER")

    ax.set_xscale("log")
    ax.set_xlim(65, 2500)
    ax.set_ylim(2.2, 18.4)
    ax.set_xlabel("8-Chip TPU End-to-End Throughput on test-other (× Realtime, log scale) →")
    ax.set_ylabel("LibriSpeech test-other WER (%, Whisper-Norm) ↓")
    ax.set_title("(c) Batch Evaluation Speed–Accuracy Frontier (5.34h test-other)")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.0)

    # Panel (d): Long-Form 54.3-Min Speckled Band Speed (x Realtime) vs WER
    ax = axes[1, 1]
    w_lf_multi = [k for k in multi_models if wdata[k].get("speckled_band")]
    w_lf_en = [k for k in en_models if wdata[k].get("speckled_band")]

    if w_lf_multi:
        sx = [wdata[k]["speckled_band"]["x_realtime_e2e"] for k in w_lf_multi]
        sy = [100.0 * wdata[k]["speckled_band"]["wer_whisper_norm"] for k in w_lf_multi]
        ax.scatter(sx, sy, s=65, marker="o", color="#64748b", label="Whisper Multilingual (8× TPU v5e)")
        for k, x, y in zip(w_lf_multi, sx, sy):
            lbl = wdata[k]["short_name"].replace("whisper-", "")
            off = (0, 7)
            if lbl == "tiny":
                off = (0, 8)
            elif lbl == "large-v3":
                off = (-4, 8)
            elif lbl == "medium":
                off = (0, 8)
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=8.0, color="#475569")

    if w_lf_en:
        sx = [wdata[k]["speckled_band"]["x_realtime_e2e"] for k in w_lf_en]
        sy = [100.0 * wdata[k]["speckled_band"]["wer_whisper_norm"] for k in w_lf_en]
        ax.scatter(sx, sy, s=70, marker="s", color="#0284c7", label="Whisper English-Only (8× TPU v5e)")
        for k, x, y in zip(w_lf_en, sx, sy):
            lbl = wdata[k]["short_name"].replace("whisper-", "")
            off = (0, -13)
            if lbl == "tiny.en":
                off = (-6, -13)
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=8.1, color="#0369a1", fontweight="bold")

    ax.scatter([v2_sb_g["x_realtime"]], [100.0 * v2_sb_g["wer_whisper_norm"]], s=145, marker="D", color="#1d4ed8",
               edgecolors="black", linewidths=0.7, zorder=8,
               label=f"EG2 v2 Greedy: {v2_sb_g['x_realtime']:.0f}× RT, {100*v2_sb_g['wer_whisper_norm']:.2f}% WER")
    ax.scatter([v2_sb_b["x_realtime"]], [100.0 * v2_sb_b["wer_whisper_norm"]], s=210, marker="*", color="#059669",
               edgecolors="black", linewidths=0.7, zorder=9,
               label=f"EG2 v2 + 4-Gram: {v2_sb_b['x_realtime']:.0f}× RT, {100*v2_sb_b['wer_whisper_norm']:.2f}% WER")
    ax.scatter([v3_sb_g["x_realtime"]], [100.0 * v3_sb_g["wer_whisper_norm"]], s=125, marker="D", color="#d97706",
               edgecolors="black", linewidths=0.6, zorder=7,
               label=f"EG2 v3 Greedy: {v3_sb_g['x_realtime']:.0f}× RT, {100*v3_sb_g['wer_whisper_norm']:.2f}% WER")
    ax.scatter([v3_sb_b["x_realtime"]], [100.0 * v3_sb_b["wer_whisper_norm"]], s=170, marker="*", color="#d97706",
               edgecolors="black", linewidths=0.6, zorder=7,
               label=f"EG2 v3 + 4-Gram: {v3_sb_b['x_realtime']:.0f}× RT, {100*v3_sb_b['wer_whisper_norm']:.2f}% WER")

    ax.set_xscale("log")
    ax.set_xlim(120, 4200)
    ax.set_ylim(0.85, 6.2)
    ax.set_xlabel("8-Chip TPU End-to-End Throughput (× Realtime, log scale) →")
    ax.set_ylabel("Long-Form WER (%, Whisper-Norm) ↓")
    ax.set_title("(d) Single-File 54.3-Min Audio Speed–Accuracy (Speckled Band)")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.9)

    plt.tight_layout(rect=[0, 0, 1, 0.962])
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
        print(f"Saved Whisper comparison -> {p}")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(repo_root / "results"))
    ap.add_argument("--whisper-json", default=str(repo_root / "results" / "whisper_tpu_benchmark.json"))
    ap.add_argument("--out-dir", default=str(repo_root / "assets"))
    ap.add_argument("--extra-out-dir", default=None)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    tc_outs = [out_dir / "training_curves.png"]
    wc_outs = [out_dir / "whisper_comparison.png"]
    if args.extra_out_dir:
        extra = Path(args.extra_out_dir)
        tc_outs.append(extra / "training_curves.png")
        wc_outs.append(extra / "whisper_comparison.png")

    plot_training_curves(res_dir, tc_outs)
    w_path = Path(args.whisper_json)
    if w_path.exists():
        plot_whisper_comparison(res_dir, w_path, wc_outs)


if __name__ == "__main__":
    main()
