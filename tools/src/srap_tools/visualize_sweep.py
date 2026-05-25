#!/usr/bin/env python3
"""
visualize_sweep.py — Ji et al. (2024) SRAP-Agent スイープ / POA 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv があれば
ポリシー因子 (入室条件 × 資源サブセット) 依存の SW を，poa_history.csv があれば
POA の適応度収束曲線を可視化する．

Usage:
    uv run srap-tools visualize-sweep
    uv run srap-tools visualize-sweep --sweep_dir results/20260525_120000_sweep

Outputs (存在するデータに応じて):
    output_dir/
    ├── sweep_sw_by_subset.png   ← 資源サブセット別の平均 SW (入室条件で色分け)
    └── poa_convergence.png      ← POA 適応度 f(π) の世代収束曲線
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
ENTRY_COLORS = ["#FF9800", "#03A9F4", "#8BC34A", "#E91E63", "#795548"]


def save_sweep_sw_by_subset(df: pd.DataFrame, out_path: str) -> None:
    """資源サブセット別の平均 SW を入室条件で色分けして棒グラフで描く．"""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — 資源サブセットと社会的厚生", fontsize=14)

    subsets = sorted(df["resource_subset"].unique())
    entries = sorted(df["entry_condition"].unique())
    x = np.arange(len(subsets))
    width = 0.8 / max(len(entries), 1)

    for i, entry in enumerate(entries):
        sub = df[df["entry_condition"] == entry]
        means = [
            sub[sub["resource_subset"] == s]["final_sw"].mean() if not sub.empty else np.nan
            for s in subsets
        ]
        ax.bar(
            x + i * width,
            means,
            width=width,
            color=ENTRY_COLORS[i % len(ENTRY_COLORS)],
            label=f"E={entry}",
        )

    ax.set_xticks(x + width * (len(entries) - 1) / 2)
    ax.set_xticklabels(subsets)
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xlabel("資源サブセット R_queue")
    ax.set_ylabel("平均 社会的厚生 SW")
    ax.set_title("論文の知見: r_size 最高 / r_random 最低 (同一シード比較で顕著)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_poa_convergence(df: pd.DataFrame, out_path: str) -> None:
    """POA の最良/平均適応度の世代収束曲線を描く (Fig.4 風)．"""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — POA 適応度収束 (Fig.4 風)", fontsize=14)

    ax.plot(df["generation"], df["best_fitness"], color="#1565C0", lw=2.4, marker="o", label="最良 f(π)")
    if "mean_fitness" in df.columns:
        ax.plot(
            df["generation"],
            df["mean_fitness"],
            color="#9E9E9E",
            lw=1.6,
            linestyle="--",
            marker="s",
            label="個体群平均 f(π)",
        )
    ax.set_xlabel("世代 (generation)")
    ax.set_ylabel("適応度 f(π)")
    ax.set_title("エリート保存 → 最良適応度は単調非減少 (Phase-3 最小スタブ)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="srap-tools visualize-sweep",
        description="Ji et al. (2024) SRAP-Agent スイープ / POA 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="スイープ / POA の出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先 (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Ji et al. (2024) SRAP-Agent スイープ / POA 可視化 ===")
    print(f"結果:   {args.sweep_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    sweep_path = os.path.join(args.sweep_dir, "sweep_summary.csv")
    poa_path = os.path.join(args.sweep_dir, "poa_history.csv")
    produced = False

    if os.path.exists(sweep_path):
        print("[sweep] 資源サブセット別 SW を保存中 ...")
        df = pd.read_csv(sweep_path)
        save_sweep_sw_by_subset(df, os.path.join(out_dir, "sweep_sw_by_subset.png"))
        produced = True
    if os.path.exists(poa_path):
        print("[poa] 適応度収束曲線を保存中 ...")
        df = pd.read_csv(poa_path)
        save_poa_convergence(df, os.path.join(out_dir, "poa_convergence.png"))
        produced = True

    if not produced:
        raise FileNotFoundError(
            f"sweep_summary.csv も poa_history.csv も見つかりません: {args.sweep_dir}"
        )

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:30s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
