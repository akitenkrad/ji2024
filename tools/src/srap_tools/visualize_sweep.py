#!/usr/bin/env python3
"""
visualize_sweep.py — Ji et al. (2024) SRAP-Agent スイープ / POA 可視化スクリプト

`sweep` の親 run (または `poa` の run) を読み，ポリシー因子 (入室条件 × 資源
サブセット) 依存の SW と，POA の適応度収束曲線を可視化する．旧
`sweep_summary.csv` の 1 行は «セル 1 つの試行 1 本» で，いまはセル座標を
`parameters` に持つ子 run 1 本にあたる (`srap_tools.runs.sweep_table` が
同じ列の表に組み直す)．POA の世代パネルは `x.ji2024.poa_generation` イベント．

Usage:
    uv run srap-tools visualize-sweep
    uv run srap-tools visualize-sweep --sweep_dir results/20260525_120000_sweep  # legacy も可
    uv run srap-tools visualize-sweep --subcommand poa

Outputs (存在するデータに応じて):
    <results-root>/<experiment>/figures/<run_slug>/
    ├── sweep_sw_by_subset.png   ← 資源サブセット別の平均 SW (入室条件で色分け)
    └── poa_convergence.png      ← POA 適応度 f(π) の世代収束曲線
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from srap_tools import runs

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
        default=None,
        help="スイープ / POA の run ディレクトリ (省略時は runvault path --latest)",
    )
    p.add_argument(
        "--subcommand",
        default="sweep",
        choices=("sweep", "poa"),
        help="--sweep_dir 省略時にどの run を探すか (default: sweep)",
    )
    p.add_argument(
        "--results_root",
        "--results-root",
        default="results",
        help="results root (default: results)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先 (default: <results-root>/<experiment>/figures/<run_slug>)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = runs.resolve_run_dir(
        args.sweep_dir, subcommand=args.subcommand, results_root=args.results_root
    )
    out_dir = str(runs.analysis_output_dir(run_dir, args.output_dir))

    print("=== Ji et al. (2024) SRAP-Agent スイープ / POA 可視化 ===")
    print(f"結果:   {run_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    produced = False

    subcommand = (
        runs.subcommand_of(run_dir) if runs.is_runvault_run(run_dir) else None
    )
    if subcommand in (None, "sweep"):
        try:
            df = runs.sweep_table(run_dir)
        except (FileNotFoundError, SystemExit):
            df = None
        if df is not None and not df.empty:
            print("[sweep] 資源サブセット別 SW を保存中 ...")
            save_sweep_sw_by_subset(df, os.path.join(out_dir, "sweep_sw_by_subset.png"))
            produced = True

    history = runs.poa_history(run_dir)
    if history:
        print("[poa] 適応度収束曲線を保存中 ...")
        # `poa` は 1 目標なのでそのまま描く (2 目標の `reproduce` は
        # `srap-tools reproduce` が Fig.4 として重ねて描く)．
        _, df = sorted(history.items())[0]
        save_poa_convergence(df, os.path.join(out_dir, "poa_convergence.png"))
        produced = True

    if not produced:
        raise FileNotFoundError(
            f"スイープの子 run も POA の世代パネルも見つかりません: {run_dir}"
        )

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:30s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
