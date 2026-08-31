#!/usr/bin/env python3
"""
visualize.py — Ji et al. (2024) SRAP-Agent 単一実行結果 可視化スクリプト

`run` の親 run (指定が無ければ `runvault path --latest --subcommand run`) から
試行ごとの子 run を辿り，
(1) 満足度・公平性指標のラウンド時系列 (社会的厚生 SW・Avg r_size・Gini・F(V,NV)) と，
(2) 最終ラウンドの指標サマリ (棒グラフ)
を生成する．試行は子 run に分かれているので，旧 `metrics.csv` と同じ
(run, round + 指標列) の表に組み直してから，ラウンドごとに試行平均を取って描画する．

Usage:
    uv run srap-tools visualize
    uv run srap-tools visualize --results_dir results/20260525_120000   # legacy も可
    uv run srap-tools visualize --output_dir out

Outputs:
    <results-root>/<experiment>/figures/<run_slug>/
    ├── welfare_timeseries.png   ← SW / Avg r_size の時系列 (満足度)
    ├── fairness_timeseries.png  ← Gini / F(V,NV) の時系列 (公平性)
    └── final_metrics.png        ← 最終ラウンドの指標サマリ (棒グラフ)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from srap_tools import runs

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_SW = "#1565C0"
COLOR_RSIZE = "#2E7D32"
COLOR_GINI = "#C62828"
COLOR_FVNV = "#6A1B9A"


def round_means(df: pd.DataFrame) -> pd.DataFrame:
    """ラウンドごとに試行平均を取る (複数 run を集約)．"""
    return df.groupby("round").mean(numeric_only=True).reset_index()


def save_welfare_timeseries(agg: pd.DataFrame, out_path: str) -> None:
    """満足度 (SW / Avg r_size) の時系列を描く．"""
    fig, ax1 = plt.subplots(figsize=(11, 5.5), facecolor=COLOR_BG)
    ax1.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — 満足度の推移", fontsize=14)

    ax1.plot(agg["round"], agg["sw"], color=COLOR_SW, lw=2.4, marker="o", label="社会的厚生 SW")
    ax1.set_xlabel("配分ラウンド t")
    ax1.set_ylabel("社会的厚生 SW", color=COLOR_SW)
    ax1.tick_params(axis="y", labelcolor=COLOR_SW)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        agg["round"],
        agg["avg_rsize"],
        color=COLOR_RSIZE,
        lw=2,
        marker="s",
        linestyle="--",
        label="一人当たり平均面積 Avg r_size",
    )
    ax2.set_ylabel("一人当たり平均面積 Avg r_size", color=COLOR_RSIZE)
    ax2.tick_params(axis="y", labelcolor=COLOR_RSIZE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_fairness_timeseries(agg: pd.DataFrame, out_path: str) -> None:
    """公平性 (Gini / F(V,NV)) の時系列を描く．"""
    fig, ax1 = plt.subplots(figsize=(11, 5.5), facecolor=COLOR_BG)
    ax1.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — 公平性の推移", fontsize=14)

    ax1.plot(agg["round"], agg["co_gini"], color=COLOR_GINI, lw=2.4, marker="o", label="Gini 係数")
    ax1.set_xlabel("配分ラウンド t")
    ax1.set_ylabel("Gini 係数 (0=平等, 1=不平等)", color=COLOR_GINI)
    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis="y", labelcolor=COLOR_GINI)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        agg["round"],
        agg["f_vnv"],
        color=COLOR_FVNV,
        lw=2,
        marker="s",
        linestyle="--",
        label="脆弱層ギャップ F(V,NV)",
    )
    ax2.set_ylabel("F(V,NV) (大=不公平)", color=COLOR_FVNV)
    ax2.tick_params(axis="y", labelcolor=COLOR_FVNV)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_final_metrics(agg: pd.DataFrame, out_path: str) -> None:
    """最終ラウンドの指標サマリ (棒グラフ)．"""
    last = agg.iloc[-1]
    labels = ["SW", "Avg r_size", "Avg WT", "Var r_size", "Rop", "Gini", "F(V,NV)"]
    values = [
        last["sw"],
        last["avg_rsize"],
        last["avg_wt"],
        last["var_rsize"],
        last["rop"],
        last["co_gini"],
        last["f_vnv"],
    ]
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — 最終ラウンドの指標", fontsize=14)
    colors = ["#1565C0", "#2E7D32", "#00897B", "#F9A825", "#EF6C00", "#C62828", "#6A1B9A"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_ylabel("値")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="srap-tools visualize",
        description="Ji et al. (2024) SRAP-Agent 単一実行結果 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path --latest --subcommand run)",
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
        help="図の保存先ディレクトリ (default: <results-root>/<experiment>/figures/<run_slug>)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = runs.resolve_run_dir(
        args.results_dir, subcommand="run", results_root=args.results_root
    )
    out_dir = str(runs.analysis_output_dir(run_dir, args.output_dir))

    print("=== Ji et al. (2024) SRAP-Agent 単一実行結果 可視化 ===")
    print(f"結果:   {run_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    df = runs.pooled_metrics(run_dir)
    agg = round_means(df)
    print(f"      {df['round'].nunique()} ラウンド × {df['run'].nunique()} 試行")

    print("[1/3] 満足度の時系列を保存中 ...")
    save_welfare_timeseries(agg, os.path.join(out_dir, "welfare_timeseries.png"))
    print("[2/3] 公平性の時系列を保存中 ...")
    save_fairness_timeseries(agg, os.path.join(out_dir, "fairness_timeseries.png"))
    print("[3/3] 最終指標サマリを保存中 ...")
    save_final_metrics(agg, os.path.join(out_dir, "final_metrics.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:30s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
