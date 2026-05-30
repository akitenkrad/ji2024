#!/usr/bin/env python3
"""srap-tools reproduce — Ji et al. (2024) 論文 Table 2/3・Fig.4 の図示と照合．

Rust の `srap reproduce` が書き出した再現ディレクトリ (`results/reproduce_<ts>/`) を
読み，論文の知見を図とサマリで提示する:

  - Table 2: 入室条件 × 資源サブセット の社会的厚生 SW (棒グラフ)．論文の知見
    «r_size 最高 / r_random 最低» を matched-seed paired win-rate で照合する．
  - Table 3: POA 最適化ポリシー π_s* (満足度志向) / π_f* (公平性志向)．
  - Figure 4: POA の適応度 f(π) の世代収束曲線 (満足度 / 公平性)．

論文値は GPT-3.5-turbo の合成環境固有なので **絶対値ではなく順序・符号** を再現目標と
する (設計書 §7)．`reproduce_summary.json` の観測 vs 論文知見と PASS / off-anchor を
そのまま表示し，図を `<results-dir>/figures/` に書き出す．

Usage:
    srap-tools reproduce --results-dir results/latest
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
SUBSET_COLORS = {"r_size": "#1565C0", "r_rent": "#43A047", "r_random": "#E53935"}
OBJ_COLORS = {"satisfaction": "#1565C0", "fairness": "#8E24AA"}


def _save_table2_fig(df: pd.DataFrame, out_path: str) -> None:
    """入室条件 × 資源サブセット の平均 SW を棒グラフで描く (Table 2)．"""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — Table 2: ポリシーと社会的厚生", fontsize=14)

    entries = sorted(df["entry_condition"].unique())
    subsets = ["r_size", "r_rent", "r_random"]
    subsets = [s for s in subsets if s in set(df["resource_subset"])]
    x = np.arange(len(entries))
    width = 0.8 / max(len(subsets), 1)

    for i, sub in enumerate(subsets):
        means = [
            df[(df["entry_condition"] == e) & (df["resource_subset"] == sub)]["mean_sw"].mean()
            for e in entries
        ]
        ax.bar(
            x + i * width,
            means,
            width=width,
            color=SUBSET_COLORS.get(sub, "#888"),
            label=f"R={sub}",
        )

    ax.set_xticks(x + width * (len(subsets) - 1) / 2)
    ax.set_xticklabels(entries)
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xlabel("入室条件 E_queue")
    ax.set_ylabel("平均 社会的厚生 SW")
    ax.set_title("論文の知見: r_size 最高 / r_random 最低 (matched-seed 比較で顕著)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _save_fig4(histories: dict[str, pd.DataFrame], out_path: str) -> None:
    """POA の f(π) 世代収束曲線を描く (Figure 4; 満足度 / 公平性)．"""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.suptitle("Ji et al. (2024) SRAP-Agent — Figure 4: POA 適応度収束", fontsize=14)

    for obj, df in histories.items():
        color = OBJ_COLORS.get(obj, "#555")
        ax.plot(
            df["generation"],
            df["best_fitness"],
            color=color,
            lw=2.4,
            marker="o",
            label=f"{obj} 最良 f(π)",
        )
        if "mean_fitness" in df.columns:
            ax.plot(
                df["generation"],
                df["mean_fitness"],
                color=color,
                lw=1.4,
                linestyle="--",
                alpha=0.6,
                label=f"{obj} 個体群平均",
            )

    ax.set_xlabel("世代 (generation)")
    ax.set_ylabel("適応度 f(π)")
    ax.set_title("エリート保存 → 最良適応度は単調非減少 (GA + 予測器 f̃)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "off-anchor"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="srap-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    out_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "reproduce_summary.json")
    table2_path = os.path.join(results_dir, "table2_sw_by_policy.csv")
    table3_path = os.path.join(results_dir, "table3_optimized_policies.csv")
    if not os.path.exists(summary_path):
        print(
            f"error: {summary_path} が見つかりません．\n"
            f"  先に `cargo run --release -- reproduce --mock` を実行してください．",
            file=sys.stderr,
        )
        return 1

    summary = json.load(open(summary_path))

    print("=" * 70)
    print("Ji et al. (2024) SRAP-Agent — Table 2/3・Fig.4 再現")
    print("=" * 70)
    cfg = summary.get("config", {})
    print(
        f"mode={summary.get('mode')} quick={summary.get('quick')} "
        f"n_applicants={cfg.get('n_applicants')} runs={cfg.get('runs')} "
        f"POA iters={cfg.get('poa_iterations')} pool={cfg.get('poa_pool_size')}\n"
    )

    # ── Table 2 ──────────────────────────────────────────────────────────────
    t2 = summary.get("table2", {})
    print("[Table 2] 入室条件 × 資源サブセット の社会的厚生 (headline 条件 p_select):")
    print(
        f"  平均 SW: r_size={t2.get('headline_mean_sw_r_size', float('nan')):.2f} "
        f"r_rent={t2.get('headline_mean_sw_r_rent', float('nan')):.2f} "
        f"r_random={t2.get('headline_mean_sw_r_random', float('nan')):.2f}"
    )
    print(
        f"  最高 平均 SW ポリシー: E={t2.get('best_policy_entry')} "
        f"R={t2.get('best_policy_subset')} (SW={t2.get('best_policy_mean_sw', float('nan')):.2f}) "
        f"[論文: p_select + r_size]"
    )
    for name, chk in t2.get("checks", {}).items():
        print(
            f"  {name:<22} = {str(chk['observed']):<22} "
            f"(論文: {chk['paper']}; {_verdict(chk['pass'])})"
        )

    # ── Table 3 ──────────────────────────────────────────────────────────────
    print("\n[Table 3] POA 最適化ポリシー:")
    for row in summary.get("table3", []):
        print(
            f"  {row['objective']:<13} π*: {row['policy']} | "
            f"f(π) {row['initial_fitness']:.2f} → {row['final_fitness']:.2f} "
            f"({row['improvement_pct']:+.1f}%) | "
            f"フル評価 {row['full_evals']} 省略 {row['evals_saved']}"
        )

    # ── checks ────────────────────────────────────────────────────────────────
    print("\n[照合]")
    for name, chk in summary.get("checks", {}).items():
        print(f"  {name:<28} = {str(chk['observed']):<10} (論文: {chk['paper']}; {_verdict(chk['pass'])})")
    overall = summary.get("overall_pass", False)
    print(f"\n総合判定: {_verdict(overall)}")

    # ── 図の生成 ────────────────────────────────────────────────────────────────
    print("\n[図の生成]")
    if os.path.exists(table2_path):
        _save_table2_fig(pd.read_csv(table2_path), os.path.join(out_dir, "table2_sw_by_policy.png"))
    # Fig.4: poa_history_<objective>.csv を全て読む．
    histories: dict[str, pd.DataFrame] = {}
    for obj in ("satisfaction", "fairness"):
        p = os.path.join(results_dir, f"poa_history_{obj}.csv")
        if os.path.exists(p):
            histories[obj] = pd.read_csv(p)
    if histories:
        _save_fig4(histories, os.path.join(out_dir, "fig4_poa_convergence.png"))

    print("=" * 70)
    if os.path.exists(table3_path):
        print(f"[reproduce] Table 3 CSV: {table3_path}")
    print(f"[reproduce] 図: {out_dir}/")
    return 0 if overall else 0  # 図示ツールなので verdict によらず正常終了．


if __name__ == "__main__":
    raise SystemExit(main())
