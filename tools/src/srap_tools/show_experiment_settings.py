"""srap-tools show-experiment-settings — 実行結果の設定表示．

runvault の run ディレクトリの `config.json` (封筒．条件は `parameters` の下) を
読み，実行時に使われた全パラメータを整形表示する．どのサブコマンドの run かは
`run.json` の `subcommand` が答える．旧 `llm_meta.json` の中身は run スコープの指標
(`llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate`) と `run.json` の `llm`
ブロックへ移ったので，そちらから拾う．legacy な flat `config.json` /
`sweep_config.json` / `poa_config.json` / `llm_meta.json` も従来どおり読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment srap-allocation --latest --subcommand run
    runvault path --experiment srap-allocation --latest --subcommand sweep

Usage:
    srap-tools show-experiment-settings
    srap-tools show-experiment-settings --results-dir results/20260525_120000
    srap-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, run_scope_metrics

from srap_tools import runs

# config キー → 表示ラベル (右コロン位置を揃えるため空白パディング済み)．
FIELD_LABELS = {
    "n_applicants": "応募者数 n_applicants ",
    "pool_ratio": "プール比 pool_ratio   ",
    "n_resources": "資源数 n_resources    ",
    "entry_condition": "入室条件 E_queue      ",
    "sort_strategy": "並び替え S_queue      ",
    "resource_subset": "資源サブセット R_queue",
    "m": "キュー数 m            ",
    "k": "k-deferrals k         ",
    "c": "選択キュー容量係数 c  ",
    "max_rounds": "最大ラウンド          ",
    "visible_subset_size": "可視サブセットサイズ  ",
    "decision_mode": "駆動方式 (mock/live)  ",
    "seed": "シード (コア)         ",
    "base_seed": "シード基点            ",
    "runs": "試行数 runs           ",
    "objective": "最適化目標 objective  ",
    "iterations": "反復世代数 iterations ",
    "pool_size": "個体群サイズ          ",
    "mutation_rate": "突然変異率            ",
    "tournament_size": "トーナメントサイズ    ",
    "use_predictor": "予測器で枝刈り        ",
    "poa_iterations": "POA 世代数            ",
    "poa_pool_size": "POA 個体群サイズ      ",
    "poa_mutation_rate": "POA 突然変異率        ",
    "poa_tournament_size": "POA トーナメント      ",
    "poa_use_predictor": "POA 予測器で枝刈り    ",
    "quick": "短縮モード quick      ",
    "llm_temperature": "LLM 温度              ",
    "llm_seed": "LLM seed              ",
    "llm_cache_path": "LLM キャッシュ        ",
    # legacy 用．
    "queues": "キュー数 m            ",
    "fitness_kind": "適応度評価            ",
    "output_dir": "出力先                ",
}

#: 格子 (親 run) が持つリスト項目．
LIST_KEYS = {
    "entry_conditions": "入室条件 候補     ",
    "resource_subsets": "資源サブセット候補",
    "queue_strategies": "並び替え 候補     ",
}


def render_config(cfg: dict, run_dir: Path, subcommand: str) -> str:
    lines = ["=" * 70, f"実行設定 ({subcommand})", "=" * 70, f"run: {run_dir}", "-" * 70]
    for key, label in LIST_KEYS.items():
        if key in cfg:
            lines.append(f"{label}: {', '.join(str(v) for v in cfg[key])}")
    for key, label in FIELD_LABELS.items():
        if key in cfg:
            lines.append(f"{label}: {cfg[key]}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm(meta: dict, scoped: dict) -> str:
    """`run.json` の `llm` ブロックと LLM 指標を表示する (旧 `llm_meta.json`)．"""
    llm = meta.get("llm") or {}
    lines = ["", "LLM 実行メタデータ (run.json の llm ブロック + run スコープ指標)", "-" * 70]
    lines.append(f"provider      : {llm.get('provider', '-')}")
    lines.append(f"モデル        : {llm.get('model_snapshot', '-')}")
    lines.append(f"温度          : {llm.get('temperature', '-')}")
    for name, label in (("llm_calls", "呼び出し総数  "), ("llm_cache_hits", "cache-hit     ")):
        if name in scoped:
            lines.append(f"{label}: {int(scoped[name])}")
    if "llm_cache_hit_rate" in scoped:
        lines.append(f"cache-hit 率  : {scoped['llm_cache_hit_rate'] * 100:.1f}%")
    if "final_round" in scoped:
        lines.append(f"最終ラウンド  : {int(scoped['final_round'])}")
    lines.append("=" * 70)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# legacy (移行前の flat な JSON)
# --------------------------------------------------------------------------- #

LEGACY_FILES = (("config.json", "run"), ("sweep_config.json", "sweep"), ("poa_config.json", "poa"))


def _legacy(results_dir: Path) -> tuple[dict, str, dict | None]:
    for name, kind in LEGACY_FILES:
        path = results_dir / name
        if path.exists():
            with path.open() as f:
                cfg = json.load(f)
            meta_path = results_dir / "llm_meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            return cfg, kind, meta
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        "  期待: run.json (runvault) / config.json / sweep_config.json / poa_config.json"
    )


def render_legacy_llm(meta: dict) -> str:
    lines = ["", "LLM 実行メタデータ (llm_meta.json)", "-" * 70]
    lines.append(f"モデル        : {meta.get('llm_model', '-')}")
    lines.append(f"endpoint      : {meta.get('llm_endpoint', '-')}")
    lines.append(f"温度          : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed          : {meta.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数  : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit     : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率  : {rate * 100:.1f}%")
    lines.append(f"最終ラウンド  : {meta.get('final_round', '-')}")
    lines.append(f"最終 SW       : {meta.get('final_sw', '-')}")
    lines.append(f"配分人数      : {meta.get('final_n_allocated', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="srap-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path --latest)",
    )
    parser.add_argument(
        "--subcommand",
        default="run",
        help="--results-dir 省略時にどの run を探すか (default: run)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="results root (default: results)",
    )
    parser.add_argument("--json", action="store_true", help="表ではなく JSON 形式で出力する．")
    args = parser.parse_args(argv)

    run_dir = runs.resolve_run_dir(
        args.results_dir, subcommand=args.subcommand, results_root=args.results_root
    )
    if not run_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {run_dir}", file=sys.stderr)
        return 1

    if runs.is_runvault_run(run_dir):
        meta = load_run_meta(run_dir)
        assert meta is not None
        cfg = config_parameters(run_dir) or {}
        scoped = run_scope_metrics(run_dir)
        subcommand = meta["subcommand"]
        if args.json:
            payload = {
                "run_dir": str(run_dir),
                "subcommand": subcommand,
                "parameters": cfg,
                "llm": meta.get("llm"),
                "run_scope_metrics": scoped,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(render_config(cfg, run_dir, subcommand))
            if meta.get("llm") is not None:
                print(render_llm(meta, scoped))
        return 0

    try:
        cfg, kind, legacy_meta = _legacy(run_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {"run_dir": str(run_dir), "kind": kind, "config": cfg, "llm_meta": legacy_meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_config(cfg, run_dir, f"{kind}; legacy"))
        if legacy_meta is not None:
            print(render_legacy_llm(legacy_meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
