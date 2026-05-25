"""srap-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run) / sweep_config.json (sweep) /
poa_config.json (poa) を読み，実行時に使われた全パラメータを整形表示する．
存在すれば llm_meta.json の LLM 情報も併せて表示する．`results/latest` も解決される．

Usage:
    srap-tools show-experiment-settings
    srap-tools show-experiment-settings --results-dir results/20260525_120000
    srap-tools show-experiment-settings --results-dir results/latest --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_results_dir(arg: str) -> Path:
    """ユーザ指定の results_dir を絶対パスに解決する (symlink も実体へ)．"""
    p = Path(arg)
    if not p.is_absolute():
        candidates = [Path.cwd() / arg, p]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            p = candidates[0]
    return Path(os.path.realpath(p))


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run) / sweep_config.json (sweep) / poa_config.json (poa) を探す．"""
    for name, kind in (
        ("config.json", "run"),
        ("sweep_config.json", "sweep"),
        ("poa_config.json", "poa"),
    ):
        path = results_dir / name
        if path.exists():
            return path, kind
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待: config.json (run) / sweep_config.json (sweep) / poa_config.json (poa)"
    )


def _load_json(results_dir: Path, name: str) -> dict | None:
    path = results_dir / name
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def render_run_config(cfg: dict, source: Path) -> str:
    lines = ["=" * 70, "実行設定 (run)", "=" * 70, f"設定ファイル: {source}", "-" * 70]
    lines.append(f"応募者数 n_applicants : {cfg.get('n_applicants', '-')}")
    lines.append(f"プール比 pool_ratio   : {cfg.get('pool_ratio', '-')}")
    lines.append(f"資源数 n_resources    : {cfg.get('n_resources', '-')}")
    lines.append(f"入室条件 E_queue      : {cfg.get('entry_condition', '-')}")
    lines.append(f"並び替え S_queue      : {cfg.get('sort_strategy', '-')}")
    lines.append(f"資源サブセット R_queue: {cfg.get('resource_subset', '-')}")
    lines.append(f"キュー数 m            : {cfg.get('m', '-')}")
    lines.append(f"k-deferrals k         : {cfg.get('k', '-')}")
    lines.append(f"選択キュー容量係数 c  : {cfg.get('c', '-')}")
    lines.append(f"最大ラウンド          : {cfg.get('max_rounds', '-')}")
    lines.append(f"可視サブセットサイズ  : {cfg.get('visible_subset_size', '-')}")
    lines.append(f"シード (コア)         : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度              : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed              : {cfg.get('llm_seed', '-')}")
    lines.append(f"出力先                : {cfg.get('output_dir', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines = ["=" * 70, "実行設定 (sweep)", "=" * 70, f"設定ファイル: {source}", "-" * 70]
    lines.append(f"入室条件 候補    : {', '.join(map(str, cfg.get('entry_conditions', [])))}")
    lines.append(f"資源サブセット候補: {', '.join(map(str, cfg.get('resource_subsets', [])))}")
    lines.append(f"並び替え 候補    : {', '.join(map(str, cfg.get('queue_strategies', [])))}")
    lines.append(f"応募者数         : {cfg.get('n_applicants', '-')}")
    lines.append(f"プール比         : {cfg.get('pool_ratio', '-')}")
    lines.append(f"m / k / c        : {cfg.get('queues', '-')} / {cfg.get('k', '-')} / {cfg.get('c', '-')}")
    lines.append(f"最大ラウンド     : {cfg.get('max_rounds', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_poa_config(cfg: dict, source: Path) -> str:
    lines = ["=" * 70, "実行設定 (poa; Phase-3 最小スタブ)", "=" * 70, f"設定ファイル: {source}", "-" * 70]
    lines.append(f"最適化目標 objective : {cfg.get('objective', '-')}")
    lines.append(f"反復世代数 iterations: {cfg.get('iterations', '-')}")
    lines.append(f"個体群サイズ pool_size: {cfg.get('pool_size', '-')}")
    lines.append(f"突然変異率           : {cfg.get('mutation_rate', '-')}")
    lines.append(f"トーナメントサイズ   : {cfg.get('tournament_size', '-')}")
    lines.append(f"応募者数 / プール比  : {cfg.get('n_applicants', '-')} / {cfg.get('pool_ratio', '-')}")
    lines.append(f"最大ラウンド         : {cfg.get('max_rounds', '-')}")
    lines.append(f"シード               : {cfg.get('seed', '-')}")
    note = cfg.get("note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
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
    note = meta.get("determinism_note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
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
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument("--json", action="store_true", help="表ではなく JSON 形式で出力する．")
    args = parser.parse_args(argv)

    results_dir = _resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    cfg_path, kind = _find_config_file(results_dir)
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = _load_json(results_dir, "llm_meta.json")

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm_meta": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        elif kind == "sweep":
            print(render_sweep_config(cfg, cfg_path))
        else:
            print(render_poa_config(cfg, cfg_path))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
