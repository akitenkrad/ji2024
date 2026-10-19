"""runs.py — どの run を読むか，そして試行をどう束ねるか．

Rust 側は `run` / `sweep` 1 回を «親 run 1 本 + 試行ごとの子 run» として記録する．
旧 `metrics.csv` が持っていた «同一条件の試行を 1 枚にプールした表» は，runvault
では子 run に分かれている — `metrics.csv` の主キーは (`run_uid`, `step`,
`step_unit`, `scope`, `name`) なので，seed の違う行が同じ `round` を名乗ることは
できないからである．旧 `sweep` の `metrics.csv` に至ってはセル列すら無く，
どの行がどのセルのものかは seed からしか辿れなかった．

このモジュールはその表を読み側で組み直す．旧形式と同じ列 (`run`, `round`, …) を
返すので，`visualize` の計算そのものは移行前後で 1 セルも変わらない．`sweep` は
セル座標が子 run の `parameters` にあるので，旧 `sweep_summary.csv` と同じ列を
持つ表として組み直せる．

run ディレクトリの解決は `runvault path` に任せる — `results/` を走査して
新しそうなディレクトリを当てにいかない．run_slug には条件と環境のハッシュが
入るので，こちらで名前を組み立てることはできないし，できたとしてもすべきでない．
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from runvault.read import (
    config_parameters,
    load_run_meta,
    run_scope_metrics,
    run_subcommand,
    sweep_children,
)
from runvault.read import runvault_path as _runvault_path

EXPERIMENT = "srap-allocation"

#: 試行を子 run として持つ («親» の) サブコマンド．
PARENT_SUBCOMMANDS = ("run", "sweep")

#: 旧 `metrics.csv` の指標列 (この順)．
METRIC_NAMES = [
    "sw",
    "avg_rsize",
    "avg_wt",
    "var_rsize",
    "rop",
    "co_gini",
    "f_vnv",
    "n_allocated",
]


def resolve_run_dir(
    results_dir: str | os.PathLike | None,
    *,
    subcommand: str = "run",
    results_root: str = "results",
) -> Path:
    """読む run ディレクトリを決める．

    `results_dir` が与えられていればそれを使う (legacy な `results/<timestamp>/` も
    そのまま渡せる)．与えられていなければ `runvault path --latest` に聞く．
    """
    if results_dir is not None:
        path = Path(results_dir)
        # legacy の `results/latest` シンボリックリンクは実体へ解決する．
        if path.is_symlink():
            return Path(os.path.realpath(path))
        return path
    return Path(_runvault_path(EXPERIMENT, results_root, subcommand=subcommand))


def is_runvault_run(run_dir: str | os.PathLike) -> bool:
    """runvault の run ディレクトリか (`run.json` があるか)．"""
    return load_run_meta(run_dir, required=False) is not None


def trial_dirs(run_dir: str | os.PathLike) -> list[Path]:
    """試行 1 本ずつの run ディレクトリ．

    親 run を渡すとその子を，単独の run を渡すとそれ自身 1 本を返す．legacy な
    ディレクトリもそれ自身 1 本になる (1 枚の CSV に試行が入っているため)．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return [run_dir]
    if run_subcommand(run_dir) in PARENT_SUBCOMMANDS:
        children = [Path(c) for c in sweep_children(run_dir)]
        if not children:
            raise SystemExit(
                f"エラー: 親 run に子がありません: {run_dir}\n"
                "  子は lineage.parent_run_uid で親を指します．親と子が同じ"
                " results root にあるか確認してください．"
            )
        return children
    return [run_dir]


def subcommand_of(run_dir: str | os.PathLike) -> str:
    """その run がどのサブコマンドの実行か (`run` / `sweep-point` / `poa` / …)．"""
    return run_subcommand(run_dir)


def replicate_index(run_dir: str | os.PathLike) -> int:
    """その run がどの試行か．旧 `metrics.csv` の `run` 列にあたる．"""
    meta = load_run_meta(run_dir)
    assert meta is not None  # required=True raises rather than returning None
    index = (meta.get("rng") or {}).get("replicate_index")
    return 0 if index is None else int(index)


def master_seed(run_dir: str | os.PathLike) -> int:
    """その run を駆動したシード．旧 `sweep_summary.csv` の `seed` 列にあたる．"""
    meta = load_run_meta(run_dir)
    assert meta is not None
    seed = (meta.get("rng") or {}).get("master_seed")
    if seed is None:
        raise SystemExit(f"エラー: master_seed を持たない run です: {run_dir}")
    return int(seed)


def _metrics_wide(metrics_path: Path) -> pd.DataFrame:
    """long 形式の `metrics.csv` を «1 ラウンド 1 行» に戻す．

    `runvault.read.metrics_wide` と同じ形を返すが，読み込みに
    `float_precision="round_trip"` を渡す点だけが違う．pandas の既定の C パーサは
    f64 を 1 ULP 落とすことがあり (例: `0.26666666666666666` →
    `0.2666666666666666`)，移行前後の値の照合が «実際には一致しているのに
    一致しない» と出る．記録された数をそのまま読むために，ここでは自前で読む．
    """
    df = pd.read_csv(metrics_path, float_precision="round_trip")
    stepped = df[df["step"].notna()]
    return (
        stepped.pivot_table(index="step", columns="name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .astype({"step": int})
        .sort_values("step")
        .reset_index(drop=True)
    )


def pooled_metrics(run_dir: str | os.PathLike) -> pd.DataFrame:
    """試行をプールしたラウンド指標．旧 `metrics.csv` と同じ列を返す．

    子 run の long 形式 `metrics.csv` を wide に戻し，`run` 列を `run.json` の
    `replicate_index` から補って縦に積む．legacy な wide `metrics.csv` は
    そのまま返す．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")

    frames = []
    for child in trial_dirs(run_dir):
        wide = _metrics_wide(child / "metrics.csv").rename(columns={"step": "round"})
        wide.insert(0, "run", replicate_index(child))
        frames.append(wide)
    return pd.concat(frames, ignore_index=True)


def sweep_table(run_dir: str | os.PathLike) -> pd.DataFrame:
    """試行 1 本 1 行の表．旧 `sweep_summary.csv` と同じ列を返す．

    セル座標 (入室条件・資源サブセット・並び替え戦略) は子 run の `parameters` に
    あり，最終ラウンドの値は子の `metrics.csv` の最後の行にある．旧 CSV が
    «セル列を持たない `metrics.csv`» と «セル列を持つ `sweep_summary.csv`» に
    割れていたのが，ここでは 1 つの表に戻る．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "sweep_summary.csv", float_precision="round_trip")

    rows = []
    for child in trial_dirs(run_dir):
        params = config_parameters(child) or {}
        scoped = run_scope_metrics(child)
        wide = _metrics_wide(child / "metrics.csv")
        last = wide.iloc[-1]
        row = {
            "entry_condition": params.get("entry_condition"),
            "resource_subset": params.get("resource_subset"),
            "queue_strategy": params.get("sort_strategy"),
            "run": replicate_index(child),
            "seed": master_seed(child),
            "final_round": scoped.get("final_round"),
        }
        for name in METRIC_NAMES:
            column = name if name == "n_allocated" else f"final_{name}"
            row[column] = last[name]
        row["cache_hit_rate"] = scoped.get("llm_cache_hit_rate")
        row["run_dir"] = str(child)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["entry_condition", "resource_subset", "queue_strategy", "run"])
        .reset_index(drop=True)
    )


def _events(run_dir: Path, kind: str) -> list[dict]:
    """`events.jsonl` から 1 種別を読む．

    `runvault.read.events_table` を使わないのは，こちらが空のときに例外ではなく
    空リストを返してほしいためである (`poa` の run に Table 2 は無い)．
    """
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") == kind:
                rows.append(row)
    return rows


def poa_history(run_dir: str | os.PathLike) -> dict[str, pd.DataFrame]:
    """最適化目標ごとの世代パネル．旧 `poa_history*.csv` と同じ列を返す．

    `poa` は 1 目標，`reproduce` は 2 目標．どちらも
    `x.ji2024.poa_generation` イベントの `unit_id` が目標名を持つ．
    legacy なディレクトリは `poa_history.csv` / `poa_history_<目標>.csv` を読む．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        out = {}
        for path in sorted(run_dir.glob("poa_history*.csv")):
            label = path.stem.replace("poa_history", "").lstrip("_") or "poa"
            out[label] = pd.read_csv(path, float_precision="round_trip")
        return out

    frames: dict[str, list[dict]] = {}
    for event in _events(run_dir, "x.ji2024.poa_generation"):
        row = {k: v for k, v in event.items() if k not in ("schema", "run_uid", "ts", "unit_id")}
        row["generation"] = int(row.pop("t"))
        row.pop("t_unit", None)
        frames.setdefault(event["unit_id"], []).append(row)
    return {
        label: pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
        for label, rows in frames.items()
    }


def poa_best_policies(run_dir: str | os.PathLike) -> pd.DataFrame:
    """最適化目標ごとの π\\*．旧 `table3_optimized_policies.csv` に対応する．

    `initial_fitness` は記録に無い (世代 0 の `best_fitness` そのものなので
    2 箇所に置かない)．表として要るのでここで世代パネルから補う．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "table3_optimized_policies.csv", float_precision="round_trip")

    history = poa_history(run_dir)
    rows = []
    for event in _events(run_dir, "x.ji2024.poa_best_policy"):
        objective = event["objective"]
        first = history[objective].iloc[0]
        rows.append(
            {
                "objective": objective,
                "entry_condition": event["entry_condition"],
                "sort_strategy": event["sort_strategy"],
                "resource_subset": event["resource_subset"],
                "m": event["m"],
                "k": event["k"],
                "c": event["c"],
                "initial_fitness": float(first["best_fitness"]),
                "final_fitness": event["final_fitness"],
                "improvement_pct": event["improvement_pct"],
                "full_evals": event["total_full_evals"],
                "evals_saved": event["total_evals_saved"],
            }
        )
    return pd.DataFrame(rows)


def table2(run_dir: str | os.PathLike) -> pd.DataFrame:
    """Table 2 のセル表．旧 `table2_sw_by_policy.csv` と同じ列を返す．"""
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "table2_sw_by_policy.csv", float_precision="round_trip")
    rows = [
        {
            "entry_condition": e["entry_condition"],
            "resource_subset": e["resource_subset"],
            "mean_sw": e["mean_sw"],
            "runs": e["runs"],
        }
        for e in _events(run_dir, "x.ji2024.table2_cell")
    ]
    return pd.DataFrame(rows)


def reproduce_verdict(run_dir: str | os.PathLike) -> dict:
    """`reproduce` の判定 (`artifacts/reproduce_summary.json`)．

    95% 信頼区間による順序判定は論文の報告値ではなく `artifacts/` にある．
    legacy なディレクトリでは run
    ディレクトリ直下にあり，中身の形も違う (観測値を同居させていた)．
    """
    run_dir = Path(run_dir)
    if is_runvault_run(run_dir):
        path = run_dir / "artifacts" / "reproduce_summary.json"
    else:
        path = run_dir / "reproduce_summary.json"
    if not path.exists():
        raise SystemExit(
            f"エラー: {path} が見つかりません．\n"
            "  先に `cargo run --release -- reproduce --mock` を実行してください．"
        )
    with path.open() as f:
        summary = json.load(f)
    if not is_runvault_run(run_dir):
        # legacy は Table 2 の判定を `table2.checks` に，残りを `checks` に分けて
        # 持っていた．表示側が 1 箇所だけ見れば済むように束ねる．
        checks = dict(summary.get("table2", {}).get("checks", {}))
        checks.update(summary.get("checks", {}))
        summary["checks"] = checks
    return summary


def analysis_output_dir(run_dir: str | os.PathLike, override: str | None) -> Path:
    """図の置き場．

    `manifest.csv` は `finish()` が確定させるので，run が終わったあとに作ったものを
    `artifacts/` に置くとハッシュを持たない (＝記録の一部でない) ファイルが混ざる．
    runvault の run には `runvault.read.figures_dir` が示す run ディレクトリの外を
    使う．
    """
    if override is not None:
        out = Path(override)
    elif is_runvault_run(run_dir):
        from runvault.read import figures_dir

        out = Path(figures_dir(run_dir))
    else:
        out = Path(run_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out
