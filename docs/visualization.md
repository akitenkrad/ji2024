[English](visualization.md) | [日本語](visualization.ja.md)

# Visualization

The Python package `srap-tools` (module `srap_tools`) renders figures from the Rust outputs. Install with `uv sync` at the workspace root; run with `uv run srap-tools <subcommand>`.

## `visualize` — single run

Resolves the `run` parent with `runvault path --latest --subcommand run` (or takes `--results_dir`), reads every trial child's `metrics.csv`, rebuilds the old `(run, round, …)` table and averages per round across trials.

```bash
uv run srap-tools visualize
uv run srap-tools visualize --results_dir results/20260525_120000   # legacy directories still work
```

Outputs to `<results-root>/srap-allocation/figures/<run_slug>/` — outside the run directory, because `manifest.csv` is settled by `finish()` and a figure drawn afterwards would carry no hash:

- `welfare_timeseries.png` — SW and Avg r_size over rounds (satisfaction).
- `fairness_timeseries.png` — Gini and F(V,NV) over rounds (fairness).
- `final_metrics.png` — the final-round metric summary (bar chart).

## `visualize-sweep` — sweep / POA

Reads the sweep parent's trial children (or a `poa` run's generation panel). `--sweep_dir` takes a directory; otherwise `--subcommand sweep|poa` picks which run `runvault path --latest` should return.

```bash
uv run srap-tools visualize-sweep
uv run srap-tools visualize-sweep --subcommand poa
uv run srap-tools visualize-sweep --sweep_dir results/20260525_120000_sweep   # legacy も可
```

Outputs (depending on what the run recorded):

- `sweep_sw_by_subset.png` — average SW by resource subset, coloured by entry condition. A scripted-mock sweep does not assess the paper's policy ordering; see [Reproduction](reproduction.md).
- `poa_convergence.png` — POA best/mean fitness over generations (Fig. 4-style; best is monotonically non-decreasing under elitism).

## `show-experiment-settings`

Prints a run's `parameters` (from `config.json`), its `llm` block and its run-scope LLM metrics — the old `llm_meta.json` split into those two places. `--results-dir` takes a directory (legacy flat `config.json` / `sweep_config.json` / `poa_config.json` / `llm_meta.json` still read), `--subcommand` picks which run `--latest` returns, `--json` prints JSON.

## `reproduce`

Reads a `reproduce` run: the Table 2 cells and the POA panels from `events.jsonl`, the verdicts from `artifacts/reproduce_summary.json`. Re-prints the observed-vs-paper checks and writes `table2_sw_by_policy.png` (policy social-welfare bars) and `fig4_poa_convergence.png` (POA fitness convergence) into the figures directory. `--results-dir` selects a directory; without it the latest `reproduce` run is used. `initial_fitness` is not recorded — it is generation 0's `best_fitness` — so the table rebuilds it from the panel.

## Resolving a run

`runvault path` is invoked as `$RUNVAULT`, else `runvault` on PATH. Without an installed binary, either point at it (`export RUNVAULT=<rs-runvault>/target/debug/runvault`) or pass the run directory directly.
