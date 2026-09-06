[English](cli.md) | [日本語](cli.ja.md)

# CLI reference

The Rust binary is `srap` (built by `cargo build --release`). Subcommands: `run`, `sweep`, `poa`, `reproduce`.

## Common environment flags (`run`, `sweep`, `poa`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-applicants` | 60 | number of synthetic applicants |
| `--pool-ratio` | 0.5 | n_resources / n_applicants (scarcity) |
| `--max-rounds` | 10 | round cap (early stop on pool exhaustion / all settled) |
| `--visible-subset-size` | 5 | size cap of the visible resource subset V(p_j) |

> The synthetic-environment parameters (applicant count, resource count, profile distributions) are not fully specified in the paper body (see [Reproduction](reproduction.md)); they are externalised here with sensible defaults.

## `run` — single policy

```bash
cargo run --release -- run \
    --entry-condition p_select --resource-subset r_size --queue-strategy fifo \
    --queues 3 --k 3 --c 2 --runs 10 --seed 42
```

| Flag | Default | Values / meaning |
|------|---------|------------------|
| `--entry-condition` | `p_select` | `p_budget` / `p_family` / `p_select` (E_queue) |
| `--resource-subset` | `r_size` | `r_size` / `r_rent` / `r_random` (R_queue) |
| `--queue-strategy` | `fifo` | `fifo` / `vfa` / `vfr` (S_queue) |
| `--queues` | 3 | m (number of queues) |
| `--k` | 3 | k-deferrals |
| `--c` | 2 | selection-queue capacity factor |
| `--runs` | 1 | independent trials (each derived from the seed) |
| `--seed` | random | core RNG seed |
| `--temperature` | 0.0 | LLM temperature |
| `--llm-seed` | 0 | LLM backend seed |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache (ignored with `--mock`) |
| `--output-dir` | `results` | output base |
| `--mock` | false | use a scripted mock instead of a live LLM (offline) |

Records one **parent run** (the condition and the trial count) plus one **child run per trial** (`subcommand=run-replicate`), under `<output-dir>/srap-allocation/<run_slug>/`. runvault names the directories; there is no timestamp directory and no `latest` symlink. Each child holds `metrics.csv` (per-round `sw` / `avg_rsize` / `avg_wt` / `var_rsize` / `rop` / `co_gini` / `f_vnv` / `n_allocated`, plus run-scope `final_round` and `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate`), the condition under `config.json`'s `parameters`, and the model/endpoint/temperature in `run.json`'s `llm` block.

## `sweep` — policy-factor sensitivity

```bash
cargo run --release -- sweep \
    --entry-conditions p_budget,p_family,p_select \
    --resource-subsets r_size,r_rent,r_random \
    --queue-strategies fifo --runs 30 --seed 42
```

Records one **parent run** holding the grid definition plus one **child run per trial** (`subcommand=sweep-point`). The cell coordinates (`entry_condition`, `resource_subset`, `sort_strategy`) live in each child's `parameters`, so the old `metrics.csv` — which stacked every cell's rows without a cell column — is gone: trials of the same cell now share a `config_hash`, and `srap_tools.runs.sweep_table` rebuilds the old `sweep_summary.csv` columns from the children.

## `poa` — Policy Optimization Agent

```bash
cargo run --release -- poa --objective satisfaction \
    --iterations 50 --pool-size 50 --use-predictor --seed 42 --mock
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--objective` | `satisfaction` | `satisfaction` / `fairness` (the f(π) weighting) |
| `--iterations` | 20 | GA generations M |
| `--pool-size` | 12 | population size |
| `--mutation-rate` | 0.2 | per-gene mutation probability |
| `--tournament-size` | 3 | tournament size |
| `--use-predictor` | off | enable the surrogate `f̃` to prune full evaluations |
| `--mock` | off | scripted-mock fitness (offline); omit for live-LLM fitness |
| `--cache-path` | `.llm_cache/cache.json` | persistent prompt cache (live fitness; shared across generations) |
| `--seed` | 42 | RNG seed |

Fitness = one SRAP allocation run scored by `f_pi(metrics, objective)`. Records one run whose generation panel goes to `events.jsonl` as `x.ji2024.poa_generation` (`unit_id` = objective, `t` = generation, `t_unit` = `generation`: best/mean fitness, cumulative `full_evals` / `evals_saved`, and the generation's best policy), and whose final π\* goes to `x.ji2024.poa_best_policy` (policy, `final_fitness`, `improvement_pct`, `total_full_evals`, `total_evals_saved`). The panel is not in `metrics.csv` because `reproduce` carries two objectives on the same generation axis and `metrics.csv` has no series column — one shape for both subcommands. Best fitness is monotonically non-decreasing (elitism); with `--use-predictor` the surrogate skips evaluations that cannot beat the current elite.

## `reproduce` — Table 2/3 + Fig. 4

```bash
cargo run --release -- reproduce --mock --seed 42      # --quick for a fast smoke
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--runs` | 10 | trials per matched-seed condition (Table 2) |
| `--poa-iterations` | 20 | POA generations (Table 3 / Fig. 4) |
| `--poa-pool-size` | 16 | POA population size |
| `--mock` | off | scripted-mock evaluation (offline); omit for live LLM |
| `--quick` | off | shrink the POA budget / applicants for a fast smoke |
| `--seed` | 42 | RNG seed |

Reproduces Table 2 (matched-seed paired social-welfare ordering; the paper's `r_size` highest / `r_random` lowest finding, reported as a paired win-rate), Table 3 (POA-optimized `π_s*` / `π_f*`) and Figure 4 (POA convergence). It is **one run**, not a sweep: the paired comparison spans conditions, so splitting the conditions into child runs would claim executions that never happened. Table 2's cells go to `events.jsonl` as `x.ji2024.table2_cell`, the two optimizations as `x.ji2024.poa_generation` / `x.ji2024.poa_best_policy`, the win-rates as run-scope metrics (`n_matched_environments`, `paired_wins_*`, `paired_win_rate_*`), and the PASS / off-anchor verdicts as `artifacts/reproduce_summary.json` — a verdict and its band are neither an observation nor a number the paper printed. Render the figures with `uv run srap-tools reproduce`.
