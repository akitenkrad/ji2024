[English](reproduction.md) | [日本語](reproduction.ja.md)

# Reproduction

## Quantitative targets (paper)

| Condition | Metric | Paper | Reproduction target |
|-----------|--------|-------|---------------------|
| entry `p_select`, subset `r_size` | SW (social welfare) | 419.9 (highest) | sign/ordering |
| subset `r_random` | SW | 83.8 (lowest) | sign/ordering |
| POA after 50 iters | f(π) improvement | ≈ +20% | direction |
| satisfaction-oriented π_s* | SW improvement | +1.8% | direction |
| fairness-oriented π_f* | social-fairness improvement | +13.2% | direction |

## What is exact vs qualitative

- **Deterministic allocation/metrics path** — the allocation rule (S_queue + k-deferrals, no double-allocation, pool capacity respected) and all welfare/fairness metrics (SW, Avg r_size, Avg WT, Var r_size, Rop, co-Gini, F(V,NV)) are pure deterministic computations and reproduce bit-for-bit given a seed.
- **LLM applicant behavior** — qualitative only. The local default model (`llama3.2`) differs from the paper's `gpt-3.5-turbo-0301`, so exact paper numbers are not expected.

## What the scripted mock verifies

The scripted mock (`--mock`, `scripted_first_home`) always applies for the first visible home, independent of the applicant's profile. Under it the subset policy only permutes which applicant receives which home, so E[SW(r_size) − SW(r_random)] = 0 by construction (500 seeds: mean −5.2 ± 14.7). It verifies determinism, allocation invariants and the absence of subset-dependent bias. It cannot assess the paper's policy ordering; that assessment requires a live LLM.

## Design uncertainties (paper §7)

- The synthetic environment (applicant count, resource count, profile distributions) is not specified in the paper body — externalised via CLI (`--n-applicants`, `--pool-ratio`, `--visible-subset-size`) with defaults (60 applicants, ratio 0.5).
- The GA hyperparameters (pool size, crossover/mutation rates) and the predictor `f̃` margin are appendix-dependent and externalised as CLI flags with standard defaults. The `poa` subcommand evaluates fitness with either the deterministic mock or the live LLM, and the predictor `f̃` (a nearest-neighbour surrogate) prunes full evaluations; because the appendix values are not in the paper body, the live-LLM SW *ordering* and the POA *improvement direction* — not absolute fitness — are the reproduction targets.

## One-command reproduction (`reproduce`)

```bash
cargo run --release -- reproduce --mock --seed 42      # --quick for a fast smoke
uv run srap-tools reproduce   # render figures
```

`reproduce` runs Table 2 as a **matched-seed paired** comparison (same environment, vary only the resource subset). It records strict wins, ties and losses together with the paired mean difference and its standard error. With a live LLM, the ordering verdict is `PASS` when the lower bound of mean difference ± 1.96·SE is above zero, `reversed` when the upper bound is below zero, and `inconclusive` otherwise. With `--mock`, the ordering verdict is `not_applicable` and is excluded from the overall verdict, while all numerical observations remain recorded and displayed. The command then runs POA for both objectives and emits the per-objective convergence history.

## Real-Ollama smoke (for the main session)

```bash
ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=llama3.2:latest
# Small to bound cost: ~ n_applicants × rounds LLM calls (cache makes a rerun free).
cargo run --release -- run --entry-condition p_select --resource-subset r_size \
    --queues 3 --k 3 --c 2 --runs 1 --seed 42 --n-applicants 12 --max-rounds 3
```
