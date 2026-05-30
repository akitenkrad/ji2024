**English** | [日本語](README.ja.md)

# SRAP-Agent: LLM-Agent Scarce Resource Allocation Policy Simulation — Ji et al. (2024)

A reimplementation of the SRAP-Agent framework of Ji, Li, Liu, Du, Wei, Shen, Qi & Lin (2024), ["SRAP-Agent: Simulating and Optimizing Scarce Resource Allocation Policy with LLM-based Agent"](https://doi.org/10.48550/arXiv.2410.14152) (Findings of EMNLP 2024). LLM-driven *applicant* agents apply to a **central pool** of a public scarce resource (the paper's case study is **public housing**); a deterministic allocation rule then allocates homes under a configurable policy. The allocation policy is

> π = (E_queue *entry condition*, S_queue *sorting strategy*, R_queue *resource subset*, m *queues*, k *k-deferrals*, c *selection-queue capacity*).

Each round (= one engine tick): the entry condition builds m waiting queues → each active applicant chooses a desired home from a visible subset V(p_j) (this is the **LLM** decision, Eq. 2 `R_j* = D(p_j, V(p_j))`, or ∅ = drop out) → a deterministic allocation rule sorts by S_queue (FIFO / vulnerable-first VFA / VFR) and allocates the R_queue resources with k-deferrals → welfare and fairness are evaluated → applicant memory is updated. The model is a **central allocation** — non-spatial and non-network — so it depends only on [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) `socsim-core` + `socsim-engine` + `socsim-llm` (no `socsim-grid` / `socsim-net`).

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — synthetic applicant/pool initialization, entry-condition queue building, the deterministic allocation rule (S_queue sorting + k-deferrals, no double-allocation, pool capacity respected), and all welfare/fairness metrics (SW, Avg r_size, Avg WT, Var r_size, Rop inverse-order-pairs, co-Gini ∈ [0,1], F(V,NV) vulnerable gap), and memory updates. Given a seed these reproduce bit-for-bit (ChaCha20 `SimRng`, two streams: `RNG_WORLD_INIT=0`, `RNG_ENGINE=1`).
- **Non-deterministic LLM layer** — the single `Decision` mechanism (`ApplyDecision`), where each active applicant chooses a desired home from its visible resources, profile and memory. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses. Each run writes `llm_meta.json` recording the model, endpoint, temperature, seed and cache-hit rate. Because the local default model (`llama3.2`) differs from the paper's `gpt-3.5-turbo-0301`, the LLM-driven reproduction target is **qualitative**; the deterministic allocation/metrics path is the quantitative-ish core (the SW *ordering* across policies — `p_select`+`r_size` highest, `r_random` lowest — should hold qualitatively).

> This project standardises on the `socsim-llm` crate for the LLM layer; it does **not** use `reqwest` or `sha2` (socsim-llm owns the HTTP transport and the prompt-cache hashing), overriding the design doc's original `reqwest`+`sha2` plan to stay consistent with the han2023 / li2024 / zhao2024 / chuang2024 siblings.

## Capabilities

- **`run`**: `SrapWorld` + the 5 mechanisms + the LLM client layer + a single-policy run (welfare & fairness metrics).
- **`sweep`**: the policy-factor sensitivity sweep (entry conditions × resource subsets × sort strategies).
- **`poa`** — the Policy Optimization Agent: a genetic-algorithm outer loop (tournament selection / uniform crossover / per-gene mutation / 1-elitism) over the allocation-policy parameter space `(E_queue, S_queue, R_queue, m, k, c)`. Fitness is one SRAP allocation run scored by `f_pi(metrics, objective)`, evaluated either with the deterministic scripted mock (`--mock`, offline / bit-deterministic) or with the live LLM (Ollama→OpenAI + persistent cache). A predictor `f̃` surrogate (weighted nearest-neighbour regression over already-evaluated policies) prunes full evaluations for candidates that the surrogate predicts cannot beat the current elite, cutting expensive evaluations. Elitism keeps the best fitness monotonically non-decreasing across generations.
- **`reproduce`**: a one-command reproduction of the paper's Table 2 (policy-ordering social welfare), Table 3 (POA-optimized satisfaction/fairness policies) and Figure 4 (POA convergence), writing CSVs + `reproduce_summary.json` (observed vs paper-finding with PASS / off-anchor) and figures.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# Make sure a local Ollama is running and a model is pulled, e.g.:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# Optional OpenAI fallback:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-3.5-turbo

# Base experiment: single policy (highest-SW condition p_select + r_size)
cargo run --release -- run \
    --entry-condition p_select --resource-subset r_size \
    --queues 3 --k 3 --c 2 --runs 10 --seed 42

# Install the Python visualization tools (at the workspace root)
uv sync

# Visualize the most recent run (welfare & fairness time-series, final-metric summary)
uv run srap-tools visualize

# Inspect the run's settings and LLM metadata
uv run srap-tools show-experiment-settings --results-dir results/latest
```

### Offline (no-LLM) smoke

The full round loop, output writers and Python visualization can be exercised without any live LLM via a scripted mock client (used in CI / network-blocked sandboxes):

```bash
# Dedicated example
cargo run --release --example mock_smoke -- results

# Or pass --mock to run / sweep / poa / reproduce for the same offline behaviour
cargo run --release -- run --entry-condition p_select --resource-subset r_size \
    --queues 3 --k 3 --c 2 --runs 3 --seed 42 --mock
uv run srap-tools visualize
```

### Sensitivity analysis (sweep) and POA

```bash
# Sweep the three policy factors (entry conditions × resource subsets)
cargo run --release -- sweep \
    --entry-conditions p_budget,p_family,p_select \
    --resource-subsets r_size,r_rent,r_random \
    --runs 30 --seed 42            # add --mock for offline
uv run srap-tools visualize-sweep

# POA policy optimization with the predictor f̃ (add --mock for offline)
cargo run --release -- poa --objective satisfaction \
    --iterations 50 --pool-size 50 --use-predictor --seed 42
uv run srap-tools visualize-sweep   # plots the POA convergence curve
```

### Paper reproduction (Table 2/3 + Fig. 4)

```bash
# Reproduce the policy-ordering finding + POA optimization (offline)
cargo run --release -- reproduce --mock --seed 42        # --quick for a fast smoke
# Render the figures and re-print the observed-vs-paper verdict
uv run srap-tools reproduce --results-dir results/latest
```

## Testing & Linting

```bash
cargo test --release   # mock (ScriptedClient) driven, no live LLM (52 tests)
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Documentation

- [Architecture](docs/architecture.md) — `SrapWorld`, the 5 mechanisms / 6 phases, RNG streams, two-layer LLM.
- [CLI reference](docs/cli.md) — `run` / `sweep` / `poa` / `reproduce` flags and outputs.
- [Reproduction](docs/reproduction.md) — quantitative targets, the policy-ordering finding, and design uncertainties.
- [Visualization](docs/visualization.md) — the Python `srap-tools` figures.

## References

- Ji, J., Li, Y., Liu, H., Du, Z., Wei, Z., Shen, W., Qi, Q., & Lin, Y. (2024). SRAP-Agent: Simulating and Optimizing Scarce Resource Allocation Policy with LLM-based Agent. *Findings of the Association for Computational Linguistics: EMNLP 2024*, 267–293.
- socsim: [rs-social-simulation-tools](https://github.com/akitenkrad/rs-social-simulation-tools).

## License

MIT — see [LICENSE](LICENSE).

---
*This file was generated by Claude Code.*
