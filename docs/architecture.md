[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

SRAP-Agent maps the paper's Setup → Simulation → Evaluation phases onto socsim's `WorldState` + a 6-phase `Mechanism` loop. One engine tick = one allocation round.

## WorldState: `SrapWorld` (`simulation/src/world.rs`)

A central-allocation world (no grid / network):

| Field | Type | Role |
|-------|------|------|
| `applicants` | `BTreeMap<AgentId, Applicant>` | profiles (income, rent, family, preferences, memory, active, vulnerable) |
| `pool` | `Vec<Resource>` | scarce-resource pool (public housing: size, rent, allocated) |
| `policy` | `Policy` | π = (E_queue, S_queue, R_queue, m, k, c) |
| `queues` | `Vec<Queue>` | m waiting queues built by the entry condition |
| `allocations` | `BTreeMap<AgentId, Option<ResourceId>>` | per-applicant allocation R_j* |
| `metrics` | `SrapMetrics` | this round's welfare/fairness aggregate |

`agent_ids()` returns the `BTreeMap` keys (sorted `AgentId`) → determinism. An applicant's `utility(resource)` is the deterministic satisfaction used for SW.

## Policy: `Policy` (`simulation/src/policy.rs`)

`Policy { entry_condition: {PBudget, PFamily, PSelect}, sort_strategy: {Fifo, Vfa, Vfr}, resource_subset: {RSize, RRent, RRandom}, m, k, c }`.

## The 5 mechanisms × 6 phases (`simulation/src/mechanisms.rs`)

| Mechanism | Phase | Role |
|-----------|-------|------|
| `PolicySetup` | Environment | refill/confirm the pool; build queues q_1..q_m by entry condition E_queue (PBudget/PFamily use the income/family median threshold; PSelect admits everyone and defers dropping out to the Decision phase) |
| `ApplyDecision` | Decision | **★LLM here**. Each active applicant chooses a desired home from its visible subset V(p_j) (R_queue-sorted top-`visible_subset_size`) or ∅ = drop out. Synchronous: desires are collected into the scratch blackboard, allocation happens next |
| `AllocationRule` | Interaction | deterministic: sort applicants by S_queue (FIFO = AgentId order; VFA = vulnerable-first by attribute; VFR = vulnerable-first by family/income ranking), then allocate with k-deferrals (first choice, then up to k−1 size-ranked backups). No double-allocation; pool capacity respected |
| `EvaluateWelfare` | Reward | accumulate allocated applicants' outcomes and compute SW / Avg r_size / Avg WT / Var r_size / Rop / co-Gini / F(V,NV); push a `metrics.csv` row |
| `UpdateMemory` | PostStep | allocated → inactive; unallocated → wait_time++ (and drop out if the pool is empty); append a memory summary; `request_stop()` on pool exhaustion or all-settled |

## Update semantics & scheduler

Synchronous: each round snapshots the active applicants, collects all desires in `ApplyDecision`, then allocates once in `AllocationRule`. The scheduler is `RandomActivationScheduler` (the arrival-pattern stochasticity is carried by the engine RNG shuffle); the allocation order itself is fixed deterministically by S_queue.

## RNG streams (`simulation/src/simulation.rs`)

```
const RNG_WORLD_INIT: u64 = 0;   // applicant profiles, resource pool
const RNG_ENGINE: u64    = 1;    // scheduler / activation shuffle
// &[2] reserved for POA (the GA stream)
```

`derive_seed(root, &[RNG_WORLD_INIT])` seeds `init_world`; `derive_seed(root, &[RNG_ENGINE])` seeds `SimulationBuilder`.

## Two-layer LLM (`simulation/src/llm.rs`)

`SrapClient = CachingClient<Box<dyn LlmClient>>`. Production = `FallbackClient<OllamaClient, OpenAiClient>` type-erased into `Box<dyn LlmClient>`. Tests / `--mock` inject `socsim_llm::mock::ScriptedClient`. The LLM is confined to `ApplyDecision`, called once per active applicant per round.

## POA — Policy Optimization Agent (`simulation/src/poa.rs`)

A genetic-algorithm outer loop over the policy parameter space: individual = `Policy = (E_queue, S_queue, R_queue, m, k, c)`, with tournament selection + uniform crossover + per-gene mutation + 1-elitism. Fitness = `f_pi(metrics, objective)` from one SRAP allocation run, evaluated either with the deterministic scripted mock (`FitnessKind::Mock`, offline / bit-deterministic) or with the live LLM (`FitnessKind::Live { cache_path }`, Ollama→OpenAI + a persistent prompt cache shared across generations). Elitism makes the best fitness monotonically non-decreasing across generations.

The **predictor `f̃`** (`Predictor`) is a surrogate model: a weighted k-nearest-neighbour regression over already-evaluated `(policy-feature-vector → fitness)` samples (with an exact-match cache for re-encountered policies). When enabled (`use_predictor`), a candidate whose predicted fitness falls below `incumbent_best − margin` is pruned — its expensive simulation is skipped and the surrogate value is used instead. The initial generation is always fully evaluated to seed the surrogate, and the prune margin is scaled from the initial fitness spread. `PoaResult` reports `full_evals` and `evals_saved`.

## Reproduction (`reproduce` subcommand)

Runs the paper's headline results in one command: Table 2 (matched-seed paired comparison of social welfare across `(entry_condition, resource_subset)`, FIFO fixed), Table 3 (POA-optimized satisfaction `π_s*` and fairness `π_f*` policies) and Figure 4 (per-objective POA convergence history). Writes `table2_sw_by_policy.csv`, `table3_optimized_policies.csv`, `poa_history_<objective>.csv` and `reproduce_summary.json` (observed vs paper-finding with PASS / off-anchor); `--mock` runs offline, `--quick` shrinks the POA budget.

---
*This file was generated by Claude Code.*
