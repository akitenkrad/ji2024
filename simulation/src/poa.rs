//! POA (Policy Optimization Algorithm) — Phase 3 の **最小スタブ**．
//!
//! > [!IMPORTANT] これは Phase-3 のミニマル足場であり完成版ではない
//! > 論文の POA は «決定論的+LLM シミュレーション 1 回 = 1 ポリシー評価» を適応度
//! > 関数とし，予測器 f̃ で評価を高速化しつつトーナメント選択・交叉・突然変異を M 回
//! > 反復してポリシー π を最適化する遺伝的アルゴリズム外側ループである (論文 §4.6)．
//! > 本モジュールは «その外側 GA の骨格» だけを実装する:
//! > - 適応度 = **1 回の決定論的 `--mock` シミュレーション実行** (サンドボックスで
//! >   ライブ LLM なしにテスト可能にするため; 完成版はライブ LLM 評価 + 予測器 f̃)．
//! > - 予測器 f̃・ライブ LLM 適応度・論文 Fig.4/Table の一括再現は **未実装** (Phase 3
//! >   で差し替える)．
//! >
//! > 過剰実装しない: GA の最小要素 (個体 = Policy ベクトル，トーナメント選択，
//! > 一様交叉，遺伝子単位の突然変異，エリート保存) のみを置く．

use serde::Serialize;
use socsim_core::{derive_seed, SimRng};

use crate::config::Config;
use crate::metrics::{f_pi, Objective};
use crate::policy::{EntryCondition, Policy, ResourceSubset, SortStrategy};

/// POA (GA) のハイパーパラメータ．
///
/// > [!NOTE] 論文付録依存 (設計書 §7 の不確実性)
/// > pool_size・交叉率・突然変異率は論文本文に明示がない．標準的な既定値を置き，
/// > 付録判明後に差し替える．
#[derive(Debug, Clone)]
pub struct PoaConfig {
    /// 最適化目標 (満足度 / 公平性)．
    pub objective: Objective,
    /// 反復世代数 M．
    pub iterations: usize,
    /// 個体群サイズ (pool_size)．
    pub pool_size: usize,
    /// 突然変異率 (各遺伝子が変異する確率)．
    pub mutation_rate: f64,
    /// トーナメントサイズ．
    pub tournament_size: usize,
    /// 適応度評価に使う基準 [`Config`] (ポリシー以外の合成環境設定)．
    pub base_config: Config,
    /// 乱数シード基点．
    pub seed: u64,
}

impl Default for PoaConfig {
    fn default() -> Self {
        PoaConfig {
            objective: Objective::Satisfaction,
            iterations: 20,
            pool_size: 12,
            mutation_rate: 0.2,
            tournament_size: 3,
            base_config: Config::default(),
            seed: 42,
        }
    }
}

/// 1 世代の履歴 (poa_history.csv 行)．
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PoaHistoryRow {
    /// 世代 (0 始まり)．
    pub generation: usize,
    /// その世代の最良適応度 f(π)．
    pub best_fitness: f64,
    /// その世代の平均適応度．
    pub mean_fitness: f64,
    /// 最良個体の入室条件．
    pub best_entry_condition: String,
    /// 最良個体の並び替え戦略．
    pub best_sort_strategy: String,
    /// 最良個体の資源サブセット．
    pub best_resource_subset: String,
    /// 最良個体の m．
    pub best_m: usize,
    /// 最良個体の k．
    pub best_k: usize,
    /// 最良個体の c．
    pub best_c: usize,
}

/// POA の実行結果．
pub struct PoaResult {
    /// 各世代の履歴．
    pub history: Vec<PoaHistoryRow>,
    /// 最終的に得られた最良ポリシー π*．
    pub best_policy: Policy,
    /// 最終最良適応度．
    pub best_fitness: f64,
}

impl PoaResult {
    /// 改善率 (%): (最終 - 初期) / |初期| × 100．
    pub fn improvement_pct(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let first = self.history.first().unwrap().best_fitness;
        let last = self.history.last().unwrap().best_fitness;
        if first.abs() < 1e-12 {
            0.0
        } else {
            (last - first) / first.abs() * 100.0
        }
    }
}

/// 適応度関数: 与えられたポリシーで 1 回 `--mock` シミュレーションを実行し f(π) を返す．
///
/// **Phase-3 スタブ**: 完成版はライブ LLM 適応度 + 予測器 f̃ に差し替える．ここでは
/// サンドボックスでテスト可能にするため，決定論的 scripted mock で評価する．
pub fn mock_fitness(policy: Policy, base: &Config, objective: Objective) -> f64 {
    use crate::llm::wrap_client;
    use crate::simulation::run_with_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    let cfg = Config {
        policy: policy.normalized(),
        // mock 評価では cache を持たない (in-memory)．
        llm: crate::config::LlmSettings {
            cache_path: None,
            ..base.llm.clone()
        },
        ..base.clone()
    };
    // «最初の可視 home を選ぶ» 決定論的 mock．
    let backend = ScriptedClient::new("mock-poa", |prompt: &str| {
        if let Some(idx) = prompt.find("home ") {
            let rest = &prompt[idx + "home ".len()..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num.is_empty() {
                return format!("{{\"choice\": {num}}}");
            }
        }
        "{\"choice\": -1}".to_string()
    });
    let client = wrap_client(backend, PromptCache::in_memory());
    match run_with_client(&cfg, client, 0) {
        Ok(r) => f_pi(&r.final_metrics, objective),
        Err(_) => f64::NEG_INFINITY,
    }
}

/// ランダムなポリシー個体を生成する．
fn random_policy(rng: &mut SimRng) -> Policy {
    use rand::Rng;
    let entries = EntryCondition::all();
    let sorts = SortStrategy::all();
    let subsets = ResourceSubset::all();
    Policy {
        entry_condition: entries[rng.gen_range(0..entries.len())],
        sort_strategy: sorts[rng.gen_range(0..sorts.len())],
        resource_subset: subsets[rng.gen_range(0..subsets.len())],
        m: rng.gen_range(1..=5),
        k: rng.gen_range(1..=5),
        c: rng.gen_range(1..=3),
    }
}

/// 一様交叉: 2 親の各遺伝子を 50% でどちらかから受け継ぐ．
fn crossover(a: &Policy, b: &Policy, rng: &mut SimRng) -> Policy {
    use rand::Rng;
    Policy {
        entry_condition: if rng.gen_bool(0.5) {
            a.entry_condition
        } else {
            b.entry_condition
        },
        sort_strategy: if rng.gen_bool(0.5) {
            a.sort_strategy
        } else {
            b.sort_strategy
        },
        resource_subset: if rng.gen_bool(0.5) {
            a.resource_subset
        } else {
            b.resource_subset
        },
        m: if rng.gen_bool(0.5) { a.m } else { b.m },
        k: if rng.gen_bool(0.5) { a.k } else { b.k },
        c: if rng.gen_bool(0.5) { a.c } else { b.c },
    }
}

/// 遺伝子単位の突然変異 (各遺伝子を `rate` の確率でランダム化)．
fn mutate(p: &mut Policy, rate: f64, rng: &mut SimRng) {
    use rand::Rng;
    let entries = EntryCondition::all();
    let sorts = SortStrategy::all();
    let subsets = ResourceSubset::all();
    if rng.gen_bool(rate) {
        p.entry_condition = entries[rng.gen_range(0..entries.len())];
    }
    if rng.gen_bool(rate) {
        p.sort_strategy = sorts[rng.gen_range(0..sorts.len())];
    }
    if rng.gen_bool(rate) {
        p.resource_subset = subsets[rng.gen_range(0..subsets.len())];
    }
    if rng.gen_bool(rate) {
        p.m = rng.gen_range(1..=5);
    }
    if rng.gen_bool(rate) {
        p.k = rng.gen_range(1..=5);
    }
    if rng.gen_bool(rate) {
        p.c = rng.gen_range(1..=3);
    }
}

/// トーナメント選択: `size` 個をサンプルして最良を返す (index)．
fn tournament(fitness: &[f64], size: usize, rng: &mut SimRng) -> usize {
    use rand::Rng;
    let mut best = rng.gen_range(0..fitness.len());
    for _ in 1..size.max(1) {
        let challenger = rng.gen_range(0..fitness.len());
        if fitness[challenger] > fitness[best] {
            best = challenger;
        }
    }
    best
}

/// 個体群を評価する (各個体の適応度ベクトル)．
fn evaluate_population(pop: &[Policy], base: &Config, objective: Objective) -> Vec<f64> {
    pop.iter()
        .map(|p| mock_fitness(*p, base, objective))
        .collect()
}

/// POA (GA 外側ループ) を実行する．適応度 = [`mock_fitness`] (Phase-3 スタブ)．
///
/// エリート保存つき: 各世代で最良個体を次世代へ無条件にコピーするため，最良適応度は
/// 世代をまたいで **単調非減少** になる (テストで検証)．
pub fn run_poa(poa: &PoaConfig) -> PoaResult {
    let mut rng = SimRng::from_seed(derive_seed(poa.seed, &[2]));

    // 初期個体群．
    let pool_size = poa.pool_size.max(2);
    let mut population: Vec<Policy> = (0..pool_size).map(|_| random_policy(&mut rng)).collect();

    let mut fitness = evaluate_population(&population, &poa.base_config, poa.objective);
    let mut history: Vec<PoaHistoryRow> = Vec::with_capacity(poa.iterations.max(1));

    let mut best_idx = argmax(&fitness);
    let mut best_policy = population[best_idx];
    let mut best_fitness = fitness[best_idx];

    for generation in 0..poa.iterations.max(1) {
        // 記録 (現世代の最良)．
        history.push(PoaHistoryRow {
            generation,
            best_fitness,
            mean_fitness: fitness.iter().sum::<f64>() / fitness.len() as f64,
            best_entry_condition: best_policy.entry_condition.label().to_string(),
            best_sort_strategy: best_policy.sort_strategy.label().to_string(),
            best_resource_subset: best_policy.resource_subset.label().to_string(),
            best_m: best_policy.m,
            best_k: best_policy.k,
            best_c: best_policy.c,
        });

        // 次世代生成 (エリート保存 1)．
        let mut next: Vec<Policy> = Vec::with_capacity(pool_size);
        next.push(best_policy);
        while next.len() < pool_size {
            let pa = population[tournament(&fitness, poa.tournament_size, &mut rng)];
            let pb = population[tournament(&fitness, poa.tournament_size, &mut rng)];
            let mut child = crossover(&pa, &pb, &mut rng);
            mutate(&mut child, poa.mutation_rate, &mut rng);
            next.push(child.normalized());
        }

        population = next;
        fitness = evaluate_population(&population, &poa.base_config, poa.objective);
        best_idx = argmax(&fitness);
        // エリート保存により best は単調非減少．
        if fitness[best_idx] > best_fitness {
            best_fitness = fitness[best_idx];
            best_policy = population[best_idx];
        }
    }

    PoaResult {
        history,
        best_policy,
        best_fitness,
    }
}

/// 最大値の index (空なら 0)．
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_poa() -> PoaConfig {
        PoaConfig {
            objective: Objective::Satisfaction,
            iterations: 5,
            pool_size: 6,
            mutation_rate: 0.3,
            tournament_size: 3,
            base_config: Config {
                n_applicants: 12,
                pool_ratio: 0.5,
                max_rounds: 3,
                seed: Some(7),
                ..Config::default()
            },
            seed: 1,
        }
    }

    #[test]
    fn poa_best_fitness_is_non_decreasing() {
        let r = run_poa(&small_poa());
        assert_eq!(r.history.len(), 5);
        for w in r.history.windows(2) {
            assert!(
                w[1].best_fitness >= w[0].best_fitness - 1e-9,
                "elitism → best fitness non-decreasing: {} -> {}",
                w[0].best_fitness,
                w[1].best_fitness
            );
        }
    }

    #[test]
    fn poa_is_deterministic() {
        let a = run_poa(&small_poa());
        let b = run_poa(&small_poa());
        assert_eq!(a.best_fitness, b.best_fitness);
        assert_eq!(a.best_policy, b.best_policy);
    }

    #[test]
    fn mock_fitness_is_finite() {
        let cfg = Config {
            n_applicants: 10,
            max_rounds: 3,
            seed: Some(1),
            ..Config::default()
        };
        let f = mock_fitness(Policy::default(), &cfg, Objective::Satisfaction);
        assert!(f.is_finite());
    }
}
