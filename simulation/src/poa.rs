//! POA (Policy Optimization Agent) — 配分ポリシー π を最適化する遺伝的アルゴリズム．
//!
//! 論文 (Ji et al. 2024, §4.6) の POA は «決定論的+LLM シミュレーション 1 回 =
//! 1 ポリシー評価» を適応度関数とし，予測器 f̃ で高価な評価を間引きつつトーナメント
//! 選択・一様交叉・遺伝子単位の突然変異を M 世代反復してポリシー π を最適化する
//! 遺伝的アルゴリズム外側ループである．本モジュールはその完成版を実装する:
//!
//! - **適応度** = 1 回の SRAP 配分シミュレーション実行から得た [`f_pi`]．評価経路は 2 つ:
//!   - **mock** ([`FitnessKind::Mock`]): «最初の可視 home を選ぶ» 決定論的 scripted
//!     mock．サンドボックステスト・bit 決定論用 (ライブ LLM 不要)．
//!   - **live** ([`FitnessKind::Live`]): 応募者を実 LLM (Ollama→OpenAI + 永続キャッシュ)
//!     で駆動する．`socsim-llm` のキャッシュ + `temperature=0` で擬似決定論化する．
//! - **予測器 f̃** ([`Predictor`]): すでに評価済みのポリシーから特徴空間上の重み付き
//!   最近傍回帰で未評価ポリシーの適応度を安価に近似する．世代内で予測値が «現行
//!   エリート − margin» を下回る個体は実評価をスキップ (枝刈り) し，高価な評価回数を
//!   削減する (論文の «予測器でフル評価を間引く» に対応)．予測の信頼性が低いうち
//!   (学習サンプルが少ないうち) は枝刈りせず実評価する．
//! - **エリート保存**: 各世代で最良個体を次世代へ無条件にコピーするため，最良適応度は
//!   世代をまたいで **単調非減少** になる (テストで検証)．

use std::collections::HashMap;

use serde::Serialize;
use socsim_core::{derive_seed, SimRng};

use crate::config::{Config, LlmSettings};
use crate::metrics::{f_pi, Objective};
use crate::policy::{EntryCondition, Policy, ResourceSubset, SortStrategy};

// =========================================================================== //
// 適応度評価の種類 (mock / live)
// =========================================================================== //

/// 適応度評価に使うシミュレーション駆動方式．
#[derive(Debug, Clone)]
pub enum FitnessKind {
    /// «最初の可視 home を選ぶ» 決定論的 scripted mock (オフライン・bit 決定論)．
    Mock,
    /// 応募者を実 LLM (Ollama→OpenAI + 永続キャッシュ) で駆動する．
    ///
    /// `cache_path` を渡すと評価間でプロンプトキャッシュを共有・永続化し，同一
    /// ポリシーの再評価を cache-hit で安価にする (POA は同じ π を複数世代で再評価
    /// しうるため効果が大きい)．
    Live { cache_path: Option<String> },
}

impl FitnessKind {
    /// CLI / JSON ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            FitnessKind::Mock => "mock",
            FitnessKind::Live { .. } => "live",
        }
    }
}

// =========================================================================== //
// POA 設定
// =========================================================================== //

/// POA (GA) のハイパーパラメータ．
///
/// > [!NOTE] 論文付録依存 (設計書 §7 の不確実性)
/// > pool_size・交叉率・突然変異率・予測器 margin は論文本文に明示がない．標準的な
/// > 既定値を置き，付録判明後に差し替える．
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
    /// 適応度評価方式 (mock / live)．
    pub fitness_kind: FitnessKind,
    /// 予測器 f̃ による枝刈りを有効にするか．
    pub use_predictor: bool,
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
            fitness_kind: FitnessKind::Mock,
            use_predictor: false,
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
    /// その世代で予測器によりフル評価を省略した個体数 (累積)．
    pub evals_saved: usize,
    /// その世代までに実行したフル評価回数 (累積)．
    pub full_evals: usize,
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
    /// 実行したフル (シミュレーション) 評価の総数．
    pub full_evals: usize,
    /// 予測器によりフル評価を省略した総数．
    pub evals_saved: usize,
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

// =========================================================================== //
// 適応度関数 (mock / live)
// =========================================================================== //

/// ポリシー π で 1 回シミュレーションを実行し f(π) を返す (mock 評価)．
///
/// «最初の可視 home を選ぶ» 決定論的 scripted mock で評価する (サンドボックスで
/// ライブ LLM なしにテスト可能・bit 決定論的)．
pub fn mock_fitness(policy: Policy, base: &Config, objective: Objective) -> f64 {
    use crate::llm::wrap_client;
    use crate::simulation::run_with_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    let cfg = Config {
        policy: policy.normalized(),
        // mock 評価では cache を持たない (in-memory)．
        llm: LlmSettings {
            cache_path: None,
            ..base.llm.clone()
        },
        ..base.clone()
    };
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

/// ポリシー π で 1 回シミュレーションを実行し f(π) を返す (live-LLM 評価)．
///
/// 応募者を実 LLM (Ollama→OpenAI + プロンプトキャッシュ) で駆動する．`cache_path`
/// を渡すと評価間でキャッシュを共有・永続化し，同一プロンプトの再評価を cache-hit で
/// 省略する (POA は同じ π を複数世代で再評価しうるため効果が大きい)．LLM 層は socsim
/// の bit 再現性の外側にあり，`temperature=0` + 固定 seed + cache で擬似決定論化する．
pub fn live_fitness(
    policy: Policy,
    base: &Config,
    objective: Objective,
    cache_path: Option<String>,
) -> f64 {
    use crate::llm::build_live_client;
    use crate::simulation::run_with_client;

    let cfg = Config {
        policy: policy.normalized(),
        llm: LlmSettings {
            cache_path,
            ..base.llm.clone()
        },
        ..base.clone()
    };
    match build_live_client(&cfg.llm) {
        Ok(client) => match run_with_client(&cfg, client, 0) {
            Ok(r) => f_pi(&r.final_metrics, objective),
            Err(_) => f64::NEG_INFINITY,
        },
        Err(_) => f64::NEG_INFINITY,
    }
}

/// ポリシー π を `kind` に応じて mock / live で評価する．
fn evaluate_fitness(
    policy: Policy,
    base: &Config,
    objective: Objective,
    kind: &FitnessKind,
) -> f64 {
    match kind {
        FitnessKind::Mock => mock_fitness(policy, base, objective),
        FitnessKind::Live { cache_path } => {
            live_fitness(policy, base, objective, cache_path.clone())
        }
    }
}

// =========================================================================== //
// 予測器 f̃ (surrogate)
// =========================================================================== //

/// 予測器 f̃: 評価済みポリシーから特徴空間上の重み付き最近傍回帰で未評価ポリシーの
/// 適応度を安価に近似するサロゲートモデル．
///
/// 論文の «予測器でフル評価を間引く» に対応する．真の適応度は 1 回のシミュレーション
/// 実行 (mock なら数 ms，live なら LLM 呼び出しで数十秒) を要するため，すでに評価済みの
/// (ポリシー特徴ベクトル → 適応度) ペアを蓄積し，新個体の適応度を距離重み付き平均で
/// 予測する．予測値が現行エリートを下回る見込みの個体はフル評価を省略する (枝刈り)．
///
/// - 同一ポリシーが再出現したらキャッシュ的に正確な値を返す (距離 0 → 重み ∞)．
/// - 学習サンプルが `min_samples` 未満なら予測は信頼できないので `None` を返す
///   (呼び出し側はフル評価する)．
#[derive(Default)]
pub struct Predictor {
    /// 評価済み (特徴ベクトル, 適応度) の表．キーは離散特徴の正確一致用．
    exact: HashMap<[u64; 6], f64>,
    /// 近傍回帰用の (特徴ベクトル, 適応度) サンプル列．
    samples: Vec<([f64; 6], f64)>,
}

impl Predictor {
    /// 予測に必要な最小サンプル数 (これ未満なら `predict` は `None`)．
    const MIN_SAMPLES: usize = 4;
    /// 最近傍回帰で使う近傍数 (k-NN の k)．
    const NEIGHBORS: usize = 3;

    /// 評価結果を予測器に学習させる．
    pub fn observe(&mut self, policy: Policy, fitness: f64) {
        if !fitness.is_finite() {
            return;
        }
        self.exact.insert(discrete_key(&policy), fitness);
        self.samples.push((feature_vector(&policy), fitness));
    }

    /// ポリシー π の適応度を予測する (`None` = 信頼できる予測ができない)．
    pub fn predict(&self, policy: Policy) -> Option<f64> {
        // 同一ポリシーを評価済みなら厳密値を返す．
        if let Some(&exact) = self.exact.get(&discrete_key(&policy)) {
            return Some(exact);
        }
        if self.samples.len() < Self::MIN_SAMPLES {
            return None;
        }
        let target = feature_vector(&policy);
        // 距離の昇順で近傍 K 件を取り，距離の逆数で重み付き平均する．
        let mut dists: Vec<(f64, f64)> = self
            .samples
            .iter()
            .map(|(f, y)| (feature_distance(&target, f), *y))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = Self::NEIGHBORS.min(dists.len());
        let mut wsum = 0.0;
        let mut acc = 0.0;
        for (d, y) in dists.iter().take(k) {
            let w = 1.0 / (d + 1e-6);
            wsum += w;
            acc += w * y;
        }
        if wsum <= 0.0 {
            None
        } else {
            Some(acc / wsum)
        }
    }

    /// 学習済みサンプル数．
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }
}

/// ポリシーの離散特徴キー (厳密一致キャッシュ用)．
fn discrete_key(p: &Policy) -> [u64; 6] {
    [
        p.entry_condition as u64,
        p.sort_strategy as u64,
        p.resource_subset as u64,
        p.m as u64,
        p.k as u64,
        p.c as u64,
    ]
}

/// ポリシーの連続特徴ベクトル (近傍距離計算用)．
///
/// カテゴリ変数 (E/S/R) は one-hot ではなく単純な序数を [0,1] 正規化で並べ，数値変数
/// (m,k,c) も典型レンジで正規化して «似たポリシーは近い距離» になるようにする．
fn feature_vector(p: &Policy) -> [f64; 6] {
    [
        p.entry_condition as u64 as f64 / 2.0,
        p.sort_strategy as u64 as f64 / 2.0,
        p.resource_subset as u64 as f64 / 2.0,
        (p.m as f64 - 1.0) / 4.0,
        (p.k as f64 - 1.0) / 4.0,
        (p.c as f64 - 1.0) / 2.0,
    ]
}

/// 特徴ベクトル間のユークリッド距離．
fn feature_distance(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// =========================================================================== //
// GA 演算子
// =========================================================================== //

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

// =========================================================================== //
// 評価ループ (予測器による枝刈り)
// =========================================================================== //

/// 個体群を評価する (予測器 f̃ による枝刈りつき)．
///
/// `use_predictor` が真かつ予測器に十分なサンプルがある場合，予測適応度が現行エリート
/// (`incumbent_best`) から `prune_margin` 以上劣る個体はフル評価を省略し，予測値を
/// 代理適応度として採用する (劣る見込みの個体に高価な評価を費やさない)．省略した
/// 個体数を `*evals_saved` に，実評価回数を `*full_evals` に加算する．予測器は
/// フル評価したサンプルのみで学習する (予測値を学習に混ぜると誤差が累積するため)．
#[allow(clippy::too_many_arguments)]
fn evaluate_population(
    pop: &[Policy],
    base: &Config,
    objective: Objective,
    kind: &FitnessKind,
    predictor: &mut Predictor,
    use_predictor: bool,
    incumbent_best: f64,
    prune_margin: f64,
    full_evals: &mut usize,
    evals_saved: &mut usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(pop.len());
    for &p in pop {
        let pred = if use_predictor {
            predictor.predict(p)
        } else {
            None
        };
        // 予測が «現行エリート − margin» を明確に下回るなら枝刈り (フル評価を省略)．
        let should_prune = matches!(pred, Some(v) if incumbent_best.is_finite()
            && v < incumbent_best - prune_margin);
        if should_prune {
            *evals_saved += 1;
            // 予測値を代理適応度として採用する (エリートには勝てない見込み)．
            out.push(pred.unwrap());
        } else {
            let f = evaluate_fitness(p, base, objective, kind);
            *full_evals += 1;
            predictor.observe(p, f);
            out.push(f);
        }
    }
    out
}

// =========================================================================== //
// POA 外側ループ
// =========================================================================== //

/// POA (GA 外側ループ) を実行する．
///
/// エリート保存つき: 各世代で最良個体を次世代へ無条件にコピーするため，最良適応度は
/// 世代をまたいで **単調非減少** になる (テストで検証)．予測器 f̃ を使う場合，劣る
/// 見込みの個体のフル評価を省略して評価回数を削減する．
pub fn run_poa(poa: &PoaConfig) -> PoaResult {
    let mut rng = SimRng::from_seed(derive_seed(poa.seed, &[2]));

    let pool_size = poa.pool_size.max(2);
    let mut population: Vec<Policy> = (0..pool_size).map(|_| random_policy(&mut rng)).collect();

    let mut predictor = Predictor::default();
    let mut full_evals = 0usize;
    let mut evals_saved = 0usize;

    // 枝刈り margin: f(π) の典型スケールに依存するため，初期世代を全評価してから
    // 観測適応度の散らばりで決める (まずは初期個体群を実評価)．
    let initial_fitness = evaluate_population(
        &population,
        &poa.base_config,
        poa.objective,
        &poa.fitness_kind,
        &mut predictor,
        false, // 初期世代は予測器を使わず全評価して学習サンプルを貯める．
        f64::NEG_INFINITY,
        0.0,
        &mut full_evals,
        &mut evals_saved,
    );
    let prune_margin = estimate_margin(&initial_fitness);

    let mut fitness = initial_fitness;
    let mut history: Vec<PoaHistoryRow> = Vec::with_capacity(poa.iterations.max(1));

    let mut best_idx = argmax(&fitness);
    let mut best_policy = population[best_idx];
    let mut best_fitness = fitness[best_idx];

    for generation in 0..poa.iterations.max(1) {
        history.push(PoaHistoryRow {
            generation,
            best_fitness,
            mean_fitness: fitness
                .iter()
                .filter(|f| f.is_finite())
                .copied()
                .sum::<f64>()
                / fitness.iter().filter(|f| f.is_finite()).count().max(1) as f64,
            evals_saved,
            full_evals,
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
        fitness = evaluate_population(
            &population,
            &poa.base_config,
            poa.objective,
            &poa.fitness_kind,
            &mut predictor,
            poa.use_predictor,
            best_fitness,
            prune_margin,
            &mut full_evals,
            &mut evals_saved,
        );
        // エリート (index 0) は前世代の best をコピーしているが，予測器枝刈りで実評価を
        // 省略されると代理適応度になりうる．best_fitness はエリートの真値を保持する．
        best_idx = argmax(&fitness);
        if fitness[best_idx] > best_fitness {
            best_fitness = fitness[best_idx];
            best_policy = population[best_idx];
        }
    }

    PoaResult {
        history,
        best_policy,
        best_fitness,
        full_evals,
        evals_saved,
    }
}

/// 枝刈り margin を初期適応度の散らばりから推定する (標準偏差の 0.25 倍, 最低 1e-6)．
///
/// margin が小さすぎると «僅差で劣るだけの有望個体» まで枝刈りしてしまい，大きすぎると
/// 枝刈りがほぼ起きない．初期世代の f(π) のばらつきにスケールを合わせる．
fn estimate_margin(fitness: &[f64]) -> f64 {
    let finite: Vec<f64> = fitness.iter().copied().filter(|f| f.is_finite()).collect();
    if finite.len() < 2 {
        return 1e-6;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|f| (f - mean) * (f - mean)).sum::<f64>() / finite.len() as f64;
    (var.sqrt() * 0.25).max(1e-6)
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
            fitness_kind: FitnessKind::Mock,
            use_predictor: false,
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
    fn poa_best_fitness_non_decreasing_with_predictor() {
        let mut cfg = small_poa();
        cfg.use_predictor = true;
        cfg.iterations = 8;
        let r = run_poa(&cfg);
        for w in r.history.windows(2) {
            assert!(
                w[1].best_fitness >= w[0].best_fitness - 1e-9,
                "elitism holds even with predictor pruning"
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
    fn predictor_reduces_full_evaluations() {
        // 予測器ありの方がフル評価回数が少ない (または同等) であるべき．
        let mut with_pred = small_poa();
        with_pred.use_predictor = true;
        with_pred.iterations = 12;
        let mut without = small_poa();
        without.use_predictor = false;
        without.iterations = 12;
        let r_with = run_poa(&with_pred);
        let r_without = run_poa(&without);
        assert!(
            r_with.full_evals <= r_without.full_evals,
            "predictor should not increase full evals: with={} without={}",
            r_with.full_evals,
            r_without.full_evals
        );
        // 予測器あり経路では少なくとも一部の評価が省略されることを期待する
        // (探索が収束し再評価が増えると evals_saved>0 になる)．
        assert_eq!(
            r_with.full_evals + r_with.evals_saved,
            r_without.full_evals + r_without.evals_saved,
            "total candidate evaluations (full+saved) must match across both paths"
        );
    }

    #[test]
    fn predictor_exact_recall() {
        // 同一ポリシーを学習させたら厳密値を返す (距離 0 のキャッシュ)．
        let mut p = Predictor::default();
        let pol = Policy::default();
        p.observe(pol, 123.45);
        assert_eq!(p.predict(pol), Some(123.45));
    }

    #[test]
    fn predictor_needs_min_samples() {
        let mut p = Predictor::default();
        // 1 サンプルのみ (≠ target) → 信頼できる予測なし．
        p.observe(Policy::default(), 10.0);
        let other = Policy {
            m: 5,
            k: 5,
            c: 3,
            ..Policy::default()
        };
        assert_eq!(p.predict(other), None, "fewer than MIN_SAMPLES → None");
    }

    #[test]
    fn predictor_interpolates_when_trained() {
        // 十分なサンプルがあれば近傍回帰で有限値を返す．
        let mut p = Predictor::default();
        let pols = [
            Policy::default(),
            Policy {
                m: 2,
                ..Policy::default()
            },
            Policy {
                k: 2,
                ..Policy::default()
            },
            Policy {
                c: 1,
                ..Policy::default()
            },
            Policy {
                m: 4,
                ..Policy::default()
            },
        ];
        for (i, pol) in pols.iter().enumerate() {
            p.observe(*pol, 100.0 + i as f64);
        }
        let query = Policy {
            m: 3,
            k: 3,
            c: 2,
            ..Policy::default()
        };
        let pred = p.predict(query);
        assert!(pred.is_some());
        assert!(pred.unwrap().is_finite());
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
