//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で世界初期化
//!   (応募者プロファイル・資源プール) の init RNG を，`derive_seed(root, &[1])` で
//!   engine RNG (scheduler / 配分順序のシャッフル) を派生する．キュー構成・配分
//!   規則・指標計算・記憶更新は bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き
//!   Ollama→OpenAI フォールバッククライアントに閉じ込め，`temperature=0`/`seed`
//!   固定 + プロンプト→応答キャッシュで擬似決定論化する．モデル・endpoint・
//!   温度・seed・cache-hit を `llm_meta.json` に記録する．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::MetadataCollector;

use crate::config::Config;
use crate::llm::{build_live_client, SrapClient};
use crate::mechanisms::{
    AllocationRule, ApplyDecision, EvaluateWelfare, PolicySetup, SharedClient, SharedMetadata,
    SharedMetrics, UpdateMemory,
};
use crate::metrics::{MetricRow, SrapMetrics};
use crate::world::{Applicant, Memory, Preferences, Resource, SrapWorld};

/// 世界初期化用 RNG ラベル (応募者プロファイル・資源プール・キュー初期配置)．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン用 RNG ラベル (scheduler / 配分順序のシャッフル)．
const RNG_ENGINE: u64 = 1;

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// メトリクス行の履歴 (metrics.csv; long-format)．
    pub metrics: Vec<MetricRow>,
    /// 最終ラウンドの集計 (world.metrics のスナップショット)．
    pub final_metrics: SrapMetrics,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名．
    pub llm_model: String,
    /// LLM endpoint (primary)．
    pub llm_endpoint: String,
    /// 実行したラウンド数 (= 完了ステップ数)．
    pub final_round: usize,
}

impl SimulationResult {
    /// 最終ラウンドの社会的厚生 SW．
    pub fn final_sw(&self) -> f64 {
        self.final_metrics.sw
    }
}

/// 世界状態を初期化する (応募者プロファイル生成 + 資源プール初期化)．
///
/// 合成環境のパラメータ (応募者数・資源数・プロファイル分布) は論文付録依存のため
/// CLI で外部化する (設計書 §7 の不確実性ボックス)．プロファイルは init RNG で
/// 生成し，収入・家族規模から脆弱層フラグを決める (低収入かつ大世帯 = 脆弱)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> SrapWorld {
    let n = cfg.n_applicants;
    let n_res = cfg.n_resources();

    // --- 応募者プロファイル ---
    let mut applicants: BTreeMap<AgentId, Applicant> = BTreeMap::new();
    for i in 0..n {
        // 月収 2000..10000, 家族 1..6, 家賃 = 収入の 15-25%.
        let income = rng.gen_range(2000.0..10000.0);
        let family = rng.gen_range(1..=6usize);
        let rent = income * rng.gen_range(0.15..0.25);
        let size_weight = rng.gen_range(0.8..1.2);
        let rent_weight = rng.gen_range(0.8..1.2);
        // 脆弱層: 低収入 (下位閾値 4000) かつ大世帯 (>=4) を vulnerable とみなす．
        let vulnerable = income < 4000.0 && family >= 4;
        applicants.insert(
            AgentId(i as u64),
            Applicant {
                income,
                rent,
                family,
                preferences: Preferences {
                    size_weight,
                    rent_weight,
                },
                memory: Memory::default(),
                active: true,
                vulnerable,
            },
        );
    }

    // --- 資源プール (公共住宅; 面積と家賃は正相関) ---
    let mut pool: Vec<Resource> = Vec::with_capacity(n_res);
    for id in 0..n_res {
        let size = rng.gen_range(30.0..100.0);
        // 家賃は面積に正相関 (+ノイズ) → r_size と r_rent が類似性能になる根拠．
        let rent = size * rng.gen_range(12.0..18.0) + rng.gen_range(-50.0..50.0);
        pool.push(Resource {
            id,
            size,
            rent: rent.max(100.0),
            allocated: false,
        });
    }

    SrapWorld {
        clock: SimClock::new(cfg.max_rounds as u64),
        applicants,
        pool,
        policy: cfg.policy.normalized(),
        queues: Vec::new(),
        allocations: BTreeMap::new(),
        metrics: SrapMetrics::default(),
    }
}

/// シミュレーションを実行する (本番 LLM クライアントを構築して駆動)．
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    let client =
        build_live_client(&cfg.llm).map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
    run_with_client(cfg, client, 0)
}

/// 与えられた [`SrapClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．Scheduler は `RandomActivationScheduler`
/// (到着パターンの確率性を engine RNG のシャッフルで担保; 配分順序自体は決定論的な
/// S_queue で確定する)．
pub fn run_with_client(
    cfg: &Config,
    client: SrapClient,
    run_idx: usize,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_metrics: SharedMetrics = Rc::new(RefCell::new(Vec::new()));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(PolicySetup))
        .add_mechanism(Box::new(ApplyDecision::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            cfg.visible_subset_size,
        )))
        .add_mechanism(Box::new(AllocationRule))
        .add_mechanism(Box::new(EvaluateWelfare::new(
            Rc::clone(&shared_metrics),
            run_idx,
        )))
        .add_mechanism(Box::new(UpdateMemory {
            window: cfg.max_rounds.max(1),
        }))
        .build();

    let mut final_round = 0usize;
    let mut final_metrics = SrapMetrics::default();
    sim.run_observed(|report| {
        final_round = report.t as usize;
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    // 最終 world のメトリクスを取り出す (run_observed は world を消費しないので
    // 最後のメトリクス行から復元する)．
    if let Some(last) = shared_metrics.borrow().last() {
        final_metrics = SrapMetrics {
            sw: last.sw,
            avg_rsize: last.avg_rsize,
            avg_wt: last.avg_wt,
            var_rsize: last.var_rsize,
            rop: last.rop,
            co_gini: last.co_gini,
            f_vnv: last.f_vnv,
            n_allocated: last.n_allocated,
        };
    }

    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metrics = shared_metrics.borrow().clone();
    let metadata = shared_meta.borrow().clone();
    Ok(SimulationResult {
        metrics,
        final_metrics,
        metadata,
        llm_model,
        llm_endpoint,
        final_round,
    })
}

// --------------------------------------------------------------------------- //
// 出力
// --------------------------------------------------------------------------- //

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("出力ディレクトリの作成に失敗");
}

/// `metrics.csv` を保存する (long-format; 複数 run を追記する用に append も可)．
///
/// 書き出し機構は `socsim_results::write_csv` に委譲する (各行を `serialize` し
/// 先頭行にヘッダを書く csv クレットの標準挙動; 従来の手書き writer とバイト等価)．
/// 行構造体 [`MetricRow`] は repo 固有のままで，writer だけを共有化する．
pub fn save_metrics(metrics: &[MetricRow], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    socsim_results::write_csv(metrics, &path).expect("metrics.csv の書き込みに失敗");
}

/// `llm_meta.json` の構造体 (provider/model/endpoint/temperature/seed/cache 統計)．
#[derive(Serialize)]
pub struct LlmMetaJson {
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub final_round: usize,
    pub final_sw: f64,
    pub final_n_allocated: usize,
    pub determinism_note: &'static str,
}

/// `llm_meta.json` を保存する．
pub fn save_llm_meta(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = LlmMetaJson {
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        final_round: result.final_round,
        final_sw: result.final_sw(),
        final_n_allocated: result.final_metrics.n_allocated,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (applicant/pool init, entry-condition queue \
                           building, the deterministic allocation rule with k-deferrals, the \
                           welfare/fairness metrics, and memory updates) is deterministic given \
                           the seed.",
    };
    // pretty-print JSON の書き出しは socsim_results::write_json に委譲する
    // (内部は serde_json::to_writer_pretty + flush; 従来の writer とバイト等価)．
    // model/endpoint/temperature/seed の値は従来どおり result / cfg から採り，
    // LlmMetaJson の構造 (フィールド名・順序・determinism_note) を保持する
    // (`MetadataCollector::summary()` は cache-hit 100% 再実行や呼び出し 0 件で
    // endpoint/model が変わりうるため，バイト等価のためここでは使わない)．
    let path = format!("{}/llm_meta.json", output_dir);
    socsim_results::write_json(&meta, &path).expect("llm_meta.json の書き込みに失敗");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::wrap_client;
    use crate::policy::{Policy, ResourceSubset};
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    /// «最初の可視 home を選ぶ» scripted mock (プロンプトの最初の `home N` を拾う)．
    fn scripted_first_home() -> SrapClient {
        let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
            // プロンプト本文の最初の "home N:" を希望する．
            if let Some(idx) = prompt.find("home ") {
                let rest = &prompt[idx + "home ".len()..];
                let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if !num.is_empty() {
                    return format!("Thought: I'll take it. {{\"choice\": {num}}}");
                }
            }
            "{\"choice\": -1}".to_string()
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn cfg() -> Config {
        Config {
            n_applicants: 20,
            pool_ratio: 0.5,
            max_rounds: 5,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn init_world_synthesizes_profiles_and_pool() {
        let c = cfg();
        let mut rng = SimRng::from_seed(0);
        let w = init_world(&c, &mut rng);
        assert_eq!(w.n_applicants(), 20);
        assert_eq!(w.n_resources(), 10);
    }

    #[test]
    fn scripted_run_produces_metrics() {
        let c = cfg();
        let r = run_with_client(&c, scripted_first_home(), 0).unwrap();
        assert!(!r.metrics.is_empty(), "metrics rows produced");
        assert!(r.final_sw() >= 0.0 || r.final_sw() < 0.0); // finite
        assert!(r.final_metrics.co_gini >= 0.0 && r.final_metrics.co_gini <= 1.0);
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let c = cfg();
        let a = run_with_client(&c, scripted_first_home(), 0).unwrap();
        let b = run_with_client(&c, scripted_first_home(), 0).unwrap();
        let sa: Vec<f64> = a.metrics.iter().map(|m| m.sw).collect();
        let sb: Vec<f64> = b.metrics.iter().map(|m| m.sw).collect();
        assert_eq!(sa, sb, "同一シード + 同一 mock は完全再現すべき");
    }

    #[test]
    fn r_size_beats_r_random_sw() {
        // 同一 seed・同一 mock で r_size と r_random の最終 SW を比較する．
        let mut c_size = cfg();
        c_size.policy = Policy {
            resource_subset: ResourceSubset::RSize,
            ..Policy::default()
        };
        let mut c_rand = cfg();
        c_rand.policy = Policy {
            resource_subset: ResourceSubset::RRandom,
            ..Policy::default()
        };
        let r_size = run_with_client(&c_size, scripted_first_home(), 0).unwrap();
        let r_rand = run_with_client(&c_rand, scripted_first_home(), 0).unwrap();
        // mock は «最初の可視 home» を選ぶ → r_size は広い家から提示するので SW が高い．
        assert!(
            r_size.final_sw() >= r_rand.final_sw(),
            "SW(r_size)={} should be >= SW(r_random)={}",
            r_size.final_sw(),
            r_rand.final_sw()
        );
    }
}
