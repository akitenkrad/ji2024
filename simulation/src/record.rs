//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `poa` / `reproduce` のどれでも
//! 同じ論文を指すので，ここ 1 箇所で組み立てる．実験条件 (`parameters`)，
//! ステップごとの指標，run 全体を 1 つの値で表す指標，POA の世代パネルと
//! `reproduce` の条件表もここに集める．
//!
//! # 反復 (replicate) をどう置いたか
//!
//! 旧 `metrics.csv` は `run` (試行 index) と `round` を第 1・第 2 列に持ち，
//! 1 ファイルに同一条件の反復を複数本入れていた．runvault の `metrics.csv` は
//! (`run_uid`, `step`, `step_unit`, `scope`, `name`) が主キーなので，この形は
//! そのままでは載らない — 同じ `round` の行が試行の数だけ同じ主キーを名乗る．
//!
//! `sweep` の旧 `metrics.csv` はさらに悪く，`entry_condition` /
//! `resource_subset` / `queue_strategy` の列を持たないまま 3 セル分の行を
//! 1 ファイルに積んでいた．`run` 列は各セルの中の 0..runs なので，3 セルが同じ
//! `run` 番号を名乗り，どの行がどのセルのものかは seed からしか辿れなかった．
//!
//! **試行 1 本を子 run 1 本にした．** 決め手は «反復が時系列を持つか» である．
//! `run` も `sweep` も試行ごとに全ラウンドの指標行を残しているので，試行ごとに
//! `metrics.csv` が要る (ステップ系列の置き場はそこしかない)．そこで親が
//! [`runvault::RunOptions::sweep_parent`] を名乗って条件 (または格子) と反復数を
//! 宣言し，子がそれぞれ自分の `master_seed` と `replicate_index`，そしてセル座標を
//! 持つ．セル座標が子の `parameters` に入るので，上の «セル列が無い» 欠陥は
//! 消える — 同一セルの試行は `config_hash` が一致し，どれとどれが同じ条件の
//! 繰り返しかは機械が答える．
//!
//! # POA と reproduce
//!
//! `poa` は世代を時間軸に持つ 1 回の最適化，`reproduce` は 2 つの目標について
//! 最適化を回したうえで条件どうしを比較して判定を出す 1 回の実行である．
//! `reproduce` の «paired win-rate» は条件をまたぐ比較そのものなので，条件を
//! 子 run に割ると起きていない実行を主張することになる．どちらも run は 1 本で，
//! 世代パネルは `x.ji2024.poa_generation`，条件表は `x.ji2024.table2_cell` /
//! `x.ji2024.poa_best_policy` の名前空間つきイベントに置く．
//!
//! 世代パネルを `metrics.csv` に降ろさないのは，`reproduce` が同じ時間軸の上に
//! 2 本の系列 (満足度 / 公平性) を持ち，`metrics.csv` がそれを表せないからである．
//! 片方だけイベントにすると 1 つの表が 2 つの形に割れるので，`poa` も同じ形に
//! 揃えた．世代ごとの最良ポリシーはラベル (`p_select` / `fifo` / `r_size`) を
//! 含む 1 行なので，そもそも指標には割れない．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::config::Config;
use crate::metrics::MetricRow;
use crate::poa::{PoaHistoryRow, PoaResult};
use crate::policy::Policy;
use crate::simulation::SimulationResult;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "srap-allocation";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "ji2024";
/// 分野．応募者プロファイル・資源プール・スケジューラを乱数で引くので
/// `master_seed` が必須の `simulation` を名乗る．LLM 側の同一性は `llm` ブロックが
/// 持つ (測っているのはモデルの安全性ではなく配分ポリシーの厚生なので
/// `llm-safety` ではない)．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．モデルの刻みは配分ラウンドなので語彙の `round`．
pub const T_UNIT: &str = "round";
/// POA の時間軸の単位．世代なので語彙の `generation`．
pub const GENERATION_UNIT: &str = "generation";

/// 指標の粒度．満足度・公平性はどれも応募者全体の集約なので `run`．
const SCOPE: &str = "run";

/// POA の 1 世代を表す実験固有のイベント種別．
pub const POA_GENERATION_EVENT: &str = "x.ji2024.poa_generation";
/// POA が最後に選んだポリシー π\* を表す実験固有のイベント種別．
pub const POA_BEST_POLICY_EVENT: &str = "x.ji2024.poa_best_policy";
/// `reproduce` の Table 2 の 1 セルを表す実験固有のイベント種別．
pub const TABLE2_CELL_EVENT: &str = "x.ji2024.table2_cell";

/// 設計書 (Obsidian)．
const OBSIDIAN_NOTE: &str = "研究/98_論文レポート/80-再現実験/実装完了/ji2024/設計書.md";

// ---------------------------------------------------------------------------
// 論文メタデータ
// ---------------------------------------------------------------------------

/// この再現実験が対象としている論文．
fn work() -> Work {
    let mut work = Work::arxiv("2410.14152")
        .title(
            "SRAP-Agent: Simulating and Optimizing Scarce Resource Allocation Policy \
             with LLM-based Agent",
        )
        .year(2024)
        .source_version("published");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は arXiv 側)．
    work.paper_id = Some("P00001804".to_string());
    work
}

/// 配分ポリシーの厚生比較 (Table 2) だけを見る run — `run` / `sweep` /
/// `mock_smoke`．POA を回さない run に Table 3 / Fig.4 を名乗らせない．
pub fn replication_policy() -> Replication {
    Replication::new(work())
        .target(Target::table("table2", "Table 2"))
        .obsidian_note(OBSIDIAN_NOTE)
}

/// POA によるポリシー最適化 (Table 3 と収束曲線 Fig.4) を見る run — `poa`．
pub fn replication_poa() -> Replication {
    Replication::new(work())
        .target(Target::table("table3", "Table 3"))
        .target(Target::figure("fig4", "Figure 4"))
        .obsidian_note(OBSIDIAN_NOTE)
}

/// 3 箇所すべてを一括で見る run — `reproduce`．
pub fn replication_all() -> Replication {
    Replication::new(work())
        .target(Target::table("table2", "Table 2"))
        .target(Target::table("table3", "Table 3"))
        .target(Target::figure("fig4", "Figure 4"))
        .obsidian_note(OBSIDIAN_NOTE)
}

// `reference.csv` は書かない．
//
// 原著の Table 2 / Table 3 が印字している数は GPT-3.5-turbo と論文付録の合成環境に
// 固有で，本実装の合成環境 (応募者数・資源分布・効用関数を CLI で外部化したもの) の
// 量とは単位も尺度も違う．設計書 §7 が再現目標に置いているのは絶対値ではなく
// «r_size 最高 / r_random 最低» という順序と «POA が f(π) を改善する» という符号で，
// これは数ではない (`Target::claim` ではなく表・図を target に取っているのはこのため)．
// 判定に使う «過半数の環境で成立すれば PASS» の 0.5 はこちらが決めた帯なので，
// 論文の報告値の欄には当然置けない — 判定は `artifacts/reproduce_summary.json` と
// コンソールに残す．

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う (名前を推測で
/// 書かない)．`provider` は runvault の語彙ではなく自由記述なので，endpoint から
/// «どのゲートウェイが答えたか» を決める (`mock://…` はオフラインの scripted
/// クライアント，それ以外はホスト名で Ollama / OpenAI を分ける)．
///
/// `--mock` でも LLM 経路そのものは通る (scripted client が応答する) ので，
/// mock の run にも `llm` ブロックを書く．旧 `llm_meta.json` が mock 実行でも
/// model / endpoint を残していたのと同じ扱いである．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // プロンプトは応募者ごとに組み立てられ，固定の system prompt を持たない．
        // 無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// 実験条件 (parameters)
// ---------------------------------------------------------------------------

/// `config_hash` から外すポインタ．
///
/// キャッシュのパスは «どこに置いたか» であって条件ではない．同じ条件の run を
/// 別のキャッシュファイルで回しても同じ条件である．
pub const HASH_EXCLUDE: [&str; 1] = ["/llm_cache_path"];

/// 試行 1 本 (子 run) の seed ポインタ．
pub const TRIAL_SEED_POINTERS: [&str; 1] = ["/seed"];
/// 反復群 (親 run) の seed ポインタ．
pub const GROUP_SEED_POINTERS: [&str; 1] = ["/base_seed"];

/// LLM 経路の駆動方式．旧 `config.json` はこれを持っていなかったが，mock か
/// live かで応募者の意思決定が丸ごと変わるので，条件に入れないと «同じ条件» を
/// 名乗る 2 本の run が実は別のエージェントで回っていたことになる．
pub fn decision_mode(mock: bool) -> &'static str {
    if mock {
        "mock"
    } else {
        "live"
    }
}

/// LLM レイヤの条件．どのサブコマンドでも同じ 3 つ．
#[derive(Serialize)]
pub struct LlmParameters {
    pub decision_mode: &'static str,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    /// プロンプト → 応答キャッシュの置き場．条件ではなく置き場なので
    /// `hash_exclude` で `config_hash` から外す ([`HASH_EXCLUDE`])．
    pub llm_cache_path: Option<String>,
}

impl LlmParameters {
    /// [`Config`] と `--mock` から組み立てる．
    pub fn from_config(cfg: &Config, mock: bool) -> Self {
        LlmParameters {
            decision_mode: decision_mode(mock),
            llm_temperature: cfg.llm.temperature,
            llm_seed: cfg.llm.seed,
            llm_cache_path: cfg.llm.cache_path.clone(),
        }
    }
}

/// 合成環境の条件．ポリシーを除いた «世界» の側．
#[derive(Serialize)]
pub struct EnvParameters {
    pub n_applicants: usize,
    pub pool_ratio: f64,
    pub n_resources: usize,
    pub max_rounds: usize,
    /// 各ラウンドで応募者に提示する可視資源サブセットのサイズ上限．旧
    /// `sweep_config.json` / `poa_config.json` はこれを落としていたが，可視幅は
    /// 応募先の選択肢そのものを決めるので条件に入れる．
    pub visible_subset_size: usize,
}

impl EnvParameters {
    /// [`Config`] から取り出す．
    pub fn from_config(cfg: &Config) -> Self {
        EnvParameters {
            n_applicants: cfg.n_applicants,
            pool_ratio: cfg.pool_ratio,
            n_resources: cfg.n_resources(),
            max_rounds: cfg.max_rounds,
            visible_subset_size: cfg.visible_subset_size,
        }
    }
}

/// 配分ポリシー π = (E_queue, S_queue, R_queue, m, k, c)．
///
/// 値は [`Policy::normalized`] を通したものを書く — `init_world` が正規化済みの
/// ポリシーで世界を組むので，記録された条件が実際に効いた条件と食い違わない．
#[derive(Serialize)]
pub struct PolicyParameters {
    pub entry_condition: String,
    pub sort_strategy: String,
    pub resource_subset: String,
    pub m: usize,
    pub k: usize,
    pub c: usize,
}

impl PolicyParameters {
    /// [`Policy`] から取り出す．
    pub fn from_policy(policy: Policy) -> Self {
        let p = policy.normalized();
        PolicyParameters {
            entry_condition: p.entry_condition.label().to_string(),
            sort_strategy: p.sort_strategy.label().to_string(),
            resource_subset: p.resource_subset.label().to_string(),
            m: p.m,
            k: p.k,
            c: p.c,
        }
    }
}

/// 1 条件そのもの．シードも反復数も含まない．
#[derive(Serialize)]
pub struct ConditionParameters {
    #[serde(flatten)]
    pub env: EnvParameters,
    #[serde(flatten)]
    pub policy: PolicyParameters,
    #[serde(flatten)]
    pub llm: LlmParameters,
}

impl ConditionParameters {
    /// [`Config`] と `--mock` から組み立てる．
    pub fn from_config(cfg: &Config, mock: bool) -> Self {
        ConditionParameters {
            env: EnvParameters::from_config(cfg),
            policy: PolicyParameters::from_policy(cfg.policy),
            llm: LlmParameters::from_config(cfg, mock),
        }
    }
}

/// 試行 1 本 (子 run) の実験条件．
///
/// `seed` はこの試行が実際に使ったシードで，`master_seed` と同じ値である．
/// `seed_pointers` で seed として宣言するので `config_hash` からは外れ，
/// 同一セルの試行どうしは `config_hash` が一致する．
#[derive(Serialize)]
pub struct TrialParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub seed: u64,
}

/// 「1 条件 + その反復群」の実験条件 (`run` の親 run)．
///
/// `base_seed` は試行ごとのシードを派生させる元で，それ自体でシミュレーションを
/// 回す値ではない．親は 1 本のシミュレーションではないので `master_seed` を
/// 名乗らない．
#[derive(Serialize)]
pub struct ReplicateGroupParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub runs: usize,
    pub base_seed: u64,
}

/// スイープ親 run の実験条件 (格子の定義そのもの)．
#[derive(Serialize)]
pub struct SweepGridParameters {
    pub entry_conditions: Vec<String>,
    pub resource_subsets: Vec<String>,
    pub queue_strategies: Vec<String>,
    pub m: usize,
    pub k: usize,
    pub c: usize,
    #[serde(flatten)]
    pub env: EnvParameters,
    #[serde(flatten)]
    pub llm: LlmParameters,
    pub runs: usize,
    pub base_seed: u64,
}

/// `poa` の実験条件．
///
/// 旧 `poa_config.json` は `full_evals` / `evals_saved` (実行の結果) を設定に
/// 混ぜていたが，あれは条件ではないので指標側へ移した．代わりに旧設定が
/// 落としていた `visible_subset_size` と LLM の条件を入れてある．
#[derive(Serialize)]
pub struct PoaParameters {
    pub objective: String,
    pub iterations: usize,
    pub pool_size: usize,
    pub mutation_rate: f64,
    pub tournament_size: usize,
    pub use_predictor: bool,
    #[serde(flatten)]
    pub env: EnvParameters,
    #[serde(flatten)]
    pub llm: LlmParameters,
    pub seed: u64,
}

/// `reproduce` の実験条件．
///
/// `quick` は世代数・応募者数・ラウンド数を縮小するフラグだが，縮小後の値を
/// そのまま書くので，条件としては縮小後の値が正本である (`quick` 自体も
/// «どの経路で来たか» として残す)．旧 `reproduce_summary.json` の `config` が
/// 落としていた `visible_subset_size` / m / k / c / LLM 条件，および POA 側で
/// ハードコードされていた `mutation_rate` / `tournament_size` / `use_predictor`
/// を入れてある — どれも結果を決める値である．
#[derive(Serialize)]
pub struct ReproduceParameters {
    #[serde(flatten)]
    pub env: EnvParameters,
    pub m: usize,
    pub k: usize,
    pub c: usize,
    pub runs: usize,
    pub poa_iterations: usize,
    pub poa_pool_size: usize,
    pub poa_mutation_rate: f64,
    pub poa_tournament_size: usize,
    pub poa_use_predictor: bool,
    pub quick: bool,
    #[serde(flatten)]
    pub llm: LlmParameters,
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// 試行 1 本の記録 (run / sweep / mock_smoke の子 run)
// ---------------------------------------------------------------------------

/// 試行 1 本ぶんの記録．
///
/// ラウンドごとの 8 指標 (`round` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す `final_round` を書く．旧 `metrics.csv` の `run` 列は
/// `replicate_index`，seed は `master_seed` になったので指標にはしない．旧
/// `llm_meta.json` の `final_sw` / `final_n_allocated` も書かない — 最終ラウンドの
/// 値はステップ指標の最後の行が正本で，同じ数を 2 箇所に置かない．実行時間は
/// `status.json` の `duration_sec` が正本なので指標にしない．
pub fn log_trial(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics {
        log_round(run, m);
    }
    // `final_round` は «実行したラウンド数» (engine の最終 t) で，最後の指標行の
    // `round` (0 始まりのラウンド番号) とは 1 つずれる別の数なので書く．
    run.log_metrics(SCOPE, &[("final_round", result.final_round as f64)])
        .expect("run スコープの指標の記録に失敗");
}

/// [`MetricRow`] の数値フィールドを 1 ラウンドぶんまとめて書く．
fn log_round(run: &mut Run, m: &MetricRow) {
    run.log_metrics_at(
        m.round,
        T_UNIT,
        SCOPE,
        &[
            ("sw", m.sw),
            ("avg_rsize", m.avg_rsize),
            ("avg_wt", m.avg_wt),
            ("var_rsize", m.var_rsize),
            ("rop", m.rop),
            ("co_gini", m.co_gini),
            ("f_vnv", m.f_vnv),
            ("n_allocated", m.n_allocated as f64),
        ],
    )
    .unwrap_or_else(|e| panic!("round {} の指標の記録に失敗: {e}", m.round));
}

/// LLM 呼び出しの内訳を run スコープの指標として書く．
///
/// `tokens_in` / `tokens_out` / `cost_usd` は書かない — socsim-llm の
/// `MetadataCollector` はトークン数も費用も持たないので，予約名に入れる値が無い．
pub fn log_llm_usage(run: &mut Run, result: &SimulationResult) {
    run.log_metrics(
        SCOPE,
        &[
            ("llm_calls", result.metadata.total() as f64),
            ("llm_cache_hits", result.metadata.cache_hits() as f64),
            ("llm_cache_hit_rate", result.metadata.cache_hit_rate()),
        ],
    )
    .expect("LLM 呼び出しの内訳の記録に失敗");
}

// ---------------------------------------------------------------------------
// POA (poa / reproduce)
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く POA の 1 世代．
///
/// 予約語 `unit_id` に最適化目標を，`t` に世代を置く．`reproduce` は満足度と
/// 公平性の 2 系列を同じ世代軸の上に持つので，`metrics.csv` (主キーに系列 ID を
/// 持たない) では表せない．世代ごとの最良ポリシーはラベルなので，そもそも指標には
/// 割れない — 数とラベルが 1 行に同居するこの表は，割らずにイベントへ置く．
#[derive(Serialize)]
struct PoaGenerationEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    best_fitness: f64,
    mean_fitness: f64,
    evals_saved: usize,
    full_evals: usize,
    best_entry_condition: &'a str,
    best_sort_strategy: &'a str,
    best_resource_subset: &'a str,
    best_m: usize,
    best_k: usize,
    best_c: usize,
}

/// `events.jsonl` に書く POA の最終解 π\*．
///
/// 旧 `table3_optimized_policies.csv` の 1 行にあたる．`initial_fitness` は
/// 書かない — 世代 0 の `best_fitness` そのものなので，同じ数を 2 箇所に置かない
/// (読み側は `poa_generation` の `t=0` から取る)．`total_full_evals` /
/// `total_evals_saved` は世代パネルの累積値とは別の数である (最終世代の評価は
/// 履歴を積んだ後に走るので，パネルの最後の行には入っていない)．
#[derive(Serialize)]
struct PoaBestPolicyEvent<'a> {
    unit_id: &'a str,
    objective: &'a str,
    entry_condition: &'a str,
    sort_strategy: &'a str,
    resource_subset: &'a str,
    m: usize,
    k: usize,
    c: usize,
    final_fitness: f64,
    improvement_pct: f64,
    total_full_evals: usize,
    total_evals_saved: usize,
}

/// POA の世代パネルを書く．
pub fn log_poa_history(run: &mut Run, objective: &str, history: &[PoaHistoryRow]) {
    for h in history {
        run.log_event(
            POA_GENERATION_EVENT,
            &PoaGenerationEvent {
                unit_id: objective,
                t: h.generation as u64,
                t_unit: GENERATION_UNIT,
                best_fitness: h.best_fitness,
                mean_fitness: h.mean_fitness,
                evals_saved: h.evals_saved,
                full_evals: h.full_evals,
                best_entry_condition: &h.best_entry_condition,
                best_sort_strategy: &h.best_sort_strategy,
                best_resource_subset: &h.best_resource_subset,
                best_m: h.best_m,
                best_k: h.best_k,
                best_c: h.best_c,
            },
        )
        .unwrap_or_else(|e| panic!("世代 {} の記録に失敗: {e}", h.generation));
    }
}

/// POA の最終解 π\* を書く．
pub fn log_poa_best_policy(run: &mut Run, objective: &str, result: &PoaResult) {
    let best = result.best_policy;
    run.log_event(
        POA_BEST_POLICY_EVENT,
        &PoaBestPolicyEvent {
            unit_id: objective,
            objective,
            entry_condition: best.entry_condition.label(),
            sort_strategy: best.sort_strategy.label(),
            resource_subset: best.resource_subset.label(),
            m: best.m,
            k: best.k,
            c: best.c,
            final_fitness: result.best_fitness,
            improvement_pct: result.improvement_pct(),
            total_full_evals: result.full_evals,
            total_evals_saved: result.evals_saved,
        },
    )
    .unwrap_or_else(|e| panic!("{objective} の最終解の記録に失敗: {e}"));
}

// ---------------------------------------------------------------------------
// reproduce の条件表
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く Table 2 の 1 セル．
///
/// この表には時間軸が無く，(入室条件 × 資源サブセット) の 9 行が並ぶ．
/// `metrics.csv` に降ろすと 9 行が同じ主キーを名乗るし，条件を子 run に割ると
/// «同じ環境で資源サブセットだけを変えた» paired 比較が run をまたいでしまう．
/// 条件は «1 回の実行の中で観測された対象» なので，予約語 `unit_id` を持つ
/// イベントとして書く．
#[derive(Serialize)]
struct Table2CellEvent<'a> {
    unit_id: &'a str,
    entry_condition: &'a str,
    resource_subset: &'a str,
    mean_sw: f64,
    runs: usize,
}

/// Table 2 の 1 セルを書く．
pub fn log_table2_cell(
    run: &mut Run,
    entry_condition: &str,
    resource_subset: &str,
    mean_sw: f64,
    runs: usize,
) {
    let unit_id = format!("{entry_condition}:{resource_subset}");
    run.log_event(
        TABLE2_CELL_EVENT,
        &Table2CellEvent {
            unit_id: &unit_id,
            entry_condition,
            resource_subset,
            mean_sw,
            runs,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の記録に失敗: {e}"));
}

/// `reproduce` の run 全体を 1 つの値で表す指標．
///
/// ここに置けるのは «run に 1 つしか無い» 値に限る．条件ごとの値
/// (`mean_sw` / 最適化の結果) はイベントの担当で，こちらに降ろすと主キーが
/// 衝突する．`headline_mean_sw_*` と `best_policy_mean_sw` は旧
/// `reproduce_summary.json` が持っていたが，どれも Table 2 のセルそのものなので
/// 書かない (`p_select` 行と最大値は読み側が引ける)．
///
/// 勝ち数と勝率の両方を書くのは，比が «分母が何本だったか» を持たないためである
/// (`llm_cache_hits` と `llm_cache_hit_rate` を両方書くのと同じ)．
pub fn log_reproduce_summary(
    run: &mut Run,
    n_matched_environments: usize,
    size_wins: usize,
    rent_wins: usize,
) {
    let n = n_matched_environments as f64;
    run.log_metrics(
        SCOPE,
        &[
            ("n_matched_environments", n),
            ("paired_wins_rsize_ge_rrandom", size_wins as f64),
            ("paired_wins_rrent_ge_rrandom", rent_wins as f64),
            ("paired_win_rate_rsize_ge_rrandom", size_wins as f64 / n),
            ("paired_win_rate_rrent_ge_rrandom", rent_wins as f64 / n),
        ],
    )
    .expect("reproduce の run スコープ指標の記録に失敗");
}

// ---------------------------------------------------------------------------
// シードの派生
// ---------------------------------------------------------------------------

/// 試行 1 本のシードを base seed から派生させる (`run`)．
///
/// 旧 `derive_run_seed` と同じ (`crate::config::derive_run_seed` の別名ではなく
/// 委譲)．シードの決め方は移行で変えていない．
pub fn replicate_seed(base: u64, index: usize) -> u64 {
    crate::config::derive_run_seed(base, index)
}

/// 試行 1 本のシードを base seed とセル座標から派生させる (`sweep`)．
pub fn trial_seed(base: u64, entry: u64, subset: u64, strategy: u64, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[entry, subset, strategy, index as u64])
}

/// 環境 1 つのシードを base seed と (入室条件, 試行 index) から派生させる
/// (`reproduce` の Table 2; matched-seed 比較のため資源サブセットを含めない)．
pub fn matched_env_seed(base: u64, entry: u64, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[entry, index as u64])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{EntryCondition, ResourceSubset, SortStrategy};

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(replicate_seed(42, 3), replicate_seed(42, 3));
        assert_eq!(trial_seed(42, 1, 2, 0, 3), trial_seed(42, 1, 2, 0, 3));
        assert_eq!(matched_env_seed(42, 2, 1), matched_env_seed(42, 2, 1));
    }

    #[test]
    fn each_coordinate_changes_the_seed() {
        let base = trial_seed(42, 0, 0, 0, 0);
        assert_ne!(base, trial_seed(43, 0, 0, 0, 0), "base が効いていない");
        assert_ne!(base, trial_seed(42, 1, 0, 0, 0), "entry が効いていない");
        assert_ne!(base, trial_seed(42, 0, 1, 0, 0), "subset が効いていない");
        assert_ne!(base, trial_seed(42, 0, 0, 1, 0), "strategy が効いていない");
        assert_ne!(base, trial_seed(42, 0, 0, 0, 1), "index が効いていない");
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` が変わったときで，そのときは過去の
    /// run と結果を比較できなくなっている．Cargo.lock が socsim の commit を固定して
    /// いるので，この値は依存を上げたときにだけ動く．どれも移行前の commit
    /// (`6235b03`) が同じ引数で出す値と一致することを実測で確かめてある．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(
            trial_seed(
                42,
                EntryCondition::PSelect as u64,
                ResourceSubset::RSize as u64,
                SortStrategy::Fifo as u64,
                0,
            ),
            16_589_784_518_614_893_705,
        );
        assert_eq!(replicate_seed(42, 0), 12_668_979_023_698_771_201);
        assert_eq!(
            matched_env_seed(42, EntryCondition::PSelect as u64, 0),
            3_551_134_521_934_701_153,
        );
    }

    #[test]
    fn policy_parameters_record_the_normalized_policy() {
        let p = PolicyParameters::from_policy(Policy {
            m: 0,
            k: 0,
            c: 0,
            ..Policy::default()
        });
        assert_eq!((p.m, p.k, p.c), (1, 1, 1));
    }
}
