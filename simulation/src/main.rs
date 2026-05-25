//! Ji et al. (2024) "SRAP-Agent" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一ポリシー π で LLM 駆動の希少資源配分シミュレーションを実行し，
//!               社会的厚生 (SW)・公平性指標を計算する．
//! `sweep`     : 入室条件 × 資源サブセット (× 並び替え戦略) を走査し，最終 SW を
//!               `sweep_summary.csv` に集計する (論文 Table 2 の感度分析)．
//! `poa`       : 遺伝的アルゴリズムによるポリシー最適化 (Phase-3 最小スタブ)．
//! `reproduce` : 論文 Table 2/3・Fig.4 一括再現 (Phase 3; 現状はスタブ案内)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use srap_simulation::config::{derive_run_seed, Config, LlmSettings};
use srap_simulation::llm::wrap_client;
use srap_simulation::metrics::parse_objective;
use srap_simulation::poa::{run_poa, PoaConfig};
use srap_simulation::policy::{
    parse_entry_condition, parse_resource_subset, parse_sort_strategy, EntryCondition, Policy,
    ResourceSubset, SortStrategy,
};
use srap_simulation::simulation::{
    ensure_output_dir, run_with_client, save_llm_meta, save_metrics, SimulationResult,
};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "srap",
    about = "Ji et al. (2024) SRAP-Agent: LLM-agent scarce-resource (public housing) allocation policy simulation — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一ポリシーで LLM 駆動の希少資源配分シミュレーションを実行する．
    Run(RunArgs),
    /// 入室条件 × 資源サブセット (× 並び替え) を走査し最終 SW を集計する．
    Sweep(SweepArgs),
    /// 遺伝的アルゴリズムによるポリシー最適化 (Phase-3 最小スタブ)．
    Poa(PoaArgs),
    /// 論文 Table 2/3・Fig.4 一括再現 (Phase 3; 現状はスタブ)．
    Reproduce,
}

// --- 合成環境の共通フラグ ---

#[derive(Parser, Debug, Clone)]
struct EnvArgs {
    /// 応募者数 (合成環境; 論文付録依存のため CLI 外部化)．
    #[arg(long, default_value_t = 60)]
    n_applicants: usize,

    /// 資源プール規模 / 応募者数 比 (希少性; n_resources = pool_ratio × n_applicants)．
    #[arg(long, default_value_t = 0.5)]
    pool_ratio: f64,

    /// 最大ラウンド数 (枯渇/全員退出で早期 stop)．
    #[arg(long, default_value_t = 10)]
    max_rounds: usize,

    /// 各ラウンドで応募者に提示する可視資源サブセットのサイズ上限．
    #[arg(long, default_value_t = 5)]
    visible_subset_size: usize,
}

#[derive(Parser, Debug)]
struct RunArgs {
    #[command(flatten)]
    env: EnvArgs,

    /// 入室条件 E_queue (p_budget / p_family / p_select)．
    #[arg(long, default_value = "p_select")]
    entry_condition: String,

    /// 資源サブセット R_queue (r_size / r_rent / r_random)．
    #[arg(long, default_value = "r_size")]
    resource_subset: String,

    /// 並び替え戦略 S_queue (fifo / vfa / vfr)．
    #[arg(long, default_value = "fifo")]
    queue_strategy: String,

    /// キュー数 m．
    #[arg(long, default_value_t = 3)]
    queues: usize,

    /// k-deferrals 試行回数 k．
    #[arg(long, default_value_t = 3)]
    k: usize,

    /// 選択キュー容量係数 c．
    #[arg(long, default_value_t = 2)]
    c: usize,

    /// 独立試行数 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    /// 各応募者が «最初の可視 home» を選ぶ決定論ポリシー．
    #[arg(long, default_value_t = false)]
    mock: bool,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    #[command(flatten)]
    env: EnvArgs,

    /// カンマ区切りの入室条件リスト．
    #[arg(long, default_value = "p_budget,p_family,p_select")]
    entry_conditions: String,

    /// カンマ区切りの資源サブセットリスト．
    #[arg(long, default_value = "r_size,r_rent,r_random")]
    resource_subsets: String,

    /// カンマ区切りの並び替え戦略リスト．
    #[arg(long, default_value = "fifo")]
    queue_strategies: String,

    /// キュー数 m．
    #[arg(long, default_value_t = 3)]
    queues: usize,

    /// k-deferrals 試行回数 k．
    #[arg(long, default_value_t = 3)]
    k: usize,

    /// 選択キュー容量係数 c．
    #[arg(long, default_value_t = 2)]
    c: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 10)]
    runs: usize,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    #[arg(long, default_value_t = false)]
    mock: bool,
}

#[derive(Parser, Debug)]
struct PoaArgs {
    #[command(flatten)]
    env: EnvArgs,

    /// 最適化目標 (satisfaction / fairness)．
    #[arg(long, default_value = "satisfaction")]
    objective: String,

    /// 反復世代数 M．
    #[arg(long, default_value_t = 20)]
    iterations: usize,

    /// 個体群サイズ (pool_size)．
    #[arg(long, default_value_t = 12)]
    pool_size: usize,

    /// 突然変異率．
    #[arg(long, default_value_t = 0.2)]
    mutation_rate: f64,

    /// トーナメントサイズ．
    #[arg(long, default_value_t = 3)]
    tournament_size: usize,

    /// 乱数シード基点．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// 適応度評価を mock で行う (Phase-3 スタブでは常に mock; フラグは明示用)．
    #[arg(long, default_value_t = true)]
    mock: bool,
}

// ---------------------------------------------------------------------------
// 出力構造体
// ---------------------------------------------------------------------------

/// `sweep_summary.csv` の 1 行．
#[derive(serde::Serialize)]
struct SweepRow {
    entry_condition: String,
    resource_subset: String,
    queue_strategy: String,
    run: usize,
    seed: u64,
    final_round: usize,
    final_sw: f64,
    final_avg_rsize: f64,
    final_avg_wt: f64,
    final_var_rsize: f64,
    final_rop: f64,
    final_co_gini: f64,
    final_f_vnv: f64,
    n_allocated: usize,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    entry_conditions: Vec<String>,
    resource_subsets: Vec<String>,
    queue_strategies: Vec<String>,
    n_applicants: usize,
    pool_ratio: f64,
    queues: usize,
    k: usize,
    c: usize,
    max_rounds: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// `poa_config.json` の構造体．
#[derive(serde::Serialize)]
struct PoaConfigJson {
    command: &'static str,
    objective: String,
    iterations: usize,
    pool_size: usize,
    mutation_rate: f64,
    tournament_size: usize,
    n_applicants: usize,
    pool_ratio: f64,
    max_rounds: usize,
    seed: u64,
    note: &'static str,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// «最初の可視 home を選ぶ» 決定論的 scripted mock クライアントを作る．
fn mock_client() -> srap_simulation::llm::SrapClient {
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        if let Some(idx) = prompt.find("home ") {
            let rest = &prompt[idx + "home ".len()..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num.is_empty() {
                return format!("Thought: best fit. {{\"choice\": {num}}}");
            }
        }
        "{\"choice\": -1}".to_string()
    });
    wrap_client(backend, PromptCache::in_memory())
}

/// 1 設定を実行する (`--mock` ならライブ LLM の代わりに scripted mock を使う)．
fn run_one(cfg: &Config, run_idx: usize, mock: bool) -> Result<SimulationResult, String> {
    if mock {
        let mock_cfg = Config {
            llm: LlmSettings {
                cache_path: None,
                ..cfg.llm.clone()
            },
            ..cfg.clone()
        };
        run_with_client(&mock_cfg, mock_client(), run_idx)
    } else {
        // run() は run_idx=0 固定なので run_idx を渡すため run_with_client を直接使う．
        let client = srap_simulation::llm::build_live_client(&cfg.llm)
            .map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
        run_with_client(cfg, client, run_idx)
    }
}

/// 文字列ポリシーフラグをパースして [`Policy`] を組み立てる (panic on error)．
fn build_policy(entry: &str, subset: &str, strategy: &str, m: usize, k: usize, c: usize) -> Policy {
    Policy {
        entry_condition: parse_entry_condition(entry).unwrap_or_else(|e| panic!("{e}")),
        resource_subset: parse_resource_subset(subset).unwrap_or_else(|e| panic!("{e}")),
        sort_strategy: parse_sort_strategy(strategy).unwrap_or_else(|e| panic!("{e}")),
        m,
        k,
        c,
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let policy = build_policy(
        &args.entry_condition,
        &args.resource_subset,
        &args.queue_strategy,
        args.queues,
        args.k,
        args.c,
    );

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&output_dir);

    println!("=== Ji et al. (2024) SRAP-Agent 希少資源配分 再現実験 ===");
    println!(
        "policy: E={} S={} R={} | m={} k={} c={}",
        policy.entry_condition.label(),
        policy.sort_strategy.label(),
        policy.resource_subset.label(),
        policy.m,
        policy.k,
        policy.c,
    );
    println!(
        "env: n_applicants={} pool_ratio={} max_rounds={} | runs={}",
        args.env.n_applicants, args.env.pool_ratio, args.env.max_rounds, args.runs,
    );
    println!(
        "LLM: temp={} llm_seed={} cache={} mock={} | seed: {:?}",
        args.temperature, args.llm_seed, args.cache_path, args.mock, args.seed
    );
    println!("出力先: {output_dir}");
    println!("-------------------------------------------------");

    let base_seed = args.seed.unwrap_or(42);
    let mut all_metrics = Vec::new();
    let mut last_result: Option<SimulationResult> = None;
    let mut last_cfg: Option<Config> = None;

    for run_idx in 0..args.runs.max(1) {
        let seed = derive_run_seed(base_seed, run_idx);
        let cfg = Config {
            n_applicants: args.env.n_applicants,
            pool_ratio: args.env.pool_ratio,
            policy,
            max_rounds: args.env.max_rounds,
            visible_subset_size: args.env.visible_subset_size,
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.temperature,
                seed: args.llm_seed,
                cache_path: if args.mock {
                    None
                } else {
                    Some(args.cache_path.clone())
                },
            },
            output_dir: output_dir.clone(),
        };

        let result =
            run_one(&cfg, run_idx, args.mock).unwrap_or_else(|e| panic!("実行に失敗: {e}"));
        all_metrics.extend(result.metrics.clone());
        last_cfg = Some(cfg);
        last_result = Some(result);
    }

    // long-format metrics.csv (全 run を結合)．
    save_metrics(&all_metrics, &output_dir);

    if let (Some(result), Some(cfg)) = (&last_result, &last_cfg) {
        save_llm_meta(result, cfg, &output_dir);
        // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
        let path = format!("{output_dir}/config.json");
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");

        let m = &result.final_metrics;
        println!(
            "最終ラウンド: {} | 配分人数: {} | SW: {:.2}",
            result.final_round, m.n_allocated, m.sw,
        );
        println!(
            "満足度: Avg r_size={:.2} Avg WT={:.2} | 公平性: Var r_size={:.2} Rop={:.0} Gini={:.3} F(V,NV)={:.2}",
            m.avg_rsize, m.avg_wt, m.var_rsize, m.rop, m.co_gini, m.f_vnv,
        );
        println!(
            "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
            result.metadata.total(),
            result.metadata.cache_hits(),
            result.metadata.cache_hit_rate() * 100.0,
            result.llm_model,
        );
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);
    println!("メトリクス → {output_dir}/metrics.csv");
    println!("LLM メタ   → {output_dir}/llm_meta.json");
    println!("設定       → {output_dir}/config.json");
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let entries: Vec<EntryCondition> = split_csv(&args.entry_conditions)
        .iter()
        .map(|s| parse_entry_condition(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let subsets: Vec<ResourceSubset> = split_csv(&args.resource_subsets)
        .iter()
        .map(|s| parse_resource_subset(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let strategies: Vec<SortStrategy> = split_csv(&args.queue_strategies)
        .iter()
        .map(|s| parse_sort_strategy(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = entries.len() * subsets.len() * strategies.len() * args.runs;
    println!("=== Ji et al. (2024) SRAP-Agent ポリシー因子スイープ ===");
    println!(
        "E: {} 種 × R: {} 種 × S: {} 種 × 試行 {} = {} 実行",
        entries.len(),
        subsets.len(),
        strategies.len(),
        args.runs,
        n_total,
    );
    println!("出力先: {sweep_dir}");
    println!("-----------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut all_metrics = Vec::new();
    let mut done = 0usize;

    for &entry in &entries {
        for &subset in &subsets {
            for &strategy in &strategies {
                for run_idx in 0..args.runs {
                    let seed = socsim_core::derive_seed(
                        args.seed,
                        &[entry as u64, subset as u64, strategy as u64, run_idx as u64],
                    );
                    let policy = Policy {
                        entry_condition: entry,
                        resource_subset: subset,
                        sort_strategy: strategy,
                        m: args.queues,
                        k: args.k,
                        c: args.c,
                    };
                    let cfg = Config {
                        n_applicants: args.env.n_applicants,
                        pool_ratio: args.env.pool_ratio,
                        policy,
                        max_rounds: args.env.max_rounds,
                        visible_subset_size: args.env.visible_subset_size,
                        seed: Some(seed),
                        llm: LlmSettings {
                            temperature: args.temperature,
                            seed: args.llm_seed,
                            cache_path: if args.mock {
                                None
                            } else {
                                Some(args.cache_path.clone())
                            },
                        },
                        output_dir: sweep_dir.clone(),
                    };

                    let result = run_one(&cfg, run_idx, args.mock)
                        .unwrap_or_else(|e| panic!("実行に失敗: {e}"));
                    all_metrics.extend(result.metrics.clone());
                    let m = &result.final_metrics;
                    summary_rows.push(SweepRow {
                        entry_condition: entry.label().to_string(),
                        resource_subset: subset.label().to_string(),
                        queue_strategy: strategy.label().to_string(),
                        run: run_idx,
                        seed,
                        final_round: result.final_round,
                        final_sw: m.sw,
                        final_avg_rsize: m.avg_rsize,
                        final_avg_wt: m.avg_wt,
                        final_var_rsize: m.var_rsize,
                        final_rop: m.rop,
                        final_co_gini: m.co_gini,
                        final_f_vnv: m.f_vnv,
                        n_allocated: m.n_allocated,
                        cache_hit_rate: result.metadata.cache_hit_rate(),
                    });
                    done += 1;
                }
                println!(
                    "[{}/{}] E={} R={} S={} 完了 ({} 試行)",
                    done,
                    n_total,
                    entry.label(),
                    subset.label(),
                    strategy.label(),
                    args.runs,
                );
            }
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{sweep_dir}/sweep_summary.csv");
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }
    // metrics.csv (long-format, 全 run)．
    save_metrics(&all_metrics, &sweep_dir);
    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            entry_conditions: entries.iter().map(|e| e.label().to_string()).collect(),
            resource_subsets: subsets.iter().map(|r| r.label().to_string()).collect(),
            queue_strategies: strategies.iter().map(|s| s.label().to_string()).collect(),
            n_applicants: args.env.n_applicants,
            pool_ratio: args.env.pool_ratio,
            queues: args.queues,
            k: args.k,
            c: args.c,
            max_rounds: args.env.max_rounds,
            runs: args.runs,
            seed: args.seed,
            llm_temperature: args.temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{sweep_dir}/sweep_config.json");
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{timestamp}_sweep"));

    println!("===========================================================");
    println!("資源サブセット別の平均 SW (論文: r_size 最高 / r_random 最低):");
    for &subset in &subsets {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.resource_subset == subset.label())
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_sw = mean(&rows.iter().map(|r| r.final_sw).collect::<Vec<_>>());
        println!("  R={:<9} → 平均 SW = {avg_sw:.2}", subset.label());
    }
    println!("-----------------------------------------------------------");
    println!("サマリ → {sweep_dir}/sweep_summary.csv");
    println!("設定   → {sweep_dir}/sweep_config.json");
}

// ---------------------------------------------------------------------------
// poa (Phase-3 最小スタブ)
// ---------------------------------------------------------------------------

fn cmd_poa(args: PoaArgs) {
    let objective = parse_objective(&args.objective).unwrap_or_else(|e| panic!("{e}"));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}_poa", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);

    println!("=== Ji et al. (2024) SRAP-Agent POA (Phase-3 最小スタブ) ===");
    println!(
        "objective: {} | iterations: {} | pool_size: {} | mutation_rate: {}",
        objective.label(),
        args.iterations,
        args.pool_size,
        args.mutation_rate,
    );
    println!(
        "注: 適応度 = 1 回の決定論的 mock シミュレーション (予測器 f̃・ライブ LLM 評価は未実装)．"
    );
    println!("出力先: {output_dir}");
    println!("-------------------------------------------------");

    let base_config = Config {
        n_applicants: args.env.n_applicants,
        pool_ratio: args.env.pool_ratio,
        max_rounds: args.env.max_rounds,
        visible_subset_size: args.env.visible_subset_size,
        seed: Some(args.seed),
        ..Config::default()
    };
    let poa_cfg = PoaConfig {
        objective,
        iterations: args.iterations,
        pool_size: args.pool_size,
        mutation_rate: args.mutation_rate,
        tournament_size: args.tournament_size,
        base_config,
        seed: args.seed,
    };

    let result = run_poa(&poa_cfg);

    // poa_history.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{output_dir}/poa_history.csv");
        write_csv(&result.history, &path).expect("poa_history.csv の書き込みに失敗");
    }
    // poa_config.json
    {
        let config_json = PoaConfigJson {
            command: "poa",
            objective: objective.label().to_string(),
            iterations: args.iterations,
            pool_size: args.pool_size,
            mutation_rate: args.mutation_rate,
            tournament_size: args.tournament_size,
            n_applicants: args.env.n_applicants,
            pool_ratio: args.env.pool_ratio,
            max_rounds: args.env.max_rounds,
            seed: args.seed,
            note: "Phase-3 minimal stub: fitness = one deterministic mock simulation run. \
                   The full POA (predictor f-tilde + live-LLM fitness + paper Fig.4/Table \
                   reproduction) is deferred.",
        };
        let path = format!("{output_dir}/poa_config.json");
        write_json(&config_json, &path).expect("poa_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{timestamp}_poa"));

    let best = result.best_policy;
    println!(
        "最良ポリシー π*: E={} S={} R={} m={} k={} c={}",
        best.entry_condition.label(),
        best.sort_strategy.label(),
        best.resource_subset.label(),
        best.m,
        best.k,
        best.c,
    );
    println!(
        "適応度 f(π): 初期 {:.2} → 最終 {:.2} (改善率 {:.1}%)",
        result
            .history
            .first()
            .map(|h| h.best_fitness)
            .unwrap_or(0.0),
        result.best_fitness,
        result.improvement_pct(),
    );
    println!("履歴 → {output_dir}/poa_history.csv");
    println!("設定 → {output_dir}/poa_config.json");
}

// ---------------------------------------------------------------------------
// reproduce (Phase 3 スタブ)
// ---------------------------------------------------------------------------

fn cmd_reproduce() {
    println!("=== Ji et al. (2024) SRAP-Agent 論文 Table 2/3・Fig.4 一括再現 ===");
    println!("reproduce は Phase 3 で実装予定です (現状はスタブ)．");
    println!("Table 2 (入室条件 × 資源サブセット の SW)・Table 3 (最適化ポリシー)・");
    println!("Fig.4 (POA 収束) の一括再現を計画しています．");
    println!("現時点では `run` / `sweep` / `poa` と Python tools の");
    println!("`visualize` / `visualize-sweep` / `reproduce` を個別にご利用ください．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Poa(args) => cmd_poa(args),
        Commands::Reproduce => cmd_reproduce(),
    }
}
