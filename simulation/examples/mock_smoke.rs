//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 記録の経路 (run ディレクトリ・`metrics.csv`・`run.json` の `llm` ブロック) と
//! Python 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的に応募意思決定を駆動する．LLM ライブ呼び出しは 0 回．
//!
//! 試行は 1 本なので親 run は作らない．`run` の子 (`run-replicate`) とは
//! subcommand 名で区別する — 中身は «scripted client で駆動した 1 本» であって，
//! 本番の LLM 経路とは別物である．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```
//!
//! 擬似挙動: 各応募者は «プロンプトに最初に現れる可視 home» を希望する (R_queue が
//! r_size なら広い家から提示されるため SW が高くなる)．

use std::env;

use runvault::{Run, RunOptions};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};
use srap_simulation::config::Config;
use srap_simulation::llm::wrap_client;
use srap_simulation::record::{
    self, ConditionParameters, TrialParameters, DOMAIN, EXPERIMENT, HASH_EXCLUDE, REPO_ID,
    TRIAL_SEED_POINTERS,
};
use srap_simulation::simulation::run_with_client;

fn main() {
    let results_root = env::args().nth(1).unwrap_or_else(|| "results".to_string());

    const SEED: u64 = 42;
    let cfg = Config {
        n_applicants: 30,
        pool_ratio: 0.5,
        max_rounds: 8,
        seed: Some(SEED),
        ..Config::default()
    };

    // 最初の可視 home を希望する «満足度志向» の擬似応募者．
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
    let client = wrap_client(backend, PromptCache::in_memory());
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "mock-smoke")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&results_root)
            .parameters(&TrialParameters {
                condition: ConditionParameters::from_config(&cfg, true),
                seed: SEED,
            })
            .expect("runvault: parameters の組み立てに失敗")
            .hash_exclude(HASH_EXCLUDE)
            .seed_pointers(TRIAL_SEED_POINTERS)
            .master_seed(SEED)
            .llm(llm)
            .replication(record::replication_policy()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client, 0).expect("mock run failed");
    record::log_trial(&mut rv, &result);
    record::log_llm_usage(&mut rv, &result);
    let dir = rv.finish().expect("runvault: run の完了に失敗");

    println!("mock smoke wrote: {}", dir.display());
    println!(
        "rounds={} final_round={} final_SW={:.2} n_allocated={} gini={:.3} live_llm_calls=0 cache_hits={}",
        result.metrics.len(),
        result.final_round,
        result.final_sw(),
        result.final_metrics.n_allocated,
        result.final_metrics.co_gini,
        result.metadata.cache_hits(),
    );
}
