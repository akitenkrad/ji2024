//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / config.json / llm_meta.json) と Python 可視化を
//! 検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的に応募
//! 意思決定を駆動し，本番 `run` と同じ writer で結果を書き出す．LLM ライブ呼び出しは
//! 0 回．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```
//!
//! 擬似挙動: 各応募者は «プロンプトに最初に現れる可視 home» を希望する (R_queue が
//! r_size なら広い家から提示されるため SW が高くなる)．

use std::env;
use std::fs;

use chrono::Local;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use srap_simulation::config::Config;
use srap_simulation::llm::wrap_client;
use srap_simulation::simulation::{
    ensure_output_dir, run_with_client, save_llm_meta, save_metrics,
};

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        n_applicants: 30,
        pool_ratio: 0.5,
        max_rounds: 8,
        seed: Some(42),
        output_dir: output_dir.clone(),
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

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client, 0).expect("mock run failed");
    save_metrics(&result.metrics, &cfg.output_dir);
    save_llm_meta(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    println!("mock smoke wrote: {output_dir}");
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
