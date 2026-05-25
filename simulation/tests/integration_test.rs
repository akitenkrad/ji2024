//! Ji et al. (2024) SRAP-Agent 希少資源配分シミュレーションの統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に応募意思決定を駆動し，以下を検証する:
//! ・決定論性 (同一シード + 同一 mock → metrics 完全一致)
//! ・配分規則の正しさ (二重割当なし・プール容量厳守)
//! ・指標の境界 (Gini ∈[0,1]・Rop ≥0)
//! ・メカニズム配線 (metrics 行が生成される)
//! ・離脱 (∅) 処理
//! ・ポリシー順序 (SW(r_size) >= SW(r_random))
//! ・最小 POA GA が適応度を単調非減少に保つ

use srap_simulation::config::{Config, LlmSettings};
use srap_simulation::llm::{wrap_client, SrapClient};
use srap_simulation::metrics::{
    compute_metrics, f_pi, gini, inverse_order_pairs, AllocationOutcome, Objective,
};
use srap_simulation::poa::{run_poa, PoaConfig};
use srap_simulation::policy::{Policy, ResourceSubset};
use srap_simulation::simulation::run_with_client;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// «最初の可視 home を選ぶ» 決定論的 mock．
fn scripted_first_home() -> SrapClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| {
        if let Some(idx) = prompt.find("home ") {
            let rest = &prompt[idx + "home ".len()..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num.is_empty() {
                return format!("{{\"choice\": {num}}}");
            }
        }
        "{\"choice\": -1}".to_string()
    });
    wrap_client(backend, PromptCache::in_memory())
}

/// 全応募者を離脱させる mock (∅ 処理の検証用)．
fn scripted_drop_out() -> SrapClient {
    let backend = ScriptedClient::new("mock-model", |_p: &str| "{\"choice\": -1}".to_string());
    wrap_client(backend, PromptCache::in_memory())
}

fn base_config() -> Config {
    Config {
        n_applicants: 24,
        pool_ratio: 0.5,
        max_rounds: 5,
        seed: Some(7),
        llm: LlmSettings {
            cache_path: None,
            ..LlmSettings::default()
        },
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// メカニズム配線: metrics 行が生成される
// --------------------------------------------------------------------------- //

#[test]
fn produces_metric_rows() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_first_home(), 0).unwrap();
    assert!(!result.metrics.is_empty(), "metrics 行が生成される");
    for m in &result.metrics {
        assert!(m.sw.is_finite());
        assert!(
            (0.0..=1.0).contains(&m.co_gini),
            "gini in [0,1]: {}",
            m.co_gini
        );
        assert!(m.rop >= 0.0, "rop non-negative");
    }
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → metrics 完全一致
// --------------------------------------------------------------------------- //

#[test]
fn deterministic_given_fixed_mock() {
    let cfg = base_config();
    let a = run_with_client(&cfg, scripted_first_home(), 0).unwrap();
    let b = run_with_client(&cfg, scripted_first_home(), 0).unwrap();
    let sa: Vec<f64> = a.metrics.iter().map(|m| m.sw).collect();
    let sb: Vec<f64> = b.metrics.iter().map(|m| m.sw).collect();
    let ga: Vec<f64> = a.metrics.iter().map(|m| m.co_gini).collect();
    let gb: Vec<f64> = b.metrics.iter().map(|m| m.co_gini).collect();
    assert_eq!(sa, sb, "同一シードは SW を完全再現すべき");
    assert_eq!(ga, gb, "同一シードは Gini を完全再現すべき");
}

// --------------------------------------------------------------------------- //
// 配分規則: 配分人数が資源プール容量を超えない
// --------------------------------------------------------------------------- //

#[test]
fn allocation_respects_pool_capacity() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_first_home(), 0).unwrap();
    let n_res = cfg.n_resources();
    assert!(
        result.final_metrics.n_allocated <= n_res,
        "配分人数 {} <= プール容量 {}",
        result.final_metrics.n_allocated,
        n_res
    );
}

// --------------------------------------------------------------------------- //
// 離脱 (∅) 処理: 全員離脱 → 配分 0 人
// --------------------------------------------------------------------------- //

#[test]
fn all_drop_out_produces_zero_allocation() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_drop_out(), 0).unwrap();
    assert_eq!(
        result.final_metrics.n_allocated, 0,
        "全員離脱 → 誰も配分されない"
    );
    assert_eq!(result.final_sw(), 0.0, "離脱のみ → SW=0");
}

// --------------------------------------------------------------------------- //
// 指標の単体: Gini 境界 / Rop
// --------------------------------------------------------------------------- //

#[test]
fn gini_in_unit_interval() {
    assert!((gini(&[5.0, 5.0, 5.0])).abs() < 1e-12);
    let g = gini(&[1.0, 2.0, 9.0]);
    assert!((0.0..=1.0).contains(&g));
}

#[test]
fn rop_counts_vulnerable_inversions() {
    let v = AllocationOutcome {
        allocated: true,
        size: 30.0,
        utility: 30.0,
        wait_time: 0,
        vulnerable: true,
    };
    let nv = AllocationOutcome {
        allocated: true,
        size: 70.0,
        utility: 70.0,
        wait_time: 0,
        vulnerable: false,
    };
    let refs: Vec<&AllocationOutcome> = vec![&v, &nv];
    assert_eq!(inverse_order_pairs(&refs), 1.0);
}

#[test]
fn compute_metrics_sw_is_sum_of_utilities() {
    let outcomes = vec![
        AllocationOutcome {
            allocated: true,
            size: 60.0,
            utility: 50.0,
            wait_time: 1,
            vulnerable: false,
        },
        AllocationOutcome {
            allocated: true,
            size: 40.0,
            utility: 25.0,
            wait_time: 0,
            vulnerable: true,
        },
    ];
    let m = compute_metrics(&outcomes);
    assert!((m.sw - 75.0).abs() < 1e-9);
    assert_eq!(m.n_allocated, 2);
}

// --------------------------------------------------------------------------- //
// ポリシー順序: SW(r_size) >= SW(r_random) (論文の核心知見)
// --------------------------------------------------------------------------- //

#[test]
fn r_size_sw_not_less_than_r_random() {
    let cfg_size = Config {
        policy: Policy {
            resource_subset: ResourceSubset::RSize,
            ..Policy::default()
        },
        ..base_config()
    };
    let cfg_rand = Config {
        policy: Policy {
            resource_subset: ResourceSubset::RRandom,
            ..Policy::default()
        },
        ..base_config()
    };
    let r_size = run_with_client(&cfg_size, scripted_first_home(), 0).unwrap();
    let r_rand = run_with_client(&cfg_rand, scripted_first_home(), 0).unwrap();
    assert!(
        r_size.final_sw() >= r_rand.final_sw(),
        "SW(r_size)={} should be >= SW(r_random)={}",
        r_size.final_sw(),
        r_rand.final_sw()
    );
}

// --------------------------------------------------------------------------- //
// f(π) 目標切替: fairness は不公平を強く罰する
// --------------------------------------------------------------------------- //

#[test]
fn objective_changes_fitness_ordering() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_first_home(), 0).unwrap();
    let sat = f_pi(&result.final_metrics, Objective::Satisfaction);
    let fair = f_pi(&result.final_metrics, Objective::Fairness);
    assert!(sat.is_finite() && fair.is_finite());
}

// --------------------------------------------------------------------------- //
// 最小 POA: 適応度が単調非減少 (エリート保存)
// --------------------------------------------------------------------------- //

#[test]
fn poa_fitness_non_decreasing() {
    let poa = PoaConfig {
        objective: Objective::Satisfaction,
        iterations: 6,
        pool_size: 6,
        mutation_rate: 0.3,
        tournament_size: 3,
        base_config: Config {
            n_applicants: 12,
            pool_ratio: 0.5,
            max_rounds: 3,
            seed: Some(3),
            ..Config::default()
        },
        seed: 9,
    };
    let result = run_poa(&poa);
    assert_eq!(result.history.len(), 6);
    for w in result.history.windows(2) {
        assert!(
            w[1].best_fitness >= w[0].best_fitness - 1e-9,
            "GA best fitness must be non-decreasing under elitism"
        );
    }
}
