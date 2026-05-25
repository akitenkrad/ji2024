//! 評価指標 (満足度・公平性) — 論文 §4.3 Eq.3 / Table 2-3．
//!
//! 配分結果から社会的厚生 (満足度) と公平性の各指標を計算する．すべて
//! **LLM 非依存・決定論的** な純数値計算であり，二層アーキテクチャの下層 (socsim
//! コア) に属する．`metrics.csv` (long-format: run, round + 指標列) の行型
//! [`MetricRow`] も提供する．
//!
//! | 指標 | 定義 | 公平性/満足度 |
//! |------|------|--------------|
//! | `sw` | 社会的厚生 = 配分された応募者の効用総和 | 満足度 |
//! | `avg_rsize` | 一人当たり平均居住面積 | 満足度 |
//! | `avg_wt` | 平均待機時間 | 満足度 |
//! | `var_rsize` | 居住面積の分散 | 公平性 |
//! | `rop` | 逆順ペア数 (vulnerable がより小さい面積を得たペア数) | 公平性 |
//! | `co_gini` | 配分面積の Gini 係数 ∈[0,1] | 公平性 |
//! | `f_vnv` | 脆弱層 V と非脆弱層 NV の平均効用ギャップ | 公平性 |
//! | `f_pi` | ポリシー評価指標 f(π) (重み付き和) | 最適化目標 |

use serde::{Deserialize, Serialize};

/// 1 応募者の配分結果 (指標計算の入力)．
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AllocationOutcome {
    /// 配分されたか (false なら離脱/未配分)．
    pub allocated: bool,
    /// 得た資源の面積 (未配分なら 0)．
    pub size: f64,
    /// 得た効用 (満足度; 未配分なら 0)．
    pub utility: f64,
    /// 待機ラウンド数．
    pub wait_time: usize,
    /// 脆弱層か．
    pub vulnerable: bool,
}

/// 当該ラウンドの満足度・公平性集計 (world に保持する)．
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SrapMetrics {
    /// 社会的厚生 SW = 配分された応募者の効用総和．
    pub sw: f64,
    /// 一人当たり平均居住面積 Avg r_size．
    pub avg_rsize: f64,
    /// 平均待機時間 Avg WT．
    pub avg_wt: f64,
    /// 居住面積の分散 Var r_size (公平性)．
    pub var_rsize: f64,
    /// 逆順ペア数 Rop (公平性)．
    pub rop: f64,
    /// 配分面積の Gini 係数 ∈[0,1] (公平性)．
    pub co_gini: f64,
    /// 脆弱層 V と非脆弱層 NV の平均効用ギャップ F(V,NV) (公平性)．
    pub f_vnv: f64,
    /// 配分された応募者数．
    pub n_allocated: usize,
}

/// `metrics.csv` の 1 行 (long-format: 1 行 = 1 試行 1 ラウンド)．
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricRow {
    /// 試行 index (0 始まり)．
    pub run: usize,
    /// ラウンド (0 始まり)．
    pub round: u64,
    /// 社会的厚生 SW．
    pub sw: f64,
    /// 一人当たり平均居住面積．
    pub avg_rsize: f64,
    /// 平均待機時間．
    pub avg_wt: f64,
    /// 居住面積の分散．
    pub var_rsize: f64,
    /// 逆順ペア数．
    pub rop: f64,
    /// Gini 係数．
    pub co_gini: f64,
    /// 脆弱層 vs 非脆弱層の効用ギャップ．
    pub f_vnv: f64,
    /// 配分された応募者数．
    pub n_allocated: usize,
}

/// ポリシー評価指標 f(π) の重み付き和を計算する (論文 Eq.3)．
///
/// 最適化目標 (`Objective`) で重み w_j を切り替える:
/// - 満足度志向: SW を主とし，公平性のペナルティを軽く引く．
/// - 公平性志向: 公平性指標 (Gini・F(V,NV)・Var) のペナルティを重く引く．
pub fn f_pi(m: &SrapMetrics, objective: Objective) -> f64 {
    match objective {
        // SW を最大化しつつ，極端な不公平を軽く罰する．
        Objective::Satisfaction => m.sw - 0.1 * m.f_vnv - 0.05 * m.var_rsize.sqrt(),
        // SW を保ちつつ，公平性 (Gini・ギャップ) を重く評価する．
        Objective::Fairness => m.sw - 50.0 * m.co_gini - 1.0 * m.f_vnv - 0.5 * m.var_rsize.sqrt(),
    }
}

/// 最適化目標 (f(π) の重み w_j を切り替える)．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// 満足度志向 π_s* (SW 重視)．
    Satisfaction,
    /// 公平性志向 π_f* (Gini / ギャップ重視)．
    Fairness,
}

impl Objective {
    /// CLI / JSON ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            Objective::Satisfaction => "satisfaction",
            Objective::Fairness => "fairness",
        }
    }
}

/// 文字列から [`Objective`] をパースする．
pub fn parse_objective(s: &str) -> Result<Objective, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "satisfaction" | "sat" | "welfare" => Ok(Objective::Satisfaction),
        "fairness" | "fair" => Ok(Objective::Fairness),
        _ => Err(format!(
            "不正な最適化目標: \"{s}\" (satisfaction / fairness)"
        )),
    }
}

/// 配分結果一式から [`SrapMetrics`] を計算する (LLM 非依存・決定論)．
pub fn compute_metrics(outcomes: &[AllocationOutcome]) -> SrapMetrics {
    let allocated: Vec<&AllocationOutcome> = outcomes.iter().filter(|o| o.allocated).collect();
    let n_allocated = allocated.len();

    let sw: f64 = allocated.iter().map(|o| o.utility).sum();

    let sizes: Vec<f64> = allocated.iter().map(|o| o.size).collect();
    let avg_rsize = mean(&sizes);
    let var_rsize = variance(&sizes);

    // 平均待機時間は全応募者を対象 (離脱者も待機したラウンドを含む)．
    let wts: Vec<f64> = outcomes.iter().map(|o| o.wait_time as f64).collect();
    let avg_wt = mean(&wts);

    let co_gini = gini(&sizes);
    let rop = inverse_order_pairs(&allocated);
    let f_vnv = vulnerable_gap(&allocated);

    SrapMetrics {
        sw,
        avg_rsize,
        avg_wt,
        var_rsize,
        rop,
        co_gini,
        f_vnv,
        n_allocated,
    }
}

/// 平均値 (空なら 0)．
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// 標本分散 (空/単一なら 0)．
pub fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64
}

/// Gini 係数 ∈ [0,1]．0 = 完全平等，1 = 完全不平等 (非負値前提)．
///
/// `G = Σ_i Σ_j |x_i - x_j| / (2 n² μ)` の標準定義．空・全 0 は 0 を返す．
pub fn gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut abs_diff_sum = 0.0;
    for &xi in values {
        for &xj in values {
            abs_diff_sum += (xi - xj).abs();
        }
    }
    let mu = total / n as f64;
    let g = abs_diff_sum / (2.0 * (n as f64) * (n as f64) * mu);
    g.clamp(0.0, 1.0)
}

/// 逆順ペア数 Rop: «脆弱層なのに非脆弱層より小さい面積を得た» ペアの数．
///
/// 公平性が満たされていれば脆弱層 (優先対象) は少なくとも非脆弱層と同等の面積を
/// 得るはず．脆弱層 V が非脆弱層 NV より «小さい» 面積を得たペアを 1 つの逆順
/// (inversion) として数える (大きいほど公平性が悪い)．
pub fn inverse_order_pairs(allocated: &[&AllocationOutcome]) -> f64 {
    let mut count = 0usize;
    for v in allocated.iter().filter(|o| o.vulnerable) {
        for nv in allocated.iter().filter(|o| !o.vulnerable) {
            if v.size < nv.size {
                count += 1;
            }
        }
    }
    count as f64
}

/// 脆弱層 V と非脆弱層 NV の平均効用ギャップ F(V,NV) = mean(U_NV) - mean(U_V)．
///
/// 正の値が大きいほど «非脆弱層が得をしている» = 不公平 (脆弱層優先 VFA/VFR が
/// 縮小すべき対象)．どちらかが空なら 0．
pub fn vulnerable_gap(allocated: &[&AllocationOutcome]) -> f64 {
    let v: Vec<f64> = allocated
        .iter()
        .filter(|o| o.vulnerable)
        .map(|o| o.utility)
        .collect();
    let nv: Vec<f64> = allocated
        .iter()
        .filter(|o| !o.vulnerable)
        .map(|o| o.utility)
        .collect();
    if v.is_empty() || nv.is_empty() {
        return 0.0;
    }
    mean(&nv) - mean(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oc(allocated: bool, size: f64, utility: f64, wait: usize, vuln: bool) -> AllocationOutcome {
        AllocationOutcome {
            allocated,
            size,
            utility,
            wait_time: wait,
            vulnerable: vuln,
        }
    }

    #[test]
    fn gini_bounds() {
        assert!((gini(&[5.0, 5.0, 5.0])).abs() < 1e-12, "equal → 0");
        let g = gini(&[0.0, 0.0, 10.0]);
        assert!((0.0..=1.0).contains(&g), "gini in [0,1]: {g}");
        assert!(g > 0.5, "very unequal → high gini: {g}");
        assert_eq!(gini(&[]), 0.0);
        assert_eq!(gini(&[0.0, 0.0]), 0.0, "all zero → 0");
    }

    #[test]
    fn inverse_order_pairs_counts_inversions() {
        // vulnerable size 30 < non-vulnerable size 60 → 1 inversion．
        let a = oc(true, 30.0, 30.0, 0, true);
        let b = oc(true, 60.0, 60.0, 0, false);
        let refs: Vec<&AllocationOutcome> = vec![&a, &b];
        assert_eq!(inverse_order_pairs(&refs), 1.0);
        // vulnerable gets MORE → 0 inversions.
        let c = oc(true, 80.0, 80.0, 0, true);
        let refs2: Vec<&AllocationOutcome> = vec![&c, &b];
        assert_eq!(inverse_order_pairs(&refs2), 0.0);
    }

    #[test]
    fn vulnerable_gap_positive_when_nv_better() {
        let v = oc(true, 30.0, 20.0, 0, true);
        let nv = oc(true, 60.0, 80.0, 0, false);
        let refs: Vec<&AllocationOutcome> = vec![&v, &nv];
        assert!((vulnerable_gap(&refs) - 60.0).abs() < 1e-9);
    }

    #[test]
    fn compute_metrics_full() {
        let outcomes = vec![
            oc(true, 60.0, 50.0, 1, false),
            oc(true, 40.0, 30.0, 2, true),
            oc(false, 0.0, 0.0, 3, true), // 離脱/未配分
        ];
        let m = compute_metrics(&outcomes);
        assert_eq!(m.n_allocated, 2);
        assert!((m.sw - 80.0).abs() < 1e-9);
        assert!((m.avg_rsize - 50.0).abs() < 1e-9, "avg of 60,40 = 50");
        // avg_wt over ALL 3 = (1+2+3)/3 = 2.
        assert!((m.avg_wt - 2.0).abs() < 1e-9);
        assert!(m.co_gini >= 0.0 && m.co_gini <= 1.0);
    }

    #[test]
    fn f_pi_objectives_differ() {
        let m = SrapMetrics {
            sw: 100.0,
            co_gini: 0.5,
            f_vnv: 10.0,
            var_rsize: 100.0,
            ..SrapMetrics::default()
        };
        let sat = f_pi(&m, Objective::Satisfaction);
        let fair = f_pi(&m, Objective::Fairness);
        // fairness penalizes gini heavily → lower score.
        assert!(fair < sat, "fairness objective penalizes inequality more");
    }

    #[test]
    fn parse_objective_roundtrip() {
        assert_eq!(parse_objective("fairness").unwrap(), Objective::Fairness);
        assert_eq!(
            parse_objective("satisfaction").unwrap(),
            Objective::Satisfaction
        );
        assert!(parse_objective("bogus").is_err());
    }
}
