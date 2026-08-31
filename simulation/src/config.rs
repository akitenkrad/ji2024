//! シミュレーション設定 (Config / Policy / LlmSettings)．
//!
//! Ji et al. (2024) SRAP-Agent のコアモデル (LLM 駆動の希少資源配分ポリシー
//! シミュレーション) と感度分析パラメータを保持する [`Config`] と，その JSON
//! シリアライズ表現を定義する．
//!
//! > [!NOTE] 設計上の不確実性 (論文 §7)
//! > 公共住宅シナリオの合成環境 (応募者数・資源数・プロファイル分布) の詳細値と
//! > POA のハイパーパラメータは論文付録に依存し本文に明示がない．本実装は標準的な
//! > 既定値 (応募者 60 人・pool_ratio 0.5 など) を CLI で外部化し，論文付録判明後に
//! > 差し替える方針とする．

use crate::policy::Policy;

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// 定義は `socsim-llm` に集約済み (各 replication で同一だった struct を統合)．
/// `crate::config::LlmSettings` パスは re-export で温存する．
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定 (合成環境 + ポリシー π + LLM)．
#[derive(Debug, Clone)]
pub struct Config {
    /// 応募者数 (合成環境; 論文付録依存のため CLI 外部化)．
    pub n_applicants: usize,
    /// 資源プール規模 / 応募者数の比 (= n_resources / n_applicants)．希少性を制御．
    pub pool_ratio: f64,
    /// 配分ポリシー π = (E_queue, S_queue, R_queue, m, k, c)．
    pub policy: Policy,
    /// 最大ラウンド数 (= 強制終了ラウンド; 枯渇/全員離脱で早期 stop)．
    pub max_rounds: usize,
    /// 各ラウンドで応募者に可視化する資源サブセットのサイズ上限 (R_queue の幅)．
    pub visible_subset_size: usize,

    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_applicants: 60,
            pool_ratio: 0.5,
            policy: Policy::default(),
            max_rounds: 10,
            visible_subset_size: 5,
            seed: Some(42),
            llm: LlmSettings::default(),
        }
    }
}

impl Config {
    /// 資源プール規模 (= round(pool_ratio × n_applicants), 最低 1)．
    pub fn n_resources(&self) -> usize {
        ((self.pool_ratio * self.n_applicants as f64).round() as usize).max(1)
    }
}

/// `run` の試行シードを派生する (試行 index で独立化する)．
pub fn derive_run_seed(base: u64, run_idx: usize) -> u64 {
    socsim_core::derive_seed(base, &[run_idx as u64])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_resources_from_ratio() {
        let cfg = Config {
            n_applicants: 60,
            pool_ratio: 0.5,
            ..Config::default()
        };
        assert_eq!(cfg.n_resources(), 30);
    }

    #[test]
    fn n_resources_min_one() {
        let cfg = Config {
            n_applicants: 1,
            pool_ratio: 0.0,
            ..Config::default()
        };
        assert_eq!(cfg.n_resources(), 1);
    }
}
