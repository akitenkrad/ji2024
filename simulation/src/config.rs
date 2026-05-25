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

use serde::Serialize;

use crate::policy::Policy;

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (temperature / seed / cache)．
///
/// プロバイダ優先順位は «Ollama 第一 → OpenAI フォールバック» 固定．モデル・
/// ホスト・API キーは環境変数で渡す (`OLLAMA_HOST` / `OLLAMA_MODEL` /
/// `OPENAI_API_KEY` / `OPENAI_MODEL`)．`temperature`/`seed` で擬似決定論化する．
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// 生成温度 (既定 0.0; 再現性のため)．
    pub temperature: f32,
    /// 生成シード (バックエンドへ渡す; Ollama は honour，OpenAI は best-effort)．
    pub seed: u64,
    /// プロンプト→応答キャッシュの保存先 (None なら in-memory)．
    pub cache_path: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None,
        }
    }
}

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
    /// 結果出力ディレクトリ．
    pub output_dir: String,
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
            output_dir: "results".to_string(),
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

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_applicants: usize,
    pub pool_ratio: f64,
    pub n_resources: usize,
    pub entry_condition: String,
    pub sort_strategy: String,
    pub resource_subset: String,
    pub m: usize,
    pub k: usize,
    pub c: usize,
    pub max_rounds: usize,
    pub visible_subset_size: usize,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_applicants: self.n_applicants,
            pool_ratio: self.pool_ratio,
            n_resources: self.n_resources(),
            entry_condition: self.policy.entry_condition.label().to_string(),
            sort_strategy: self.policy.sort_strategy.label().to_string(),
            resource_subset: self.policy.resource_subset.label().to_string(),
            m: self.policy.m,
            k: self.policy.k,
            c: self.policy.c,
            max_rounds: self.max_rounds,
            visible_subset_size: self.visible_subset_size,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
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
