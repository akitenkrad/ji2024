//! 配分ポリシー π = (E_queue, S_queue, R_queue, m, k, c)．
//!
//! SRAP-Agent (Ji et al. 2024) の核心は «入室条件・並び替え方法・資源の分類» の
//! 3 主要因子からなる配分ポリシーである (論文 §4 / Eq. 1)．本モジュールは
//! ポリシーを構成する 6 個のパラメータ
//!
//! - `entry_condition` (E_queue): 待機キューへの入室条件
//! - `sort_strategy` (S_queue): キュー内の並び替え戦略
//! - `resource_subset` (R_queue): 応募者に可視化する資源サブセットの選び方
//! - `m`: キュー数 (待機キューを m 本に分割する)
//! - `k`: k-deferrals (1 応募者が 1 ラウンドで保留できる最大オファー回数)
//! - `c`: 選択キュー容量係数 (選択キューが受け入れる候補者数 = c × 当該ラウンド資源数)
//!
//! を `enum` + 数値で表現する．POA (Phase 3) はこの `Policy` ベクトルを探索する．

use serde::{Deserialize, Serialize};

/// 入室条件 E_queue: どの応募者を待機キューへ入れるか (論文 Table 2 の行)．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryCondition {
    /// `p_budget`: 予算 (収入・支払い能力) の閾値で入室判定する．
    PBudget,
    /// `p_family`: 家族構成 (世帯規模) の閾値で入室判定する．
    PFamily,
    /// `p_select`: 自己選択 (応募者自身が応募するか決める)．論文の最高 SW 条件．
    PSelect,
}

impl EntryCondition {
    /// CLI / JSON ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            EntryCondition::PBudget => "p_budget",
            EntryCondition::PFamily => "p_family",
            EntryCondition::PSelect => "p_select",
        }
    }

    /// すべての候補 (sweep / POA 用)．
    pub fn all() -> [EntryCondition; 3] {
        [
            EntryCondition::PBudget,
            EntryCondition::PFamily,
            EntryCondition::PSelect,
        ]
    }
}

/// 並び替え戦略 S_queue: キュー内で誰を先に処理するか (論文の脆弱層優先分析)．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortStrategy {
    /// `fifo`: 到着順 (先着順)．
    Fifo,
    /// `vfa`: vulnerable-first by attribute — 脆弱層を «属性» で固定的に優先する．
    Vfa,
    /// `vfr`: vulnerable-first by ranking — 脆弱層を «ランキング» でラウンドごとに優先する．
    Vfr,
}

impl SortStrategy {
    /// CLI / JSON ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            SortStrategy::Fifo => "fifo",
            SortStrategy::Vfa => "vfa",
            SortStrategy::Vfr => "vfr",
        }
    }

    /// すべての候補．
    pub fn all() -> [SortStrategy; 3] {
        [SortStrategy::Fifo, SortStrategy::Vfa, SortStrategy::Vfr]
    }
}

/// 資源サブセット R_queue: 応募者に可視化する資源の選び方 (論文 Table 2 の列)．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceSubset {
    /// `r_size`: 面積でソートしたサブセットを提示する．論文の最高 SW 条件．
    RSize,
    /// `r_rent`: 家賃でソートしたサブセットを提示する (r_size と類似性能)．
    RRent,
    /// `r_random`: ランダムなサブセットを提示する．論文の最低 SW 条件．
    RRandom,
}

impl ResourceSubset {
    /// CLI / JSON ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            ResourceSubset::RSize => "r_size",
            ResourceSubset::RRent => "r_rent",
            ResourceSubset::RRandom => "r_random",
        }
    }

    /// すべての候補．
    pub fn all() -> [ResourceSubset; 3] {
        [
            ResourceSubset::RSize,
            ResourceSubset::RRent,
            ResourceSubset::RRandom,
        ]
    }
}

/// 配分ポリシー π = (E_queue, S_queue, R_queue, m, k, c)．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    /// 入室条件 E_queue．
    pub entry_condition: EntryCondition,
    /// 並び替え戦略 S_queue．
    pub sort_strategy: SortStrategy,
    /// 資源サブセット R_queue．
    pub resource_subset: ResourceSubset,
    /// キュー数 m (待機キューを m 本に分割する; ≥1)．
    pub m: usize,
    /// k-deferrals 試行回数 k (1 応募者が保留できる最大オファー回数; ≥1)．
    pub k: usize,
    /// 選択キュー容量係数 c (選択キュー容量 = c × 当該ラウンド資源数; ≥1)．
    pub c: usize,
}

impl Default for Policy {
    fn default() -> Self {
        // 論文の最高 SW 条件 (p_select + r_size) を既定にする．
        Policy {
            entry_condition: EntryCondition::PSelect,
            sort_strategy: SortStrategy::Fifo,
            resource_subset: ResourceSubset::RSize,
            m: 3,
            k: 3,
            c: 2,
        }
    }
}

impl Policy {
    /// クランプ済みの正規化ポリシー (m,k,c はすべて ≥1)．
    pub fn normalized(self) -> Policy {
        Policy {
            m: self.m.max(1),
            k: self.k.max(1),
            c: self.c.max(1),
            ..self
        }
    }
}

/// 文字列から [`EntryCondition`] をパースする．
pub fn parse_entry_condition(s: &str) -> Result<EntryCondition, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "p_budget" | "budget" => Ok(EntryCondition::PBudget),
        "p_family" | "family" => Ok(EntryCondition::PFamily),
        "p_select" | "select" => Ok(EntryCondition::PSelect),
        _ => Err(format!(
            "不正な入室条件: \"{s}\" (p_budget / p_family / p_select)"
        )),
    }
}

/// 文字列から [`SortStrategy`] をパースする．
pub fn parse_sort_strategy(s: &str) -> Result<SortStrategy, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fifo" => Ok(SortStrategy::Fifo),
        "vfa" => Ok(SortStrategy::Vfa),
        "vfr" => Ok(SortStrategy::Vfr),
        _ => Err(format!("不正な並び替え戦略: \"{s}\" (fifo / vfa / vfr)")),
    }
}

/// 文字列から [`ResourceSubset`] をパースする．
pub fn parse_resource_subset(s: &str) -> Result<ResourceSubset, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "r_size" | "size" => Ok(ResourceSubset::RSize),
        "r_rent" | "rent" => Ok(ResourceSubset::RRent),
        "r_random" | "random" => Ok(ResourceSubset::RRandom),
        _ => Err(format!(
            "不正な資源サブセット: \"{s}\" (r_size / r_rent / r_random)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        assert_eq!(
            parse_entry_condition("p_select").unwrap(),
            EntryCondition::PSelect
        );
        assert_eq!(parse_sort_strategy("VFA").unwrap(), SortStrategy::Vfa);
        assert_eq!(
            parse_resource_subset("r_random").unwrap(),
            ResourceSubset::RRandom
        );
        assert!(parse_entry_condition("bogus").is_err());
        assert_eq!(EntryCondition::PBudget.label(), "p_budget");
    }

    #[test]
    fn normalized_clamps_to_one() {
        let p = Policy {
            m: 0,
            k: 0,
            c: 0,
            ..Policy::default()
        }
        .normalized();
        assert_eq!((p.m, p.k, p.c), (1, 1, 1));
    }

    #[test]
    fn default_is_highest_sw_condition() {
        let p = Policy::default();
        assert_eq!(p.entry_condition, EntryCondition::PSelect);
        assert_eq!(p.resource_subset, ResourceSubset::RSize);
    }
}
