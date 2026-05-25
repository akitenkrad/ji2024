//! socsim フレームワーク上の SRAP-Agent 世界状態 (`SrapWorld`)．
//!
//! エージェント = 移動する空間主体でも固定サイトでもなく，中央の希少資源プールへ
//! 応募する応募者である (公共住宅の応募者)．したがって `socsim-grid` /
//! `socsim-net` は不使用で，応募者プロファイルを `BTreeMap<AgentId, Applicant>`，
//! 希少資源プールを `Vec<Resource>`，現ラウンドのポリシー π を `Policy` に保持する．
//! 「相互作用」は固定位相ではなく，中央の決定論的配分規則を介して間接的に生じる．
//!
//! `agent_ids()` は `BTreeMap` のキー (= `AgentId` 昇順, ソート済み) を返し決定論を
//! 担保する (socsim コア層)．`#[derive(Clone, Serialize, Deserialize)]` で
//! スナップショットと感度分析の比較実験に対応する．

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};

use crate::metrics::SrapMetrics;
use crate::policy::Policy;

/// 資源 ID (プール内 index の型付きラッパ)．
pub type ResourceId = usize;

/// 応募者の選好 (面積・家賃に対する重み)．効用関数のパラメータ．
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Preferences {
    /// 面積選好の重み (大きいほど広さを好む)．
    pub size_weight: f64,
    /// 家賃選好の重み (大きいほど安さを好む; 効用では負に寄与)．
    pub rent_weight: f64,
}

impl Default for Preferences {
    fn default() -> Self {
        Preferences {
            size_weight: 1.0,
            rent_weight: 1.0,
        }
    }
}

/// 応募者の短期/長期メモリバンク (論文の記憶反映; Generative Agents 流)．
///
/// 本実装では «過去ラウンドの試行サマリ» を文字列で保持し，プロンプトへ再注入する
/// (直近 `summaries` の末尾を使う)．
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    /// ラウンドごとの行動サマリ (古い順)．
    pub summaries: Vec<String>,
    /// 当該応募者が経験した待機ラウンド数 (待機時間 WT の集計に使う)．
    pub wait_time: usize,
}

impl Memory {
    /// 直近 `n` 件のサマリを 1 文へ結合する (プロンプト用)．
    pub fn recent(&self, n: usize) -> String {
        let start = self.summaries.len().saturating_sub(n);
        self.summaries[start..].join(" ")
    }
}

/// 応募者プロファイル (需要・収入・適格性・選好・記憶)．
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Applicant {
    /// 月収 (予算 / 支払い能力; p_budget 入室条件・適格性判定に使う)．
    pub income: f64,
    /// 希望家賃上限 (応募者が支払い可能な家賃)．
    pub rent: f64,
    /// 世帯規模 (p_family 入室条件・脆弱層判定に使う)．
    pub family: usize,
    /// 選好 (面積・家賃の重み)．
    pub preferences: Preferences,
    /// 短期/長期メモリバンク．
    pub memory: Memory,
    /// active フラグ (R_j*=∅ で離脱したら false)．
    pub active: bool,
    /// 脆弱層フラグ (vulnerable; 低所得かつ大世帯など; VFA/VFR の優先対象)．
    pub vulnerable: bool,
}

impl Applicant {
    /// 資源 `r` に対する応募者の効用 (満足度)．
    ///
    /// 面積を選好重みで評価し，家賃が予算を超えるほど効用を下げる単純な線形効用．
    /// 配分結果の評価 (社会的厚生 SW) に使う決定論的関数．
    pub fn utility(&self, r: &Resource) -> f64 {
        let size_term = self.preferences.size_weight * r.size;
        // 家賃が支払い能力を超えた分だけ負の効用を与える．
        let rent_penalty = self.preferences.rent_weight * (r.rent - self.rent).max(0.0);
        size_term - rent_penalty
    }
}

/// 希少資源 (公共住宅; 面積・家賃)．
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Resource {
    /// 資源 ID (プール内 index)．
    pub id: ResourceId,
    /// 居住面積 (r_size; 効用の主項)．
    pub size: f64,
    /// 家賃 (r_rent; 面積と正相関; 効用ペナルティ)．
    pub rent: f64,
    /// 当該ラウンドで割当済みか (allocation_rule が更新)．
    pub allocated: bool,
}

/// 1 本のキュー (待機キュー)．`policy.m` 本に応募者を分割して保持する．
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Queue {
    /// このキューに並ぶ応募者 (`policy_setup` が入室条件 E_queue で構成する)．
    pub members: Vec<AgentId>,
}

/// SRAP-Agent 希少資源配分シミュレーションの世界状態．
#[derive(Clone, Serialize, Deserialize)]
pub struct SrapWorld {
    /// シミュレーションクロック (1 tick = 1 配分ラウンド)．
    pub clock: SimClock,
    /// 応募者プロファイル (`AgentId` 昇順; `agent_ids()` がそのまま決定論順を返す)．
    pub applicants: BTreeMap<AgentId, Applicant>,
    /// 希少資源プール (公共住宅; 面積・家賃)．
    pub pool: Vec<Resource>,
    /// 現ラウンドのポリシー π = (E_queue, S_queue, R_queue, m, k, c)．
    pub policy: Policy,
    /// m 個の待機キュー (`policy_setup` が入室条件で構成する)．
    pub queues: Vec<Queue>,
    /// 各応募者への配分結果 R_j* (None = 未配分/離脱)．
    pub allocations: BTreeMap<AgentId, Option<ResourceId>>,
    /// 当該ラウンドの満足度・公平性集計 (`evaluate_welfare` が更新)．
    pub metrics: SrapMetrics,
}

impl SrapWorld {
    /// 応募者数．
    pub fn n_applicants(&self) -> usize {
        self.applicants.len()
    }

    /// 資源プール規模．
    pub fn n_resources(&self) -> usize {
        self.pool.len()
    }

    /// 現在のラウンド (0 始まり)．
    ///
    /// socsim エンジンはステップ先頭で `tick()` するため，クロックは `1..=t_max`
    /// を走る．本モデルはラウンドを 0 始まりで扱うので `t() - 1` を返す．
    pub fn current_round(&self) -> u64 {
        self.clock.t().saturating_sub(1)
    }

    /// active な (未配分かつ未離脱の) 応募者 ID のソート済みリスト．
    pub fn active_applicants(&self) -> Vec<AgentId> {
        self.applicants
            .iter()
            .filter(|(_, a)| a.active)
            .map(|(id, _)| *id)
            .collect()
    }

    /// 未割当の資源 ID リスト (プール内順)．
    pub fn available_resources(&self) -> Vec<ResourceId> {
        self.pool
            .iter()
            .filter(|r| !r.allocated)
            .map(|r| r.id)
            .collect()
    }

    /// 全資源が割当済みか (枯渇判定)．
    pub fn pool_exhausted(&self) -> bool {
        self.pool.iter().all(|r| r.allocated)
    }

    /// active な応募者が 1 人もいないか (全員配分済み or 離脱)．
    pub fn all_settled(&self) -> bool {
        self.applicants.values().all(|a| !a.active)
    }
}

impl WorldState for SrapWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap のキーは昇順 → 決定論．
        self.applicants.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applicant(income: f64, rent: f64, family: usize, vulnerable: bool) -> Applicant {
        Applicant {
            income,
            rent,
            family,
            preferences: Preferences::default(),
            memory: Memory::default(),
            active: true,
            vulnerable,
        }
    }

    fn world() -> SrapWorld {
        let mut applicants = BTreeMap::new();
        applicants.insert(AgentId(2), applicant(3000.0, 800.0, 4, true));
        applicants.insert(AgentId(0), applicant(8000.0, 1500.0, 2, false));
        applicants.insert(AgentId(1), applicant(5000.0, 1000.0, 3, false));
        let pool = vec![
            Resource {
                id: 0,
                size: 60.0,
                rent: 1200.0,
                allocated: false,
            },
            Resource {
                id: 1,
                size: 40.0,
                rent: 800.0,
                allocated: true,
            },
        ];
        SrapWorld {
            clock: SimClock::new(10),
            applicants,
            pool,
            policy: Policy::default(),
            queues: Vec::new(),
            allocations: BTreeMap::new(),
            metrics: SrapMetrics::default(),
        }
    }

    #[test]
    fn agent_ids_are_sorted() {
        let w = world();
        assert_eq!(w.agent_ids(), vec![AgentId(0), AgentId(1), AgentId(2)]);
    }

    #[test]
    fn available_resources_skips_allocated() {
        let w = world();
        assert_eq!(w.available_resources(), vec![0]);
        assert!(!w.pool_exhausted());
    }

    #[test]
    fn active_applicants_sorted_and_filtered() {
        let mut w = world();
        w.applicants.get_mut(&AgentId(1)).unwrap().active = false;
        assert_eq!(w.active_applicants(), vec![AgentId(0), AgentId(2)]);
        assert!(!w.all_settled());
    }

    #[test]
    fn utility_rewards_size_and_penalizes_overrent() {
        let a = applicant(5000.0, 1000.0, 3, false);
        let cheap = Resource {
            id: 0,
            size: 50.0,
            rent: 900.0,
            allocated: false,
        };
        let pricey = Resource {
            id: 1,
            size: 50.0,
            rent: 1400.0,
            allocated: false,
        };
        // 同面積なら家賃が予算内の方が効用が高い．
        assert!(a.utility(&cheap) > a.utility(&pricey));
    }
}
