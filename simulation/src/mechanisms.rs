//! socsim フレームワーク上の SRAP-Agent 配分メカニズム (5 Mechanism × 6 phase)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! キュー構成・配分規則・指標計算・記憶更新を担い，上層 (非決定的 LLM レイヤ) は
//! [`SrapClient`] (キャッシュ付き Ollama→OpenAI フォールバック) 越しの応募者意思
//! 決定 (`ApplyDecision`) のみを担う．
//!
//! 論文の 1 配分ラウンド (= socsim の 1 tick) を 5 Mechanism へ割り当てる:
//!
//! | Mechanism | Phase | 役割 (論文の規則) |
//! |-----------|-------|------------------|
//! | [`PolicySetup`]      | Environment | 資源プールの補充とポリシー π の確認．入室条件 E_queue でキュー q_1..q_m を構成 |
//! | [`ApplyDecision`]    | Decision    | **★LLM 所在**．各 active 応募者が可視資源 V(p_j) から希望資源を選択 R_j*=D(p_j,V(p_j))．∅ は離脱 |
//! | [`AllocationRule`]   | Interaction | 並び替え戦略 S_queue (FIFO/VFA/VFR) で順序づけ，k-deferrals で残余資源を割り当てる決定論的ポリシー関数 |
//! | [`EvaluateWelfare`]  | Reward      | 社会的厚生 SW・満足度・公平性 (Var/Rop/Gini/F(V,NV)) を計算・記録 |
//! | [`UpdateMemory`]     | PostStep    | 応募者の記憶 m_j を更新．資源枯渇・全員離脱で request_stop |
//!
//! 同期更新セマンティクス: ラウンド開始時に active 応募者をスナップショットし，
//! `apply_decision` で全員の希望をまず収集してから `allocation_rule` で一括配分する．

use std::cell::RefCell;
use std::rc::Rc;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::config::LlmSettings;
use crate::llm::{llm_config, SrapClient};
use crate::metrics::{compute_metrics, AllocationOutcome, MetricRow};
use crate::policy::{EntryCondition, ResourceSubset, SortStrategy};
use crate::prompts::{applicant_prompt, parse_choice, ApplicantBriefing};
use crate::world::{ResourceId, SrapWorld};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<SrapClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有メトリクス行バッファ (metrics.csv; long-format)．
pub type SharedMetrics = Rc<RefCell<Vec<MetricRow>>>;

/// scratch blackboard に当該ラウンドの応募者希望を渡すキー．
const SCRATCH_DESIRES: &str = "desires";

/// 当該ラウンドの応募者希望 (applicant_id → 希望資源 ID | None=離脱)．
type Desires = Vec<(AgentId, Option<ResourceId>)>;

// =========================================================================== //
// 1. PolicySetup (Environment)
// =========================================================================== //

/// 資源プールの補充とラウンドのポリシー π 確認，入室条件 E_queue による
/// キュー q_1..q_m の構成 (`Environment`; LLM 非依存)．
///
/// 各ラウンド冒頭で:
/// - 当該ラウンドの割当フラグをリセット (前ラウンドの割当は確定済みで pool から
///   既に除去されている; ここでは «まだ pool に残る» 未割当資源のみ対象)．
/// - active 応募者を入室条件 E_queue で待機キューへ振り分ける (m 本に分割)．
pub struct PolicySetup;

impl PolicySetup {
    /// 応募者が入室条件 E_queue を満たすか判定する．
    ///
    /// - `PBudget`: 収入が中央値以上 (支払い能力のある層を入室させる)．
    /// - `PFamily`: 世帯規模が中央値以上 (大世帯を入室させる)．
    /// - `PSelect`: 自己選択 — 全員入室を許し，離脱は Decision フェーズに委ねる
    ///   (論文の «自律性付与» = 最高 SW 条件)．
    fn passes_entry(world: &SrapWorld, id: AgentId, income_med: f64, family_med: f64) -> bool {
        let a = &world.applicants[&id];
        match world.policy.entry_condition {
            EntryCondition::PBudget => a.income >= income_med,
            EntryCondition::PFamily => a.family as f64 >= family_med,
            EntryCondition::PSelect => true,
        }
    }
}

impl Mechanism<SrapWorld> for PolicySetup {
    fn name(&self) -> &str {
        "policy_setup"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SrapWorld>) -> Result<()> {
        let world = &mut *ctx.world;

        // 入室条件の閾値 (中央値) を計算する．
        let mut incomes: Vec<f64> = world.applicants.values().map(|a| a.income).collect();
        let mut families: Vec<f64> = world.applicants.values().map(|a| a.family as f64).collect();
        incomes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        families.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let income_med = median_sorted(&incomes);
        let family_med = median_sorted(&families);

        // active 応募者を入室判定し，m 本のキューへラウンドロビンで分割する．
        let m = world.policy.m.max(1);
        let mut queues: Vec<crate::world::Queue> = vec![crate::world::Queue::default(); m];
        let active = world.active_applicants();
        let mut bucket = 0usize;
        for id in active {
            if Self::passes_entry(world, id, income_med, family_med) {
                queues[bucket % m].members.push(id);
                bucket += 1;
            }
        }
        world.queues = queues;
        Ok(())
    }
}

/// ソート済みスライスの中央値 (空なら 0)．
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

// =========================================================================== //
// 2. ApplyDecision (Decision, LLM)
// =========================================================================== //

/// 各 active 応募者が可視資源プール V(p_j) から希望資源を選択する
/// (`Decision`; ★LLM 所在)．
///
/// 可視資源サブセット V(p_j) はポリシー R_queue で決まる (面積/家賃/ランダムで
/// ソートした未割当資源の上位 `visible_subset_size` 件)．LLM (またはテスト時の
/// scripted mock) に希望資源 ID か離脱 (∅) を答えさせ，希望を scratch へ集める
/// (同期更新: 配分は次の `AllocationRule` で一括実行)．
pub struct ApplyDecision {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    visible_subset_size: usize,
}

impl ApplyDecision {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        visible_subset_size: usize,
    ) -> Self {
        ApplyDecision {
            client,
            metadata,
            settings,
            visible_subset_size,
        }
    }
}

impl Mechanism<SrapWorld> for ApplyDecision {
    fn name(&self) -> &str {
        "apply_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SrapWorld>) -> Result<()> {
        let round = ctx.world.current_round();
        let active = ctx.world.active_applicants();

        // R_queue でソートした未割当資源 ID 列 (全 active 応募者で共通の «見える順序»)．
        let visible_ids = visible_resource_ids(ctx.world);

        let mut desires: Desires = Vec::with_capacity(active.len());
        for id in active {
            // 可視資源サブセット V(p_j): 上位 visible_subset_size 件 (参照を集める)．
            let take = self.visible_subset_size.min(visible_ids.len());
            let visible_refs: Vec<&crate::world::Resource> = visible_ids
                .iter()
                .take(take)
                .map(|rid| &ctx.world.pool[*rid])
                .collect();
            let visible_ids_subset: Vec<usize> = visible_refs.iter().map(|r| r.id).collect();

            let applicant = &ctx.world.applicants[&id];
            let brief = ApplicantBriefing {
                applicant_id: id.0,
                round,
                applicant,
                visible: &visible_refs,
            };
            let prompt = applicant_prompt(&brief);

            let choice = if visible_ids_subset.is_empty() {
                None // 可視資源がない → 離脱．LLM を呼ばない．
            } else {
                let text = {
                    let mut client = self.client.borrow_mut();
                    let resp = client
                        .complete(&prompt, &llm_config(&self.settings))
                        .map_err(|e| {
                            SocsimError::Mechanism(format!("applicant LLM call failed: {e}"))
                        })?;
                    self.metadata.borrow_mut().record(resp.metadata.clone());
                    resp.text
                };
                parse_choice(&text, &visible_ids_subset)
            };
            desires.push((id, choice));
        }

        ctx.scratch.insert(SCRATCH_DESIRES, desires);
        Ok(())
    }
}

/// R_queue (資源サブセット戦略) で未割当資源を «見える順序» に並べた ID 列．
///
/// - `RSize`: 面積降順 (広い順)．
/// - `RRent`: 家賃昇順 (安い順)．
/// - `RRandom`: ランダム順序 (engine RNG を使わず world の clock 由来で擬似ランダム
///   …ではなく，後段で RNG を使えないため id をハッシュ的に並べ替える決定論順序)．
///   ※ 厳密なシャッフルは `AllocationRule` 側で SimRng を使う; ここでは応募者へ
///   提示する «順序» のみを決める．`RRandom` は «面積と無相関» な順序を作る．
fn visible_resource_ids(world: &SrapWorld) -> Vec<ResourceId> {
    let mut ids = world.available_resources();
    match world.policy.resource_subset {
        ResourceSubset::RSize => {
            ids.sort_by(|a, b| {
                world.pool[*b]
                    .size
                    .partial_cmp(&world.pool[*a].size)
                    .unwrap()
                    .then(a.cmp(b))
            });
        }
        ResourceSubset::RRent => {
            ids.sort_by(|a, b| {
                world.pool[*a]
                    .rent
                    .partial_cmp(&world.pool[*b].rent)
                    .unwrap()
                    .then(a.cmp(b))
            });
        }
        ResourceSubset::RRandom => {
            // 面積/家賃と無相関な決定論的順序 (id を乗算ハッシュで撹拌)．
            ids.sort_by_key(|&id| {
                (id as u64)
                    .wrapping_mul(2654435761)
                    .wrapping_add(world.current_round())
            });
        }
    }
    ids
}

// =========================================================================== //
// 3. AllocationRule (Interaction)
// =========================================================================== //

/// 決定論的配分規則: 並び替え戦略 S_queue で応募者を順序づけ，k-deferrals で
/// 残余資源を割り当てる (`Interaction`; LLM 非依存)．
///
/// アルゴリズム:
/// 1. scratch から «各応募者の希望資源» を取り出す．
/// 2. 並び替え戦略 S_queue で応募者を順序づける (キュー横断):
///    - `FIFO`: 到着順 (AgentId 昇順; シャッフルは PolicySetup/engine 側)．
///    - `VFA`: 脆弱層を属性で先頭固定 → 残りを AgentId 昇順．
///    - `VFR`: 脆弱層を «家族規模/収入比» ランキングで先頭 → 残りをランキング順．
/// 3. 各応募者を順に処理: 希望資源が空いていれば割当 (deferred-acceptance 風)．
///    希望が埋まっていれば k-deferrals: 次善の可視資源 (面積降順) を最大 k 回試す．
/// 4. 二重割当なし・プール容量厳守．離脱 (希望 None) は未配分とする．
pub struct AllocationRule;

impl Mechanism<SrapWorld> for AllocationRule {
    fn name(&self) -> &str {
        "allocation_rule"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SrapWorld>) -> Result<()> {
        let desires: Desires = ctx
            .scratch
            .get::<Desires>(SCRATCH_DESIRES)
            .cloned()
            .unwrap_or_default();

        // 並び替え戦略 S_queue で応募者順序を決める．
        let mut order: Vec<(AgentId, Option<ResourceId>)> = desires;
        sort_applicants(ctx.world, &mut order);

        // k-deferrals の «次善候補» に使う面積降順の資源順序 (毎回再計算は重いので
        // 1 度だけ; ただし割当が進むと availability が変わるため都度フィルタする)．
        let k = ctx.world.policy.k.max(1);

        for (id, desired) in order {
            let Some(want) = desired else {
                // 離脱: 未配分のまま (active=false は UpdateMemory で設定)．
                ctx.world.allocations.insert(id, None);
                continue;
            };

            let mut assigned: Option<ResourceId> = None;
            // 第一希望 → k-deferrals で次善を最大 k 回試す．
            let mut candidates: Vec<ResourceId> = Vec::with_capacity(k);
            candidates.push(want);
            // 次善候補: 面積降順の未割当資源を k-1 件まで (want と重複は除く)．
            let mut backups = ctx.world.available_resources();
            backups.sort_by(|a, b| {
                ctx.world.pool[*b]
                    .size
                    .partial_cmp(&ctx.world.pool[*a].size)
                    .unwrap()
                    .then(a.cmp(b))
            });
            for rid in backups {
                if candidates.len() >= k {
                    break;
                }
                if rid != want {
                    candidates.push(rid);
                }
            }

            for rid in candidates {
                if let Some(r) = ctx.world.pool.get(rid) {
                    if !r.allocated {
                        ctx.world.pool[rid].allocated = true;
                        assigned = Some(rid);
                        break;
                    }
                }
            }
            ctx.world.allocations.insert(id, assigned);
        }
        Ok(())
    }
}

/// 並び替え戦略 S_queue で応募者リストを in-place 並べ替える．
fn sort_applicants(world: &SrapWorld, order: &mut [(AgentId, Option<ResourceId>)]) {
    match world.policy.sort_strategy {
        SortStrategy::Fifo => {
            // 到着順 = AgentId 昇順 (active_applicants が既に昇順だが明示)．
            order.sort_by_key(|(id, _)| id.0);
        }
        SortStrategy::Vfa => {
            // 脆弱層を先頭 (属性固定)，同層内は AgentId 昇順．
            order.sort_by(|(a, _), (b, _)| {
                let va = world.applicants[a].vulnerable;
                let vb = world.applicants[b].vulnerable;
                vb.cmp(&va).then(a.0.cmp(&b.0))
            });
        }
        SortStrategy::Vfr => {
            // 脆弱層を «家族規模/収入» ランキングで先頭 (ラウンドごとに変わりうる)．
            // ランキングスコア = family / (income+1): 大世帯・低収入ほど高い → 先頭．
            order.sort_by(|(a, _), (b, _)| {
                let aa = &world.applicants[a];
                let ab = &world.applicants[b];
                let sa = aa.family as f64 / (aa.income + 1.0);
                let sb = ab.family as f64 / (ab.income + 1.0);
                sb.partial_cmp(&sa).unwrap().then(a.0.cmp(&b.0))
            });
        }
    }
}

// =========================================================================== //
// 4. EvaluateWelfare (Reward)
// =========================================================================== //

/// 社会的厚生 SW・満足度・公平性指標を計算し記録する (`Reward`; LLM 非依存)．
///
/// 当該ラウンドの配分結果から [`crate::metrics::compute_metrics`] で
/// SW / Avg r_size / Avg WT / Var r_size / Rop / Gini / F(V,NV) を計算し，world と
/// 共有メトリクスバッファ (metrics.csv) へ書き込む．SW は «累積» で記録する
/// (各ラウンドで新規配分された応募者の効用を加算; 論文の累積満足に対応)．
pub struct EvaluateWelfare {
    metrics: SharedMetrics,
    run_idx: usize,
    /// 累積 SW (ラウンドをまたいで配分済み応募者の効用を保持)．
    cumulative: CumulativeWelfare,
}

/// 累積集計 (配分済み応募者の効用・面積・待機時間)．
#[derive(Default)]
struct CumulativeWelfare {
    outcomes: Vec<AllocationOutcome>,
}

impl EvaluateWelfare {
    pub fn new(metrics: SharedMetrics, run_idx: usize) -> Self {
        EvaluateWelfare {
            metrics,
            run_idx,
            cumulative: CumulativeWelfare::default(),
        }
    }
}

impl Mechanism<SrapWorld> for EvaluateWelfare {
    fn name(&self) -> &str {
        "evaluate_welfare"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SrapWorld>) -> Result<()> {
        let round = ctx.world.current_round();

        // 当該ラウンドで新規に配分された応募者の outcome を累積へ追加する．
        // (allocations は AllocationRule が当ラウンドぶんを書いている)．
        let allocations: Vec<(AgentId, Option<ResourceId>)> = ctx
            .world
            .allocations
            .iter()
            .map(|(id, r)| (*id, *r))
            .collect();

        for (id, alloc) in allocations {
            let applicant = &ctx.world.applicants[&id];
            // 既に active=false (前ラウンドで確定済み) はスキップ．
            if !applicant.active {
                continue;
            }
            match alloc {
                Some(rid) => {
                    let r = &ctx.world.pool[rid];
                    let utility = applicant.utility(r);
                    self.cumulative.outcomes.push(AllocationOutcome {
                        allocated: true,
                        size: r.size,
                        utility,
                        wait_time: applicant.memory.wait_time,
                        vulnerable: applicant.vulnerable,
                    });
                }
                None => {
                    // 当ラウンドに未配分．まだ離脱と確定はしない (待機継続もありうる)．
                    // ただし可視資源が尽きた応募者は UpdateMemory で離脱扱いになる．
                }
            }
        }

        // 累積 outcome から指標を計算する (= ラウンド t までの社会的厚生)．
        let m = compute_metrics(&self.cumulative.outcomes);
        ctx.world.metrics = m;

        self.metrics.borrow_mut().push(MetricRow {
            run: self.run_idx,
            round,
            sw: m.sw,
            avg_rsize: m.avg_rsize,
            avg_wt: m.avg_wt,
            var_rsize: m.var_rsize,
            rop: m.rop,
            co_gini: m.co_gini,
            f_vnv: m.f_vnv,
            n_allocated: m.n_allocated,
        });

        ctx.scratch.insert("sw", m.sw);
        Ok(())
    }
}

// =========================================================================== //
// 5. UpdateMemory (PostStep)
// =========================================================================== //

/// 応募者の記憶 m_j を更新し，配分済み/離脱を確定する (`PostStep`; LLM 非依存)．
///
/// - 配分された応募者: active=false にして待機キューから外す (確定退出)．
/// - 未配分の応募者: 待機時間 +1．可視資源が尽きていれば離脱 (active=false)．
/// - 記憶: ラウンドの結果サマリを memory へ追記する (直近 window 件に切り詰め)．
/// - 資源枯渇 / 全員退出で `request_stop()`．
pub struct UpdateMemory {
    /// 記憶窓 (直近何ラウンド分のサマリを保持するか)．
    pub window: usize,
}

impl Mechanism<SrapWorld> for UpdateMemory {
    fn name(&self) -> &str {
        "update_memory"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SrapWorld>) -> Result<()> {
        let round = ctx.world.current_round();
        let any_visible_left = !ctx.world.available_resources().is_empty();

        let ids: Vec<AgentId> = ctx.world.applicants.keys().copied().collect();
        for id in ids {
            let alloc = ctx.world.allocations.get(&id).copied().flatten();
            let applicant = ctx.world.applicants.get_mut(&id).unwrap();
            if !applicant.active {
                continue;
            }
            let summary = match alloc {
                Some(rid) => {
                    applicant.active = false; // 配分確定 → 退出．
                    format!("round {round}: allocated home {rid}.")
                }
                None => {
                    applicant.memory.wait_time += 1;
                    if !any_visible_left {
                        applicant.active = false; // 資源枯渇 → 離脱．
                        format!("round {round}: no homes left, dropped out.")
                    } else {
                        format!("round {round}: not allocated, still waiting.")
                    }
                }
            };
            applicant.memory.summaries.push(summary);
            if self.window > 0 && applicant.memory.summaries.len() > self.window {
                let excess = applicant.memory.summaries.len() - self.window;
                applicant.memory.summaries.drain(0..excess);
            }
        }

        // 当ラウンドの allocations をクリア (次ラウンドへ持ち越さない)．
        ctx.world.allocations.clear();

        if ctx.world.pool_exhausted() || ctx.world.all_settled() {
            ctx.request_stop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Policy;
    use crate::world::{Applicant, Memory, Preferences, Resource};
    use socsim_core::{Blackboard, NullRecorder, SimClock, SimRng, WorldState};
    use std::collections::BTreeMap;

    fn applicant(income: f64, family: usize, vuln: bool) -> Applicant {
        Applicant {
            income,
            rent: 1000.0,
            family,
            preferences: Preferences::default(),
            memory: Memory::default(),
            active: true,
            vulnerable: vuln,
        }
    }

    fn world(policy: Policy) -> SrapWorld {
        let mut applicants = BTreeMap::new();
        applicants.insert(AgentId(0), applicant(8000.0, 1, false));
        applicants.insert(AgentId(1), applicant(3000.0, 5, true));
        applicants.insert(AgentId(2), applicant(5000.0, 3, false));
        let pool = vec![
            Resource {
                id: 0,
                size: 80.0,
                rent: 1000.0,
                allocated: false,
            },
            Resource {
                id: 1,
                size: 50.0,
                rent: 700.0,
                allocated: false,
            },
        ];
        SrapWorld {
            clock: SimClock::new(10),
            applicants,
            pool,
            policy,
            queues: Vec::new(),
            allocations: BTreeMap::new(),
            metrics: Default::default(),
        }
    }

    /// テスト用に 1 mechanism を 1 回 apply する．
    fn run_once<M: Mechanism<SrapWorld>>(
        m: &mut M,
        phase: Phase,
        w: &mut SrapWorld,
        scratch: &mut Blackboard,
    ) -> bool {
        let mut rng = SimRng::from_seed(0);
        let mut stop = false;
        let mut rec = NullRecorder;
        let order = w.agent_ids();
        let clock = *w.clock();
        let mut ctx = StepContext {
            world: w,
            clock,
            rng: &mut rng,
            recorder: &mut rec,
            agent_order: &order,
            scratch,
            stop: &mut stop,
        };
        m.apply(phase, &mut ctx).unwrap();
        stop
    }

    #[test]
    fn policy_setup_builds_m_queues() {
        let mut w = world(Policy {
            m: 2,
            entry_condition: EntryCondition::PSelect,
            ..Policy::default()
        });
        let mut sb = Blackboard::new();
        run_once(&mut PolicySetup, Phase::Environment, &mut w, &mut sb);
        assert_eq!(w.queues.len(), 2);
        // p_select → 全員入室 (3 人が 2 キューに分散)．
        let total: usize = w.queues.iter().map(|q| q.members.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn entry_condition_filters_by_family() {
        let mut w = world(Policy {
            m: 1,
            entry_condition: EntryCondition::PFamily,
            ..Policy::default()
        });
        let mut sb = Blackboard::new();
        run_once(&mut PolicySetup, Phase::Environment, &mut w, &mut sb);
        // family median of [1,3,5] = 3 → family>=3 enters: ids 1(fam5),2(fam3).
        let total: usize = w.queues.iter().map(|q| q.members.len()).sum();
        assert_eq!(total, 2, "only family>=median enter");
    }

    #[test]
    fn allocation_no_double_assign_and_respects_capacity() {
        let mut w = world(Policy {
            k: 1,
            ..Policy::default()
        });
        // 全員が home 0 を希望 → 1 人だけ割当, 残りは k-deferrals で home 1 へ.
        let mut sb = Blackboard::new();
        let desires: Desires = vec![
            (AgentId(0), Some(0)),
            (AgentId(1), Some(0)),
            (AgentId(2), Some(0)),
        ];
        sb.insert(SCRATCH_DESIRES, desires);
        run_once(&mut AllocationRule, Phase::Interaction, &mut w, &mut sb);
        let assigned: Vec<ResourceId> = w.allocations.values().flatten().copied().collect();
        // k=1 → 第一希望 home0 を 1 人だけ取り, 他は次善不可 → 1 件のみ割当.
        assert_eq!(assigned.len(), 1, "k=1 → only first-choice, no backup");
        assert!(w.pool[0].allocated);
        assert!(!w.pool[1].allocated);
        // 二重割当なし．
        assert_eq!(
            assigned
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            assigned.len()
        );
    }

    #[test]
    fn k_deferrals_allows_backup() {
        let mut w = world(Policy {
            k: 2,
            ..Policy::default()
        });
        let mut sb = Blackboard::new();
        // 2 人が home 0 を希望 → 1 人 home0, もう 1 人は k=2 で home1 へ.
        let desires: Desires = vec![(AgentId(0), Some(0)), (AgentId(1), Some(0))];
        sb.insert(SCRATCH_DESIRES, desires);
        run_once(&mut AllocationRule, Phase::Interaction, &mut w, &mut sb);
        let assigned: Vec<ResourceId> = w.allocations.values().flatten().copied().collect();
        assert_eq!(assigned.len(), 2, "k=2 → backup allocation succeeds");
        assert!(w.pool[0].allocated && w.pool[1].allocated);
    }

    #[test]
    fn vfa_orders_vulnerable_first() {
        let w = world(Policy {
            sort_strategy: SortStrategy::Vfa,
            ..Policy::default()
        });
        let mut order: Vec<(AgentId, Option<ResourceId>)> =
            vec![(AgentId(0), None), (AgentId(1), None), (AgentId(2), None)];
        sort_applicants(&w, &mut order);
        // id 1 is vulnerable → first.
        assert_eq!(order[0].0, AgentId(1));
    }

    #[test]
    fn update_memory_settles_and_stops_on_exhaustion() {
        let mut w = world(Policy::default());
        // 両資源を割当済みにして枯渇させる．
        w.pool[0].allocated = true;
        w.pool[1].allocated = true;
        w.allocations.insert(AgentId(0), Some(0));
        w.allocations.insert(AgentId(1), None);
        w.allocations.insert(AgentId(2), None);
        let mut sb = Blackboard::new();
        let stop = run_once(
            &mut UpdateMemory { window: 5 },
            Phase::PostStep,
            &mut w,
            &mut sb,
        );
        assert!(!w.applicants[&AgentId(0)].active, "allocated → inactive");
        // 未配分かつ枯渇 → 離脱．
        assert!(!w.applicants[&AgentId(1)].active);
        assert!(stop, "pool exhausted → request_stop");
    }

    #[test]
    fn drop_out_when_no_visible() {
        // 可視資源が空の応募者は ApplyDecision で離脱希望 (None) になる．
        let mut w = world(Policy::default());
        w.pool[0].allocated = true;
        w.pool[1].allocated = true;
        // ApplyDecision は LLM を呼ばずに None を入れるはず (visible empty)．
        // mock client が無いので, ここでは visible_resource_ids が空であることを確認．
        assert!(visible_resource_ids(&w).is_empty());
    }
}
