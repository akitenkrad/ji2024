//! Ji, Li, Liu, Du, Wei, Shen, Qi & Lin (2024) "SRAP-Agent: Simulating and
//! Optimizing Scarce Resource Allocation Policy with LLM-based Agent"
//! (Findings of EMNLP 2024) の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した «公共希少資源 (公共住宅) を中央プールへ配分
//! するポリシーの LLM 駆動シミュレーション» の公開 API を提供する．配分ポリシー
//! `policy`・世界状態 `world`・LLM クライアント層 `llm`・プロンプト生成と応答パース
//! `prompts`・更新メカニズム `mechanisms`・実行ドライバ `simulation`・評価指標
//! `metrics`・ポリシー最適化 `poa` (Phase-3 スタブ) をモジュールとして公開し，
//! バイナリ (`srap`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (応募者/資源プール初期化・入室条件のキュー構成・k-deferrals 配分
//! 規則・満足度/公平性指標・記憶更新) は seed から bit 単位で決定論的である．LLM
//! レイヤ (応募者の資源選択意思決定) は socsim の bit 再現性の **外側** にあり，
//! `socsim-llm` のキャッシュ + `temperature=0` + `seed` 固定で擬似決定論化する
//! (詳細は `crate::llm`)．設計書 §4.2/§7 は当初 `reqwest` + `sha2` を挙げていたが，
//! 本スイートは han2023 / li2024 / zhao2024 / chuang2024 と統一して `socsim-llm`
//! (issue #21/#26) に標準化したため `reqwest` / `sha2` は使わない．
//!
//! # フェーズ
//!
//! - **Phase 1** (本実装): SrapWorld + 5 mechanism + LLM クライアント層 + 単一
//!   ポリシー `run`．
//! - **Phase 2** (本実装): ポリシー因子の `sweep` + 可視化．
//! - **Phase 3** (`poa` は最小スタブ): 遺伝的アルゴリズムによるポリシー最適化．
//!   完成版 (予測器 f̃ + ライブ LLM 適応度 + 論文 Fig.4/Table 一括再現) は未実装．

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod poa;
pub mod policy;
pub mod prompts;
pub mod simulation;
pub mod world;
