[English](README.md) | **日本語**

# SRAP-Agent: LLM エージェントによる希少資源配分ポリシーのシミュレーション — Ji et al. (2024)

Ji, Li, Liu, Du, Wei, Shen, Qi & Lin (2024) ["SRAP-Agent: Simulating and Optimizing Scarce Resource Allocation Policy with LLM-based Agent"](https://doi.org/10.48550/arXiv.2410.14152) (Findings of EMNLP 2024) の再現実装である．LLM 駆動の **応募者** エージェントが公共希少資源 (ケーススタディは **公共住宅**) の **中央プール** へ応募し，決定論的な配分規則が設定可能なポリシーのもとで住宅を割り当てる．配分ポリシーは

> π = (E_queue *入室条件*, S_queue *並び替え戦略*, R_queue *資源サブセット*, m *キュー数*, k *k-deferrals*, c *選択キュー容量*)

である．各ラウンド (= 1 engine tick): 入室条件で m 本の待機キューを構成 → 各 active 応募者が可視サブセット V(p_j) から希望住宅を選択 (これが **LLM** の意思決定; Eq. 2 `R_j* = D(p_j, V(p_j))`, ∅ = 離脱) → 決定論的配分規則が S_queue (FIFO / 脆弱層優先 VFA・VFR) で順序づけ k-deferrals で R_queue の資源を割り当てる → 満足度・公平性を評価 → 応募者の記憶を更新．本モデルは **中央配分** であり非空間・非ネットワークなので [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) の `socsim-core` + `socsim-engine` + `socsim-llm` のみに依存する (`socsim-grid` / `socsim-net` 不使用)．

## 二層決定論 (最初に読むこと)

LLM 出力は socsim の bit 単位再現性の **外側** にある．したがって設計を二層に分ける:

- **決定論的 socsim コア** — 合成応募者/資源プール初期化，入室条件によるキュー構成，決定論的配分規則 (S_queue 並び替え + k-deferrals; 二重割当なし・プール容量厳守)，満足度/公平性の各指標 (SW, Avg r_size, Avg WT, Var r_size, Rop 逆順ペア数, co-Gini ∈ [0,1], F(V,NV) 脆弱層ギャップ)，記憶更新．seed が与えられれば bit 単位で再現する (ChaCha20 `SimRng`; ストリーム 2 本: `RNG_WORLD_INIT=0`, `RNG_ENGINE=1`)．
- **非決定的 LLM レイヤ** — 単一の `Decision` メカニズム (`ApplyDecision`)．各 active 応募者がプロファイル・可視資源・記憶から希望住宅を選択する．`socsim-llm` の `CachingClient` (`hash(prompt+model)` → 応答キャッシュ)・`temperature=0`・固定 seed で擬似決定論化する．プロバイダ順序は **Ollama 第一 → OpenAI フォールバック** (`FallbackClient`)．

再現性の本体はモデルではなくキャッシュである．各実行は `llm_meta.json` にモデル・endpoint・温度・seed・cache-hit 率を記録する．ローカル既定モデル (`llama3.2`) は論文の `gpt-3.5-turbo-0301` と異なるため，LLM 駆動の再現目標は **定性的** である．決定論的な配分/指標パスが定量的なコアであり，ポリシー間の SW の **順序** (`p_select`+`r_size` が最高，`r_random` が最低) は定性的に保たれるはずである．

> 本プロジェクトは LLM レイヤを `socsim-llm` クレートに標準化し，`reqwest` / `sha2` は使わない (socsim-llm が HTTP とプロンプトハッシュを所有する)．これは han2023 / li2024 / zhao2024 / chuang2024 の sibling と統一するため，設計書の当初の `reqwest`+`sha2` 案を上書きするものである．

## 機能

- **`run`**: `SrapWorld` + 5 メカニズム + LLM クライアント層 + 単一ポリシー実行 (満足度・公平性指標)．
- **`sweep`**: ポリシー因子の感度スイープ (入室条件 × 資源サブセット × 並び替え戦略)．
- **`poa`** — Policy Optimization Agent: 配分ポリシー空間 `(E_queue, S_queue, R_queue, m, k, c)` 上の遺伝的アルゴリズム外側ループ (トーナメント選択 / 一様交叉 / 遺伝子単位の突然変異 / 1-エリート保存)．適応度は 1 回の SRAP 配分実行を `f_pi(metrics, objective)` で評価し，決定論的 scripted mock (`--mock`, オフライン・bit 決定論) かライブ LLM (Ollama→OpenAI + 永続キャッシュ) のいずれかで計算する．予測器 `f̃` サロゲート (評価済みポリシーの重み付き最近傍回帰) が «現行エリートに勝てない見込み» の個体のフル評価を枝刈りし，高価な評価を削減する．エリート保存により最良適応度は世代をまたいで単調非減少．
- **`reproduce`**: 論文 Table 2 (ポリシー順序の社会的厚生)・Table 3 (POA 最適化ポリシー)・Figure 4 (POA 収束) を一括再現し，CSV + `reproduce_summary.json` (観測 vs 論文知見の PASS / off-anchor) と図を書き出す．

## インストールとクイックスタート

```bash
# Rust シミュレーションをビルド (socsim と socsim-llm の Ollama+OpenAI バックエンドを取得)
cargo build --release

# ローカル Ollama を起動しモデルを pull しておく:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# (任意) OpenAI フォールバック:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-3.5-turbo

# 基本実験: 単一ポリシー (最高 SW 条件 p_select + r_size)
cargo run --release -- run \
    --entry-condition p_select --resource-subset r_size \
    --queues 3 --k 3 --c 2 --runs 10 --seed 42

# Python 可視化ツールのインストール (workspace ルートで)
uv sync

# 直近実行の可視化 (満足度・公平性の時系列, 最終指標サマリ)
uv run srap-tools visualize

# 設定と LLM メタデータの確認
uv run srap-tools show-experiment-settings --results-dir results/latest
```

### オフライン (LLM 不要) スモーク

ライブ LLM なしで全ラウンドループ・出力 writer・Python 可視化を検証できる (CI・ネットワーク遮断サンドボックス用の scripted mock):

```bash
# 専用 example
cargo run --release --example mock_smoke -- results

# または run / sweep / poa / reproduce に --mock を付けて同じオフライン挙動
cargo run --release -- run --entry-condition p_select --resource-subset r_size \
    --queues 3 --k 3 --c 2 --runs 3 --seed 42 --mock
uv run srap-tools visualize
```

### 感度分析 (sweep) と POA

```bash
# 3 主要因子のスイープ (入室条件 × 資源サブセット)
cargo run --release -- sweep \
    --entry-conditions p_budget,p_family,p_select \
    --resource-subsets r_size,r_rent,r_random \
    --runs 30 --seed 42            # オフラインなら --mock を付ける
uv run srap-tools visualize-sweep

# POA ポリシー最適化 (予測器 f̃ あり; オフラインなら --mock)
cargo run --release -- poa --objective satisfaction \
    --iterations 50 --pool-size 50 --use-predictor --seed 42
uv run srap-tools visualize-sweep   # POA 収束曲線を描く
```

### 論文再現 (Table 2/3 + Fig.4)

```bash
# ポリシー順序の知見 + POA 最適化を一括再現 (オフライン)
cargo run --release -- reproduce --mock --seed 42        # 高速スモークは --quick
# 図を生成し観測 vs 論文の判定を再表示
uv run srap-tools reproduce --results-dir results/latest
```

## テストと Lint

```bash
cargo test --release   # mock (ScriptedClient) 駆動, ライブ LLM 不要 (52 テスト)
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## ドキュメント

- [アーキテクチャ](docs/architecture.ja.md) — `SrapWorld`, 5 メカニズム / 6 フェーズ, RNG ストリーム, 二層 LLM．
- [CLI リファレンス](docs/cli.ja.md) — `run` / `sweep` / `poa` / `reproduce` のフラグと出力．
- [再現](docs/reproduction.ja.md) — 定量目標, ポリシー順序の知見, 設計上の不確実性．
- [可視化](docs/visualization.ja.md) — Python `srap-tools` の図．

## 参考文献

- Ji, J., Li, Y., Liu, H., Du, Z., Wei, Z., Shen, W., Qi, Q., & Lin, Y. (2024). SRAP-Agent: Simulating and Optimizing Scarce Resource Allocation Policy with LLM-based Agent. *Findings of the Association for Computational Linguistics: EMNLP 2024*, 267–293.
- socsim: [rs-social-simulation-tools](https://github.com/akitenkrad/rs-social-simulation-tools).

## ライセンス

MIT — [LICENSE](LICENSE) を参照．

---
*This file was generated by Claude Code.*
