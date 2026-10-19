[English](reproduction.md) | [日本語](reproduction.ja.md)

# 再現

## 定量目標 (論文)

| 条件 | 指標 | 論文値 | 再現目標 |
|------|------|--------|----------|
| 入室 `p_select`, サブセット `r_size` | SW (社会的厚生) | 419.9 (最高) | 符号/順序 |
| サブセット `r_random` | SW | 83.8 (最低) | 符号/順序 |
| POA 50 反復後 | f(π) 改善率 | 約 +20% | 方向 |
| 満足度志向 π_s* | SW 改善率 | +1.8% | 方向 |
| 公平性志向 π_f* | 社会的公平性改善率 | +13.2% | 方向 |

## 厳密 vs 定性

- **決定論的な配分/指標パス** — 配分規則 (S_queue + k-deferrals; 二重割当なし・プール容量厳守) と全満足度/公平性指標 (SW, Avg r_size, Avg WT, Var r_size, Rop, co-Gini, F(V,NV)) は純決定論計算で seed が与えられれば bit 単位で再現する．
- **LLM 応募者挙動** — 定性のみ．ローカル既定モデル (`llama3.2`) は論文の `gpt-3.5-turbo-0301` と異なるため論文値との厳密一致は期待しない．

## scripted mock で確かめられること

scripted mock（`--mock`，`scripted_first_home`）は，応募者のプロファイルに関係なく常に先頭の可視資源を選ぶ．このため資源サブセットのポリシーは «誰がどの家を得るか» を並べ替えるだけになり，E[SW(r_size) − SW(r_random)] は構成上 0 になる（500 シードで平均 −5.2 ± 14.7）．mock で確かめられるのは決定論性・配分の不変量・サブセットに依存した偏りが無いことまでで，論文のポリシー順序は確かめられない．順序の評価にはライブ LLM が要る．

## 設計上の不確実性 (論文 §7)

- 合成環境 (応募者数・資源数・プロファイル分布) は論文本文に明示がない — CLI (`--n-applicants`, `--pool-ratio`, `--visible-subset-size`) で外部化し既定値 (応募者 60, 比 0.5) を置く．
- GA ハイパーパラメータ (プール規模・交叉/突然変異率)・予測器 `f̃` の margin は付録依存で，CLI フラグとして標準既定値で外部化する．`poa` サブコマンドは決定論 mock かライブ LLM のいずれかで適応度を評価し，予測器 `f̃` (最近傍サロゲート) がフル評価を枝刈りする．付録値は論文本文にないため，ライブ LLM での SW の *順序* と POA の *改善方向* (絶対適応度ではない) を再現目標とする．

## 一括再現 (`reproduce`)

```bash
cargo run --release -- reproduce --mock --seed 42      # 高速スモークは --quick
uv run srap-tools reproduce   # 図を生成
```

`reproduce` は Table 2 を **matched-seed paired** 比較 (同一環境で資源サブセットのみを変える) として実行し，厳密な wins・ties・losses と対応差の平均・標準誤差を記録する．ライブ LLM では平均差 ± 1.96·SE の下限が 0 より大きければ `PASS`，上限が 0 より小さければ `reversed`，それ以外は `inconclusive` と判定する．`--mock` では順序判定を `not_applicable` として総合判定から外すが，数値は記録・表示する．続けて両目標で POA を実行し，目標別の収束履歴を出力する．

## 実 Ollama スモーク (メインセッション用)

```bash
ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=llama3.2:latest
# コストを抑えるため小さく: 約 n_applicants × rounds 回の LLM 呼び出し (cache で再実行は無料)．
cargo run --release -- run --entry-condition p_select --resource-subset r_size \
    --queues 3 --k 3 --c 2 --runs 1 --seed 42 --n-applicants 12 --max-rounds 3
```
