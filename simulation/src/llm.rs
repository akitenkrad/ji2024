//! LLM クライアント層 (Ollama 第一 → OpenAI フォールバック + キャッシュ)．
//!
//! 本モジュールは `socsim-llm` の合成 API に対する薄いビルダである．二層
//! アーキテクチャの **上層 (非決定的 LLM レイヤ)** をここに閉じ込め，下層の
//! 決定論的 socsim コア (キュー構成・配分規則・指標計算・記憶更新) からは
//! [`SrapClient`] 型エイリアス経由でのみ触れる．LLM 呼び出しは `apply_decision`
//! Mechanism (Decision フェーズ; 応募者の資源選択) のみに閉じ込める．
//!
//! # 合成 (Ollama 第一 → OpenAI フォールバック → キャッシュ)
//!
//! ```text
//! CachingClient< Box<dyn LlmClient> >
//!   └─ cache: PromptCache (prompt → response; 擬似決定論の本体)
//!      └─ backend (型消去): FallbackClient< OllamaClient, OpenAiClient >
//!         primary:   OllamaClient   (OLLAMA_HOST / OLLAMA_MODEL)
//!         secondary: OpenAiClient   (OPENAI_API_KEY / OPENAI_MODEL)
//! ```
//!
//! `FallbackClient` は socsim-llm が提供する (自前実装しない)．`CachingClient` は
//! その上にプロンプト→応答キャッシュを被せ，`temperature=0` / `seed` 固定と合わせて
//! 再実行を擬似決定論化する．socsim-llm がプロンプトハッシュ (cache key) を所有
//! するため `sha2` は不要．
//!
//! 設計書 §4.2/§7 は当初 `reqwest` + `sha2` + 手書き `llm.rs` を挙げていたが，
//! 本スイートは han2023 / li2024 / zhao2024 / chuang2024 と統一して `socsim-llm`
//! クレート (issue #21/#26) に標準化したため `reqwest` / `sha2` は使わず，HTTP と
//! ハッシュは socsim-llm が所有する．
//!
//! テストでは `socsim-llm::mock::ScriptedClient` を `Box<dyn LlmClient>` として
//! 同じ [`SrapClient`] に流し込める．

use socsim_llm::{
    CachingClient, FallbackClient, LlmClient, LlmConfig, LlmError, OllamaClient, OpenAiClient,
    PromptCache,
};

use crate::config::LlmSettings;

/// 本シミュレーションが用いるキャッシュ付きクライアント型．
///
/// バックエンドは `Box<dyn LlmClient>` に型消去してあり，本番は
/// `FallbackClient<OllamaClient, OpenAiClient>`，テストは `ScriptedClient` を
/// 注入できる．`socsim-llm` の `impl LlmClient for Box<T>` (issue #26) により
/// 専用 newtype なしで `CachingClient` の `C: LlmClient` 境界を満たす．
pub type SrapClient = CachingClient<Box<dyn LlmClient>>;

/// 本番用の «Ollama 第一 → OpenAI フォールバック + キャッシュ» クライアントを
/// 環境変数から構築する．
///
/// - Ollama: `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL`
///   (既定 `llama3.2:latest`)．
/// - OpenAI: `OPENAI_API_KEY` / `OPENAI_MODEL` (既定 `gpt-4o-mini`; 原論文は
///   `gpt-3.5-turbo-0301`)．未設定なら空キーのフォールバックを置く (Ollama が
///   成功すれば呼ばれない; 両方失敗時のみ設定エラーになる)．
/// - キャッシュ: `settings.cache_path` があればその JSON ファイル，なければ
///   in-memory．
pub fn build_live_client(settings: &LlmSettings) -> Result<SrapClient, LlmError> {
    // 既定モデルを llama3.2:latest にする (OLLAMA_MODEL 未設定時)．socsim-llm の
    // OllamaClient::from_env 既定は llama3.1 のため，明示的に上書きする．
    if std::env::var("OLLAMA_MODEL").is_err() {
        std::env::set_var("OLLAMA_MODEL", "llama3.2:latest");
    }
    let ollama = OllamaClient::from_env();
    // OPENAI_API_KEY が無い環境でも Ollama 単独で動かせるよう，from_env が失敗
    // した場合は空キーのプレースホルダを置く (Ollama 失敗時のみ Config エラー)．
    let openai = OpenAiClient::from_env().unwrap_or_else(|_| {
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        OpenAiClient::new("", model)
    });

    let fallback = FallbackClient::new(ollama, openai);
    let backend: Box<dyn LlmClient> = Box::new(fallback);

    let cache = match &settings.cache_path {
        Some(path) => PromptCache::open(path)?,
        None => PromptCache::in_memory(),
    };
    Ok(CachingClient::new(backend, cache))
}

/// 任意の [`LlmClient`] (例: `mock::ScriptedClient`) をキャッシュで包んだ
/// [`SrapClient`] を作る (主にテスト・`--mock` 用)．
pub fn wrap_client<C: LlmClient + 'static>(backend: C, cache: PromptCache) -> SrapClient {
    let boxed: Box<dyn LlmClient> = Box::new(backend);
    CachingClient::new(boxed, cache)
}

/// [`LlmSettings`] から socsim-llm の [`LlmConfig`] を組み立てる．
pub fn llm_config(settings: &LlmSettings) -> LlmConfig {
    LlmConfig::deterministic()
        .with_temperature(settings.temperature)
        .with_seed(settings.seed)
}
