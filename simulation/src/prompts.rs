//! LLM プロンプト生成と応答パース (応募者の資源選択意思決定)．
//!
//! SRAP-Agent の Decision プロンプト (論文 Eq.2 `R_j* = D(p_j, V(p_j))`) を，
//! 応募者プロファイル + 可視資源プール V(p_j) + 記憶 m_j から構築する．LLM には
//! «希望する資源 ID» または «離脱 (∅)» を REACT スタイル (Thought → Action) で
//! 答えさせ，末尾に JSON で結論を出させる (論文 §5 の発話=REACT 法の注記に対応)．
//!
//! 応答パースは「JSON `{"choice": id}` → 本文中の数値 → フォールバック」の多段で
//! 頑健化する (ローカルモデルは厳密 JSON を返さないことがある)．`"choice": -1` /
//! `"drop"` / `none` は離脱 (∅) として扱う．

use crate::world::{Applicant, Resource};

/// 応募者の意思決定プロンプトに渡すブリーフィング．
pub struct ApplicantBriefing<'a> {
    /// 応募者の生 `AgentId` (= applicant_id)．
    pub applicant_id: u64,
    /// このラウンド番号 (0 始まり)．
    pub round: u64,
    /// 応募者プロファイル．
    pub applicant: &'a Applicant,
    /// 可視資源プール V(p_j) (ポリシー R_queue でサブセット化済み; プール内の参照)．
    pub visible: &'a [&'a Resource],
}

/// 応募者の意思決定 (`R_j* = D(p_j, V(p_j))`) のプロンプトを構築する．
///
/// LLM には «choice» (希望資源 ID か離脱) を JSON で答えさせる．プロンプト末尾の
/// 固定文でキャッシュキー (= プロンプト全文 + モデル名) を決定論化する．
pub fn applicant_prompt(brief: &ApplicantBriefing) -> String {
    let a = brief.applicant;
    let mut s = String::new();

    // --- Profile ---
    s.push_str("## Your Profile\n");
    s.push_str(&format!(
        "You are applicant #{}, applying for public housing. Your monthly income is {:.0}, \
         your affordable rent is up to {:.0}, and your household has {} member(s).\n",
        brief.applicant_id, a.income, a.rent, a.family
    ));
    if a.vulnerable {
        s.push_str(
            "You belong to a vulnerable group (low income relative to a large household).\n",
        );
    }
    s.push_str(&format!(
        "You value living space (weight {:.2}) and prefer affordable rent (weight {:.2}).\n\n",
        a.preferences.size_weight, a.preferences.rent_weight
    ));

    // --- Round / Rules ---
    s.push_str("## Round Rules\n");
    s.push_str(&format!("- This is allocation round {}.\n", brief.round));
    s.push_str(
        "- Public housing is a scarce resource: not everyone will be allocated a home.\n\
         - You may apply for ONE of the visible homes below, or choose to drop out this round.\n\n",
    );

    // --- Visible resources V(p_j) ---
    if brief.visible.is_empty() {
        s.push_str(
            "## Available Homes\nThere are no homes visible to you this round; you must drop out.\n\n",
        );
    } else {
        s.push_str("## Available Homes (visible to you)\n");
        for r in brief.visible {
            s.push_str(&format!(
                "- home {}: size {:.0} m^2, rent {:.0}\n",
                r.id, r.size, r.rent
            ));
        }
        s.push('\n');
    }

    // --- Memory (m_j; bounded recent summaries) ---
    let mem = a.memory.recent(5);
    if !mem.is_empty() {
        s.push_str("## Your Memory\n");
        s.push_str(&mem);
        s.push_str("\n\n");
    }

    // --- Decision (REACT-style) ---
    s.push_str(
        "## Decision\n\
         Think step by step (REACT): briefly reason about which home maximizes your \
         satisfaction given your budget and household, then decide.\n\
         If no home is worth applying for, drop out.\n\
         End your answer with JSON only on the final line, e.g. {\"choice\": 3} to apply for \
         home 3, or {\"choice\": -1} to drop out.\n",
    );
    s
}

/// 応募者の意思決定 (choice = 希望資源 ID | 離脱) をパースする．
///
/// 1. JSON `{"choice": id}` を試す (id < 0 / "drop" / "none" は離脱 = `None`)．
/// 2. 失敗時は本文中の最初の数値を拾う．
/// 3. それも失敗なら `None` (= 離脱として扱う)．
///
/// `visible_ids` に含まれない ID は無効として `None` (離脱) を返す (可視外の選択を
/// 防ぐ)．
pub fn parse_choice(text: &str, visible_ids: &[usize]) -> Option<usize> {
    // 明示的な離脱語．
    let lower = text.to_ascii_lowercase();
    if lower.contains("\"choice\": -1")
        || lower.contains("\"choice\":-1")
        || lower.contains("\"drop\"")
        || lower.contains("drop out")
    {
        // -1 を意図する場合は離脱．ただし下で数値も確認する．
        if let Some(json) = extract_json_object(text) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(c) = v.get("choice").and_then(|x| x.as_i64()) {
                    if c < 0 {
                        return None;
                    }
                    let id = c as usize;
                    return if visible_ids.contains(&id) {
                        Some(id)
                    } else {
                        None
                    };
                }
            }
        }
        return None;
    }

    if let Some(json) = extract_json_object(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(c) = v.get("choice").and_then(|x| x.as_i64()) {
                if c < 0 {
                    return None;
                }
                let id = c as usize;
                return if visible_ids.contains(&id) {
                    Some(id)
                } else {
                    None
                };
            }
        }
    }

    // 本文中の最初の整数を拾う．
    if let Some(n) = first_uint(text) {
        if visible_ids.contains(&n) {
            return Some(n);
        }
    }
    None
}

/// 文字列から最初の `{ … }` ブロックを切り出す (末尾の JSON も拾えるよう rfind)．
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.rfind('{')?;
    let end = text[start..].find('}').map(|i| start + i)?;
    Some(text[start..=end].to_string())
}

/// 本文中の最初の非負整数を拾う．
fn first_uint(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if let Ok(n) = text[start..i].parse::<usize>() {
                return Some(n);
            }
        } else {
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Memory, Preferences};

    fn applicant() -> Applicant {
        Applicant {
            income: 5000.0,
            rent: 1000.0,
            family: 3,
            preferences: Preferences::default(),
            memory: Memory::default(),
            active: true,
            vulnerable: true,
        }
    }

    #[test]
    fn prompt_lists_visible_and_profile() {
        let a = applicant();
        let r0 = Resource {
            id: 0,
            size: 60.0,
            rent: 900.0,
            allocated: false,
        };
        let r1 = Resource {
            id: 1,
            size: 40.0,
            rent: 700.0,
            allocated: false,
        };
        let visible = vec![&r0, &r1];
        let brief = ApplicantBriefing {
            applicant_id: 7,
            round: 2,
            applicant: &a,
            visible: &visible,
        };
        let p = applicant_prompt(&brief);
        assert!(p.contains("applicant #7"));
        assert!(p.contains("home 0"));
        assert!(p.contains("home 1"));
        assert!(p.contains("vulnerable"));
        assert!(p.contains("\"choice\""));
    }

    #[test]
    fn parse_choice_json() {
        assert_eq!(parse_choice("{\"choice\": 3}", &[1, 3, 5]), Some(3));
        assert_eq!(parse_choice("I pick {\"choice\": 1}", &[0, 1]), Some(1));
    }

    #[test]
    fn parse_choice_drop_out() {
        assert_eq!(parse_choice("{\"choice\": -1}", &[0, 1]), None);
        assert_eq!(parse_choice("I will drop out.", &[0, 1]), None);
    }

    #[test]
    fn parse_choice_invalid_id_drops() {
        // 99 is not visible → treated as drop-out (None).
        assert_eq!(parse_choice("{\"choice\": 99}", &[0, 1]), None);
    }

    #[test]
    fn parse_choice_prose_fallback() {
        assert_eq!(parse_choice("I choose home 2", &[0, 2]), Some(2));
        assert_eq!(parse_choice("hmm not sure", &[0, 2]), None);
    }
}
