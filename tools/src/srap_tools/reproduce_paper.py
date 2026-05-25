"""srap-tools reproduce — 論文 Table 2/3・Fig.4 一括再現 (Phase 3 スタブ)．

論文 Ji et al. (2024) の Table 2 (入室条件 × 資源サブセット の SW)・Table 3 (満足度
志向 π_s* / 公平性志向 π_f* の最適化ポリシー)・Figure 4 (POA の f(π) 収束) を一括で
再現する計画．現状は Phase 3 の実装待ちのため，案内メッセージを表示するスタブである．

Phase 1/2 で利用できるもの:
  - 単一実行の満足度・公平性時系列 → `srap-tools visualize`
  - ポリシー因子 (入室条件 × 資源サブセット) の SW → `srap-tools visualize-sweep`
  - POA 収束曲線 (Phase-3 最小スタブ)         → `srap-tools visualize-sweep`

Usage:
    srap-tools reproduce
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="srap-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)

    print("=== Ji et al. (2024) SRAP-Agent 論文 Table 2/3・Fig.4 一括再現 ===")
    print("reproduce は Phase 3 で実装予定です (現状はスタブ)．")
    print("Table 2/3・Fig.4 の一括再現 (予測器 f̃ + ライブ LLM 適応度) を計画しています．")
    print("")
    print("現時点では以下を個別にご利用ください:")
    print("  - 満足度 / 公平性の時系列   :  uv run srap-tools visualize")
    print("  - ポリシー因子の SW 依存     :  uv run srap-tools visualize-sweep")
    print("  - POA 収束曲線 (最小スタブ)  :  cargo run --release -- poa --mock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
