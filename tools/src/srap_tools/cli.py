"""srap-tools — Ji et al. (2024) SRAP-Agent 再現実装 ツール統合 CLI．

Usage:
    srap-tools visualize [...]
    srap-tools visualize-sweep [...]
    srap-tools show-experiment-settings [...]
    srap-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示/
再現の実体 (visualize / visualize_sweep / show_experiment_settings / reproduce_paper)
は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="srap-tools",
    description="Ji et al. (2024) SRAP-Agent 希少資源配分シミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (満足度・公平性の時系列 + 配分面積分布) の可視化",
            "srap_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (ポリシー因子依存) / POA 収束曲線の可視化",
            "srap_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / poa_config / llm_meta) の表示",
            "srap_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "論文 Table 2/3・Fig.4 一括再現 (Phase 3; 現状はスタブ案内)",
            "srap_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
