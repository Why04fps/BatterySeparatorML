"""
SHAP 可解释性分析入口脚本
====================
用法：
    python run_shap.py                # 对所有已训练模型生成蜂群图
    python run_shap.py --target Tg_K  # 仅对指定目标生成蜂群图
    python run_shap.py --out reports  # 指定输出目录

说明：
    1. 需要先运行 `python run_train.py` 训练好模型
    2. 蜂群图保存到 reports/ 目录（{目标}_shap_beeswarm.png）
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import shap_analysis


def main():
    parser = argparse.ArgumentParser(description="SHAP 可解释性分析")
    parser.add_argument("--target", type=str, default=None, help="目标变量名（默认全部）")
    parser.add_argument("--out", type=str, default=None, help="图片输出目录（默认 reports/）")
    args = parser.parse_args()

    print("=" * 60)
    print("SHAP 可解释性分析")
    print("=" * 60)

    saved = shap_analysis.run_all(target_name=args.target, save_dir=args.out)
    if not saved:
        print("\n❌ 未生成任何蜂群图，请检查模型是否已训练。")
        sys.exit(1)

    print("\n🎉 SHAP 分析完成！图片保存在:")
    for p in saved:
        print(f"   - {p}")


if __name__ == "__main__":
    main()
