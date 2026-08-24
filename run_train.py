"""
训练入口脚本
====================
用法：
    python run_train.py                 # 训练全部 3 个目标（Tg、Td、离子电导率）
    python run_train.py --task Tg       # 只训练指定任务（Tg/Td/conductivity）
    python run_train.py --task Tg,Td    # 训练多个任务

说明：
    1. Tg/Td 是纯聚合物任务（RDKit 描述符特征）
    2. conductivity 是多组分任务（RDKit + 盐混合编码 + 浓度 + 温度）
    3. 训练完成后模型保存到 models/ 目录
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config, train


TASK_NAMES = list(config.POLYMER_TASKS.keys()) + ["conductivity"]


def main():
    parser = argparse.ArgumentParser(description="训练锂电池隔膜性能预测模型")
    parser.add_argument("--task", type=str, default=None,
                        help=f"任务名，逗号分隔（可选: {', '.join(TASK_NAMES)}）。默认全部。")
    args = parser.parse_args()

    config.ensure_dirs()

    # 解析要训练的任务
    if args.task:
        tasks = [t.strip() for t in args.task.split(",")]
        invalid = [t for t in tasks if t not in TASK_NAMES]
        if invalid:
            print(f"❌ 未知任务: {invalid}，可选: {TASK_NAMES}")
            sys.exit(1)
    else:
        tasks = TASK_NAMES

    print("=" * 60)
    print("锂电池隔膜性能预测 - 模型训练")
    print("=" * 60)
    print(f"任务列表: {tasks}")

    results = {}
    for task in tasks:
        if task in config.POLYMER_TASKS:
            res = train.train_polymer_task(task, config.POLYMER_TASKS[task])
            results[task] = res
        elif task == "conductivity":
            res = train.train_conductivity_task()
            results["conductivity"] = res

    # 保存模型
    print("\n正在保存模型...")
    train.save_all(results)

    # 输出汇总
    print("\n" + "=" * 60)
    print("📊 训练结果汇总")
    print("=" * 60)
    header = f"{'目标':<20}{'CV R2':<14}{'测试R2':<10}{'MAE':<10}{'RMSE':<10}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(f"{name:<20}{res['cv_r2_mean']:<14.3f}"
              f"{res['test_metrics']['R2']:<10.3f}"
              f"{res['test_metrics']['MAE']:<10.3f}"
              f"{res['test_metrics']['RMSE']:<10.3f}")

    print("\n🎉 训练完成！模型已保存到 models/ 目录")
    print("   下一步: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
