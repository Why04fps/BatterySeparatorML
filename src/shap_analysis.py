"""
SHAP 可解释性分析模块
====================
为已训练模型计算 SHAP 值并生成蜂群图（beeswarm），用于解释各特征对预测的贡献。

**重要**（交接文档踩坑经验）：
- 需用新版 API：`shap.plots.beeswarm(Explanation)`
- 旧版 `shap.summary_plot(plot_type="beeswarm")` 会渲染不出散点
- matplotlib 中文乱码已通过 `SimHei` 字体 + `axes.unicode_minus=False` 解决
"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # 无界面环境保存图片
import matplotlib.pyplot as plt
import numpy as np
import shap

from . import config
from . import predict


def _init_matplotlib():
    """初始化 matplotlib 中文字体，避免中文乱码"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def compute_shap_values(model, X):
    """
    用 TreeExplainer 计算 SHAP 值。

    参数
    ----
    model : xgboost 模型
    X : pandas.DataFrame
        特征数据

    返回
    ----
    Explanation 对象（新版 SHAP 接口，可直接传给 shap.plots.beeswarm）
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)  # 生成 Explanation 对象
    return shap_values


def plot_beeswarm(shap_values, feature_names, save_path):
    """
    生成 SHAP 蜂群图并保存。

    参数
    ----
    shap_values : Explanation 对象
    feature_names : list
        特征名列表
    save_path : str
        图片保存路径
    """
    _init_matplotlib()
    shap.plots.beeswarm(shap_values, max_display=len(feature_names), show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def run_all(target_name=None, save_dir=None, max_samples=2000):
    """
    为指定目标（或全部已加载模型）生成 SHAP 蜂群图。

    参数
    ----
    target_name : str 或 None
        目标变量名；None 时对所有已加载模型执行
    save_dir : str 或 None
        图片输出目录；None 时用 config.REPORT_DIR
    max_samples : int
        SHAP 计算用的最大样本数（避免大数据集过慢）

    返回
    ----
    list: 生成的图片路径列表
    """
    from . import features

    if save_dir is None:
        save_dir = config.REPORT_DIR
    os.makedirs(save_dir, exist_ok=True)

    # 加载全部模型
    models = predict.load_models()
    if not models:
        print("未找到已训练模型，请先运行 `python run_train.py`")
        return []

    if target_name is not None and target_name not in models:
        print(f"目标变量 {target_name} 不在已加载模型中: {list(models.keys())}")
        return []

    targets = [target_name] if target_name is not None else list(models.keys())

    saved = []
    for name in targets:
        print(f"\n{'='*60}")
        print(f"计算 SHAP: {name}")
        print(f"{'='*60}")

        # 根据任务类型加载对应的特征数据
        if name in config.POLYMER_TASKS:
            _, X, _ = features.load_polymer_data(config.POLYMER_TASKS[name])
        elif name == "conductivity":
            _, X, _, _, _ = features.load_conductivity_data()
        else:
            print(f"  跳过未知任务: {name}")
            continue

        # 抽样（避免 SHAP 计算过慢）
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=config.RANDOM_STATE)

        model = models[name]
        shap_values = compute_shap_values(model, X)
        save_path = os.path.join(save_dir, f"{name}_shap_beeswarm.png")
        plot_beeswarm(shap_values, list(X.columns), save_path)
        print(f"  蜂群图已保存: {save_path}")
        saved.append(save_path)

    return saved
