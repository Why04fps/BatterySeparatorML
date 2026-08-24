# 锂电池隔膜筛选工具

> 用机器学习从聚合物分子结构快速预测隔膜关键性能，加速候选材料筛选

**作者：韦宏宇**

## 1. 项目简介

隔膜是锂电池最关键的部件之一，性能直接决定电池安全。传统依赖实验筛选候选材料，周期长（数月）、效率低。本项目使用 **RDKit 分子描述符 + XGBoost** 构建机器学习模型，从聚合物 SMILES 秒级预测隔膜的 **Tg（玻璃化转变温度）、Td（热分解温度）、离子电导率**，并配套 **Streamlit 网页筛选工具**，将筛选周期从数月压缩到秒级。

## 2. 项目背景

锂电池隔膜材料的开发依赖大量实验试错：合成 → 制膜 → 测试 → 筛选，单轮周期数月。材料创新需求（高能量密度、高安全性）要求更高效的筛选手段。本工具的目标是：**输入分子结构，快速预测性能，先筛后验**，让研究者把精力集中在最有希望的候选材料上。

## 3. 数据来源

整合自多个公开数据集 + 文献补充数据，覆盖 3 个性能目标：

| 目标 | 数据源 | 清洗后样本 | 说明 |
|------|--------|-----------|------|
| **Tg** | Polymer_UQ（Tg_Original + Tg_OOD_EXP） | 7125 | 实验 Tg，去重 |
| **Td** | Polymer_UQ（Td） | 1412 | 实验 Td，去重 |
| **离子电导率** | PolymerElectrolyteData（16009 条） | 14094 | 聚合物+盐+浓度+温度，log 尺度 |

**数据增强（文献依据）**：
- **5 条人工负样本**：PE/PP/PS+LiTFSI、PEO/PMMA 无盐（1e-10 S/cm），教会模型"什么结构不导电"
- **12 条文献补充数据**：PEO+LiClO4（8 条）+ PVDF-HFP+LiTFSI（4 条），修正盐类型排序与含氟聚合物外推偏差

原始数据源（E 盘 `BaiduNetdiskDownload/数据集`）经 `prepare_data.py` 清洗，产出 `data/` 目录下的干净训练数据。

## 4. 方法概述

### 两套特征体系

1. **纯聚合物本征性能（Tg/Td）**：RDKit 从 SMILES 计算 **12 个分子描述符**（MolWt、RingCount、NumRotatableBonds、TPSA、氢键供/受体、杂原子数等）

2. **聚合物电解质电导率（多组分）**：
   - RDKit 12 描述符
   - **盐类型混合编码**：Top5 one-hot（LiTFSI/LiTFO/LiClO4/LiAsF6/LiPF6）+ 6~15 名频率编码 + 其余归并
   - 盐浓度（molality）、温度（°C）
   - **物理约束特征**（VFT 理论）：`Tg`、`T_minus_Tg`、`is_glassy`
   - **硬规则特征**：`has_heteroatom`（纯烃链为 0）

### 模型架构

- **算法**：XGBoost 回归
- **调优**：GridSearchCV（n_estimators×learning_rate×max_depth×subsample = 108 组合）
- **验证**：5 折交叉验证 + 独立验证集早停（`early_stopping_rounds=50`）
- **数据划分**：训练 68% / 验证 12% / 测试 20%
- **电导率用 log 尺度**（跨度 15 个数量级）

### 物理约束设计

- **VFT 理论**：电导率取决于 T - Tg（测试温度距 Tg 多远），而非绝对温度 → `T_minus_Tg` 成为第 2 重要特征
- **硬规则**：纯烃链无法络合锂盐、本征不导电 → `has_heteroatom` 硬约束修正

## 5. 模型性能

| 目标 | 有效样本 | CV R² | 测试 R² | MAE | RMSE |
|------|---------|-------|---------|-----|------|
| **Tg** | 7125 | 0.839 ± 0.010 | 0.842 | 32.52 °C | 43.57 °C |
| **Td** | 1412 | 0.644 ± 0.064 | 0.667 | 39.73 °C | 54.59 °C |
| **离子电导率(log)** | 14094 | 0.937 ± 0.006 | 0.871 | 0.48 (log) | 0.69 (log) |

**特征重要性**：
- Tg/Td：`RingCount` 主导（环结构↑刚性↑ → Tg↑、热稳定性↑）
- 电导率：`MolWt` + `T_minus_Tg` + 盐类型 + `has_heteroatom` 共同决定

**可解释性**：SHAP 蜂群图（`reports/*_shap_beeswarm.png`）验证模型学到的规律与高分子物理/电化学理论一致。

## 6. 模型适用范围

### ✅ 适用范围（高置信度）
- 含 O/N 等杂原子的极性聚合物（如 PEO、PMMA、PAN 等）
- 聚合物 + 锂盐（LiTFSI、LiPF6、LiClO4）体系
- 温度范围：**25-80°C**
- 盐浓度范围：**0.5-2.0 mol/kg**
- 预测用途：材料预筛选、相对排序

### ❌ 失效范围（低置信度/不可用）
- 纯烃链聚合物（PE、PP、PS 等）→ 已通过硬规则修正为低电导率，但缺乏真实数据验证
- 含氟聚合物（PVDF、PVDF-HFP 等）→ 训练数据极少，预测为外推，仅供参考
- 不含杂原子的聚合物 → 模型强制预测为低电导率
- 温度超出 25-80°C 范围 → 外推不可靠
- 非锂盐体系（钠盐、镁盐等）→ 模型未训练
- 含增塑剂、多孔结构等复杂隔膜体系 → 模型仅基于纯聚合物 + 盐数据训练

### 💡 使用建议
- 本工具适用于从大量候选材料中快速筛选出值得进一步研究的对象
- 预测结果**不能替代实验验证**，建议结合文献和实验交叉验证
- 对预测结果存疑时，优先查阅相关文献或进行实验验证

## 7. 快速开始

### 环境准备

```bash
conda create -n separator python=3.13 -y
conda activate separator
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

### 训练模型

```bash
python prepare_data.py            # （可选）从原始数据清洗训练数据
python run_train.py               # 训练全部 3 个目标（Tg、Td、电导率）
python run_train.py --task Tg     # 只训练指定目标
```

### SHAP 可解释性分析

```bash
python run_shap.py                # 生成全部目标的 SHAP 蜂群图
```

### 启动 Web 工具

```bash
streamlit run app/streamlit_app.py
```

浏览器访问 `http://localhost:8501`。

**网页功能**：
- **Tab1 单分子预测**：输入 PSMILES 预测 Tg/Td；输入 PSMILES + 盐 + 浓度 + 温度预测电导率
- **Tab2 批量筛选**（3 种模式）：
  - 热性能批量预测（Tg/Td）+ 排序 + 导出 CSV
  - 电导率批量预测（固定条件：统一盐/浓度/温度，对多个分子批量预测）
  - 电导率批量预测（CSV 逐行条件：上传 CSV，每行含 SMILES/salt/molality/temperature）
- **Tab3 模型性能**：三目标性能指标 + 特征重要性可视化
- 侧边栏含"模型适用范围与失效范围"提示

## 8. 项目结构

```
BatterySeparatorML/
├── data/                    # 清洗后的训练数据（tg.csv / td.csv / conductivity.csv）
├── models/                  # 训练好的模型（Tg/Td/conductivity 的 .joblib + .json）
├── src/
│   ├── config.py            # 全局配置（两套任务、参数网格、特征列）
│   ├── features.py          # RDKit 描述符 + 盐编码 + 物理约束 + 硬规则特征
│   ├── train.py             # 训练 + 调优 + 交叉验证
│   ├── predict.py           # 预测接口（纯聚合物 + 电导率）
│   └── shap_analysis.py     # SHAP 可解释性分析
├── app/
│   └── streamlit_app.py     # Streamlit 网页筛选工具
├── prepare_data.py          # 数据清洗脚本
├── run_train.py             # 训练入口
├── run_shap.py              # SHAP 分析入口
├── run_external_validation.py  # 外部验证脚本
├── reports/                 # 案例报告 + SHAP 蜂群图
├── requirements.txt
├── README.md
└── PROJECT_NOTES.md         # 项目交接文档（完整技术细节）
```

## 9. 技术栈

**Python** · **RDKit**（分子描述符）· **XGBoost**（回归）· **scikit-learn**（调优/验证）· **SHAP**（可解释性）· **Streamlit**（Web 应用）· **pandas / numpy / matplotlib**

## 10. 作者与致谢

**作者**：韦宏宇

**数据致谢**：
- Polymer_UQ 数据集（Tg/Td）
- PolymerElectrolyteData 数据集（离子电导率）
- Table S3（外部验证，聚醚电解质基准数据）
- 文献补充数据（PEO-LiClO4、PVDF-HFP 电导率）

**学术声明**：本项目用于材料筛选辅助，预测结果需实验验证。
