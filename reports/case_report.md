# 锂电池隔膜筛选工具 - 案例报告

> 文档目的：展示本工具的方法论、关键成果与可解释性分析，作为对外汇报与项目交付的支撑材料。
> 数据基础：3 个目标（Tg、Td、离子电导率）已用正式数据训练完成（2026-08-20）。氧化稳定性因数据缺失暂缓。

---

## 1. 项目背景

隔膜是锂电池最关键的部件之一，**性能直接决定电池安全**。传统上依赖实验筛选候选材料，**周期长（数月）、效率低**，难以应对材料创新需求。

本项目使用 **机器学习从聚合物分子结构（PSMILES）快速预测隔膜关键性能**，将筛选周期从数月压缩到秒级，加速材料设计。

### 关键性能指标（4 个目标）

| 指标 | 物理意义 | 工程意义 |
|------|----------|----------|
| **Tg** 玻璃化转变温度 | 聚合物链段开始运动的温度 | 决定工作温度上限 |
| **Td** 热分解温度 | 材料开始降解的温度 | 决定热稳定性与安全性 |
| **离子电导率** | 锂离子在隔膜中迁移的快慢 | 决定电池倍率与快充能力 |
| **氧化稳定性** | 抵抗高电压氧化的能力 | 决定高压电池安全性 |

---

## 2. 方法论

### 2.1 特征工程：RDKit 12 个分子描述符

从 PSMILES（聚合物 SMILES）出发，用 RDKit 提取 12 个分子描述符：

| 类别 | 描述符 | 物理意义 |
|------|--------|----------|
| 体积 | `MolWt`, `HeavyAtomCount`, `NumHeteroatoms` | 分子大小、杂原子数 |
| 拓扑 | `NumRotatableBonds` | 链柔性 |
| 氢键 | `NumHDonors`, `NumHAcceptors` | 极性相互作用 |
| 环 | `NumAromaticRings`, `NumSaturatedRings`, `NumAliphaticRings`, `RingCount` | 链刚性、环结构 |
| 极性/亲疏 | `TPSA`, `MolLogP` | 极性表面积、疏水性 |

**PSMILES 特殊处理**：
- 含 `[*]` 虚拟原子，RDKit 可直接解析（已验证）
- 含 `[R]` 标记的部分结构无法解析，**已自动剔除**
- `FractionCsp3` 等部分描述符在含 `[*]` 分子上会抛异常，**已规避**

### 2.2 建模：XGBoost + GridSearchCV + 5 折 CV

**流程**：
1. **数据划分**：训练 68% / 验证 12%（早停监控）/ 测试 20%
2. **5 折交叉验证**（`KFold(5, shuffle=True, random_state=42)`），评估 CV R² 与 RMSE
3. **GridSearchCV 调优**（`scoring='r2'`，超参数网格：`n_estimators×learning_rate×max_depth×subsample` = 4×3×3×3 = 108 种组合）
4. **最佳参数重训**：`early_stopping_rounds=50`，**用独立验证集监控**（避免数据泄漏）
5. **测试集评估**：MAE、RMSE、R²

**超参数网格**：
```python
{
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
}
```

---

## 3. 正式训练结果（3 个目标）

### 3.1 性能汇总

| 目标 | 有效样本 | 特征数 | CV R² | 测试 R² | 测试 MAE | 测试 RMSE |
|------|---------|--------|-------|---------|----------|-----------|
| **Tg** | 7125 | 12 | 0.839 ± 0.010 | 0.842 | 32.52 °C | 43.57 °C |
| **Td** | 1412 | 12 | 0.644 ± 0.064 | 0.667 | 39.73 °C | 54.59 °C |
| **电导率(log)** | 14094 | 25 | 0.937 ± 0.006 | 0.871 | 0.48 (log) | 0.69 (log) |

### 3.2 特征重要性（XGBoost gain）

**Tg 特征重要性 Top 5**：

| 排序 | 特征 | 重要性 |
|------|------|--------|
| 1 | **RingCount** | **0.731** |
| 2 | NumRotatableBonds | 0.074 |
| 3 | NumHDonors | 0.052 |
| 4 | NumAromaticRings | 0.032 |
| 5 | TPSA | 0.020 |

**Td 特征重要性 Top 5**：RingCount(0.36) > NumSaturatedRings(0.11) > NumAromaticRings(0.087) > NumRotatableBonds(0.078) > NumHAcceptors(0.077)

**电导率特征重要性 Top 8**（含物理约束特征）：

| 排序 | 特征 | 重要性 | 含义 |
|------|------|--------|------|
| 1 | MolWt | 0.200 | 分子量 |
| 2 | **T_minus_Tg** | 0.127 | 测试温度 - Tg（VFT 物理量） |
| 3 | NumHeteroatoms | 0.085 | 杂原子数 |
| 4 | TPSA | 0.079 | 极性表面积 |
| 5 | salt_LiPF6 | 0.078 | 盐类型（LiPF6） |
| 6 | HeavyAtomCount | 0.071 | 重原子数 |
| 7 | salt_LiTFSI | 0.063 | 盐类型（LiTFSI） |
| 8 | temperature | 0.055 | 温度 |

**关键发现**：
- **Tg/Td 均由 RingCount 主导** → 环结构提高链刚性，同时提升 Tg 与热稳定性
- **电导率由分子量 + T_minus_Tg（测试温度距 Tg 多远）+ 盐类型共同决定**
  - **T_minus_Tg 重要性(0.127)远超温度本身(0.055)** → 物理约束特征有效，符合 VFT 理论（电导率取决于 T 相对 Tg 的位置，而非绝对温度）

### 3.3 SHAP 可解释性分析

| 目标 | 蜂群图 | 关键结论 |
|------|--------|----------|
| Tg | `Tg_shap_beeswarm.png` | RingCount 主导，环↑→Tg↑ |
| Td | `Td_shap_beeswarm.png` | 环/芳香环↑→热稳定性↑ |
| 电导率 | `conductivity_shap_beeswarm.png` | 温度↑、特定盐→电导率↑ |

**与高分子物理/电化学理论的一致性**：
- 环结构增加链段运动阻力 → Tg 升高 ✅
- 芳香环增强热稳定性 → Td 升高 ✅
- 温度升高 → 离子迁移加快 → 电导率升高（Arrhenius/VFT 规律）✅
- 盐类型影响离子解离与迁移 → 电导率差异 ✅

→ 三个模型学到的规律均与领域理论高度吻合，证明特征工程与建模方法合理。

---

## 4. 工程应用：Streamlit 网页工具

启动 `streamlit run app/streamlit_app.py` 后可使用以下功能：

| 模块 | 功能 |
|------|------|
| 单分子预测 | 输入 PSMILES，预测所有目标性能 |
| 批量筛选 | 输入多个候选分子，按指定目标排序筛选，导出 CSV |
| 模型性能 | 查看每个目标的 CV R²、测试 R²、MAE、RMSE、特征重要性 |

---

## 5. 待补充内容

- [ ] **氧化稳定性目标**（数据缺失；后续可用 HOMO/LUMO 能级 proxy 或找新数据）
- [ ] **案例研究**：选 2-3 个真实候选聚合物，预测 + 解读
- [ ] **跨目标相关性分析**：环结构是否对所有性能都重要

---

## 6. 代码与可复现性

- **环境**：`conda env pythonproject2`（Python 3.13.14）
- **数据整理**：`python prepare_data.py`（从原始数据清洗出 data/*.csv）
- **训练**：`python run_train.py`（或 `--task Tg/Td/conductivity`）
- **SHAP**：`python run_shap.py`（生成 `reports/{目标}_shap_beeswarm.png`）
- **网页**：`streamlit run app/streamlit_app.py`

详细方法、踩坑经验与项目交接见 `PROJECT_NOTES.md`。
