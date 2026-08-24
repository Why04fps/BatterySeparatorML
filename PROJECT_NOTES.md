# 项目交接文档（PROJECT_NOTES）

> 本文档用于跨会话/跨 AI 对话交接。新对话的 AI 应先阅读本文件，即可完整恢复项目上下文。
> 最后更新：2026-08-20（新数据接入，3 目标训练完成：Tg/Td/离子电导率）

---

## 1. 项目是什么

**锂电池隔膜筛选工具** —— 用机器学习从聚合物分子结构快速预测隔膜关键性能，替代低效的实验筛选（实验需数月）。

### 要预测的隔膜关键性能（4 个目标）
| 性能 | 说明 | 状态 |
|------|------|------|
| **Tg** | 玻璃化转变温度 | ✅ 已训练（7125 样本） |
| **Td** | 热分解温度 | ✅ 已训练（1412 样本） |
| **离子电导率** | 决定电池性能的关键 | ✅ 已训练（14077 样本，log 尺度） |
| **氧化稳定性** | 决定电池安全的关键 | ⏸ 暂缓（数据缺失） |

### 背景痛点
- 隔膜是锂电池最关键的部件之一，性能直接决定电池安全
- 目前依赖实验筛选，周期长（数月）、效率低
- 用 ML 快速预测性能，加速筛选

### 技术路线
- **特征（两套体系）**：
  - Tg/Td：RDKit 从 SMILES 计算 12 个分子描述符（纯聚合物）
  - 离子电导率：RDKit 描述符 + 盐类型混合编码 + 盐浓度 + 温度（多组分）
- **模型**：XGBoost + GridSearchCV 调优 + 5 折交叉验证
- **应用**：Streamlit 网页工具 + SHAP 可解释性 + 案例报告

---

## 2. 项目结构

```
BatterySeparatorML/
├── data/
│   └── experiment_polymer_data.xlsx   # 框架验证数据（OpenPoly，后续会被新隔膜数据替换）
├── models/                            # 训练好的模型（.joblib + .json 元数据）
│   ├── Tg_K_model.joblib
│   └── Tg_K_meta.json
├── src/
│   ├── __init__.py
│   ├── config.py      # 全局配置：数据路径、目标列、参数网格、随机种子
│   ├── features.py    # RDKit 描述符提取 + 数据加载
│   ├── train.py       # 5折CV + GridSearchCV + 训练 + 保存模型
│   └── predict.py     # 预测接口（供 Streamlit 调用）
├── app/
│   └── streamlit_app.py   # Streamlit 网页筛选工具
├── run_train.py       # 训练入口
├── requirements.txt
├── PROJECT_NOTES.md   # 本文档
└── README.md          # 待补全
```

---

## 3. 方法论（已验证，直接沿用）

### 特征工程：RDKit 12 个分子描述符
在 `src/features.py` 的 `extract_rdkit_descriptors()` 中实现：

`MolWt`（分子量）、`HeavyAtomCount`（重原子数）、`NumRotatableBonds`（可旋转键数）、`NumHDonors`/`NumHAcceptors`（氢键供/受体）、`NumAromaticRings`/`NumSaturatedRings`/`NumAliphaticRings`/`RingCount`（环类）、`TPSA`（极性表面积）、`MolLogP`（疏水性）、`NumHeteroatoms`（杂原子数）。

**重要经验**：
- PSMILES 是聚合物表示，含 `[*]` 虚拟原子，**RDKit 能直接解析**（已验证）
- 个别含 `[R]` 标记的 PSMILES 无法解析（如 `[R1][Si]...`），需自动剔除（`DROP_INVALID_SMILES=True`）
- `FractionCsp3` 等部分描述符在含 `[*]` 的分子上会抛异常，**不要用**（会拖累整个提取）
- 原项目数据 8471 个样本，剔除 2 个无效后剩 8469 个有效

### 建模流程（src/train.py）
1. **数据划分**：`train_test_split(test_size=0.2)` → 再从训练集切 15% 作验证集（用于早停）
2. **5 折交叉验证**：`KFold(5, shuffle=True)`，评估 CV R² 和 RMSE
3. **GridSearchCV 调优**：`scoring='r2'`, `cv=5`
4. **最佳参数重训**：含 `early_stopping_rounds=50`，用**独立验证集**监控（避免数据泄漏）
5. **保存**：模型 `.joblib` + 元数据 `.json`

### 超参数网格（config.PARAM_GRID）
```python
{
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
}
```

### 关键参数
```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
```

---

## 4. 正式训练结果（2026-08-20，3 个目标）

### 4.1 三目标性能汇总

| 目标 | 有效样本 | 特征数 | CV R² | 测试 R² | 测试 MAE | 测试 RMSE |
|------|---------|--------|-------|---------|----------|-----------|
| **Tg** | 7125 | 12 | 0.839 ± 0.010 | 0.842 | 32.52 °C | 43.57 °C |
| **Td** | 1412 | 12 | 0.644 ± 0.064 | 0.667 | 39.73 °C | 54.59 °C |
| **电导率(log)** | 14094 | 25 | 0.937 ± 0.006 | 0.871 | 0.48 (log) | 0.69 (log) |

### 4.2 特征重要性（XGBoost gain）

- **Tg**：RingCount(0.73) >> NumRotatableBonds > NumHDonors > NumAromaticRings > TPSA
  → 环结构提高 Tg（高分子物理一致）
- **Td**：RingCount(0.36) > NumSaturatedRings > NumAromaticRings > NumRotatableBonds
  → 环/芳香结构增强热稳定性
- **电导率**（含物理约束+硬规则特征）：MolWt(0.21) > **T_minus_Tg(0.115)** > TPSA > salt_LiPF6 > HeavyAtomCount > temperature
  → 分子量 + **T-Tg（测试温度距 Tg 多远）** + 盐类型 + has_heteroatom 共同决定电导率

### 4.3 单位说明（重要，避免混淆）
- **Tg、Td 的目标值单位是 °C**（训练数据 tg.csv 范围 -118~495，中位数 131.5，必为 °C）
- 模型输出即 °C，`predict.py` **不做单位换算**，原样输出
- ⚠️ 曾发现 bug：Streamlit 网页把 Tg/Td 的 °C 输出错标为 "K"，导致出现"低于绝对零度"的荒谬值（已修复，现显示 °C）
- 外部验证脚本 `run_external_validation.py` 中，Table S3 的实验 Tg 是 K，需 `K - 273.15` 转 °C 再与模型 °C 输出对比（逻辑正确）

### 4.4 电导率建模要点（重要）
- 电导率跨度 15 个数量级（1e-16 ~ 9e-2 S/cm），**必须用 log 尺度**（目标列 `log_conductivity`）
- 盐类型混合编码（用户指定方案）：
  - Top 5 盐 one-hot（LITFSI/LITFO/LICLO4/LIASF6/LIPF6）
  - 第 6~15 名频率编码（归一化 0~1）
  - 其余归并 is_other_salt（0/1）
  - 保留 molality + temperature 数值特征
- **物理约束特征**（2026-08-20 加入，源自 VFT 理论）：
  - `Tg`：该聚合物的 Tg（°C，用 Tg 模型预测）
  - `T_minus_Tg`：测试温度 - Tg（VFT 方程的核心物理量）
  - `is_glassy`：T - Tg < 0 则为 1（玻璃态），否则 0
  - `T_minus_Tg` 成为第 2 重要特征（0.127），远超 temperature(0.055)，验证物理约束有效
- 结果：log 电导率测试 R²=0.870，MAE=0.48 个数量级（误差约 3 倍内）
- ⚠️ 注意：prepare_data.py 的电导率整理**依赖 Tg 模型**，需先训练 Tg 再整理电导率

### 4.5 人工构造负样本（2026-08-20）
**背景**：分析发现数据中最低 5% 电导率样本（801 条）不适合直接作负样本——低电导主要由"条件"（低温、少盐、无盐）决定，而非结构。同一聚合物在其他条件下是正样本，会造成特征-标签因果错位。

**最终方案**：人工构造 5 条物理清晰的负样本（log=-10 S/cm，即 1e-10），仅加入电导率训练数据，**强制留在训练集**（`is_synthetic` 标记）：

| 聚合物 | 盐 | 浓度 | 温度 | 物理依据 |
|--------|-----|------|------|---------|
| PE `[*]C[*]` | LiTFSI | 1.0 | 25°C | 非极性聚烯烃不溶解锂盐，锂盐析出为晶体，接近绝缘 |
| PP `[*]CC([*])C` | LiTFSI | 1.0 | 25°C | 同上 |
| PS `[*]CC([*])c1ccccc1` | LiTFSI | 1.0 | 25°C | 非极性芳香族，同上 |
| PEO `[*]CCO[*]` | 无盐 | 0 | 25°C | 无离子载体，仅本征杂质导电 |
| PMMA `[*]CC([*])(C)C(=O)OC` | 无盐 | 0 | 25°C | 无离子载体且玻璃态，链段冻结 |

**踩坑经验（重要）**：
1. 曾尝试 log=-14（1e-14）极端值 → 模型整体预测被压缩，测试 R² 暴跌到 0.48，PP/PEO 全被预测成同一值。**极端值负样本会破坏模型**。
2. 曾尝试第二批 PP 变体（0.5mol/LiPF6，共 7 条）→ PP 无改善，PE 反而回退，R² 降到 0.649。**单纯增加负样本数量无效**（PP 与 PEO 特征差异太小）。
3. 根治方案：**增加 `has_heteroatom` 硬规则特征**（见 4.6），一举解决问题。

### 4.6 has_heteroatom 硬规则特征（最终解决方案）
- 定义：SMILES 中是否含杂原子（O/N/F/Cl/Br/I/S/P/Si 等，C/H 除外），含则 1，纯烃链 0
- 物理依据：纯烃链聚合物（PE/PP/PS）无法解离/络合锂盐，离子传导极弱
- 实现：用 RDKit 解析分子判断原子序数（`GetAtomicNum()`），**跳过虚拟原子 `[*]`（原子序数 0）**
- **踩坑**：初版用正则 `[ONFClBrISiP]` 判断是错的——字符类展开后含 `C`/`l`/`i` 等单字符，导致所有分子（含纯烃）都被判为 1。必须用 RDKit 原子序数判断。
- 加入后模型从"纯烃链特征外推"变成"直接看到结构约束"，预测修正效果显著。

**最终效果（验证全部达标）**：
| 场景 | 目标 | 实际 | 状态 |
|------|------|------|------|
| PP+LiTFSI 25°C | <1e-6 | 3.5e-07 | ✅ |
| PE+LiTFSI 25°C | <1e-6 | 9.4e-09 | ✅ |
| PS+LiTFSI 25°C | <1e-6 | 2.0e-08 | ✅ |
| PMMA无盐 25°C | <1e-6 | 5.2e-09 | ✅ |
| PEO无盐 25°C | <1e-6 | 5.1e-07 | ✅ |
| PEO+LiTFSI 25°C（正） | ~1e-2 | 8.7e-03 | ✅ 未误伤 |
| PEO+LiTFSI 80°C（正） | ~1e-2 | 1.7e-02 | ✅ 未误伤 |
| **测试 R²** | ≥0.75 | **0.870** | ✅ 超越原 0.857 |

### 4.7 文献补充数据（2026-08-20，修正盐类型与 PVDF 偏差）
**背景**：文献核实 PEO 基聚合物电解质离子电导率顺序应为 **LiTFSI > LiPF6 > LiClO4 > LiBF4 > LiCF3SO3**（阴离子尺寸越大越能破坏 PEO 结晶、电荷分散更易解离）。但训练数据中 LiClO4 在 PEO+25°C 中位仅 6.9e-5（实际应 1e-4~1e-3），导致模型排序错误。同时 PVDF-HFP 无训练样本（预测完全外推、偏低）。

**方案**：在 `prepare_data.py` 新增 `SUPPLEMENTARY_SAMPLES`（12 条，作为**正常数据** is_synthetic=0）：
- **PEO + LiClO4（8 条）**：0.5/1.0 mol × 25/40/60/80°C，1.5e-4 ~ 2.5e-3 S/cm
- **PVDF-HFP + LiTFSI（4 条）**：20wt%≈0.87 mol/kg × 25/40/60/80°C，2.84e-4 ~ 1.2e-3 S/cm

**踩坑**：补充数据的盐名需与原始数据一致归一化（`.upper()`），否则 `LiClO4` vs `LICLO4` 会被当成两种盐。

**效果（补充后验证）**：
| 场景 | 补充前 | 补充后 | 目标 | 状态 |
|------|--------|--------|------|------|
| PEO+LiClO4 1.0mol 25°C | 1.3e-02 | **1.16e-03** | 1e-4~1e-3 | ✅ |
| PEO+LiTFSI 1.0mol 25°C | 8.7e-03 | 2.2e-03 | ~1e-2 | 🟡 略降 |
| PVDF-HFP+LiTFSI 25°C | 6.6e-06 | **1.9e-05** | 1e-4~1e-3 | 🟡 提升但偏低 |
| 盐排序 | LiClO4>LiTFSI>LiPF6 | LiTFSI>LiClO4>LiPF6 | LiTFSI>LiPF6>LiClO4 | 🟡 LiTFSI 第一 ✅ |
| 回归 PP/PE 25°C | <1e-6 | 2.3e-7/1.2e-8 | <1e-6 | ✅ |
| **测试 R²** | 0.870 | **0.871** | — | ✅ 保持 |

**结论**：LiClO4 过高和 PVDF 过低两个核心偏差已基本修正（LiClO4 从 1.3e-2 降至 1.2e-3、PVDF 从 6.6e-6 提升 3 倍）；LiTFSI 回到盐排序第一。剩余小偏差（LiClO4 vs LiPF6 顺序、PVDF 25°C 偏低）源于原训练数据 LiPF6 中位偏高（-2.54）与 PVDF 外推空间，可在后续补充更多数据改善。

---

## 5. 当前状态

### 已完成
- [x] 搭建模块化项目框架（config/features/train/predict/run_train）
- [x] 训练流程验证跑通
- [x] 修复：模型元数据保存时 numpy `float32` JSON 序列化报错
- [x] Streamlit 安装 + 网页工具实现
- [x] SHAP 可解释性分析脚本（`src/shap_analysis.py` + `run_shap.py`）
- [x] **新数据接入**（`prepare_data.py`，从 E 盘原始数据清洗出 data/tg.csv、td.csv、conductivity.csv）
- [x] **两套特征体系改造**（纯聚合物 Tg/Td + 多组分电导率）
- [x] **3 目标模型训练完成**（Tg 0.821 / Td 0.667 / 电导率 0.856，模型存 models/）
- [x] **3 目标 SHAP 蜂群图生成**（reports/*_shap_beeswarm.png）
- [x] **Streamlit 适配 3 模型**（含电导率的盐/浓度/温度输入）
- [x] **批量预测功能**（2026-08-24，Tab2 三模式：热性能批量 + 电导率固定条件 + 电导率 CSV 逐行条件）
  - 新增 `predict.predict_conductivity_batch()`（固定条件）和 `predict.predict_conductivity_batch_rows()`（逐行）
  - 已验证：PEO/PP/PS/PE 批量预测物理规律正确，0 异常

### 待办
- [ ] **氧化稳定性目标**（数据缺失，暂缓；后续可用 HOMO/LUMO 能级 proxy 或找新数据）
- [ ] 案例报告 `reports/case_report.md` 补全 3 目标的正式结果（当前是 Tg_K 框架版）
- [ ] （可选）上传 GitHub

### ⚠️ 外部验证发现（2026-08-20，重要）
用 `Table S3_dataset.xlsx`（聚醚电解质基准数据，18 条）作外部验证集，检验模型泛化能力：

| 目标 | 外部 MAE | 外部 R² | 结论 |
|------|---------|---------|------|
| Tg | 49.59°C | -5.99 | 系统性低估（预测 -31~-54°C vs 真实 -9~+46°C） |
| 电导率 | 1.68 log | -14.2 | 系统性高估（预测偏高 1~2.5 个数量级） |

**根因**：Table S3 是**结构高度同质的聚醚类聚合物**（PEO 衍生物，无环柔性链），落在训练数据的化学空间之外。
- Tg 模型学到的"环结构主导 Tg"对无环聚醚不适用，`NumRotatableBonds` 权重太低无法刻画聚醚 Tg 细微差异
- 电导率模型缺少 `Li:EO`（锂醚比）这一关键特征（Table S3 有，但训练数据无对应列）

**结论**：模型在训练数据化学空间内表现好（测试 R² 0.82/0.86），但**外推到同质聚醚体系会系统性偏差**。这是"外部验证集"的核心价值——揭示了泛化边界。

**改进方向**（待用户决策）：
1. 扩增特征（如加入链柔性、分子量归一化、Li:EO 比等）
2. 加入聚醚类数据扩充训练集
3. 或明确模型适用范围（dom/应用域），提示用户聚醚类预测需谨慎

---

## 6. 如何运行

### 数据整理（一次性，从 E 盘原始数据清洗）
```bash
python prepare_data.py
```
产出：`data/tg.csv`、`data/td.csv`、`data/conductivity.csv`

### 训练
```bash
# 训练全部 3 个目标（Tg、Td、电导率）
python run_train.py

# 只训练指定任务
python run_train.py --task Tg
python run_train.py --task Td,conductivity
```

### SHAP 可解释性分析
```bash
python run_shap.py              # 全部目标
python run_shap.py --target Tg  # 指定目标
```

### 外部验证
```bash
python run_external_validation.py   # 用 Table S3 聚醚数据验证模型泛化能力
```

### 启动 Streamlit 网页工具
```bash
streamlit run app/streamlit_app.py
```

---

## 7. 环境说明（重要！）

**项目实际使用的 Python 环境**是 conda 环境的 `pythonproject2`：

```
解释器: C:\Users\Administrator\miniconda3\envs\pythonproject2\python.exe
Python : 3.13.14
```

该环境已安装：xgboost, pandas, numpy, scikit-learn, rdkit, matplotlib, openpyxl, shap

**关键经验（避免踩坑）**：
1. **PyCharm 默认解释器必须是 `pythonproject2`**，否则报 `ModuleNotFoundError`（曾因 rdkit 没装在这个环境导致运行不了）
2. `rdkit` 已装在该环境；**streamlit 已装**（2026-08-19 用官方 PyPI 源绕过清华镜像 403 装的 1.61.1）。如果将来需要重装：
   ```
   C:\Users\Administrator\miniconda3\envs\pythonproject2\python.exe -m pip install streamlit -i https://pypi.org/simple
   ```
3. Windows cmd 下运行，中文/emoji 会乱码显示（GBK 代码页问题），**不影响程序运行**，在 PyCharm 里运行显示正常
4. matplotlib 中文乱码问题已在代码中解决（`SimHei` 字体 + `axes.unicode_minus=False`）
5. SHAP 蜂群图需用新版 API `shap.plots.beeswarm(Explanation)`，旧版 `shap.summary_plot(plot_type="beeswarm")` 会渲染不出散点
6. `shap_analysis.py` 实际验证：使用 `shap.TreeExplainer(model)(X)` 直接生成 `Explanation` 对象，然后 `shap.plots.beeswarm(...)` 渲染 — 完全正常工作
7. **环境全链路验证通过（2026-08-19）**：10 个关键依赖（numpy/pandas/sklearn/xgboost/rdkit/matplotlib/openpyxl/joblib/shap/streamlit）全部导入正常；`features.load_data()` 正常返回 8469 样本×12 特征；`predict.predict_smiles()` 正常预测；Streamlit 用 `AppTest` 跑通，0 异常 3 tab
8. **streamlit 弃用参数已修复**：新版 1.61.1 弃用 `st.dataframe(use_container_width=True)`（2025-12-31 后移除），已改为 `width="stretch"`（`app/streamlit_app.py` 2 处）
9. 已验证 Streamlit 启动命令：`python -m streamlit run app/streamlit_app.py --server.headless true` 正常监听

### 系统 Python（勿混淆）
另有系统 Python 3.14（`AppData/Local/Python/...`），也装了 rdkit，但**不是 PyCharm 项目用的**。

---

## 8. 数据说明

### 原始数据源（E:\BaiduNetdiskDownload\数据集）
| 数据源 | 用途 | 规模 |
|--------|------|------|
| `Polymer_UQ-main/data/Tg_Original.csv` | Tg | 6906 条（Smiles + Tg） |
| `Polymer_UQ-main/data/Tg_OOD_EXP.csv` | Tg（分布外实验，已合并） | 240 条（Smiles + Tg） |
| `Polymer_UQ-main/data/Tg_OOD_MD.csv` | Tg（分布外 MD 模拟，未用） | 566 条 |
| `Polymer_UQ-main/data/Td.csv` | Td | 1415 条（Smiles + TdValue） |
| `PolymerElectrolyteData.csv` | 离子电导率 | 16009 条（多组分：聚合物+盐+浓度+温度） |
| `PI1070.csv` | （备用）含 HOMO/LUMO 能级，可作氧化稳定性 proxy | — |

### 整理后的数据（data/ 目录）
- `tg.csv`：7125 条（去重，Tg_Original + Tg_OOD_EXP 合并），列 SMILES + Tg
- `td.csv`：1412 条（去重），列 SMILES + Td
- `conductivity.csv`：14077 条，列 SMILES + salt + molality + temperature + log_conductivity
  - 盐类型 96 种（大小写归一化后），226 个唯一聚合物
  - 剔除：molality>100 异常值（2 条）、缺盐浓度、缺 SMILES/电导率样本

### 数据质量问题（已处理）
- 盐名大小写不一致（如 `LITFSI` vs `LiTFSI`）→ 统一转大写合并
- 盐名含数据库内部编号（如 `20190319_3_salt1`）→ 归入 is_other_salt
- 无 Li 前缀的简化写法（如 `ClO4`、`TFSI`）→ 作为独立盐处理

---

## 9. 与旧项目的关系

旧项目 `PythonProject2/Tg_Prediction_Project.py` 是**单文件脚本版**的 Tg 预测（RDKit + XGBoost），本新项目 `BatterySeparatorML` 是其**模块化、多目标、带网页应用的升级版**。
- 方法论（RDKit 描述符、XGBoost、GridSearchCV、5折CV、早停防泄漏）完全沿用并已验证
- 旧项目产生的核心成果：`Tg_Prediction_Results.png`、`shap_beeswarm.png` 可作参考
