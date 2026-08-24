# -*- coding: utf-8 -*-
"""
数据整理脚本
====================
从 E 盘原始数据集清洗出 3 个干净的训练数据文件，输出到 data/ 目录：

1. data/tg.csv          纯聚合物 Tg（Smiles + Tg）
2. data/td.csv          纯聚合物 Td（Smiles + Td）
3. data/conductivity.csv 聚合物电解质电导率（SMILES + 盐 + 浓度 + 温度 + log电导率）

用法：
    python prepare_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# 将项目根加入 sys.path，以便导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_BASE = r"E:\BaiduNetdiskDownload\数据集"
OUT_DIR = os.path.join(PROJECT_ROOT, "data")

# ==================== 人工构造的负样本（电导率） ====================
# 渐进式方案：先加 5 条（log=-10）验证效果，再逐步增加。
# 格式: (SMILES, salt, molality, temperature, conductivity_S_cm)
# 来源与物理依据见 PROJECT_NOTES.md 第 4.5 节。
NEGATIVE_SAMPLES = [
    # --- 第一批（5 条，最终方案）：各结构代表 + LiTFSI 或无盐，25°C，log=-10 ---
    ("[*]C[*]", "LiTFSI", 1.0, 25, 1e-10),                  # PE + LiTFSI 25°C
    ("[*]CC([*])C", "LiTFSI", 1.0, 25, 1e-10),              # PP + LiTFSI 25°C
    ("[*]CC([*])c1ccccc1", "LiTFSI", 1.0, 25, 1e-10),       # PS + LiTFSI 25°C
    ("[*]CCO[*]", None, 0.0, 25, 1e-10),                    # PEO 无盐 25°C
    ("[*]CC([*])(C)C(=O)OC", None, 0.0, 25, 1e-10),         # PMMA 无盐 25°C
    # 注：第二批 PP 变体（0.5mol/LiPF6）经验证无效（5→7 条导致 PE 回退、R² 下降），已移除。
    # 见 PROJECT_NOTES.md 第 4.6 节。
]

# --- 备用负样本（验证第一批效果后再逐步放开，值域调整为 1e-10）---
_NEGATIVE_SAMPLES_BACKUP = [
    # 第 1 类扩展：非极性聚烯烃 + LiTFSI（多浓度/多温度）
    ("[*]C[*]", "LiTFSI", 0.5, 25, 1e-10),      # PE
    ("[*]C[*]", "LiTFSI", 0.5, 60, 1e-10),      # PE
    ("[*]C[*]", "LiTFSI", 1.0, 60, 1e-10),      # PE
    ("[*]CC([*])C", "LiTFSI", 0.5, 60, 1e-10),  # PP
    ("[*]CC([*])C", "LiTFSI", 1.0, 60, 1e-10),  # PP
    # 第 2 类扩展：非极性芳香族 + LiTFSI
    ("[*]CC([*])c1ccccc1", "LiTFSI", 0.5, 25, 1e-10),  # PS
    ("[*]CC([*])c1ccccc1", "LiTFSI", 1.0, 60, 1e-10),  # PS
    ("[*]CC([*])c1ccccc1", "LiTFSI", 0.5, 60, 1e-10),  # PS
    # 第 3 类扩展：纯聚合物无盐（多温度）
    ("[*]CCO[*]", None, 0.0, 60, 1e-10),                  # PEO 无盐 60°C
    ("[*]CC([*])(C)C(=O)OC", None, 0.0, 60, 1e-10),       # PMMA 无盐 60°C
    # 第 4 类：PP + 不同盐（排除盐类型影响）
    ("[*]CC([*])C", "LiPF6", 1.0, 60, 1e-10),   # PP + LiPF6
    ("[*]CC([*])C", "LiClO4", 1.0, 25, 1e-10),  # PP + LiClO4
    ("[*]CC([*])C", "LiClO4", 1.0, 60, 1e-10),  # PP + LiClO4
]


# ==================== 文献补充数据（电导率，正常数据非负样本） ====================
# 修正两个数据偏差：
#   1. LiClO4 在 PEO 中 25°C 电导率偏低（训练数据中位 6.9e-5，文献应为 1e-4~1e-3）
#   2. PVDF-HFP 无训练样本（含氟聚合物，预测完全外推）
# 来源依据见 PROJECT_NOTES.md 第 4.7 节。
# 格式: (SMILES, salt, molality, temperature, conductivity_S_cm)
# PVDF 浓度的单位换算：20wt% LiTFSI（MW≈287 g/mol）≈ 0.87 mol/kg
PVDF_HFP_SMILES = "[*]CC(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)([*])-C(F)(F)-C(CF)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-C(F)(F)-[*]"

SUPPLEMENTARY_SAMPLES = [
    # --- PEO + LiClO4（8 条，修正盐类型排序偏差）---
    ("[*]CCO[*]", "LiClO4", 0.5, 25, 1.5e-4),
    ("[*]CCO[*]", "LiClO4", 1.0, 25, 2.0e-4),
    ("[*]CCO[*]", "LiClO4", 0.5, 40, 5.0e-4),
    ("[*]CCO[*]", "LiClO4", 1.0, 40, 6.0e-4),
    ("[*]CCO[*]", "LiClO4", 0.5, 60, 1.0e-3),
    ("[*]CCO[*]", "LiClO4", 1.0, 60, 1.2e-3),
    ("[*]CCO[*]", "LiClO4", 0.5, 80, 2.0e-3),
    ("[*]CCO[*]", "LiClO4", 1.0, 80, 2.5e-3),
    # --- PVDF-HFP + LiTFSI（4 条，20wt% ≈ 0.87 mol/kg）---
    (PVDF_HFP_SMILES, "LiTFSI", 0.87, 25, 2.84e-4),
    (PVDF_HFP_SMILES, "LiTFSI", 0.87, 40, 5.0e-4),
    (PVDF_HFP_SMILES, "LiTFSI", 0.87, 60, 8.0e-4),
    (PVDF_HFP_SMILES, "LiTFSI", 0.87, 80, 1.2e-3),
]


def build_supplementary_samples():
    """将文献补充数据转换为 DataFrame（is_synthetic=0，作为正常训练数据）"""
    rows = []
    for smiles, salt, molality, temp, cond in SUPPLEMENTARY_SAMPLES:
        rows.append({
            "SMILES": smiles,
            "salt": salt,
            "molality": molality,
            "temperature": temp,
            "log_conductivity": float(np.log10(cond)),
            "is_synthetic": 0,  # 正常补充数据（可进入测试集）
        })
    return pd.DataFrame(rows)


def build_negative_samples():
    """将人工负样本转换为 DataFrame（含 log_conductivity 和 is_synthetic 标记）"""
    rows = []
    for smiles, salt, molality, temp, cond in NEGATIVE_SAMPLES:
        rows.append({
            "SMILES": smiles,
            "salt": salt,
            "molality": molality,
            "temperature": temp,
            "log_conductivity": float(np.log10(cond)),
            "is_synthetic": 1,  # 人工构造标记（用于训练划分时强制留在训练集）
        })
    return pd.DataFrame(rows)


def prepare_tg_td():
    """整理 Tg、Td 数据（纯聚合物 → 性能）"""
    print("=" * 60)
    print("整理 Tg / Td 数据")
    print("=" * 60)

    uq_data = os.path.join(SRC_BASE, "Polymer_UQ-main", "Polymer_UQ-main", "data")
    tg_path = os.path.join(uq_data, "Tg_Original.csv")
    tg_ood_path = os.path.join(uq_data, "Tg_OOD_EXP.csv")  # 分布外实验 Tg（补充数据）
    td_path = os.path.join(uq_data, "Td.csv")

    tg = pd.read_csv(tg_path, encoding="utf-8")
    tg_ood = pd.read_csv(tg_ood_path, encoding="utf-8")
    td = pd.read_csv(td_path, encoding="utf-8")

    # 规范化 Tg（合并 Original + OOD 实验数据）
    tg = tg[["Smiles", "Tg"]].rename(columns={"Smiles": "SMILES"})
    tg_ood = tg_ood[["Smiles", "Tg"]].rename(columns={"Smiles": "SMILES"})
    tg = pd.concat([tg, tg_ood], ignore_index=True)
    tg = tg.dropna().drop_duplicates(subset="SMILES").reset_index(drop=True)
    tg_out = os.path.join(OUT_DIR, "tg.csv")
    tg.to_csv(tg_out, index=False, encoding="utf-8")
    print(f"  ✅ Tg: {len(tg)} 条（Original + OOD 实验，去重后） -> {tg_out}")

    # 规范化 Td
    td = td[["Smiles", "TdValue"]].rename(columns={"Smiles": "SMILES", "TdValue": "Td"})
    td = td.dropna().drop_duplicates(subset="SMILES").reset_index(drop=True)
    td_out = os.path.join(OUT_DIR, "td.csv")
    td.to_csv(td_out, index=False, encoding="utf-8")
    print(f"  ✅ Td: {len(td)} 条 -> {td_out}")

    return tg, td


def _load_tg_model():
    """加载已训练的 Tg 模型（用于派生物理约束特征）"""
    from src import predict
    models = predict.load_models()
    if "Tg" not in models:
        raise RuntimeError("未找到 Tg 模型，请先运行 `python run_train.py --task Tg` 再整理电导率数据")
    return models["Tg"]


def _derive_tg_features(df, tg_model):
    """
    为电导率数据派生三个物理约束特征：
      Tg         : 该聚合物的 Tg (°C，用 Tg 模型预测)
      T_minus_Tg : 测试温度 - Tg (°C)
      is_glassy  : T - Tg < 0 则为 1，否则 0
    """
    from src.features import extract_rdkit_descriptors
    import pandas as pd

    # 对每个唯一聚合物预测 Tg（避免重复计算）
    unique_smiles = df["SMILES"].unique()
    tg_map = {}
    for s in unique_smiles:
        d = extract_rdkit_descriptors(s)
        if d is None:
            tg_map[s] = None
        else:
            try:
                X = pd.DataFrame([d])
                tg_map[s] = float(tg_model.predict(X)[0])
            except Exception:
                tg_map[s] = None

    df = df.copy()
    df["Tg"] = df["SMILES"].map(tg_map)

    # 丢弃 Tg 无法预测的行
    n_before = len(df)
    df = df.dropna(subset=["Tg"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  剔除 Tg 无法预测的样本: {n_before - len(df)} 条")

    # 派生物理特征
    df["T_minus_Tg"] = df["temperature"] - df["Tg"]
    df["is_glassy"] = (df["T_minus_Tg"] < 0).astype(int)

    return df


def prepare_conductivity():
    """整理离子电导率数据（聚合物 + 盐 + 浓度 + 温度 → log 电导率）"""
    print("\n" + "=" * 60)
    print("整理离子电导率数据")
    print("=" * 60)

    # 加载 Tg 模型（物理约束特征依赖）
    tg_model = _load_tg_model()
    print("  ✅ 已加载 Tg 模型")

    src = os.path.join(SRC_BASE, "PolymerElectrolyteData.csv")
    df = pd.read_csv(src, encoding="utf-8", low_memory=False, on_bad_lines="skip")

    # 选列
    keep = {
        "S1 SMILES": "SMILES",
        "Salt 1": "salt",
        "Salt1 Molality (mol salt/kg polymer)": "molality",
        "Temperature (oC)": "temperature",
        "log Conductivity (S/cm)": "log_conductivity",
    }
    df = df[list(keep.keys())].rename(columns=keep)

    # 盐名清洗：去首尾空格 + 大小写归一化
    df["salt"] = df["salt"].astype(str).str.strip().str.upper()

    # 丢弃无 SMILES 或电导率缺失的行
    df = df.dropna(subset=["SMILES", "log_conductivity"]).reset_index(drop=True)

    # 盐浓度缺失的行：盐存在但浓度缺失则丢弃（浓度是关键特征，不能随意填 0）
    df = df.dropna(subset=["molality"]).reset_index(drop=True)

    # 清洗异常浓度（> 100 mol/kg 明显异常，正常范围 0~10）
    n_before = len(df)
    df = df[df["molality"] <= 100].reset_index(drop=True)
    print(f"  剔除 molality>100 异常值: {n_before - len(df)} 条")

    # 合并人工构造的负样本（仅加入电导率数据，不影响 Tg/Td）
    df["is_synthetic"] = 0  # 原始数据非人工构造
    # 合并文献补充数据（正常数据，is_synthetic=0；盐名同样归一化）
    sup_df = build_supplementary_samples()
    sup_df["salt"] = sup_df["salt"].astype(str).str.strip().str.upper()
    df = pd.concat([df, sup_df], ignore_index=True)
    print(f"  ➕ 合并文献补充数据: {len(sup_df)} 条（LiClO4 8条 + PVDF-HFP 4条）")

    # 合并人工负样本
    neg_df = build_negative_samples()
    # 无盐样本的 salt 用 NaN 表示（None -> NaN），避免被 astype(str) 变成 "NONE"
    neg_df["salt"] = neg_df["salt"].apply(lambda x: np.nan if x is None else x)
    df = pd.concat([df, neg_df], ignore_index=True)
    print(f"  ➕ 合并人工负样本: {len(neg_df)} 条（其中无盐 {neg_df['salt'].isna().sum()} 条）")

    # 派生物理约束特征（Tg、T_minus_Tg、is_glassy）
    df = _derive_tg_features(df, tg_model)

    out = os.path.join(OUT_DIR, "conductivity.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  ✅ 电导率: {len(df)} 条 -> {out}")
    print(f"     盐种类数(归一化后): {df['salt'].nunique()}")
    print(f"     唯一聚合物 SMILES: {df['SMILES'].nunique()}")
    print(f"     新增物理特征: Tg, T_minus_Tg, is_glassy")
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    prepare_tg_td()
    prepare_conductivity()
    print("\n🎉 数据整理完成！")
