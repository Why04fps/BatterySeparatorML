# -*- coding: utf-8 -*-
"""
外部验证脚本
====================
用 Table S3_dataset.xlsx（聚醚电解质基准数据，18 条）作为外部验证集，
检验已训练的 Tg 模型和电导率模型在新数据上的泛化能力。

验证内容：
    1. Tg 模型：SMILES → Tg（K→°C 换算后对比）
    2. 电导率模型：SMILES + LiTFSI + Molality + 80°C → log10 电导率

用法：
    python run_external_validation.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src import config, predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Table S3 路径
S3_FILE = r"E:\BaiduNetdiskDownload\数据集\Table S3_dataset.xlsx"
K_TO_C = 273.15


def validate_tg(df):
    """验证 Tg 模型"""
    print("=" * 60)
    print("验证 Tg 模型（外部数据 Table S3）")
    print("=" * 60)

    model = predict.load_models().get("Tg")
    if model is None:
        print("未找到 Tg 模型，请先训练。")
        return None

    true_tg_c = df["Tg (K)"] - K_TO_C  # K → °C
    pred_tg = []
    for s in df["SMILES"]:
        p = predict.predict_smiles(s, {"Tg": model})
        pred_tg.append(p["Tg"] if p else np.nan)

    pred_tg = np.array(pred_tg, dtype=float)
    valid = ~np.isnan(pred_tg)
    true = true_tg_c[valid].values
    pred = pred_tg[valid]

    if len(pred) == 0:
        print("所有 SMILES 都无法预测。")
        return None

    result = {
        "n": len(pred),
        "mae": mean_absolute_error(true, pred),
        "rmse": np.sqrt(mean_squared_error(true, pred)),
        "r2": r2_score(true, pred),
    }
    print(f"  有效样本: {result['n']}/{len(df)}")
    print(f"  MAE  = {result['mae']:.2f} °C")
    print(f"  RMSE = {result['rmse']:.2f} °C")
    print(f"  R²   = {result['r2']:.3f}")

    # 逐条对比
    print("\n  逐条对比（真实 vs 预测，°C）:")
    names = df["Name"].values[valid]
    for i in range(len(pred)):
        print(f"    {names[i]:<8} 真实={true[i]:6.1f}  预测={pred[i]:6.1f}  误差={pred[i]-true[i]:+.1f}")
    return result


def validate_conductivity(df):
    """验证电导率模型"""
    print("\n" + "=" * 60)
    print("验证电导率模型（外部数据 Table S3）")
    print("=" * 60)

    models = predict.load_models()
    model = models.get("conductivity")
    if model is None:
        print("未找到电导率模型，请先训练。")
        return None

    meta = predict.load_meta("conductivity")
    salt_meta = meta.get("salt_meta", {}) if meta else {}

    # 电导率目标列（注意数据里列名末尾缺右括号的笔误）
    true_col = None
    for c in df.columns:
        if "exp" in c and "log10" in c:
            true_col = c
            break
    if true_col is None:
        print("未找到实验电导率 log10 列。")
        return None

    true_log = df[true_col].astype(float)

    # 温度：353 K → 80 °C（我们的模型用 °C）
    temp_c = df["T_exp (K)"] - K_TO_C

    pred_log = []
    for s, mol, t in zip(df["SMILES"], df["Molality (mol kg-1)"], temp_c):
        p = predict.predict_conductivity(s, "LiTFSI", mol, t, model, salt_meta)
        pred_log.append(p if p is not None else np.nan)

    pred_log = np.array(pred_log, dtype=float)
    valid = ~np.isnan(pred_log)
    true = true_log[valid].values
    pred = pred_log[valid]

    if len(pred) == 0:
        print("所有样本都无法预测。")
        return None

    result = {
        "n": len(pred),
        "mae": mean_absolute_error(true, pred),
        "rmse": np.sqrt(mean_squared_error(true, pred)),
        "r2": r2_score(true, pred),
    }
    print(f"  有效样本: {result['n']}/{len(df)}")
    print(f"  MAE  = {result['mae']:.3f} (log10 S/cm)")
    print(f"  RMSE = {result['rmse']:.3f} (log10 S/cm)")
    print(f"  R²   = {result['r2']:.3f}")

    print("\n  逐条对比（真实 vs 预测，log10 S/cm）:")
    names = df["Name"].values[valid]
    for i in range(len(pred)):
        print(f"    {names[i]:<8} 真实={true[i]:6.2f}  预测={pred[i]:6.2f}  误差={pred[i]-true[i]:+.2f}")
    return result


def main():
    print("=" * 60)
    print("外部验证：Table S3 聚醚电解质基准数据（18 条）")
    print("=" * 60)

    if not os.path.exists(S3_FILE):
        print(f"❌ 找不到数据文件: {S3_FILE}")
        sys.exit(1)

    df = pd.read_excel(S3_FILE, sheet_name="PEMD")
    print(f"数据加载: {len(df)} 条\n")

    res_tg = validate_tg(df)
    res_cond = validate_conductivity(df)

    print("\n" + "=" * 60)
    print("外部验证汇总")
    print("=" * 60)
    if res_tg:
        print(f"  Tg:       MAE={res_tg['mae']:.2f}°C  RMSE={res_tg['rmse']:.2f}°C  R²={res_tg['r2']:.3f}")
    if res_cond:
        print(f"  电导率:   MAE={res_cond['mae']:.3f}  RMSE={res_cond['rmse']:.3f}  R²={res_cond['r2']:.3f}")


if __name__ == "__main__":
    main()
