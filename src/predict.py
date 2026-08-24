"""
预测接口模块
====================
加载已训练模型，对新的分子结构（PSMILES）进行性能预测。
供 Streamlit 网页应用调用。
"""

import os
import json

import joblib
import pandas as pd

from . import config
from .features import extract_rdkit_descriptors


def load_models(model_dir=None):
    """加载 models 目录下所有已训练模型，返回 {target_name: model}"""
    if model_dir is None:
        model_dir = config.MODEL_DIR
    models = {}
    if not os.path.isdir(model_dir):
        return models
    for f in os.listdir(model_dir):
        if f.endswith("_model.joblib"):
            target = f[: -len("_model.joblib")]
            models[target] = joblib.load(os.path.join(model_dir, f))
    return models


def load_meta(target_name, model_dir=None):
    """加载指定目标变量的元数据（参数、指标、特征重要性）"""
    if model_dir is None:
        model_dir = config.MODEL_DIR
    meta_path = os.path.join(model_dir, f"{target_name}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_smiles(smiles, models):
    """
    对单个 PSMILES 预测所有目标性能。

    参数
    ----
    smiles : str
        分子结构
    models : dict
        {target_name: model}

    返回
    ----
    dict: {target_name: predicted_value}；解析失败返回 None
    """
    desc = extract_rdkit_descriptors(smiles)
    if desc is None:
        return None
    X = pd.DataFrame([desc])
    return {target: float(model.predict(X)[0]) for target, model in models.items()}


def predict_batch(smiles_list, models):
    """
    批量预测多个 PSMILES。

    返回
    ----
    pd.DataFrame: 每行为一个分子，列为各目标性能预测值
    """
    rows = []
    for s in smiles_list:
        pred = predict_smiles(s, models)
        if pred is not None:
            rows.append({"PSMILES": s, **pred})
    return pd.DataFrame(rows)


def predict_conductivity_batch(smiles_list, salt, molality, temperature,
                               model, salt_meta, tg_model=None):
    """
    批量预测电导率（固定条件：所有分子使用相同的盐/浓度/温度）。

    参数
    ----
    smiles_list : list
        聚合物 SMILES 列表
    salt, molality, temperature : 固定条件
    model : 电导率模型
    salt_meta : 盐编码元信息
    tg_model : Tg 模型（物理特征派生）

    返回
    ----
    pd.DataFrame: PSMILES + log10 电导率 + 电导率
    """
    rows = []
    for s in smiles_list:
        log_sigma = predict_conductivity(s, salt, molality, temperature,
                                         model, salt_meta, tg_model=tg_model)
        if log_sigma is not None:
            rows.append({
                "PSMILES": s,
                "log10(电导率 S/cm)": round(log_sigma, 4),
                "电导率 (S/cm)": 10 ** log_sigma,
            })
    return pd.DataFrame(rows)


def predict_conductivity_batch_rows(df, model, salt_meta, tg_model=None):
    """
    批量预测电导率（逐行条件：每行数据带自己的 SMILES/盐/浓度/温度）。

    参数
    ----
    df : pandas.DataFrame
        必须含列: SMILES(或PSMILES)、salt、molality、temperature
    model : 电导率模型
    salt_meta : 盐编码元信息
    tg_model : Tg 模型（物理特征派生）

    返回
    ----
    pd.DataFrame: 原始列 + log10 电导率 + 电导率 + 是否预测成功
    """
    import numpy as np

    smiles_col = "SMILES" if "SMILES" in df.columns else "PSMILES"
    out = df.copy()

    log_col, cond_col, ok_col = [], [], []
    for _, row in out.iterrows():
        try:
            log_sigma = predict_conductivity(
                row[smiles_col], row["salt"], row["molality"], row["temperature"],
                model, salt_meta, tg_model=tg_model,
            )
            if log_sigma is not None:
                log_col.append(round(float(log_sigma), 4))
                cond_col.append(10 ** float(log_sigma))
                ok_col.append(True)
            else:
                log_col.append(np.nan)
                cond_col.append(np.nan)
                ok_col.append(False)
        except Exception:
            log_col.append(np.nan)
            cond_col.append(np.nan)
            ok_col.append(False)

    out["log10(电导率 S/cm)"] = log_col
    out["电导率 (S/cm)"] = cond_col
    out["预测成功"] = ok_col
    return out


# ==================== 电导率预测 ====================

def encode_salt_single(salt, salt_meta):
    """
    根据训练时保存的 salt_meta，对单个盐名做混合编码，返回特征 dict。
    """
    top_salts = salt_meta.get("top_salts", [])
    freq_map = salt_meta.get("freq_map", {})
    freq_salts = salt_meta.get("freq_salts", [])

    salt = str(salt).strip().upper()

    feat = {}
    for s in top_salts:
        feat[f"salt_{s}"] = 1 if salt == s else 0
    feat["salt_freq_rank"] = freq_map.get(salt, 0.0)
    known = set(top_salts) | set(freq_salts)
    feat["is_other_salt"] = 0 if salt in known else 1
    return feat


def predict_conductivity(smiles, salt, molality, temperature, model, salt_meta, tg_model=None):
    """
    预测单个聚合物电解质体系的 log 电导率。

    参数
    ----
    smiles : str
        聚合物 SMILES
    salt : str
        盐类型
    molality : float
        盐浓度 (mol salt/kg polymer)
    temperature : float
        温度 (oC)
    model : 训练好的 XGBoost 模型
    salt_meta : dict
        盐编码元信息
    tg_model : 可选
        Tg 模型，用于派生物理约束特征（Tg、T_minus_Tg、is_glassy）

    返回
    ----
    float: log10(电导率 S/cm)；解析失败返回 None
    """
    from .features import extract_rdkit_descriptors, has_heteroatom
    desc = extract_rdkit_descriptors(smiles)
    if desc is None:
        return None

    salt_feat = encode_salt_single(salt, salt_meta)

    # 硬规则特征：是否含杂原子（与训练时的 has_heteroatom 一致）
    het_feat = {"has_heteroatom": has_heteroatom(smiles)}

    # 物理约束特征：用 Tg 模型预测该聚合物的 Tg，再派生 T_minus_Tg、is_glassy
    if tg_model is not None:
        tg = float(tg_model.predict(pd.DataFrame([desc]))[0])
    else:
        tg = None
    if tg is None:
        return None
    t_minus_tg = temperature - tg
    is_glassy = 1 if t_minus_tg < 0 else 0

    row = {
        **desc,
        **salt_feat,
        "molality": molality,
        "temperature": temperature,
        **het_feat,
        "Tg": tg,
        "T_minus_Tg": t_minus_tg,
        "is_glassy": is_glassy,
    }
    X = pd.DataFrame([row])
    return float(model.predict(X)[0])

