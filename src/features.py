"""
特征工程模块
====================
使用 RDKit 从分子结构（PSMILES）计算分子描述符，作为模型输入特征。
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import pandas as pd

from . import config

# 抑制 RDKit 解析不合法 SMILES 时的错误日志
RDLogger.DisableLog("rdApp.error")


def extract_rdkit_descriptors(smiles):
    """
    用 RDKit 从 PSMILES 计算分子描述符。
    解析或计算失败返回 None（该样本会被剔除）。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "MolWt": Descriptors.MolWt(mol),                                  # 分子量
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),                # 重原子数
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),          # 可旋转键数（链柔性）
            "NumHDonors": Descriptors.NumHDonors(mol),                        # 氢键供体数
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),                  # 氢键受体数
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),            # 芳香环数（链刚性）
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),          # 饱和环数
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),          # 脂肪环数
            "RingCount": Descriptors.RingCount(mol),                          # 环总数
            "TPSA": Descriptors.TPSA(mol),                                    # 极性表面积
            "MolLogP": Descriptors.MolLogP(mol),                              # 脂水分配系数（疏水性）
            "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),                # 杂原子数
        }
    except Exception:
        return None


def build_features_from_smiles(smiles_series, drop_invalid=None):
    """
    从 SMILES 序列构建描述符特征 DataFrame。

    参数
    ----
    smiles_series : pandas.Series
        分子结构列（PSMILES）
    drop_invalid : bool 或 None
        是否剔除无法解析的样本；None 时取 config.DROP_INVALID_SMILES

    返回
    ----
    (X, valid_mask) : (DataFrame, ndarray)
        X          : 描述符特征
        valid_mask : 长度为 len(smiles_series) 的布尔数组，True 表示有效样本
    """
    if drop_invalid is None:
        drop_invalid = config.DROP_INVALID_SMILES

    feature_list = []
    valid_mask = []
    for s in smiles_series:
        desc = extract_rdkit_descriptors(s)
        if desc is None:
            valid_mask.append(False)
            continue
        feature_list.append(desc)
        valid_mask.append(True)

    X = pd.DataFrame(feature_list)
    return X, pd.Series(valid_mask, index=smiles_series.index)


def load_data(data_file=None):
    """
    加载数据，返回 (df, X, target_df)。
    仅加载目标列存在的样本；剔除无法解析的 SMILES 样本。
    """
    if data_file is None:
        data_file = config.DATA_FILE

    df = pd.read_excel(data_file)

    # 只保留目标列非缺失的样本
    df = df.dropna(subset=config.TARGET_COLUMNS).reset_index(drop=True)

    # 提取特征
    X, valid_mask = build_features_from_smiles(df[config.SMILES_COL])
    df = df[valid_mask].reset_index(drop=True)

    target_df = df[config.TARGET_COLUMNS].reset_index(drop=True)
    return df, X, target_df


def load_polymer_data(task_cfg):
    """
    加载纯聚合物任务数据（Tg/Td），返回 (X, y)。
    数据文件为 CSV，含 smiles 列和目标列。
    """
    df = pd.read_csv(task_cfg["data_file"])
    df = df.dropna(subset=[task_cfg["target_col"]]).reset_index(drop=True)

    X, valid_mask = build_features_from_smiles(df[task_cfg["smiles_col"]])
    df = df[valid_mask.values].reset_index(drop=True)
    X = X[valid_mask.values].reset_index(drop=True)

    y = df[task_cfg["target_col"]].reset_index(drop=True)
    return df, X, y


# ==================== 聚合物电解质电导率特征 ====================

def encode_salt(salt_series):
    """
    盐类型混合编码（频率 + One-Hot 混合方案）。

    编码逻辑（见 config）：
      - Top K 盐      -> one-hot 编码（每盐一列 0/1）
      - 第 K+1 ~ N 名 -> 频率编码（归一化 0~1，仅 1 列）
      - 其余盐        -> is_other_salt（0/1 二值列）

    参数
    ----
    salt_series : pandas.Series
        盐类型列（已大小写归一化）

    返回
    ----
    (salt_df, salt_meta) :
        salt_df  : 编码后的盐特征 DataFrame
        salt_meta: 字典，记录编码信息（Top 盐列表、频率映射），供预测时复用
    """
    k = config.SALT_TOP_K
    lo, hi = config.SALT_FREQ_RANGE

    # 统计盐频率排名（剔除 NaN 后再排名）
    counts = salt_series.value_counts(dropna=True)
    ordered = list(counts.index)

    top_salts = list(ordered[:k])
    freq_salts = list(ordered[k:hi])

    # 频率映射（第 K+1 ~ N 名盐 -> 归一化频率）
    total = len(salt_series)
    freq_map = {s: counts.get(s, 0) / total for s in freq_salts}

    # 构建 one-hot（Top K）
    onehot = pd.DataFrame(index=salt_series.index)
    for s in top_salts:
        onehot[f"salt_{s}"] = (salt_series == s).astype(int)

    # 频率编码（第 K+1 ~ N 名）
    freq_col = salt_series.map(freq_map).fillna(0.0)
    onehot["salt_freq_rank"] = freq_col

    # 其余盐（含 NaN）归并 is_other_salt
    known = set(top_salts) | set(freq_salts)
    onehot["is_other_salt"] = (~salt_series.isin(known)).astype(int)

    salt_meta = {
        "top_salts": top_salts,
        "freq_salts": freq_salts,
        "freq_map": freq_map,
    }
    return onehot, salt_meta


def _derive_tg_features_from_model(smiles_series, tg_model):
    """
    用 Tg 模型为每个 SMILES 预测 Tg，并派生物理约束特征。
    返回 DataFrame：Tg、T_minus_Tg、is_glassy（T_minus_Tg 需外部提供 temperature）
    这里仅返回 Tg 预测值，T_minus_Tg 由调用方结合 temperature 计算。
    """
    tg_list = []
    for s in smiles_series:
        d = extract_rdkit_descriptors(s)
        if d is None or tg_model is None:
            tg_list.append(None)
        else:
            try:
                X = pd.DataFrame([d])
                tg_list.append(float(tg_model.predict(X)[0]))
            except Exception:
                tg_list.append(None)
    return pd.Series(tg_list, index=smiles_series.index)


def has_heteroatom(smiles):
    """
    硬规则特征：SMILES 中是否含杂原子（碳以外的重原子：O、N、F、Cl、Br、I、S、P、Si 等）。
    含任意一个返回 1，纯烃链（只有 C/H）返回 0。

    物理依据：纯烃链聚合物（PE/PP/PS）无法解离/络合锂盐，离子传导能力极弱；
    含杂原子（尤其 O/N）的聚合物才能提供离子传导位点。

    注：用 RDKit 解析分子判断（比正则可靠，能正确区分 Cl/Br/Si 等双字符元素符号）。
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    for atom in mol.GetAtoms():
        # 跳过虚拟原子 [*]（聚合物端基占位符）
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetAtomicNum() != 6:  # 非碳（Z≠6）即杂原子
            return 1
    return 0


def build_conductivity_features(df, tg_model=None):
    """
    构建电导率任务的特征矩阵：
      RDKit 描述符 + 盐混合编码 + 盐浓度 + 温度 + 物理约束特征（Tg、T_minus_Tg、is_glassy）
      + 硬规则特征（has_heteroatom）

    参数
    ----
    df : pandas.DataFrame
        原始电导率数据（含 SMILES、salt、molality、temperature 列）
    tg_model : 可选
        Tg 模型；若 df 中已有 Tg 列则用已有值，否则用该模型预测

    返回
    ----
    (X, valid_mask, salt_meta) :
        X          : 特征 DataFrame
        valid_mask : 有效样本布尔数组（SMILES 无法解析的样本）
        salt_meta  : 盐编码元信息（供预测复用）
    """
    # 1. RDKit 描述符
    desc_X, valid_mask = build_features_from_smiles(df[config.CONDUCTIVITY_TASK["smiles_col"]])
    desc_X = desc_X.reset_index(drop=True)

    # 2. 盐混合编码
    salt_df, salt_meta = encode_salt(df[config.CONDUCTIVITY_TASK["salt_col"]].reset_index(drop=True))
    salt_df = salt_df.reset_index(drop=True)

    # 3. 数值特征（盐浓度、温度）
    num_df = pd.DataFrame({
        "molality": df[config.CONDUCTIVITY_TASK["molality_col"]].reset_index(drop=True),
        "temperature": df[config.CONDUCTIVITY_TASK["temperature_col"]].reset_index(drop=True),
    })

    # 3.5 硬规则特征（has_heteroatom）
    hard_df = pd.DataFrame({
        "has_heteroatom": df[config.CONDUCTIVITY_TASK["smiles_col"]].apply(has_heteroatom).reset_index(drop=True),
    })

    # 4. 物理约束特征（Tg、T_minus_Tg、is_glassy）
    tg_col = config.CONDUCTIVITY_TASK["tg_col"]
    if tg_col in df.columns:
        # 已有 Tg 值（prepare_data 生成或用户提供）
        tg_vals = df[tg_col].reset_index(drop=True)
    else:
        # 用 Tg 模型预测
        tg_vals = _derive_tg_features_from_model(
            df[config.CONDUCTIVITY_TASK["smiles_col"]].reset_index(drop=True), tg_model
        )

    temp_vals = df[config.CONDUCTIVITY_TASK["temperature_col"]].reset_index(drop=True)
    phys_df = pd.DataFrame({
        tg_col: tg_vals,
        config.CONDUCTIVITY_TASK["t_minus_tg_col"]: temp_vals - tg_vals,
        config.CONDUCTIVITY_TASK["is_glassy_col"]: ((temp_vals - tg_vals) < 0).astype(int),
    })

    # 拼接
    X = pd.concat([desc_X, salt_df, num_df, hard_df, phys_df], axis=1)
    return X, valid_mask, salt_meta


def load_conductivity_data(data_file=None, tg_model=None):
    """
    加载电导率数据，返回 (df, X, y, salt_meta, is_synthetic)。
    剔除无法解析的 SMILES 样本及物理特征缺失的样本。
    is_synthetic: 布尔数组，True 表示人工构造的负样本（训练时应强制留在训练集）。
    """
    if data_file is None:
        data_file = config.CONDUCTIVITY_TASK["data_file"]

    df = pd.read_csv(data_file)

    # 目标列缺失剔除
    df = df.dropna(subset=[config.CONDUCTIVITY_TASK["target_col"]]).reset_index(drop=True)

    # 构建特征
    X, valid_mask, salt_meta = build_conductivity_features(df, tg_model=tg_model)

    # 剔除 SMILES 无法解析 或 物理特征(Tg)缺失 的样本
    tg_col = config.CONDUCTIVITY_TASK["tg_col"]
    mask = valid_mask.values & X[tg_col].notna().values
    df = df[mask].reset_index(drop=True)
    X = X[mask].reset_index(drop=True)

    y = df[config.CONDUCTIVITY_TASK["target_col"]].reset_index(drop=True)
    is_synthetic = df["is_synthetic"].astype(int).values if "is_synthetic" in df.columns else None
    return df, X, y, salt_meta, is_synthetic
