"""
全局配置文件
====================
集中管理：数据路径、目标变量、特征配置、超参数网格、随机种子等。
新数据到位后，只需修改这里的配置即可适配。
"""

import os

# ==================== 路径配置 ====================
# 项目根目录（src 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

# 数据文件路径（新数据替换时修改这里）
DATA_FILE = os.path.join(DATA_DIR, "experiment_polymer_data.xlsx")

# ==================== 数据列配置 ====================
# 分子结构列（特征来源）
SMILES_COL = "PSMILES"

# ==================== 任务配置（两套特征体系） ====================
# 任务 1：纯聚合物本征性能（Tg、Td）
#   特征 = RDKit 分子描述符（12 个）
# 任务 2：聚合物电解质离子电导率（多组分）
#   特征 = RDKit 描述符 + 盐类型混合编码 + 盐浓度 + 温度

# 纯聚合物任务的数据文件与目标列
POLYMER_TASKS = {
    "Tg": {"data_file": os.path.join(DATA_DIR, "tg.csv"), "target_col": "Tg", "smiles_col": "SMILES"},
    "Td": {"data_file": os.path.join(DATA_DIR, "td.csv"), "target_col": "Td", "smiles_col": "SMILES"},
}

# 电导率任务的数据文件与列
CONDUCTIVITY_TASK = {
    "data_file": os.path.join(DATA_DIR, "conductivity.csv"),
    "target_col": "log_conductivity",   # 目标：log10(电导率 S/cm)
    "smiles_col": "SMILES",
    "salt_col": "salt",                 # 盐类型（类别）
    "molality_col": "molality",         # 盐浓度
    "temperature_col": "temperature",   # 温度 (oC)
    # 物理约束特征（由 Tg 模型派生，单位 °C）
    "tg_col": "Tg",                     # 该聚合物的 Tg (°C)
    "t_minus_tg_col": "T_minus_Tg",     # 测试温度 - Tg
    "is_glassy_col": "is_glassy",       # 若 T - Tg < 0 则为 1，否则 0
}

# 盐类型混合编码配置
SALT_TOP_K = 5          # Top 5 盐做 one-hot
SALT_FREQ_RANGE = (5, 15)  # 第 6~15 名盐用频率编码（归一化 0~1）
# 其余盐（第 16 名及以后）归并为 is_other_salt (0/1)

# 需要剔除的非特征列（id、名称、结构、文献等）
EXCLUDE_COLS = ["id", "Name", "PSMILES", "Reference"]

# ==================== 特征配置 ====================
# 是否剔除 PSMILES 无法被 RDKit 解析的样本
DROP_INVALID_SMILES = True

# ==================== 模型配置 ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15        # 从训练集中再划分验证集用于早停

# GridSearchCV 超参数网格
PARAM_GRID = {
    "n_estimators": [100, 200, 500, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
}

# 交叉验证折数
CV_FOLDS = 5

# 早停轮数
EARLY_STOPPING_ROUNDS = 50


def ensure_dirs():
    """确保必要的目录存在"""
    for d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)
