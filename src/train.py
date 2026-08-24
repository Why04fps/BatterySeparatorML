"""
模型训练模块
====================
为每个目标变量训练独立的 XGBoost 模型：
  数据划分 -> 5 折交叉验证 -> GridSearchCV 调优 -> 用最佳参数重训 -> 保存模型
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from . import config


def cross_validate(X, y):
    """5 折交叉验证，返回 (r2_scores, rmse_scores)"""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    neg_mse = cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse)
    return r2_scores, rmse_scores


def tune_hyperparams(X, y):
    """GridSearchCV 调优，返回 (best_params, best_score)"""
    grid_model = XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        estimator=grid_model,
        param_grid=config.PARAM_GRID,
        scoring="r2",
        cv=config.CV_FOLDS,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_


def train_one_target(X, y, target_name, train_only_mask=None):
    """
    训练单个目标变量的模型。

    参数
    ----
    X, y : 特征与目标
    target_name : 目标名
    train_only_mask : 可选布尔数组
        为 True 的样本强制留在训练集（不进入测试集）。
        用于人工构造的负样本——它们的作用是教会模型，不适合作为测试样本。

    返回
    ----
    dict: {
        model, best_params, cv_r2, cv_rmse, test_metrics, target_name, feature_importance
    }
    """
    # 1. 划分数据集（训练/验证/测试）
    if train_only_mask is not None and train_only_mask.any():
        # 强制留训练集的样本索引
        keep_idx = np.where(~train_only_mask)[0]
        only_idx = np.where(train_only_mask)[0]

        X_rest, X_only = X.iloc[keep_idx], X.iloc[only_idx]
        y_rest, y_only = y.iloc[keep_idx], y.iloc[only_idx]

        # 从其余样本中划分测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_rest, y_rest, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

        # 把强制留训练集的样本加入训练集
        X_train = pd.concat([X_train, X_only], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_only], axis=0).reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

    # 再从训练集中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
    )

    # 2. 交叉验证
    r2_scores, rmse_scores = cross_validate(X, y)

    # 3. 超参数调优
    best_params, best_cv_score = tune_hyperparams(X, y)

    # 4. 用最佳参数重训（含早停）
    model = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 5. 测试集评估
    y_pred = model.predict(X_test)
    test_metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    return {
        "target_name": target_name,
        "model": model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
        "cv_rmse_mean": float(rmse_scores.mean()),
        "test_metrics": test_metrics,
        "feature_importance": dict(zip(X.columns, model.feature_importances_)),
    }


def train_all(X, target_df):
    """为每个目标变量训练模型，并返回结果列表"""
    results = {}
    for col in target_df.columns:
        print(f"\n{'='*60}")
        print(f"正在训练目标: {col}")
        print(f"{'='*60}")
        y = target_df[col]
        res = train_one_target(X, y, col)
        results[col] = res
        print(f"  最佳参数: {res['best_params']}")
        print(f"  CV R2: {res['cv_r2_mean']:.3f} ± {res['cv_r2_std']:.3f}")
        print(f"  测试集: R2={res['test_metrics']['R2']:.3f}, "
              f"MAE={res['test_metrics']['MAE']:.2f}, RMSE={res['test_metrics']['RMSE']:.2f}")
    return results


def save_model(result, target_name):
    """保存模型及元数据到 models 目录"""
    config.ensure_dirs()
    base = os.path.join(config.MODEL_DIR, target_name)

    # 保存模型
    joblib.dump(result["model"], base + "_model.joblib")

    # 将 numpy 标量转为 Python 原生类型，避免 JSON 序列化报错
    def _to_py_type(v):
        if hasattr(v, "item"):  # numpy 标量
            return v.item()
        return v

    # 保存元数据（参数、指标、特征重要性）
    meta = {
        "target_name": target_name,
        "best_params": {k: _to_py_type(v) for k, v in result["best_params"].items()},
        "best_cv_score": _to_py_type(result["best_cv_score"]),
        "cv_r2_mean": _to_py_type(result["cv_r2_mean"]),
        "cv_r2_std": _to_py_type(result["cv_r2_std"]),
        "cv_rmse_mean": _to_py_type(result["cv_rmse_mean"]),
        "test_metrics": {k: _to_py_type(v) for k, v in result["test_metrics"].items()},
        "feature_importance": {k: _to_py_type(v) for k, v in result["feature_importance"].items()},
    }

    # 若结果包含盐编码元信息（电导率任务），一并保存
    if "salt_meta" in result and result["salt_meta"] is not None:
        meta["salt_meta"] = result["salt_meta"]

    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 模型已保存: {base}_model.joblib")
    print(f"  ✅ 元数据已保存: {base}_meta.json")


def save_all(results):
    """保存所有目标模型"""
    for target_name, res in results.items():
        save_model(res, target_name)


def train_polymer_task(task_name, task_cfg):
    """训练纯聚合物任务（Tg/Td），返回结果 dict"""
    from . import features
    print(f"\n{'='*60}")
    print(f"正在训练聚合物目标: {task_name}")
    print(f"{'='*60}")

    df, X, y = features.load_polymer_data(task_cfg)
    print(f"  有效样本: {len(X)} 个，特征: {X.shape[1]} 个")

    res = train_one_target(X, y, task_name)
    _print_result(res)
    return res


def train_conductivity_task(salt_meta=None):
    """训练电导率任务，返回结果 dict（含 salt_meta）"""
    from . import features
    print(f"\n{'='*60}")
    print("正在训练离子电导率目标 (log_conductivity)")
    print(f"{'='*60}")

    df, X, y, salt_meta, is_synthetic = features.load_conductivity_data()
    print(f"  有效样本: {len(X)} 个，特征: {X.shape[1]} 个")
    if is_synthetic is not None and is_synthetic.any():
        print(f"  （其中人工负样本 {int(is_synthetic.sum())} 条，强制留在训练集）")

    res = train_one_target(X, y, "conductivity", train_only_mask=is_synthetic)
    res["salt_meta"] = salt_meta
    _print_result(res)
    return res


def _print_result(res):
    print(f"  最佳参数: {res['best_params']}")
    print(f"  CV R2: {res['cv_r2_mean']:.3f} ± {res['cv_r2_std']:.3f}")
    print(f"  测试集: R2={res['test_metrics']['R2']:.3f}, "
          f"MAE={res['test_metrics']['MAE']:.2f}, RMSE={res['test_metrics']['RMSE']:.2f}")
