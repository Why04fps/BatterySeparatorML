"""
锂电池隔膜筛选工具 - Streamlit 网页应用
========================================
用法：
    streamlit run app/streamlit_app.py

功能：
    1. 输入 PSMILES 预测 Tg、Td（纯聚合物性能）
    2. 输入 PSMILES + 盐 + 浓度 + 温度 预测离子电导率
    3. 批量预测候选聚合物结构并排序筛选
    4. 展示已训练模型的性能指标与特征重要性
"""

import os
import sys

import pandas as pd
import streamlit as st

# 将项目根目录加入 sys.path，以便导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config, predict


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="锂电池隔膜筛选工具",
    page_icon="🔋",
    layout="wide",
)

st.title("🔋 锂电池隔膜筛选工具")
st.markdown(
    "基于 **RDKit 分子描述符 + XGBoost** 的机器学习模型，"
    "快速预测聚合物隔膜的关键性能，加速候选材料筛选。"
)


# ==================== 加载模型 ====================
@st.cache_resource
def load_models():
    return predict.load_models()


models = load_models()

# 区分纯聚合物模型与电导率模型
POLYMER_TARGETS = [k for k in models if k in config.POLYMER_TASKS]
COND_TARGET = "conductivity" if "conductivity" in models else None


# ==================== 侧边栏 ====================
st.sidebar.header("📦 模型信息")
if models:
    st.sidebar.write(f"已加载 {len(models)} 个目标模型:")
    for name, m in models.items():
        meta = predict.load_meta(name)
        if meta:
            st.sidebar.write(f"- **{name}**  (测试R²={meta['test_metrics']['R2']:.3f})")
        else:
            st.sidebar.write(f"- {name}")
else:
    st.sidebar.warning("未找到已训练模型！请先运行 `python run_train.py`")
    st.sidebar.info(f"模型目录: `{config.MODEL_DIR}`")


# ==================== 模型适用范围提示 ====================
with st.sidebar.expander("⚠️ 模型适用范围与失效范围", expanded=False):
    st.markdown(
        """
**✅ 适用范围（高置信度）**
- 含 O/N 等杂原子的极性聚合物（如 PEO、PMMA、PAN 等）
- 聚合物 + 锂盐（LiTFSI、LiPF6、LiClO4）体系
- 温度范围：**25-80°C**
- 盐浓度范围：**0.5-2.0 mol/kg**
- 预测用途：材料预筛选、相对排序

**❌ 失效范围（低置信度/不可用）**
- 纯烃链聚合物（PE、PP、PS 等）→ 已通过硬规则修正为低电导率，但缺乏真实数据验证
- 含氟聚合物（PVDF、PVDF-HFP 等）→ 训练数据极少，预测为外推，仅供参考
- 不含杂原子的聚合物 → 模型强制预测为低电导率
- 温度超出 25-80°C 范围 → 外推不可靠
- 非锂盐体系（钠盐、镁盐等）→ 模型未训练
- 含增塑剂、多孔结构等复杂隔膜体系 → 模型仅基于纯聚合物 + 盐数据训练

**💡 使用建议**
- 本工具适用于从大量候选材料中快速筛选出值得进一步研究的对象
- 预测结果**不能替代实验验证**，建议结合文献和实验进行交叉验证
- 对预测结果存疑时，优先查阅相关文献或进行实验验证
"""
    )


# ==================== 主功能区 ====================
tab1, tab2, tab3 = st.tabs(["🔍 单分子预测", "📋 批量筛选", "📊 模型性能"])


# ---------- Tab 1: 单分子预测 ----------
with tab1:
    # --- 纯聚合物性能预测（Tg/Td） ---
    st.subheader("① 聚合物热性能预测（Tg / Td）")
    smiles_input = st.text_input(
        "输入聚合物 PSMILES 分子结构",
        placeholder="例如: [*]CC([*])O",
        value="",
        key="smiles_polymer",
    )

    if st.button("预测 Tg / Td", key="btn_polymer"):
        if not POLYMER_TARGETS:
            st.error("未找到 Tg/Td 模型，请先训练")
        elif not smiles_input.strip():
            st.warning("请输入分子结构")
        else:
            poly_models = {t: models[t] for t in POLYMER_TARGETS}
            result = predict.predict_smiles(smiles_input.strip(), poly_models)
            if result is None:
                st.error("❌ 无法解析该 PSMILES，请检查结构是否正确")
            else:
                st.success("✅ 预测完成")
                cols = st.columns(len(result))
                for col, (target, val) in zip(cols, result.items()):
                    unit = "°C" if target in ("Tg", "Td") else ""
                    with col:
                        st.metric(label=f"{target} ({unit})", value=f"{val:.2f}")

    st.divider()

    # --- 离子电导率预测 ---
    st.subheader("② 离子电导率预测（聚合物 + 盐 + 条件）")
    if not COND_TARGET:
        st.info("未找到电导率模型，请先训练。")
    else:
        cond_meta = predict.load_meta(COND_TARGET)
        salt_meta = cond_meta.get("salt_meta", {}) if cond_meta else {}
        top_salts = salt_meta.get("top_salts", [])
        freq_salts = salt_meta.get("freq_salts", [])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cond_smiles = st.text_input(
                "聚合物 PSMILES",
                placeholder="例如: [*]CCO[*] (PEO)",
                key="smiles_cond",
            )
        with c2:
            # 盐选项：Top5 + 频率盐 + 手动输入
            salt_options = list(top_salts) + list(freq_salts)
            salt_choice = st.selectbox(
                "盐类型",
                options=salt_options,
                key="salt_cond",
            )
        with c3:
            molality = st.number_input(
                "盐浓度 (mol/kg)",
                min_value=0.0, max_value=20.0, value=1.0, step=0.1,
                key="molality_cond",
            )
        with c4:
            temperature = st.number_input(
                "温度 (°C)",
                min_value=-70.0, max_value=200.0, value=25.0, step=5.0,
                key="temp_cond",
            )

        if st.button("预测电导率", key="btn_cond"):
            if not cond_smiles.strip():
                st.warning("请输入聚合物 PSMILES")
            else:
                tg_model = models.get("Tg")  # 用于派生物理约束特征
                log_sigma = predict.predict_conductivity(
                    cond_smiles.strip(), salt_choice, molality, temperature,
                    models[COND_TARGET], salt_meta, tg_model=tg_model,
                )
                if log_sigma is None:
                    st.error("❌ 无法解析该 PSMILES 或缺少 Tg 模型")
                else:
                    sigma = 10 ** log_sigma
                    st.success("✅ 预测完成")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("log10(电导率 S/cm)", value=f"{log_sigma:.3f}")
                    with m2:
                        st.metric("电导率 (S/cm)", value=f"{sigma:.3e}")


# ---------- Tab 2: 批量筛选 ----------
with tab2:
    st.subheader("批量筛选")
    bulk_mode = st.radio(
        "选择批量预测模式",
        options=["热性能批量预测（Tg/Td）", "电导率批量预测（固定条件）", "电导率批量预测（CSV 逐行条件）"],
        horizontal=True,
    )

    # ---------- 模式 1：热性能批量预测（Tg/Td） ----------
    if bulk_mode == "热性能批量预测（Tg/Td）":
        st.caption("对纯聚合物热性能（Tg/Td）批量预测并排序。")
        bulk_text = st.text_area(
            "输入多个分子结构（每行一个）",
            height=150,
            placeholder="[*]CC([*])O\n[*]CC([*])C1=CC=CC=C1\n...",
            key="bulk_polymer_text",
        )

        sort_target = st.selectbox(
            "按哪个性能排序？",
            options=POLYMER_TARGETS if POLYMER_TARGETS else ["Tg"],
            key="bulk_sort_target",
        )

        if st.button("批量预测并筛选", key="btn_bulk"):
            if not POLYMER_TARGETS:
                st.error("未找到 Tg/Td 模型，请先训练")
            elif not bulk_text.strip():
                st.warning("请输入分子结构")
            else:
                smiles_list = [s.strip() for s in bulk_text.splitlines() if s.strip()]
                poly_models = {t: models[t] for t in POLYMER_TARGETS}
                results = predict.predict_batch(smiles_list, poly_models)
                if results.empty:
                    st.warning("所有输入都无法解析")
                else:
                    if sort_target in results.columns:
                        results = results.sort_values(sort_target, ascending=False).reset_index(drop=True)
                st.success(f"✅ 成功预测 {len(results)} 个分子")
                st.dataframe(results, width="stretch")
                st.download_button(
                    "⬇️ 下载结果 CSV",
                    data=results.to_csv(index=False).encode("utf-8-sig"),
                    file_name="screening_results.csv",
                    mime="text/csv",
                )

    # ---------- 模式 2：电导率批量预测（固定条件） ----------
    elif bulk_mode == "电导率批量预测（固定条件）":
        st.caption("对多个聚合物分子，用**相同的盐/浓度/温度**统一预测电导率。")
        if not COND_TARGET:
            st.info("未找到电导率模型，请先训练。")
        else:
            cond_meta2 = predict.load_meta(COND_TARGET)
            salt_meta2 = cond_meta2.get("salt_meta", {}) if cond_meta2 else {}
            top_salts2 = salt_meta2.get("top_salts", [])
            freq_salts2 = salt_meta2.get("freq_salts", [])

            c1, c2, c3 = st.columns(3)
            with c1:
                salt_fixed = st.selectbox(
                    "盐类型（固定）",
                    options=list(top_salts2) + list(freq_salts2),
                    key="salt_fixed",
                )
            with c2:
                mol_fixed = st.number_input(
                    "盐浓度 (mol/kg)（固定）",
                    min_value=0.0, max_value=20.0, value=1.0, step=0.1,
                    key="mol_fixed",
                )
            with c3:
                temp_fixed = st.number_input(
                    "温度 (°C)（固定）",
                    min_value=-70.0, max_value=200.0, value=25.0, step=5.0,
                    key="temp_fixed",
                )

            bulk_cond_text = st.text_area(
                "输入多个聚合物 PSMILES（每行一个）",
                height=150,
                placeholder="[*]CCO[*]\n[*]CC([*])C\n[*]CC([*])c1ccccc1\n...",
                key="bulk_cond_text",
            )

            if st.button("批量预测电导率", key="btn_bulk_cond"):
                smiles_list = [s.strip() for s in bulk_cond_text.splitlines() if s.strip()]
                if not smiles_list:
                    st.warning("请输入分子结构")
                else:
                    tg_model = models.get("Tg")
                    results = predict.predict_conductivity_batch(
                        smiles_list, salt_fixed, mol_fixed, temp_fixed,
                        models[COND_TARGET], salt_meta2, tg_model=tg_model,
                    )
                    if results.empty:
                        st.warning("所有输入都无法解析")
                    else:
                        st.success(f"✅ 成功预测 {len(results)} 个分子（条件: {salt_fixed} {mol_fixed} mol/kg {temp_fixed}°C）")
                        st.dataframe(results, width="stretch")
                        st.download_button(
                            "⬇️ 下载结果 CSV",
                            data=results.to_csv(index=False).encode("utf-8-sig"),
                            file_name="conductivity_batch_results.csv",
                            mime="text/csv",
                        )

    # ---------- 模式 3：电导率批量预测（CSV 逐行条件） ----------
    elif bulk_mode == "电导率批量预测（CSV 逐行条件）":
        st.caption("上传 CSV，每行包含自己的 **SMILES / salt / molality / temperature**，逐行预测电导率。")
        if not COND_TARGET:
            st.info("未找到电导率模型，请先训练。")
        else:
            st.markdown("**CSV 格式要求**（第一行为列名）：")
            st.code("SMILES,salt,molality,temperature\n[*]CCO[*],LiTFSI,1.0,25\n[*]CC([*])C,LiTFSI,1.0,60", language="csv")
            uploaded = st.file_uploader("上传 CSV 文件", type=["csv"], key="csv_upload")

            if uploaded is not None:
                try:
                    df_in = pd.read_csv(uploaded)
                    required = ["SMILES", "salt", "molality", "temperature"]
                    missing = [c for c in required if c not in df_in.columns]
                    if missing:
                        st.error(f"CSV 缺少必需列: {missing}（应为 SMILES, salt, molality, temperature）")
                    elif st.button("开始批量预测", key="btn_csv_batch"):
                        cond_meta3 = predict.load_meta(COND_TARGET)
                        salt_meta3 = cond_meta3.get("salt_meta", {}) if cond_meta3 else {}
                        tg_model = models.get("Tg")
                        results = predict.predict_conductivity_batch_rows(
                            df_in, models[COND_TARGET], salt_meta3, tg_model=tg_model,
                        )
                        n_ok = results["预测成功"].sum()
                        st.success(f"✅ 处理 {len(results)} 行，成功预测 {n_ok} 行")
                        st.dataframe(results, width="stretch")
                        st.download_button(
                            "⬇️ 下载结果 CSV",
                            data=results.to_csv(index=False).encode("utf-8-sig"),
                            file_name="conductivity_csv_results.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"读取 CSV 失败: {e}")


# ---------- Tab 3: 模型性能 ----------
with tab3:
    st.subheader("模型性能指标")
    if not models:
        st.info("暂无模型性能数据，请先训练模型。")
    else:
        perf_rows = []
        for name in models:
            meta = predict.load_meta(name)
            if meta:
                perf_rows.append({
                    "目标变量": name,
                    "CV R²(均值±std)": f"{meta['cv_r2_mean']:.3f} ± {meta['cv_r2_std']:.3f}",
                    "测试集 R²": f"{meta['test_metrics']['R2']:.3f}",
                    "MAE": f"{meta['test_metrics']['MAE']:.3f}",
                    "RMSE": f"{meta['test_metrics']['RMSE']:.3f}",
                })
        if perf_rows:
            st.dataframe(pd.DataFrame(perf_rows), width="stretch")
        else:
            st.info("模型目录中没有元数据文件")

        # 特征重要性
        st.subheader("特征重要性")
        feat_target = st.selectbox("选择目标变量查看特征重要性", options=list(models.keys()))
        meta = predict.load_meta(feat_target)
        if meta and "feature_importance" in meta:
            imp = pd.Series(meta["feature_importance"]).sort_values(ascending=True)
            st.bar_chart(imp)


st.markdown("---")
st.caption(f"模型目录: `{config.MODEL_DIR}` | 数据目录: `{config.DATA_DIR}`")
