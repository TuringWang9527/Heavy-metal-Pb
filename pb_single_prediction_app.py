import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# ---------------------- 1. 基础配置 ----------------------
st.set_page_config(page_title="Pb Adsorption Predictor", layout="wide")

# ---------------------- 2. 自定义 CSS ----------------------
st.markdown(
    """
<style>
body { background-color: #f5f7fa; font-family: "Helvetica Neue", Arial, sans-serif; }
.card { background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; }
.section-title { font-size: 18px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 15px; }
.label-col { text-align: left !important; width: 260px; padding-right: 10px; font-size: 13px; font-weight: 600; color: #555; }
.input-col { flex: 1; }
div[class*="stText"], div[class*="stNumberInput"], div[class*="stSelectbox"] { text-align: left !important; }
.stButton>button { background-color: #3498db !important; color: white !important; border-radius: 6px !important; padding: 10px 20px !important; border: 2px solid white !important; }
.stButton>button:hover { background-color: #2980b9 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- 3. 加载模型 ----------------------
# 注意：这个模型必须是用下面 15 个 Pb 数据集特征训练的。
# 如果仍然使用旧 SMX 数据集训练的 XGBoost.joblib，预测会报错或结果无效。
MODEL_PATH = Path("catboost_model.joblib")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "未找到 XGBoost.joblib。请把基于 Pb 数据集训练好的模型文件放在本 app.py 同级目录。"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------------- 4. Pb 数据集特征范围 ----------------------
# 数据来源：Pb插值后.xlsx；前 15 列为输入特征，最后一列 Qe(mg/g) 为目标值。
# default 使用该列均值，min/max 使用该列最小/最大值。
feature_ranges = {
    "C(%)": {"min": 8.47, "max": 88.3, "default": 61.317},
    "H(%)": {"min": 0.75, "max": 11.38, "default": 3.072},
    "O(%)": {"min": 0.3, "max": 46.17, "default": 16.903},
    "N(%)": {"min": 0.15, "max": 25.6, "default": 2.559},
    "(O+N)/C": {"min": 0.018, "max": 1.14, "default": 0.336},
    "O/C": {"min": 0.004, "max": 32.37, "default": 2.087},
    "H/C": {"min": 0.018, "max": 6.56, "default": 0.572},
    "pH of Biochar": {"min": 3.02, "max": 12.62, "default": 9.238},
    "SSA(m²/g)": {"min": 0.738, "max": 1224.0, "default": 67.285},
    "Initial Pb concentration(mg/L)": {"min": 2.0, "max": 1830.0, "default": 340.968},
    "Stirring speed(rpm)": {"min": 100.0, "max": 4000.0, "default": 348.013},
    "Volume (L)": {"min": 0.01, "max": 0.75, "default": 0.139},
    "Concentration of biochar in water(g/L)": {"min": 0.0001, "max": 50.0, "default": 2.776},
    "Adsorption temperature(℃)": {"min": 5.0, "max": 60.0, "default": 25.913},
    "Adsorption time(min)": {"min": 0.0, "max": 4760.0, "default": 421.520},
}
feature_names = list(feature_ranges.keys())


def get_model_feature_names(m):
    """尽量读取模型训练时保存的特征名；读不到时返回 None。"""
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "get_booster"):
        booster = m.get_booster()
        if getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    return None


trained_feature_names = get_model_feature_names(model)
if trained_feature_names is not None and trained_feature_names != feature_names:
    st.error("模型特征名与当前 Pb 数据集特征名不一致。请重新训练模型，或修改 feature_names 的顺序。")
    st.write("当前代码特征：", feature_names)
    st.write("模型内特征：", trained_feature_names)
    st.stop()
elif trained_feature_names is None:
    st.warning(
        "未能从模型文件读取训练特征名。当前代码会按 Pb 数据集列顺序输入特征；请确认模型训练时使用的是完全相同的 15 个特征及顺序。"
    )


def choose_step(min_value, max_value):
    span = max_value - min_value
    if span <= 1:
        return 0.001
    if span <= 10:
        return 0.01
    if span <= 100:
        return 0.1
    return 1.0


def choose_format(step):
    return "%.4f" if step < 0.01 else "%.3f"


def predict_single(input_df):
    pred = model.predict(input_df)
    return float(np.asarray(pred).ravel()[0])


def compute_shap_values(input_df):
    """只用于单样本解释；失败时返回 None，不影响预测。"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        shap_row = shap_values[0] if shap_values.ndim == 2 else shap_values

        base_value = explainer.expected_value
        base_value = float(np.asarray(base_value).ravel()[0])
        return shap_row, base_value
    except Exception:
        return None, None


# ---------------------- 5. 单样本预测界面：只保留原来的第一个功能 ----------------------
st.title("Pb Adsorption Capacity Predictor")
st.caption("Single prediction only. Target: Qe(mg/g)")

with st.container():
    st.markdown(
        '<div class="card"><h3 class="section-title">Experimental Parameters</h3>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    feature_values = []

    for idx, (feature, props) in enumerate(feature_ranges.items()):
        min_v = float(props["min"])
        max_v = float(props["max"])
        default_v = float(props["default"])
        step = choose_step(min_v, max_v)

        with cols[idx % 3]:
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                f'<div class="label-col">{feature}</div><div class="input-col">',
                unsafe_allow_html=True,
            )
            value = st.number_input(
                label=feature,
                min_value=min_v,
                max_value=max_v,
                value=default_v,
                step=step,
                format=choose_format(step),
                label_visibility="collapsed",
                key=f"input_{idx}",
            )
            feature_values.append(value)
            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict Result", type="primary", use_container_width=True):
    input_data = pd.DataFrame([feature_values], columns=feature_names)

    try:
        pred_value = predict_single(input_data)
    except Exception as e:
        st.error("预测失败。最常见原因是模型不是用当前 Pb 数据集的 15 个特征训练的，或特征顺序不一致。")
        st.exception(e)
        st.stop()

    shap_row, base_value = compute_shap_values(input_data)

    st.session_state.result = {
        "pred": pred_value,
        "shap": shap_row,
        "base": base_value,
        "input": input_data,
    }

if "result" in st.session_state:
    res = st.session_state.result

    st.markdown("### Prediction Dashboard")
    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.info("Predicted Adsorption Capacity")
        st.metric(label="Qe (mg/g)", value=f"{res['pred']:.4f}", delta="Model Output")
        if res["base"] is not None:
            st.write("Base Value (Average):", f"{res['base']:.4f}")

    with col_res2:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=res["pred"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Adsorption Capacity Performance"},
                gauge={
                    "axis": {"range": [0, 1000]},
                    "bar": {"color": "#3498db"},
                    "steps": [
                        {"range": [0, 100], "color": "#e0e0e0"},
                        {"range": [100, 300], "color": "#bdc3c7"},
                        {"range": [300, 1000], "color": "#95a5a6"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": res["pred"],
                    },
                },
            )
        )
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP 仍属于单样本预测解释；不再包含依赖分析、交互分析、逆向优化、全局重要性、对比分析、批量预测。
    if res["shap"] is not None:
        st.markdown("### Model Explanation (SHAP)")
        col_shap1, col_shap2 = st.columns([2, 1])

        with col_shap1:
            try:
                shap_exp = shap.Explanation(
                    values=res["shap"],
                    base_values=res["base"],
                    data=res["input"].iloc[0].values,
                    feature_names=feature_names,
                )
                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(shap_exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                st.warning(f"SHAP waterfall plot unavailable: {e}")

        with col_shap2:
            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Input Value": res["input"].iloc[0].values,
                    "SHAP Value": res["shap"],
                }
            )
            shap_df["Abs"] = shap_df["SHAP Value"].abs()
            st.dataframe(
                shap_df.sort_values("Abs", ascending=False).drop(columns="Abs"),
                height=420,
                use_container_width=True,
            )
    else:
        st.info("预测已完成；当前模型类型不支持 TreeExplainer，已跳过 SHAP 解释。")
