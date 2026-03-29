import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path


def show():
    st.title("Model Performance")
    st.caption("ML model metrics and evaluation results")

    # ── Model Metrics Cards ───────────────────────────────────
    st.subheader("Model Comparison")

    metrics_data = {
        "Model":      ["XGBoost", "FinBERT", "LSTM", "Ensemble"],
        "Primary Metric": [0.7875, 0.9846, 0.5771, 0.7875],
        "Metric Name":["ROC-AUC", "F1 Score", "ROC-AUC", "ROC-AUC"],
        "KS Stat":    [0.4342, None, 0.1323, 0.4342],
        "Gini":       [0.5750, None, 0.1541, 0.5750],
        "Status":     ["Production", "Production",
                       "Production", "Production"]
    }

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    colors = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]
    models = ["XGBoost", "FinBERT", "LSTM", "Ensemble"]
    metrics = [0.7875, 0.9846, 0.5771, 0.7875]
    labels = ["ROC-AUC", "F1 Score", "ROC-AUC", "ROC-AUC"]

    for col, model, metric, label, color in zip(
            cols, models, metrics, labels, colors):
        with col:
            st.markdown(
                f"<div style='background-color:{color}22; "
                f"padding:15px; border-radius:10px; "
                f"border-left:4px solid {color}'>"
                f"<h4 style='color:{color}'>{model}</h4>"
                f"<h2>{metric:.4f}</h2>"
                f"<p>{label}</p></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── ROC-AUC Bar Chart ─────────────────────────────────────
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("ROC-AUC Comparison")
        fig_auc = go.Figure(go.Bar(
            x=["XGBoost", "LSTM", "Ensemble"],
            y=[0.7875, 0.5771, 0.7875],
            marker_color=["#3498db", "#e67e22", "#2ecc71"],
            text=[0.7875, 0.5771, 0.7875],
            textposition="auto"
        ))
        fig_auc.add_hline(
            y=0.75,
            line_dash="dash",
            line_color="red",
            annotation_text="Industry standard (0.75)"
        )
        fig_auc.update_layout(
            yaxis_range=[0, 1],
            title="ROC-AUC by Model"
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    with col8:
        st.subheader("Banking Metrics — XGBoost")
        fig_metrics = go.Figure(go.Bar(
            x=["ROC-AUC", "KS Statistic", "Gini"],
            y=[0.7875, 0.4342, 0.5750],
            marker_color=["#3498db", "#9b59b6", "#2ecc71"],
            text=[0.7875, 0.4342, 0.5750],
            textposition="auto"
        ))
        fig_metrics.update_layout(
            yaxis_range=[0, 1],
            title="XGBoost Banking Metrics"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    # ── Industry Comparison Table ─────────────────────────────
    st.subheader("vs Industry Standards")
    comparison = pd.DataFrame({
        "Metric":    ["ROC-AUC", "KS Statistic", "Gini"],
        "CreditIQ":  [0.7875, 0.4342, 0.5750],
        "Industry":  ["0.72-0.78", "0.35-0.45", "0.44-0.56"],
        "Status":    ["✅ Above", "✅ Good", "✅ Above"]
    })
    st.dataframe(comparison, use_container_width=True,
                  hide_index=True)

    # ── SHAP Feature Importance ───────────────────────────────
    st.subheader("Top SHAP Features")
    shap_path = (Path(__file__).parent.parent.parent /
                  "models" / "xgboost" /
                  "shap_feature_importance.csv")
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path).head(15)
        fig_shap = px.bar(
            shap_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            title="Top 15 Features by SHAP Importance"
        )
        fig_shap.update_layout(height=500)
        st.plotly_chart(fig_shap, use_container_width=True)
        