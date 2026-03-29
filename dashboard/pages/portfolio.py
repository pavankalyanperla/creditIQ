import plotly.express as px
import requests
import streamlit as st


def show(api_url: str):
    st.title("Portfolio Analytics")
    st.caption("Portfolio-level credit risk metrics")

    try:
        response = requests.get(f"{api_url}/portfolio/", timeout=10)
        data = response.json()

        # ── KPI Cards ─────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications",
                      f"{data['total_applications']:,}")
        with col2:
            st.metric("Avg Credit Score",
                      f"{data['average_credit_score']:.0f}")
        with col3:
            st.metric("Default Rate",
                      f"{data['default_rate']*100:.1f}%")
        with col4:
            approve_rate = (
                data['recommendation_distribution']['APPROVE'] /
                data['total_applications'] * 100
            )
            st.metric("Approval Rate", f"{approve_rate:.1f}%")

        st.markdown("---")
        col5, col6 = st.columns(2)

        # ── Risk Distribution ─────────────────────────────────
        with col5:
            st.subheader("Risk Band Distribution")
            risk_data = data["risk_distribution"]
            fig_risk = px.pie(
                values=list(risk_data.values()),
                names=list(risk_data.keys()),
                color_discrete_map={
                    "Low Risk":       "#2ecc71",
                    "Medium Risk":    "#f39c12",
                    "High Risk":      "#e67e22",
                    "Very High Risk": "#e74c3c"
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        # ── Recommendation Distribution ───────────────────────
        with col6:
            st.subheader("Recommendation Distribution")
            rec_data = data["recommendation_distribution"]
            fig_rec = px.bar(
                x=list(rec_data.keys()),
                y=list(rec_data.values()),
                color=list(rec_data.keys()),
                color_discrete_map={
                    "APPROVE":                 "#2ecc71",
                    "APPROVE WITH CONDITIONS": "#f39c12",
                    "MANUAL REVIEW":           "#e67e22",
                    "DECLINE":                 "#e74c3c"
                }
            )
            st.plotly_chart(fig_rec, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load portfolio data: {e}")
        st.info("Make sure the API is running on port 8000")