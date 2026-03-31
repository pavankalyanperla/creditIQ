import os

import streamlit as st

st.set_page_config(
    page_title="CreditIQ Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image("https://img.shields.io/badge/CreditIQ-v1.0-blue", width=150)
st.sidebar.title("CreditIQ")
st.sidebar.caption("Intelligent Credit Risk Platform")

page = st.sidebar.selectbox(
    "Navigate", ["Loan Assessment", "Portfolio Analytics", "Model Performance"]
)

st.sidebar.markdown("---")

API_URL = os.getenv("API_BASE_URL", "https://creditiq-api.onrender.com")
st.sidebar.info(f"API: {API_URL}")

if page == "Loan Assessment":
    from dashboard.page_modules import assessment

    assessment.show(API_URL)
elif page == "Portfolio Analytics":
    from dashboard.page_modules import portfolio

    portfolio.show(API_URL)
elif page == "Model Performance":
    from dashboard.page_modules import model_performance

    model_performance.show()
