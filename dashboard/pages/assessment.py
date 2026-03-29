import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def show(api_url: str):
    st.title("Loan Assessment")
    st.caption("Submit a loan application and get an instant credit risk assessment")

    # ── Input Form ────────────────────────────────────────────
    with st.form("loan_form"):
        st.subheader("Applicant Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", 18, 70, 35)
            gender = st.selectbox("Gender", ["M", "F"])
            family_members = st.number_input("Family members", 1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)

        with col2:
            income = st.number_input("Annual income (₹)", 10000, 10000000, 180000, step=10000)
            credit_amount = st.number_input("Loan amount (₹)", 10000, 5000000, 450000, step=10000)
            annuity = st.number_input("Monthly annuity (₹)", 1000, 500000, 22500, step=1000)
            goods_price = st.number_input("Goods price (₹)", 10000, 5000000, 450000, step=10000)

        with col3:
            employment_years = st.number_input("Employment years", 0.0, 50.0, 5.0, step=0.5)
            education = st.selectbox("Education", [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree"
            ])
            family_status = st.selectbox("Family status", [
                "Married", "Single / not married",
                "Civil marriage", "Separated", "Widow"
            ])
            housing = st.selectbox("Housing type", [
                "House / apartment", "Rented apartment",
                "With parents", "Municipal apartment",
                "Office apartment"
            ])

        st.subheader("Credit History")
        col4, col5, col6 = st.columns(3)

        with col4:
            ext1 = st.slider("External Score 1", 0.0, 1.0, 0.6, 0.01)
            ext2 = st.slider("External Score 2", 0.0, 1.0, 0.7, 0.01)
            ext3 = st.slider("External Score 3", 0.0, 1.0, 0.65, 0.01)

        with col5:
            own_car = st.checkbox("Owns a car", value=True)
            own_realty = st.checkbox("Owns property", value=True)
            car_age = st.number_input("Car age (years)", 0.0, 50.0, 5.0) if own_car else 0.0

        with col6:
            income_type = st.selectbox("Income type", [
                "Working", "Commercial associate",
                "Pensioner", "State servant", "Student"
            ])
            organization = st.selectbox("Organization type", [
                "Business Entity Type 3", "School", "Government",
                "Medicine", "Self-employed", "Military", "Police"
            ])
            loan_purpose = st.text_area(
                "Loan purpose",
                "Home renovation for family property",
                height=80
            )

        submitted = st.form_submit_button(
            "Assess Application",
            use_container_width=True,
            type="primary"
        )

    # ── API Call & Results ────────────────────────────────────
    if submitted:
        payload = {
            "age_years": age,
            "gender": gender,
            "family_members": family_members,
            "children_count": children,
            "income_total": income,
            "credit_amount": credit_amount,
            "annuity_amount": annuity,
            "goods_price": goods_price,
            "employment_years": employment_years,
            "income_type": income_type,
            "organization_type": organization,
            "ext_source_1": ext1,
            "ext_source_2": ext2,
            "ext_source_3": ext3,
            "own_car": own_car,
            "own_realty": own_realty,
            "car_age": car_age,
            "education_type": education,
            "family_status": family_status,
            "housing_type": housing,
            "loan_purpose": loan_purpose,
            "contract_type": "Cash loans"
        }

        with st.spinner("Analyzing application..."):
            try:
                response = requests.post(
                    f"{api_url}/assess/",
                    json=payload,
                    timeout=60
                )
                result = response.json()

                # ── Score Display ─────────────────────────────
                st.markdown("---")
                st.subheader("Assessment Results")

                col1, col2, col3, col4 = st.columns(4)

                score = result["credit_score"]
                risk_band = result["risk_band"]
                rec = result["recommendation"]
                default_prob = result["default_probability"]

                # Color based on recommendation
                if rec == "APPROVE":
                    rec_color = "green"
                    rec_icon = "✅"
                elif rec == "APPROVE WITH CONDITIONS":
                    rec_color = "orange"
                    rec_icon = "⚠️"
                elif rec == "MANUAL REVIEW":
                    rec_color = "orange"
                    rec_icon = "🔍"
                else:
                    rec_color = "red"
                    rec_icon = "❌"

                with col1:
                    st.metric("Credit Score", score,
                              delta=f"{score - 600} from threshold")
                with col2:
                    st.metric("Risk Band", risk_band)
                with col3:
                    st.metric("Default Probability",
                              f"{default_prob*100:.1f}%")
                with col4:
                    st.metric("Inference Time",
                              f"{result['inference_time_ms']}ms")

                # Recommendation banner
                st.markdown(
                    f"<h2 style='text-align:center; color:{rec_color}'>"
                    f"{rec_icon} {rec}</h2>",
                    unsafe_allow_html=True
                )

                # ── Credit Score Gauge ────────────────────────
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Credit Score"},
                    gauge={
                        "axis": {"range": [300, 850]},
                        "bar":  {"color": "darkblue"},
                        "steps": [
                            {"range": [300, 550], "color": "#e74c3c"},
                            {"range": [550, 650], "color": "#e67e22"},
                            {"range": [650, 750], "color": "#f1c40f"},
                            {"range": [750, 850], "color": "#2ecc71"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": score
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)

                col_gauge, col_sent = st.columns(2)
                with col_gauge:
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_sent:
                    st.subheader("Sentiment Analysis")
                    sentiment = result["sentiment_label"]
                    confidence = result["sentiment_confidence"]
                    sent_color = {
                        "positive": "#2ecc71",
                        "neutral":  "#f39c12",
                        "negative": "#e74c3c"
                    }.get(sentiment, "#gray")

                    st.markdown(
                        f"<h3 style='color:{sent_color}'>"
                        f"{sentiment.upper()}</h3>",
                        unsafe_allow_html=True
                    )
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence*100:.1f}%")
                    st.caption(f"Loan purpose analyzed: '{loan_purpose[:80]}'")

                # ── SHAP Explanations ─────────────────────────
                st.subheader("Decision Explanation (SHAP)")
                col_risk, col_prot = st.columns(2)

                with col_risk:
                    st.markdown("**Top Risk Factors**")
                    risk_factors = result["top_risk_factors"]
                    if risk_factors:
                        rf_df = pd.DataFrame(risk_factors)
                        fig_risk = px.bar(
                            rf_df,
                            x="importance",
                            y="feature",
                            orientation="h",
                            color_discrete_sequence=["#e74c3c"],
                            title="Factors increasing risk"
                        )
                        fig_risk.update_layout(height=300)
                        st.plotly_chart(fig_risk,
                                        use_container_width=True)

                with col_prot:
                    st.markdown("**Top Protective Factors**")
                    prot_factors = result["top_protective_factors"]
                    if prot_factors:
                        pf_df = pd.DataFrame(prot_factors)
                        fig_prot = px.bar(
                            pf_df,
                            x="importance",
                            y="feature",
                            orientation="h",
                            color_discrete_sequence=["#2ecc71"],
                            title="Factors decreasing risk"
                        )
                        fig_prot.update_layout(height=300)
                        st.plotly_chart(fig_prot,
                                        use_container_width=True)

                # ── Raw JSON ──────────────────────────────────
                with st.expander("View raw API response"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")