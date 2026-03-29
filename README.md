# CreditIQ — Intelligent Credit Risk Intelligence Platform

> A full-stack, production-grade credit risk assessment system combining
> predictive modeling, NLP sentiment analysis, time series forecasting,
> and a live analytics dashboard — served via a REST API.

[![CI](https://github.com/pavankalyanperla/creditIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/pavankalyanperla/creditIQ/actions)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://creditiq-api.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com)

---

## Live Demo

| Resource | URL |
|---|---|
| API Docs (Swagger) | https://creditiq-api.onrender.com/docs |
| Health Check | https://creditiq-api.onrender.com/health |
| Portfolio Analytics | https://creditiq-api.onrender.com/portfolio/ |

> Note: Free tier spins down after inactivity — first request may take 30-50 seconds to wake up.

---

## Project Overview

CreditIQ is a credit risk intelligence platform built as a B.Tech major project
combining **Computer Science (Data Science specialization)** and **Finance & Banking**.

It takes a loan application and returns:
- A **credit score** (300–850 scale)
- A **risk band** (Low / Medium / High / Very High)
- A **lending recommendation** (Approve / Review / Decline)
- **SHAP explanations** — why the decision was made
- A **12-month default probability forecast**
- **NLP sentiment** analysis on loan purpose text

---

## Architecture

```
Data Sources → Feature Engineering → ML Engine → REST API → Dashboard
     ↓                ↓                  ↓            ↓           ↓
Home Credit      253 features       4 ML models   FastAPI    Streamlit
  Bureau         engineered        XGBoost        7 endpoints  3 pages
  FRED API       from 7 CSVs       FinBERT        JWT auth    Plotly charts
  yfinance                         LSTM           Redis cache  SHAP viz
                                   Ensemble
```

---

## ML Models

| Model | Purpose | Technique | Metric |
|---|---|---|---|
| XGBoost Default Predictor | Predict P(default) | XGBoost + Optuna + SHAP | ROC-AUC: **0.7875** |
| FinBERT Sentiment Engine | Score loan text | Fine-tuned FinBERT | F1: **0.9846** |
| LSTM Risk Forecaster | 12-month PD trajectory | PyTorch LSTM + Attention | ROC-AUC: 0.5771 |
| Calibrated Ensemble | Final 300-850 score | Platt scaling | Brier: **0.0661** |

### vs Industry Standards

| Metric | CreditIQ | Industry Standard | Status |
|---|---|---|---|
| ROC-AUC | 0.7875 | 0.72–0.78 | ✅ Above industry |
| KS Statistic | 0.4342 | 0.35–0.45 | ✅ Good range |
| Gini Coefficient | 0.5750 | 0.44–0.56 | ✅ Above industry |

---

## Data Sources

| Source | Description | Size |
|---|---|---|
| Home Credit Default Risk | Primary loan data (Kaggle) | 307,511 applications |
| Financial PhraseBank | NLP sentiment training (HuggingFace) | 2,264 sentences |
| FRED API | Macro indicators (unemployment, GDP, CPI) | 194 observations |
| Yahoo Finance | Market signals (S&P500, VIX, US10Y) | 4,079 data points |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data & ML | Python, Pandas, Scikit-learn, XGBoost, SHAP, Optuna, FinBERT, PyTorch, Prophet |
| API | FastAPI, Pydantic, JWT Auth, Redis |
| Dashboard | Streamlit, Plotly |
| MLOps | MLflow experiment tracking |
| Deployment | Render (API), Docker, GitHub Actions CI/CD |
| Database | SQL Server (local), SQLite (cloud) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/assess/` | Assess a single loan application |
| POST | `/assess/batch` | Batch score up to 100 applications |
| GET | `/forecast/{id}` | 12-month default probability forecast |
| GET | `/portfolio/` | Portfolio-level risk analytics |
| POST | `/auth/token` | Obtain JWT access token |
| GET | `/health` | API health check |
| GET | `/docs` | Interactive Swagger UI |

### Example Request

```bash
curl -X POST https://creditiq-api.onrender.com/assess/ \
  -H "Content-Type: application/json" \
  -d '{
    "age_years": 35,
    "gender": "M",
    "family_members": 2,
    "children_count": 0,
    "income_total": 180000,
    "credit_amount": 450000,
    "annuity_amount": 22500,
    "goods_price": 450000,
    "employment_years": 5,
    "income_type": "Working",
    "organization_type": "Business Entity Type 3",
    "ext_source_1": 0.6,
    "ext_source_2": 0.7,
    "ext_source_3": 0.65,
    "own_car": true,
    "own_realty": true,
    "education_type": "Higher education",
    "family_status": "Married",
    "housing_type": "House / apartment",
    "loan_purpose": "Home renovation for family property",
    "contract_type": "Cash loans"
  }'
```

### Example Response

```json
{
  "credit_score": 721,
  "risk_band": "Medium Risk",
  "default_probability": 0.182,
  "recommendation": "APPROVE WITH CONDITIONS",
  "xgb_default_probability": 0.243,
  "sentiment_label": "neutral",
  "sentiment_confidence": 0.997,
  "top_risk_factors": [...],
  "top_protective_factors": [...],
  "model_version": "1.0.0",
  "inference_time_ms": 176
}
```

---

## Project Structure

```
creditiq/
├── data/
│   ├── raw/                  # Home Credit CSVs (gitignored)
│   ├── processed/            # Feature engineered data + EDA plots
│   └── external/             # FRED, PhraseBank, market data
├── notebooks/                # 9 Jupyter notebooks (EDA → Ensemble)
├── models/
│   ├── xgboost/              # XGBoost model + SHAP explainer
│   ├── finbert/              # Fine-tuned FinBERT
│   ├── lstm/                 # PyTorch LSTM checkpoint
│   └── ensemble/             # Calibrated ensemble
├── api/
│   ├── main.py               # FastAPI app
│   ├── routers/              # assess, auth, forecast, portfolio
│   ├── schemas/              # Pydantic request/response models
│   └── services/             # ML inference pipeline
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   └── pages/                # assessment, portfolio, model_performance
├── tests/                    # Pytest test suite
├── scripts/                  # Data download scripts
├── .github/workflows/        # GitHub Actions CI/CD
├── render.yaml               # Render deployment config
├── docker-compose.yml        # Full local stack
└── requirements.txt          # All dependencies
```

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/pavankalyanperla/creditIQ.git
cd creditiq

# 2. Setup
python setup.py

# 3. Activate venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# 4. Add your API keys to .env

# 5. Run API
uvicorn api.main:app --reload --port 8000

# 6. Run Dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## EDA Findings

- **307,511** loan applications × 122 features
- **8.07%** default rate — severe class imbalance addressed with SMOTE
- **EXT_SOURCE_MEAN** is the strongest predictor (SHAP importance: 0.44)
- Younger borrowers (20–25) default at **12.3%** vs 4.9% for 60–70 age group
- Males default **45% more** than females (10.1% vs 7.0%)
- Higher education reduces default risk by **6x** vs lower secondary

---

## Feature Engineering

| Group | Features | Examples |
|---|---|---|
| Financial ratios | 20 | DEBT_TO_INCOME, ANNUITY_TO_INCOME, EXT_SOURCE_MEAN |
| Bureau features | 17 | BUREAU_ACTIVE_RATIO, BB_MAX_STATUS_EVER, BB_DPD_RATIO |
| Previous applications | 14 | PREV_APPROVAL_RATE, PREV_REFUSAL_RATE |
| Installment payments | 16 | INST_LATE_RATIO, INST_PAYMENT_RATIO_MEAN |
| POS Cash balance | 13 | POS_DPD_RATIO, POS_COMPLETED_RATIO |
| Credit card balance | 13 | CC_UTILIZATION_MEAN, CC_DPD_RATIO |
| One-hot encoded | 38 | Education, income type, housing type |
| **Total** | **253** | |

---

## Author

**Pavan Kalyan Perla**
B.Tech Computer Science (Data Science) | Minor: Finance & Banking

---

## License

MIT