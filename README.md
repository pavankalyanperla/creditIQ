# CreditIQ — Intelligent Credit Risk Intelligence Platform

> A full-stack, ML-powered credit risk assessment system combining predictive modeling,
> NLP sentiment analysis, time series forecasting, and a live analytics dashboard —
> served via a REST API.

---

## Architecture

```
Data Sources → Feature Engineering → ML Engine → REST API → Dashboard
```

| Layer | Tech |
|---|---|
| Data & ML | Python, Pandas, Scikit-learn, XGBoost, LightGBM, SHAP, FinBERT, PyTorch LSTM, Prophet |
| API | FastAPI, Pydantic, JWT Auth, PostgreSQL, Redis |
| Dashboard | Streamlit + Plotly |
| MLOps | MLflow |
| Deployment | Docker + Docker Compose, GitHub Actions CI/CD |

---

## Project Structure

```
creditiq/
├── data/
│   ├── raw/                  # Original Home Credit CSVs (gitignored)
│   ├── processed/            # Cleaned, merged, feature-engineered data
│   └── external/             # FRED macro data, Financial PhraseBank
│
├── notebooks/                # Jupyter notebooks (EDA, modelling, analysis)
│
├── models/
│   ├── xgboost/              # Trained XGBoost default predictor + SHAP
│   ├── finbert/              # Fine-tuned FinBERT sentiment model
│   ├── lstm/                 # PyTorch LSTM risk forecaster
│   └── ensemble/             # Calibrated ensemble score generator
│
├── api/
│   ├── main.py               # FastAPI app entry point
│   ├── routers/              # Endpoint definitions
│   ├── schemas/              # Pydantic request/response models
│   ├── services/             # ML inference, business logic
│   └── db/                   # SQLAlchemy models, migrations
│
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   ├── pages/                # Multi-page dashboard screens
│   └── components/           # Reusable chart components
│
├── tests/
│   ├── api/                  # FastAPI endpoint tests
│   └── models/               # Model inference tests
│
├── scripts/                  # Data download, preprocessing scripts
├── docs/                     # Architecture docs, reports
├── mlflow_runs/              # MLflow experiment tracking (gitignored)
├── .github/workflows/        # GitHub Actions CI/CD
├── config.py                 # Central settings (loaded from .env)
├── requirements.txt          # All Python dependencies
├── .env.example              # Environment variable template
├── docker-compose.yml        # Full stack: API + Dashboard + DB + Cache
└── setup.py                  # One-shot environment setup script
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/creditiq.git
cd creditiq

# 2. Run setup (creates venv, installs deps, copies .env)
python setup.py

# 3. Activate virtual environment
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 4. Add your API keys to .env
nano .env

# 5. Move your Home Credit CSVs to data/raw/

# 6. Run the API
uvicorn api.main:app --reload

# 7. Run the dashboard
streamlit run dashboard/app.py
```

---

## ML Models

| Model | Purpose | Technique | Key Metric |
|---|---|---|---|
| Default Predictor | Predict P(default) | XGBoost + SHAP | ROC-AUC, KS-stat, Gini |
| Sentiment Engine | Score loan text | FinBERT (fine-tuned) | F1-score |
| Risk Forecaster | 12-month PD trajectory | PyTorch LSTM / Prophet | MAE, RMSE |
| Score Generator | Final 300–850 credit score | Calibrated Ensemble | Brier score |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/assess` | Assess a single loan application |
| POST | `/batch-score` | Batch score multiple applications |
| GET | `/forecast/{loan_id}` | 12-month default probability forecast |
| GET | `/explain/{loan_id}` | SHAP explanation for a decision |
| GET | `/portfolio` | Portfolio-level analytics |
| POST | `/token` | Obtain JWT access token |
| GET | `/health` | API health check |

---

## Data Sources

- **Home Credit Default Risk** — Kaggle (primary training data)
- **Financial PhraseBank** — Hugging Face (NLP sentiment)
- **FRED API** — Federal Reserve macro indicators
- **Yahoo Finance** — Market signals via `yfinance`

---

## Model Performance

*(To be updated after training)*

| Model | ROC-AUC | KS-Statistic | Gini |
|---|---|---|---|
| XGBoost | — | — | — |
| LightGBM | — | — | — |
| Ensemble | — | — | — |

---

## License

MIT
