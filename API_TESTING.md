# CreditIQ API Testing Guide

## Prerequisites

Make sure all models are available in the expected directories:
```
models/
├── xgboost/
│   ├── xgboost_final.pkl
│   └── shap_explainer.pkl
├── lstm/
│   ├── lstm_checkpoint.pt
│   └── lstm_scaler.pkl
├── finbert/
│   └── finbert_finetuned/
├── ensemble/
│   └── calibrator.pkl
```

And test data in:
```
data/processed/
├── X_test.parquet
└── y_test.csv
```

## Running the API

### Option 1: Direct run
```bash
cd creditiq
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using Python
```bash
cd creditiq
python -c "import uvicorn; uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=True)"
```

The API will be available at: `http://localhost:8000`

## Testing the API

### Option 1: Using the Test Script
```bash
cd creditiq
python test_api.py
```

This will test:
- ✓ Health checks
- ✓ Single loan assessment
- ✓ Batch assessment (3 applications)
- ✓ Edge cases (missing optional fields)

### Option 2: Interactive API Documentation
Open your browser to: `http://localhost:8000/docs`

This provides Swagger UI with:
- Try out all endpoints
- See request/response examples
- Validate request schemas

### Option 3: Using cURL

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Assessment:**
```bash
curl -X POST http://localhost:8000/assess/ \
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
    "loan_purpose": "Home renovation",
    "contract_type": "Cash loans"
  }'
```

### Option 4: Using Python requests
```python
import requests
import json

url = "http://localhost:8000/assess/"
payload = {
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
    "own_car": True,
    "own_realty": True,
    "education_type": "Higher education",
    "family_status": "Married",
    "housing_type": "House / apartment",
    "loan_purpose": "Home renovation",
    "contract_type": "Cash loans"
}

response = requests.post(url, json=payload)
result = response.json()
print(json.dumps(result, indent=2))
```

## API Response Example

```json
{
  "credit_score": 720,
  "risk_band": "Medium Risk",
  "default_probability": 0.2456,
  "recommendation": "APPROVE WITH CONDITIONS",
  "xgb_default_probability": 0.2543,
  "lstm_default_probability": 0.5,
  "sentiment_label": "neutral",
  "sentiment_confidence": 0.85,
  "top_risk_factors": [
    {
      "feature": "DEBT_TO_INCOME",
      "importance": 0.0854,
      "direction": "increases_risk"
    },
    {
      "feature": "ANNUITY_TO_INCOME",
      "importance": 0.0712,
      "direction": "increases_risk"
    }
  ],
  "top_protective_factors": [
    {
      "feature": "EXT_SOURCE_1",
      "importance": 0.1234,
      "direction": "decreases_risk"
    }
  ],
  "model_version": "1.0.0",
  "inference_time_ms": 245
}
```

## Endpoints

### 1. Health Check
```
GET /health
```
Returns status of all models

### 2. Single Assessment
```
POST /assess/
```
Assess a single loan application

**Request Body:** `LoanApplicationRequest`
**Response:** `AssessmentResponse`

### 3. Batch Assessment
```
POST /assess/batch
```
Assess up to 100 applications at once

**Request Body:** `BatchAssessmentRequest` (max 100 applications)
**Response:** `BatchAssessmentResponse` with summary statistics

### 4. API Documentation
```
GET /docs
```
Interactive Swagger UI

## Troubleshooting

### Issue: "Feature names not loaded"
**Solution:** Make sure `data/processed/X_test.parquet` exists

### Issue: "LSTM scaler not found"
**Solution:** Make sure `models/lstm/lstm_scaler.pkl` exists

### Issue: "Calibrator not found"
**Solution:** This is optional - API will fall back to XGBoost-only predictions

### Issue: Model inference is slow
**Possible causes:**
- First inference loads FinBERT (slow)
- GPU not available (using CPU)
- Add `payment_history` for LSTM predictions (will be cached)

### Issue: "Connection refused"
**Solution:** Make sure API is running with `uvicorn api.main:app --reload`

## Performance Metrics

Expected inference times (on CPU):
- Single assessment: 200-500ms
- Batch (10 apps): 2-5 seconds
- FinBERT (first call): +1-2 seconds

## Next Steps

1. ✓ Run the test script: `python test_api.py`
2. ✓ Check API docs at `http://localhost:8000/docs`
3. ✓ Deploy with production ASGI server (gunicorn, hypercorn)
4. ✓ Set up authentication (JWT in auth.py)
5. ✓ Configure database for audit logging
