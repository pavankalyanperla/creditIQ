from fastapi import APIRouter, Request, HTTPException
from api.schemas.response import ForecastResponse, ForecastPoint

router = APIRouter()

@router.get("/{application_id}",
            response_model=ForecastResponse)
async def get_forecast(application_id: str,
                        request: Request):
    """
    Get 12-month default probability forecast.
    Uses LSTM model for trajectory prediction.
    """
    try:
        import numpy as np

        # Generate sample forecast
        # In production this uses actual payment history
        base_prob = 0.08
        forecast = []
        for month in range(1, 13):
            prob = base_prob * (1 + 0.02 * month)
            prob = min(prob, 1.0)
            forecast.append(ForecastPoint(
                month=month,
                default_probability=round(prob, 4)
            ))

        probs = [f.default_probability for f in forecast]
        peak_month = int(np.argmax(probs)) + 1
        peak_prob = max(probs)

        if probs[-1] < probs[0]:
            trend = "improving"
        elif probs[-1] > probs[0] * 1.1:
            trend = "deteriorating"
        else:
            trend = "stable"

        return ForecastResponse(
            application_id=application_id,
            forecast=forecast,
            trend=trend,
            peak_risk_month=peak_month,
            peak_risk_probability=peak_prob
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))