from typing import Dict, List

from pydantic import BaseModel, ConfigDict


class SHAPFeature(BaseModel):
    feature: str
    importance: float
    direction: str  # "increases_risk" or "decreases_risk"


class AssessmentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Core output
    credit_score: int
    risk_band: str
    default_probability: float
    recommendation: str

    # Sub-model outputs
    xgb_default_probability: float
    lstm_default_probability: float
    sentiment_label: str
    sentiment_confidence: float

    # Explainability
    top_risk_factors: List[SHAPFeature]
    top_protective_factors: List[SHAPFeature]

    # Metadata
    model_version: str = "1.0.0"
    inference_time_ms: int


class ForecastPoint(BaseModel):
    month: int
    default_probability: float


class ForecastResponse(BaseModel):
    application_id: str
    forecast: List[ForecastPoint]
    trend: str  # "improving", "stable", "deteriorating"
    peak_risk_month: int
    peak_risk_probability: float


class BatchAssessmentResponse(BaseModel):
    total: int
    processed: int
    results: List[AssessmentResponse]
    summary: Dict


class PortfolioResponse(BaseModel):
    total_applications: int
    average_credit_score: float
    default_rate: float
    risk_distribution: Dict[str, int]
    recommendation_distribution: Dict[str, int]
