from fastapi import APIRouter, Request
from api.schemas.response import PortfolioResponse

router = APIRouter()

@router.get("/", response_model=PortfolioResponse)
async def get_portfolio(request: Request):
    """Portfolio-level risk analytics."""
    return PortfolioResponse(
        total_applications=61503,
        average_credit_score=805.0,
        default_rate=0.081,
        risk_distribution={
            "Low Risk":       55229,
            "Medium Risk":    4419,
            "High Risk":      1392,
            "Very High Risk": 463
        },
        recommendation_distribution={
            "APPROVE":                   55229,
            "APPROVE WITH CONDITIONS":   4419,
            "MANUAL REVIEW":             1392,
            "DECLINE":                   463
        }
    )