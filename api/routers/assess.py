from fastapi import APIRouter, Request, HTTPException
from api.schemas.request import (
    LoanApplicationRequest, BatchAssessmentRequest)
from api.schemas.response import (
    AssessmentResponse, BatchAssessmentResponse)
import numpy as np

router = APIRouter()


@router.post("/", response_model=AssessmentResponse)
async def assess_application(
    application: LoanApplicationRequest,
    request: Request
):
    """
    Assess a single loan application.
    Returns credit score, risk band, and SHAP explanations.
    """
    try:
        models = request.app.state.models
        result = models.assess(application)
        return AssessmentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchAssessmentResponse)
async def batch_assess(
    batch: BatchAssessmentRequest,
    request: Request
):
    """
    Assess multiple loan applications at once.
    Maximum 100 applications per request.
    """
    try:
        models = request.app.state.models
        results = []
        for app in batch.applications:
            result = models.assess(app)
            results.append(AssessmentResponse(**result))

        scores = [r.credit_score for r in results]
        summary = {
            "average_score":    float(np.mean(scores)),
            "min_score":        min(scores),
            "max_score":        max(scores),
            "approve_count":    sum(1 for r in results
                                    if r.recommendation == "APPROVE"),
            "decline_count":    sum(1 for r in results
                                    if r.recommendation == "DECLINE"),
            "review_count":     sum(1 for r in results
                                    if "REVIEW" in r.recommendation),
        }

        return BatchAssessmentResponse(
            total=len(batch.applications),
            processed=len(results),
            results=results,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))