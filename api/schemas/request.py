from pydantic import BaseModel, Field
from typing import Optional, List


class LoanApplicationRequest(BaseModel):
    # Personal info
    age_years: float = Field(..., ge=18, le=100,
                              description="Applicant age in years")
    gender: str = Field(..., pattern="^[MF]$",
                         description="M or F")
    family_members: int = Field(..., ge=1, le=20)
    children_count: int = Field(0, ge=0, le=20)

    # Financial info
    income_total: float = Field(..., gt=0,
                                 description="Annual income")
    credit_amount: float = Field(..., gt=0,
                                  description="Loan amount requested")
    annuity_amount: float = Field(..., gt=0,
                                   description="Monthly annuity")
    goods_price: Optional[float] = Field(None, gt=0)

    # Employment
    employment_years: Optional[float] = Field(None, ge=0)
    income_type: str = Field("Working",
                              description="Type of income source")
    organization_type: str = Field("Business Entity Type 3")
    occupation_type: Optional[str] = None

    # Credit history
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # Property
    own_car: bool = False
    own_realty: bool = False
    car_age: Optional[float] = None

    # Education & family
    education_type: str = Field("Secondary / secondary special")
    family_status: str = Field("Single / not married")
    housing_type: str = Field("House / apartment")

    # Loan purpose (for NLP)
    loan_purpose: str = Field("General loan application",
                               description="Purpose of loan in text")

    # Contract
    contract_type: str = Field("Cash loans")

    class Config:
        json_schema_extra = {
            "example": {
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
                "loan_purpose": "Home renovation for family property",
                "contract_type": "Cash loans"
            }
        }


class BatchAssessmentRequest(BaseModel):
    applications: List[LoanApplicationRequest] = Field(
        ..., min_length=1, max_length=100)