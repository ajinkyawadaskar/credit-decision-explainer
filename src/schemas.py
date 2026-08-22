"""Pydantic request/response models for the decision API.

Validation here is deliberately strict: categorical fields accept only the
exact A-codes from the UCI codebook, and numerics carry real bounds. A credit
decision endpoint that silently accepts garbage and returns a confident score
is worse than one that refuses.

The prohibited-basis fields (sex, national origin, age) are NOT part of the
request model at all. They cannot be submitted, so they cannot influence a
decision — the constraint is enforced by the type system, not by convention.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Literal types generated from the UCI codebook (see data.py::CODE_LABELS).
CheckingStatus = Literal["A11", "A12", "A13", "A14"]
CreditHistory = Literal["A30", "A31", "A32", "A33", "A34"]
Purpose = Literal["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]
SavingsStatus = Literal["A61", "A62", "A63", "A64", "A65"]
EmploymentSince = Literal["A71", "A72", "A73", "A74", "A75"]
OtherDebtors = Literal["A101", "A102", "A103"]
PropertyMagnitude = Literal["A121", "A122", "A123", "A124"]
OtherInstallmentPlans = Literal["A141", "A142", "A143"]
Housing = Literal["A151", "A152", "A153"]
Job = Literal["A171", "A172", "A173", "A174"]
Telephone = Literal["A191", "A192"]


class ApplicantRequest(BaseModel):
    """One credit application. All 17 modeled features are required.

    No defaults on purpose: a missing field is a data problem the caller must
    fix, not something the API should quietly guess at.
    """

    # Real record: test row 96, the worked DECLINE example (P(bad)=0.9353).
    # Not hand-composed — copied verbatim so the docs example is reproducible.
    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [{
        "checking_status": "A11", "duration_months": 45, "credit_history": "A30",
        "purpose": "A49", "credit_amount": 11816, "savings_status": "A61",
        "employment_since": "A75", "installment_rate_pct_income": 2,
        "other_debtors": "A101", "residence_since_years": 4,
        "property_magnitude": "A123", "other_installment_plans": "A143",
        "housing": "A151", "existing_credits_count": 2, "job": "A173",
        "num_dependents": 1, "telephone": "A191",
    }]}}

    checking_status: CheckingStatus = Field(..., description="Status of existing checking account")
    duration_months: int = Field(..., ge=1, le=120, description="Loan term in months")
    credit_history: CreditHistory
    purpose: Purpose
    credit_amount: int = Field(..., ge=1, le=1_000_000, description="Requested amount (DM)")
    savings_status: SavingsStatus
    employment_since: EmploymentSince
    installment_rate_pct_income: int = Field(..., ge=1, le=4, description="Installment rate as % of disposable income")
    other_debtors: OtherDebtors
    residence_since_years: int = Field(..., ge=1, le=4)
    property_magnitude: PropertyMagnitude
    other_installment_plans: OtherInstallmentPlans
    housing: Housing
    existing_credits_count: int = Field(..., ge=1, le=10)
    job: Job
    num_dependents: int = Field(..., ge=1, le=10)
    telephone: Telephone

    @field_validator("*", mode="before")
    @classmethod
    def _reject_blank(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("must not be blank")
        return v


class DriverOut(BaseModel):
    """One SHAP contribution, as returned to the caller."""

    feature: str
    value: str
    display_value: str
    shap_value: float = Field(..., description=">0 pushed toward decline, <0 toward approve")
    direction: Literal["toward_decline", "toward_approve"]


class ReasonOut(BaseModel):
    """One adverse action reason. Populated once reason_mapper.py exists."""

    code: str
    text: str
    feature: str = Field(..., description="SHAP feature this reason traces to")
    shap_value: float
    rank: int


class DecisionResponse(BaseModel):
    decision: Literal["APPROVE", "DECLINE"]
    probability_bad: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    base_value: float = Field(..., description="Population-average model output, log-odds")
    top_drivers: list[DriverOut]
    reasons: Optional[list[ReasonOut]] = Field(
        None,
        description="Adverse action reasons. Null when reason generation is not wired up.",
    )
    reasons_status: str = Field(
        ..., description="'ok' | 'not_implemented' — whether reason generation ran"
    )
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    reason_pipeline: str
    n_features: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
