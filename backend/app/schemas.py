"""Pydantic response schemas for the ChurnGuard API."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class CustomerRisk(BaseModel):
    customer_id: int
    churn_probability: float
    risk_level: Literal["High", "Medium", "Low"]
    top_3_reasons: list[str] = Field(
        ...,
        description="SHAP-based plain-English risk or protective factors",
    )


class DashboardSummary(BaseModel):
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    overall_churn_rate: float = Field(
        ...,
        description="Actual churn rate from labels, as a percentage",
    )
    average_churn_probability: float
