"""FastAPI app entry point for ChurnGuard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.data_cleaning import load_and_clean
from app.model import get_risk_scores, train_model
from app.schemas import CustomerRisk, DashboardSummary, HealthResponse

# In-memory store populated once at startup
_customer_risks: list[CustomerRisk] = []
_dashboard_summary: DashboardSummary | None = None


def _build_startup_state() -> tuple[list[CustomerRisk], DashboardSummary]:
    """Load data, train once, score all customers, compute dashboard stats."""
    cleaned_df, customer_ids = load_and_clean()
    train_model()
    scores, _cutoffs = get_risk_scores(cleaned_df)

    risks: list[CustomerRisk] = []
    for idx, row in scores.iterrows():
        risks.append(
            CustomerRisk(
                customer_id=int(customer_ids.loc[idx]),
                churn_probability=float(row["churn_probability"]),
                risk_level=row["risk_level"],
                top_3_reasons=list(row["top_3_reasons"]),
            )
        )

    high = sum(1 for r in risks if r.risk_level == "High")
    medium = sum(1 for r in risks if r.risk_level == "Medium")
    low = sum(1 for r in risks if r.risk_level == "Low")
    overall_churn_rate = float(cleaned_df["Churn"].mean() * 100)
    average_churn_probability = float(
        sum(r.churn_probability for r in risks) / len(risks)
    )

    summary = DashboardSummary(
        total_customers=len(risks),
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        overall_churn_rate=round(overall_churn_rate, 2),
        average_churn_probability=round(average_churn_probability, 4),
    )
    return risks, summary


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Train model and precompute all customer risk scores on startup."""
    global _customer_risks, _dashboard_summary
    _customer_risks, _dashboard_summary = _build_startup_state()
    yield


app = FastAPI(
    title="ChurnGuard API",
    description="AI-powered customer churn prediction and retention tool",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(status="ok")


@app.get("/customers/risk", response_model=list[CustomerRisk])
def customers_risk(
    risk_level: Literal["High", "Medium", "Low"] | None = Query(
        default=None,
        description="Optional filter by risk band",
    ),
    sort_by: Literal["churn_probability", "customer_id"] = Query(
        default="churn_probability",
        description="Field to sort by (default: churn_probability descending)",
    ),
) -> list[CustomerRisk]:
    """Return all (or filtered) customers with precomputed risk scores."""
    rows = _customer_risks
    if risk_level is not None:
        rows = [r for r in rows if r.risk_level == risk_level]

    reverse = sort_by == "churn_probability"
    if sort_by == "churn_probability":
        rows = sorted(rows, key=lambda r: r.churn_probability, reverse=reverse)
    else:
        rows = sorted(rows, key=lambda r: r.customer_id, reverse=False)

    return rows


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary() -> DashboardSummary:
    """Aggregate risk and churn stats for the dashboard."""
    if _dashboard_summary is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return _dashboard_summary
