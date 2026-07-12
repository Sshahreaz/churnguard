"""FastAPI app entry point for ChurnGuard."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import CustomerRisk, DashboardSummary, HealthResponse

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RISK_SCORES_PATH = ARTIFACTS_DIR / "risk_scores.json"

# In-memory store populated once at startup from precomputed artifacts
_customer_risks: list[CustomerRisk] = []
_dashboard_summary: DashboardSummary | None = None


def _cors_origins() -> list[str]:
    """Parse CORS_ORIGINS (comma-separated); default to local Next.js."""
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _load_startup_state() -> tuple[list[CustomerRisk], DashboardSummary]:
    """Load precomputed risk scores from disk (no training / SHAP at runtime)."""
    if not RISK_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RISK_SCORES_PATH}. Run: python -m scripts.train_and_save"
        )

    payload = json.loads(RISK_SCORES_PATH.read_text(encoding="utf-8"))
    risks = [CustomerRisk(**row) for row in payload["customers"]]
    summary = DashboardSummary(**payload["dashboard"])
    return risks, summary


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load precomputed risk scores into memory on startup."""
    global _customer_risks, _dashboard_summary
    _customer_risks, _dashboard_summary = _load_startup_state()
    yield


app = FastAPI(
    title="ChurnGuard API",
    description="AI-powered customer churn prediction and retention tool",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
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
