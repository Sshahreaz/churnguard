"""Train the churn model offline and persist artifacts for the API.

Run from the backend directory:
    python -m scripts.train_and_save

Writes:
    app/artifacts/model.pkl          — fitted RandomForest (joblib)
    app/artifacts/risk_scores.json   — all customer scores + dashboard stats

The SHAP explainer is not saved: reasons are baked into risk_scores.json
at train time, so the API never recomputes SHAP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.data_cleaning import load_and_clean  # noqa: E402
from app.model import get_risk_scores, train_model  # noqa: E402

ARTIFACTS_DIR = BACKEND_ROOT / "app" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
RISK_SCORES_PATH = ARTIFACTS_DIR / "risk_scores.json"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    cleaned_df, customer_ids = load_and_clean()
    print(f"  Rows: {len(cleaned_df)}")

    print("Training RandomForest...")
    model = train_model()
    joblib.dump(model, MODEL_PATH)
    print(f"  Saved model -> {MODEL_PATH}")

    print("Scoring all customers with SHAP reasons (this may take a few minutes)...")
    scores, cutoffs = get_risk_scores(cleaned_df)

    customers = []
    for idx, row in scores.iterrows():
        customers.append(
            {
                "customer_id": int(customer_ids.loc[idx]),
                "churn_probability": float(row["churn_probability"]),
                "risk_level": row["risk_level"],
                "top_3_reasons": list(row["top_3_reasons"]),
            }
        )

    high = sum(1 for c in customers if c["risk_level"] == "High")
    medium = sum(1 for c in customers if c["risk_level"] == "Medium")
    low = sum(1 for c in customers if c["risk_level"] == "Low")
    overall_churn_rate = float(cleaned_df["Churn"].mean() * 100)
    average_churn_probability = float(
        sum(c["churn_probability"] for c in customers) / len(customers)
    )

    payload = {
        "cutoffs": cutoffs,
        "dashboard": {
            "total_customers": len(customers),
            "high_risk_count": high,
            "medium_risk_count": medium,
            "low_risk_count": low,
            "overall_churn_rate": round(overall_churn_rate, 2),
            "average_churn_probability": round(average_churn_probability, 4),
        },
        "customers": customers,
    }

    RISK_SCORES_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved risk scores -> {RISK_SCORES_PATH}")
    print(
        f"  Distribution: High={high}, Medium={medium}, Low={low} | "
        f"cutoffs p80={cutoffs['p80']:.4f}, p95={cutoffs['p95']:.4f}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
