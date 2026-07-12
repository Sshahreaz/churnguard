# ChurnGuard

AI-powered customer churn prediction and retention tool. Scores every customer by churn risk, ranks them into a prioritized call list, and explains *why* in plain English.

**Live app:** https://churnguard-zeta.vercel.app
**API:** https://churnguard-lyud.onrender.com

> Backend runs on Render's free tier, so the first request after inactivity may take 10-20 seconds to wake up.

## What it does

Retention teams usually find out a customer is unhappy after they've already left. ChurnGuard scores every customer, ranks them by risk, and gives a plain-English reason so the team knows who to call and why, before they churn.

## The interesting part

The first version explained risk using global feature importance, which ranks features by how much they matter on average. This caused a bug: a customer with a 1.5% churn probability (very safe) was shown reasons like "long distance from warehouse," phrased like a risk factor, because the logic didn't account for which direction the feature actually pushed *that* customer's prediction.

Fixed it with **SHAP values**, which calculate direction and magnitude per individual prediction. Now safe customers correctly show protective reasons ("long-standing customer, no recent complaints") instead of confusing stats.

## Features

- Ranked risk list with plain-English SHAP explanations per customer
- Percentile-based risk tiers (top 5% / next 15% / rest), so tiers stay meaningfully populated
- Filterable, searchable customer table
- Customer detail modal with a suggested next action
- Live KPI dashboard (total customers, risk counts, churn rate)

## Tech stack

**Backend:** FastAPI, scikit-learn (Random Forest), SHAP, pandas — deployed on Render
**Frontend:** Next.js, TypeScript, Tailwind CSS — deployed on Vercel

Model trains offline (`scripts/train_and_save.py`) and saves results to disk, so the live API just loads pre-computed data instead of retraining on every restart.

## Dataset

[E-Commerce Customer Churn dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (Kaggle), 5,630 customers, 20 features.

Missing values weren't random. Customers missing `Tenure` churned at nearly double the baseline rate, so each affected column got a binary `_missing` flag plus median imputation, preserving that signal instead of discarding it.

## Model performance

Accuracy 98%, Precision 99%, Recall 89%, F1 94%. Notably high largely because `Tenure` is an unusually strong signal in this dataset. I'd expect more modest numbers on messier real-world data.

## Running locally

**Backend**
```bash
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.train_and_save
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

## API endpoints

- `GET /health`
- `GET /dashboard/summary`
- `GET /customers/risk?risk_level=High|Medium|Low`

Full docs at `/docs` on the backend URL.

## V1 scope

No auth, no live database, no historical trend tracking. Data is a static CSV with pre-computed model artifacts. Likely next step: replace the CSV with a PostgreSQL database and add scheduled retraining.

## Author

Shadman Shahreaz Rhythm
[GitHub](https://github.com/Sshahreaz)
