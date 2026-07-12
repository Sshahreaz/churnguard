"""Churn prediction model logic for ChurnGuard.

Trains a balanced RandomForest on the cleaned E Comm data and exposes
per-customer risk scores with SHAP-based plain-English reasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.data_cleaning import load_and_clean

CATEGORICAL_COLUMNS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "PreferedOrderCat",
    "Gender",
    "MaritalStatus",
]

# Fixed sample IDs for __main__ comparisons across explanation methods.
SAMPLE_CUSTOMER_IDS = [50001, 50008, 50027]

# Risk-factor phrasing (positive SHAP / High-Medium risk).
FEATURE_LABELS: dict[str, str] = {
    "Churn": "Churned",
    "Tenure": "Short customer tenure",
    "CityTier": "Lives in a lower-tier city",
    "WarehouseToHome": "Long warehouse-to-home distance",
    "HourSpendOnApp": "Low time spent on the app",
    "NumberOfDeviceRegistered": "Many devices registered",
    "SatisfactionScore": "Low satisfaction score",
    "NumberOfAddress": "Many addresses on file",
    "Complain": "Filed a recent complaint",
    "OrderAmountHikeFromlastYear": "Large order-amount increase vs last year",
    "CouponUsed": "Heavy coupon usage",
    "OrderCount": "High order count",
    "DaySinceLastOrder": "Long time since last order",
    "CashbackAmount": "High cashback amount",
    "Tenure_missing": "No tenure data on file",
    "WarehouseToHome_missing": "No warehouse-distance data on file",
    "HourSpendOnApp_missing": "No app-usage time on file",
    "OrderAmountHikeFromlastYear_missing": "No order-hike data on file",
    "CouponUsed_missing": "No coupon-usage data on file",
    "OrderCount_missing": "No order-count data on file",
    "DaySinceLastOrder_missing": "No last-order date on file",
    "PreferredLoginDevice_Mobile Phone": "Prefers logging in on mobile",
    "PreferredLoginDevice_Computer": "Prefers logging in on computer",
    "PreferredLoginDevice_Phone": "Prefers logging in on phone",
    "PreferredPaymentMode_Credit Card": "Pays with credit card",
    "PreferredPaymentMode_Debit Card": "Pays with debit card",
    "PreferredPaymentMode_E wallet": "Pays with e-wallet",
    "PreferredPaymentMode_UPI": "Pays with UPI",
    "PreferredPaymentMode_COD": "Pays cash on delivery",
    "PreferredPaymentMode_CC": "Pays with credit card",
    "PreferredPaymentMode_Cash on Delivery": "Pays cash on delivery",
    "PreferedOrderCat_Mobile Phone": "Mostly orders mobile phones",
    "PreferedOrderCat_Mobile": "Mostly orders mobiles",
    "PreferedOrderCat_Fashion": "Mostly orders fashion",
    "PreferedOrderCat_Grocery": "Mostly orders grocery",
    "PreferedOrderCat_Laptop & Accessory": "Mostly orders laptops & accessories",
    "PreferedOrderCat_Others": "Orders from other categories",
    "Gender_Male": "Customer is male",
    "Gender_Female": "Customer is female",
    "MaritalStatus_Single": "Customer is single",
    "MaritalStatus_Married": "Customer is married",
    "MaritalStatus_Divorced": "Customer is divorced",
}

# Protective phrasing (negative SHAP / Low risk).
FEATURE_LABELS_PROTECTIVE: dict[str, str] = {
    "Churn": "Not churned",
    "Tenure": "Long-standing customer",
    "CityTier": "Lives in a higher-tier city",
    "WarehouseToHome": "Short warehouse-to-home distance",
    "HourSpendOnApp": "Spends meaningful time on the app",
    "NumberOfDeviceRegistered": "Stable number of registered devices",
    "SatisfactionScore": "Healthy satisfaction score",
    "NumberOfAddress": "Stable address history",
    "Complain": "No recent complaints filed",
    "OrderAmountHikeFromlastYear": "Steady order amounts vs last year",
    "CouponUsed": "Moderate coupon usage",
    "OrderCount": "Regular order activity",
    "DaySinceLastOrder": "Ordered recently",
    "CashbackAmount": "Healthy cashback engagement",
    "Tenure_missing": "Tenure data available on file",
    "WarehouseToHome_missing": "Warehouse-distance data available on file",
    "HourSpendOnApp_missing": "App-usage time available on file",
    "OrderAmountHikeFromlastYear_missing": "Order-hike data available on file",
    "CouponUsed_missing": "Coupon-usage data available on file",
    "OrderCount_missing": "Order-count data available on file",
    "DaySinceLastOrder_missing": "Last-order date available on file",
    "PreferredLoginDevice_Mobile Phone": "Stable mobile login preference",
    "PreferredLoginDevice_Computer": "Stable computer login preference",
    "PreferredLoginDevice_Phone": "Stable phone login preference",
    "PreferredPaymentMode_Credit Card": "Pays reliably with credit card",
    "PreferredPaymentMode_Debit Card": "Pays reliably with debit card",
    "PreferredPaymentMode_E wallet": "Pays reliably with e-wallet",
    "PreferredPaymentMode_UPI": "Pays reliably with UPI",
    "PreferredPaymentMode_COD": "Comfortable with cash on delivery",
    "PreferredPaymentMode_CC": "Pays reliably with credit card",
    "PreferredPaymentMode_Cash on Delivery": "Comfortable with cash on delivery",
    "PreferedOrderCat_Mobile Phone": "Consistent mobile-phone shopper",
    "PreferedOrderCat_Mobile": "Consistent mobile shopper",
    "PreferedOrderCat_Fashion": "Consistent fashion shopper",
    "PreferedOrderCat_Grocery": "Consistent grocery shopper",
    "PreferedOrderCat_Laptop & Accessory": "Consistent laptop & accessory shopper",
    "PreferedOrderCat_Others": "Consistent shopper across other categories",
    "Gender_Male": "Customer is male",
    "Gender_Female": "Customer is female",
    "MaritalStatus_Single": "Customer is single",
    "MaritalStatus_Married": "Customer is married",
    "MaritalStatus_Divorced": "Customer is divorced",
}

# Module-level artifacts filled by train_model() / get_risk_scores()
_model: RandomForestClassifier | None = None
_explainer: shap.TreeExplainer | None = None
_feature_columns: list[str] | None = None
_X_test: pd.DataFrame | None = None
_y_test: pd.Series | None = None
_X_train: pd.DataFrame | None = None
_y_train: pd.Series | None = None
# Percentile cutoffs from the last get_risk_scores() call on a customer base
_risk_cutoffs: dict[str, float] | None = None


def _plain_label(feature_name: str, *, protective: bool = False) -> str:
    """Map a model feature name to a business-friendly reason string."""
    mapping = FEATURE_LABELS_PROTECTIVE if protective else FEATURE_LABELS
    if feature_name in mapping:
        return mapping[feature_name]
    return feature_name.replace("_", " ")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode known categoricals with drop_first=True."""
    cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    return pd.get_dummies(df, columns=cols, drop_first=True)


def _align_features(encoded: pd.DataFrame) -> pd.DataFrame:
    """Align encoded columns to the training feature set (missing -> 0)."""
    if _feature_columns is None:
        raise RuntimeError("Model is not trained yet. Call train_model() first.")
    X = encoded.drop(columns=["Churn"], errors="ignore")
    X = X.reindex(columns=_feature_columns, fill_value=0)
    return X


def _churn_shap_matrix(X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for the churn (positive) class: shape (n, n_features)."""
    assert _explainer is not None
    shap_values = _explainer.shap_values(X)

    # Newer SHAP may return an Explanation object.
    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            # (n_samples, n_features, n_classes) — class 1 = churn
            return values[:, :, 1]
        return values

    # Older API: list of arrays per class, or a single 2D array.
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def train_model(
    random_state: int = 42,
) -> RandomForestClassifier:
    """Load cleaned data, encode, split, and train the RandomForest.

    Also builds a ``shap.TreeExplainer`` on the fitted model for local
    per-customer explanations.
    """
    global _model, _explainer, _feature_columns
    global _X_train, _X_test, _y_train, _y_test

    cleaned_df, _customer_ids = load_and_clean()
    encoded = encode_features(cleaned_df)

    y = encoded["Churn"].astype(int)
    X = encoded.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    _model = model
    _explainer = shap.TreeExplainer(model)
    _feature_columns = list(X_train.columns)
    _X_train, _X_test = X_train, X_test
    _y_train, _y_test = y_train, y_test

    return model


def _top_reasons_from_shap(
    shap_row: np.ndarray,
    feature_names: list[str],
    risk_level: str,
    top_n: int = 3,
) -> list[str]:
    """Pick top SHAP-driven reasons for one customer.

    High/Medium: largest positive SHAP (risk factors).
    Low: largest-magnitude negative SHAP (protective factors).
    """
    if risk_level in ("High", "Medium"):
        # Ascending sort then take last top_n of positive contributors.
        order = np.argsort(shap_row)
        chosen: list[int] = []
        for idx in order[::-1]:
            if shap_row[idx] <= 0:
                break
            chosen.append(int(idx))
            if len(chosen) >= top_n:
                break
        # If fewer than top_n positives, pad with next-largest overall.
        if len(chosen) < top_n:
            for idx in order[::-1]:
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen) >= top_n:
                    break
        return [_plain_label(feature_names[i], protective=False) for i in chosen]

    # Low risk: most negative SHAP values (protective).
    order = np.argsort(shap_row)  # most negative first
    chosen = []
    for idx in order:
        if shap_row[idx] >= 0:
            break
        chosen.append(int(idx))
        if len(chosen) >= top_n:
            break
    if len(chosen) < top_n:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= top_n:
                break
    return [_plain_label(feature_names[i], protective=True) for i in chosen]


def get_risk_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score a cleaned DataFrame with churn probability, risk level, and reasons.

    Risk bands are percentile-based on this customer set's probability
    distribution:
    - High:   probability >= 95th percentile
    - Medium: probability >= 80th percentile and < 95th percentile
    - Low:    everything else

    ``top_3_reasons`` uses per-customer SHAP values from ``TreeExplainer``:
    risk factors for High/Medium, protective factors for Low.

    Parameters
    ----------
    df :
        Cleaned feature frame as returned by ``load_and_clean()`` (may include
        Churn). Categoricals are one-hot encoded to match the trained model.

    Returns
    -------
    scores : pd.DataFrame
        Columns: churn_probability, risk_level, top_3_reasons
    cutoffs : dict[str, float]
        ``{"p80": ..., "p95": ...}`` — probability thresholds used for
        Medium and High bands.
    """
    global _risk_cutoffs

    if _model is None or _feature_columns is None or _explainer is None:
        train_model()

    assert _model is not None and _feature_columns is not None
    encoded = encode_features(df)
    X = _align_features(encoded)
    proba = _model.predict_proba(X)[:, 1]
    shap_matrix = _churn_shap_matrix(X)

    p80 = float(np.percentile(proba, 80))
    p95 = float(np.percentile(proba, 95))
    cutoffs = {"p80": p80, "p95": p95}
    _risk_cutoffs = cutoffs

    risk_levels: list[str] = []
    reasons: list[list[str]] = []
    for i, p in enumerate(proba):
        if p >= p95:
            level = "High"
        elif p >= p80:
            level = "Medium"
        else:
            level = "Low"
        risk_levels.append(level)
        reasons.append(
            _top_reasons_from_shap(shap_matrix[i], _feature_columns, level)
        )

    scores = pd.DataFrame(
        {
            "churn_probability": proba,
            "risk_level": risk_levels,
            "top_3_reasons": reasons,
        },
        index=df.index,
    )
    return scores, cutoffs


def _print_report() -> None:
    """Train (if needed) and print evaluation + percentile risk report."""
    if _model is None:
        train_model()

    assert _model is not None and _X_test is not None and _y_test is not None
    assert _feature_columns is not None

    y_pred = _model.predict(_X_test)
    print("=== Test-set metrics ===")
    print(f"Accuracy:  {accuracy_score(_y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(_y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(_y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(_y_test, y_pred):.4f}")
    print()

    importances = (
        pd.Series(_model.feature_importances_, index=_feature_columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("=== Top 10 global feature importances ===")
    for name, value in importances.items():
        print(f"  {value:.4f}  {name}  ({_plain_label(str(name))})")
    print()

    cleaned_df, customer_ids = load_and_clean()
    scores, cutoffs = get_risk_scores(cleaned_df)

    print("=== Percentile risk cutoffs ===")
    print(f"  Medium threshold (80th percentile): {cutoffs['p80']:.4f}")
    print(f"  High threshold   (95th percentile): {cutoffs['p95']:.4f}")
    print()

    dist = scores["risk_level"].value_counts()
    print("=== Risk level distribution ===")
    for level in ("High", "Medium", "Low"):
        print(f"  {level}: {int(dist.get(level, 0))}")
    print()

    print("=== Sample customers (risk score + reasons) ===")
    for cid in SAMPLE_CUSTOMER_IDS:
        idx = customer_ids[customer_ids == cid].index[0]
        row = scores.loc[idx]
        reasons = "; ".join(row["top_3_reasons"])
        print(f"CustomerID {cid}:")
        print(f"  churn_probability: {row['churn_probability']:.4f}")
        print(f"  risk_level:        {row['risk_level']}")
        print(f"  top_3_reasons:     {reasons}")
        print()


if __name__ == "__main__":
    _print_report()
