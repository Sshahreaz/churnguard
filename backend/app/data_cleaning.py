"""Data cleaning for the ChurnGuard E Comm dataset.

Decisions here are driven by EDA findings in eda_report.py:
- Duplicate category labels that mean the same thing get collapsed.
- Missingness is informative for churn, so we retain binary missingness flags
  before median-imputing the original columns.
- WarehouseToHome has extreme outliers (max 127 vs median ~14); we cap at the
  99th percentile rather than dropping rows.
"""

from pathlib import Path

import pandas as pd

# Resolve against this file so uvicorn works from project root or backend/
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ecommerce_churn.csv"

# Columns with nulls from EDA; missingness correlates with churn for several.
COLUMNS_WITH_NULLS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
]


def load_and_clean(
    data_path: Path | str = DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the churn CSV and apply cleaning only (no scaling/encoding).

    Steps
    -----
    1. Read ``data/ecommerce_churn.csv``.
    2. Pull CustomerID into its own Series and drop it from features.
       CustomerID is an identifier, not a predictive feature.
    3. Collapse inconsistent category labels discovered in EDA:
       - PreferredLoginDevice: "Phone" -> "Mobile Phone"
       - PreferredPaymentMode: "CC" -> "Credit Card", "Cash on Delivery" -> "COD"
       - PreferedOrderCat: "Mobile" -> "Mobile Phone"
    4. For each of the 7 columns with nulls:
       - Add a binary ``{column}_missing`` flag (1 if originally missing).
         Missingness is not MCAR for several fields (e.g. Tenure missing ~
         2x churn rate), so the flag preserves that signal after imputation.
       - Fill remaining nulls with the column median (robust to skew/outliers).
    5. Cap WarehouseToHome at the 99th percentile to limit extreme distance
       values without discarding rows.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Feature matrix after cleaning (CustomerID removed).
    customer_ids : pd.Series
        Original CustomerID values aligned to cleaned_df rows.
    """
    df = pd.read_csv(data_path)

    # --- 2. Separate identifier from features ---
    customer_ids = df["CustomerID"].copy()
    df = df.drop(columns=["CustomerID"])

    # --- 3. Fix inconsistent category labels ---
    # EDA showed overlapping synonyms that would otherwise become separate
    # one-hot levels and dilute signal.
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(
        {"Phone": "Mobile Phone"}
    )
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
        {
            "CC": "Credit Card",
            "Cash on Delivery": "COD",
        }
    )
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(
        {"Mobile": "Mobile Phone"}
    )

    # --- 4. Missingness flags + median imputation ---
    for col in COLUMNS_WITH_NULLS:
        # Flag first, while nulls are still present.
        df[f"{col}_missing"] = df[col].isna().astype(int)
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # --- 5. Cap WarehouseToHome outliers ---
    # Cap after imputation so the percentile is computed on a complete column.
    # 99th percentile keeps nearly all mass while clipping the extreme tail
    # (EDA: max 127 vs median 14).
    cap = df["WarehouseToHome"].quantile(0.99)
    df["WarehouseToHome"] = df["WarehouseToHome"].clip(upper=cap)

    return df, customer_ids


def _print_summary() -> None:
    """Print a short cleaning summary when this module is run directly."""
    raw = pd.read_csv(DATA_PATH)
    cleaned_df, customer_ids = load_and_clean()

    missing_flags = [c for c in cleaned_df.columns if c.endswith("_missing")]

    print("ChurnGuard data cleaning summary")
    print(f"Shape before: {raw.shape}")
    print(f"Shape after:  {cleaned_df.shape}")
    print(f"CustomerIDs retained separately: {len(customer_ids)}")
    print(f"Missing flags added ({len(missing_flags)}): {missing_flags}")
    print("Final column list:")
    for col in cleaned_df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    _print_summary()
