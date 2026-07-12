"""Exploratory data analysis report for the E Comm churn dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "E Commerce Dataset.xlsx"
SHEET_NAME = "E Comm"

# Known categoricals that may load as numeric
LIKELY_CATEGORICAL_NUMERIC = [
    "Churn",
    "CityTier",
    "SatisfactionScore",
    "Complain",
]

CATEGORICAL_COLUMNS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "PreferedOrderCat",
    "MaritalStatus",
    "Gender",
]


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_data() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)


def report_missing_values(df: pd.DataFrame) -> pd.Series:
    section("1. MISSING VALUES PER COLUMN")
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }
    ).sort_values("missing_count", ascending=False)

    print(f"Total rows: {len(df)}")
    print()
    print(report.to_string())
    print()
    cols_with_missing = report[report["missing_count"] > 0]
    if cols_with_missing.empty:
        print("No missing values found.")
    else:
        print(f"Columns with missing values: {len(cols_with_missing)}")
    return missing_count[missing_count > 0]


def report_missingness_vs_churn(df: pd.DataFrame, cols_with_missing: pd.Series) -> None:
    section("2. CHURN DISTRIBUTION BY MISSINGNESS")
    if cols_with_missing.empty:
        print("No columns with missing values — skipping.")
        return

    overall_rate = df["Churn"].mean() * 100
    print(f"Overall churn rate for reference: {overall_rate:.2f}%")
    print()

    for col in cols_with_missing.index:
        is_missing = df[col].isna()
        print(f"--- {col} ---")
        print(f"Missing rows: {is_missing.sum()} | Non-missing rows: {(~is_missing).sum()}")
        print()

        for label, mask in [("MISSING", is_missing), ("NOT MISSING", ~is_missing)]:
            subset = df.loc[mask, "Churn"]
            if len(subset) == 0:
                print(f"  {label}: no rows")
                continue
            counts = subset.value_counts().sort_index()
            rate = subset.mean() * 100
            print(f"  {label}:")
            for churn_val, count in counts.items():
                pct = count / len(subset) * 100
                print(f"    Churn={churn_val}: {count} ({pct:.2f}%)")
            print(f"    Churn rate: {rate:.2f}%")
        print()


def report_dtypes(df: pd.DataFrame) -> None:
    section("3. DATA TYPES (with categorical-as-numeric flags)")
    print(f"{'Column':<32} {'dtype':<12} Flag")
    print("-" * 72)
    for col in df.columns:
        dtype = str(df[col].dtype)
        flag = ""
        if col in LIKELY_CATEGORICAL_NUMERIC and pd.api.types.is_numeric_dtype(df[col]):
            flag = "<- likely categorical, loaded as numeric"
        print(f"{col:<32} {dtype:<12} {flag}")
    print()
    print(
        "Note: CityTier, SatisfactionScore, and Complain (and Churn itself) "
        "are ordinal/binary categories stored as numbers."
    )


def report_categorical_value_counts(df: pd.DataFrame) -> None:
    section("4. VALUE COUNTS FOR CATEGORICAL COLUMNS")
    print(
        "Look for inconsistent labels that may mean the same thing "
        '(e.g. "Mobile Phone" vs "Phone").'
    )
    print()
    for col in CATEGORICAL_COLUMNS:
        print(f"--- {col} ---")
        if col not in df.columns:
            print(f"  Column not found in dataset.")
            print()
            continue
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            pct = count / len(df) * 100
            display = "<NaN>" if pd.isna(value) else repr(value)
            print(f"  {display}: {count} ({pct:.2f}%)")
        print(f"  Unique values: {df[col].nunique(dropna=False)}")
        print()


def report_numeric_descriptives(df: pd.DataFrame) -> None:
    section("5. NUMERIC DESCRIPTIVE STATS (min, max, mean, median)")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude ID and known categorical-coded numerics from "outlier" focus,
    # but still show stats for all numeric columns.
    print(f"{'Column':<30} {'min':>10} {'max':>10} {'mean':>12} {'median':>12}")
    print("-" * 78)

    flags: list[str] = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            print(f"{col:<30} {'n/a':>10} {'n/a':>10} {'n/a':>12} {'n/a':>12}")
            continue
        mn, mx, mean, med = s.min(), s.max(), s.mean(), s.median()
        print(f"{col:<30} {mn:10.2f} {mx:10.2f} {mean:12.2f} {med:12.2f}")

        # Heuristic flags for possible outliers / data entry errors
        if col == "CustomerID":
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((s < lower) | (s > upper)).sum()
            if n_out > 0:
                flags.append(
                    f"  {col}: {n_out} values outside IQR fences "
                    f"[{lower:.2f}, {upper:.2f}] (min={mn:.2f}, max={mx:.2f})"
                )
        if mn < 0:
            flags.append(f"  {col}: has negative values (min={mn:.2f})")
        if col == "HourSpendOnApp" and mx > 24:
            flags.append(f"  {col}: max={mx:.2f} exceeds 24 hours/day")
        if col == "SatisfactionScore" and (mn < 1 or mx > 5):
            flags.append(
                f"  {col}: values outside typical 1-5 range (min={mn:.2f}, max={mx:.2f})"
            )
        if col == "Tenure" and mx > 100:
            flags.append(f"  {col}: unusually high max tenure ({mx:.2f})")
        if col == "WarehouseToHome" and mx > 100:
            flags.append(f"  {col}: unusually high distance max ({mx:.2f})")
        if col == "NumberOfAddress" and mx > 20:
            flags.append(f"  {col}: high address count max ({mx:.2f})")

    print()
    print("Potential outliers / data entry concerns:")
    if flags:
        for f in flags:
            print(f)
    else:
        print("  None flagged by simple heuristics.")


def report_churn_balance(df: pd.DataFrame) -> None:
    section("6. OVERALL CHURN RATE AND CLASS BALANCE")
    counts = df["Churn"].value_counts().sort_index()
    total = len(df)
    print(f"Total customers: {total}")
    print()
    for val, count in counts.items():
        label = "Churned" if val == 1 else "Retained"
        pct = count / total * 100
        print(f"  Churn={val} ({label}): {count} ({pct:.2f}%)")
    churn_rate = df["Churn"].mean() * 100
    print()
    print(f"Overall churn rate: {churn_rate:.2f}%")
    ratio = counts.get(0, 0) / counts.get(1, 1) if counts.get(1, 0) else float("inf")
    print(f"Class ratio (retained:churned): {ratio:.2f}:1")
    if churn_rate < 20 or churn_rate > 80:
        print("Class balance note: minority class is under 20% — imbalanced.")
    else:
        print("Class balance note: not severely imbalanced by the 20% rule of thumb.")


def main() -> None:
    print("ChurnGuard EDA Report")
    print(f"Source: {DATA_PATH.name} | Sheet: {SHEET_NAME}")
    df = load_data()
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    cols_with_missing = report_missing_values(df)
    report_missingness_vs_churn(df, cols_with_missing)
    report_dtypes(df)
    report_categorical_value_counts(df)
    report_numeric_descriptives(df)
    report_churn_balance(df)
    print()
    print("Done. No cleaning or modeling applied.")


if __name__ == "__main__":
    main()
