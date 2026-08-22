"""Load and prepare the UCI Statlog German Credit dataset.

Target convention used across this project:
    y = 1  -> BAD credit risk  -> DECLINE
    y = 0  -> GOOD credit risk -> APPROVE

So the model predicts P(bad), and a SHAP value that is POSITIVE pushes the
applicant toward a decline. "Top negative contributors" for an applicant are
therefore the features with the largest positive SHAP values.

Fair-lending note: three raw attributes are dropped before training because
they are prohibited bases under ECOA. See PROHIBITED_BASIS_COLUMNS.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "german.data"

# Column names from the UCI german.doc codebook, in file order.
COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct_income",
    "personal_status_sex",
    "other_debtors",
    "residence_since_years",
    "property_magnitude",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",
]

# ECOA (Reg B, 12 CFR 1002.6) prohibits basing a credit decision on these.
# A model about generating lawful adverse action reasons must not train on them:
# a SHAP attribution pointing at "sex" is not a reason we could ever lawfully cite.
PROHIBITED_BASIS_COLUMNS = {
    "personal_status_sex": "sex / marital status",
    "foreign_worker": "national origin",
    "age_years": "age",
}

NUMERIC_COLUMNS = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct_income",
    "residence_since_years",
    "existing_credits_count",
    "num_dependents",
]

# Human-readable expansions of the UCI A-codes, used later for reason text.
CODE_LABELS = {
    "checking_status": {
        "A11": "< 0 DM", "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assigned", "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing",
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other",
    },
    "savings_status": {
        "A61": "< 100 DM", "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM", "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years", "A75": ">= 7 years",
    },
    "other_debtors": {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    },
    "property_magnitude": {
        "A121": "real estate", "A122": "building society savings / life insurance",
        "A123": "car or other", "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank", "A142": "stores", "A143": "none",
    },
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {"A191": "none", "A192": "registered under customer name"},
}


def download() -> Path:
    """Fetch german.data into data/ if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    return RAW_PATH


def load_raw() -> pd.DataFrame:
    """Read the space-delimited raw file with proper column names."""
    return pd.read_csv(download(), sep=r"\s+", header=None, names=COLUMNS)


def load_clean(drop_prohibited: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with y=1 meaning BAD risk / decline.

    Categorical columns are left as pandas `category` dtype so XGBoost can
    consume them natively (enable_categorical=True) — this keeps feature names
    one-to-one with real credit-file concepts, which is what makes SHAP
    attributions mappable to ECOA reason codes.
    """
    df = load_raw()
    y = (df.pop("target") == 2).astype(int)  # 2 = bad in the raw encoding

    if drop_prohibited:
        df = df.drop(columns=list(PROHIBITED_BASIS_COLUMNS))

    for col in df.columns:
        if col not in NUMERIC_COLUMNS:
            df[col] = df[col].astype("category")

    return df, y


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split — the classes are imbalanced 70/30."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def describe_code(feature: str, value) -> str:
    """Expand a raw A-code into human-readable text, if we know it."""
    return CODE_LABELS.get(feature, {}).get(value, str(value))


if __name__ == "__main__":
    X, y = load_clean()
    print(f"rows={len(X)}  features={X.shape[1]}")
    print(f"class balance: good/approve={int((y == 0).sum())}  bad/decline={int((y == 1).sum())}"
          f"  ({y.mean():.1%} bad)")
    print(f"dropped as ECOA prohibited basis: {list(PROHIBITED_BASIS_COLUMNS)}")
    print(f"numeric={len(NUMERIC_COLUMNS)}  categorical={X.shape[1] - len(NUMERIC_COLUMNS)}")
    print("\nfeatures:", list(X.columns))
