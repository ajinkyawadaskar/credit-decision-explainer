"""Train and persist the XGBoost credit risk model.

Predicts P(bad risk). The decision threshold is a separate, explicit choice —
see DECISION_THRESHOLD — so that "why was this declined" is always answerable
in terms of both the score and the cutoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from data import load_clean, split

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "xgb_credit.json"

# Applicants scoring at or above this P(bad) are declined.
DECISION_THRESHOLD = 0.5

PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    min_child_weight=3,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="auc",
    random_state=42,
)


def build() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**PARAMS)


def train_and_eval(seed: int = 42):
    X, y = load_clean()
    X_train, X_test, y_train, y_test = split(X, y, seed=seed)

    model = build()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, metrics, (X_test, y_test, proba, pred)


def save(model: xgb.XGBClassifier) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    return MODEL_PATH


def load() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**PARAMS)
    model.load_model(MODEL_PATH)
    return model


if __name__ == "__main__":
    model, m, (X_test, y_test, proba, pred) = train_and_eval()

    print(f"train={m['n_train']}  test={m['n_test']}  (stratified, seed=42)")
    print()
    print(f"  ROC AUC   {m['roc_auc']:.4f}")
    print(f"  PR  AUC   {m['pr_auc']:.4f}   (base rate {y_test.mean():.3f})")
    print(f"  Accuracy  {m['accuracy']:.4f}   @ threshold {DECISION_THRESHOLD}")
    print()
    print(classification_report(y_test, pred, target_names=["approve (good)", "decline (bad)"]))

    # 5-fold-ish stability check across seeds: a 1000-row dataset moves a lot.
    aucs = [train_and_eval(seed=s)[1]["roc_auc"] for s in (0, 1, 7, 42, 2024)]
    print(f"AUC across 5 seeds: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}   "
          f"[{min(aucs):.4f}, {max(aucs):.4f}]")

    print(f"\nsaved -> {save(model)}")
