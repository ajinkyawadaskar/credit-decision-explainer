"""SHAP local attribution for individual credit decisions.

This module answers "why THIS applicant" — not "what does the model care about
in general". That distinction is the whole basis of an adverse action notice:
FCRA/ECOA require the specific principal reasons for THIS denial.

Sign convention (see data.py): the model predicts P(bad risk), so a POSITIVE
SHAP value pushes the applicant toward DECLINE. The "top negative contributors"
for an applicant are therefore the largest POSITIVE SHAP values.

Output of `explain()` is the contract handed to reason_mapper.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from data import describe_code, load_clean, split
from model import DECISION_THRESHOLD, load as load_model


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed push on a single applicant's score."""

    feature: str          # e.g. "checking_status"
    raw_value: object     # e.g. "A11" or 24
    display_value: str    # e.g. "< 0 DM"  (human-readable)
    shap_value: float     # >0 pushes toward DECLINE, <0 toward APPROVE
    direction: str        # "toward_decline" | "toward_approve"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_value"] = str(self.raw_value)
        d["shap_value"] = float(self.shap_value)
        return d


@dataclass(frozen=True)
class Explanation:
    """Everything reason_mapper.py is allowed to draw on for one applicant."""

    probability_bad: float
    decision: str                 # "APPROVE" | "DECLINE"
    threshold: float
    base_value: float             # model's average log-odds output
    contributions: list[FeatureContribution]  # all features, sorted by |shap|

    def top_decline_drivers(self, n: int = 4) -> list[FeatureContribution]:
        """The n features pushing hardest toward decline, largest first.

        Only genuinely positive contributions are returned. If an applicant has
        fewer than n features pushing toward decline, this returns fewer than n
        — never padding with features that actually helped them. Citing a
        feature that pushed toward approval as a denial reason would be false.
        """
        pushing = [c for c in self.contributions if c.shap_value > 0]
        return sorted(pushing, key=lambda c: c.shap_value, reverse=True)[:n]

    def feature_names(self) -> set[str]:
        """Feature vocabulary for this applicant — the validator's ground truth."""
        return {c.feature for c in self.contributions}


class CreditExplainer:
    """Wraps the model with per-record SHAP attribution.

    Uses XGBoost's built-in `pred_contribs` rather than shap.TreeExplainer:
    it is exact TreeSHAP, and it handles pandas `category` dtypes natively,
    which the generic explainer path does not reliably do.
    """

    def __init__(self, model=None):
        self.model = model if model is not None else load_model()
        self.booster = self.model.get_booster()
        self.feature_names = list(self.booster.feature_names)

    def _contribs(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values. Returns (n_rows, n_features + 1).

        The final column is the base value (expected model output in log-odds).
        """
        import xgboost as xgb

        dm = xgb.DMatrix(X, enable_categorical=True)
        return self.booster.predict(dm, pred_contribs=True)

    def explain(self, X: pd.DataFrame) -> Explanation:
        """Explain exactly one applicant (single-row DataFrame)."""
        if len(X) != 1:
            raise ValueError(f"explain() takes exactly one row, got {len(X)}")

        contribs = self._contribs(X)[0]
        shap_values, base_value = contribs[:-1], float(contribs[-1])

        proba = float(self.model.predict_proba(X)[:, 1][0])
        row = X.iloc[0]

        contributions = [
            FeatureContribution(
                feature=name,
                raw_value=row[name],
                display_value=describe_code(name, row[name]),
                shap_value=float(sv),
                direction="toward_decline" if sv > 0 else "toward_approve",
            )
            for name, sv in zip(self.feature_names, shap_values)
        ]
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return Explanation(
            probability_bad=proba,
            decision="DECLINE" if proba >= DECISION_THRESHOLD else "APPROVE",
            threshold=DECISION_THRESHOLD,
            base_value=base_value,
            contributions=contributions,
        )

    def verify_additivity(self, X: pd.DataFrame, tol: float = 1e-4) -> bool:
        """SHAP's core guarantee: base + sum(shap) == model output (log-odds).

        If this fails, every downstream reason is built on sand.
        """
        contribs = self._contribs(X)
        reconstructed = contribs.sum(axis=1)
        import xgboost as xgb

        actual = self.booster.predict(
            xgb.DMatrix(X, enable_categorical=True), output_margin=True
        )
        return bool(np.allclose(reconstructed, actual, atol=tol))


if __name__ == "__main__":
    X, y = load_clean()
    _, X_test, _, y_test = split(X, y)

    ex = CreditExplainer()

    ok = ex.verify_additivity(X_test)
    print(f"additivity check on {len(X_test)} test rows: {'PASS' if ok else 'FAIL'}")
    print("  (base_value + sum of SHAP == model log-odds, per row)\n")

    proba_all = ex.model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(proba_all))  # most confidently declined applicant
    record = X_test.iloc[[idx]]

    e = ex.explain(record)

    print("=" * 68)
    print(f"APPLICANT  (test row {idx}, actual outcome = "
          f"{'BAD' if y_test.iloc[idx] == 1 else 'GOOD'})")
    print("=" * 68)
    print(f"  P(bad risk) = {e.probability_bad:.4f}   threshold = {e.threshold}")
    print(f"  DECISION    = {e.decision}")
    print(f"  base value  = {e.base_value:+.4f} log-odds (population average)")
    print(f"  sum of SHAP = {sum(c.shap_value for c in e.contributions):+.4f}")

    print(f"\n  TOP 4 DRIVERS TOWARD DECLINE")
    print(f"  {'-' * 64}")
    print(f"  {'feature':<28} {'applicant value':<26} {'SHAP':>7}")
    print(f"  {'-' * 64}")
    for c in e.top_decline_drivers(4):
        print(f"  {c.feature:<28} {c.display_value:<26} {c.shap_value:>+7.4f}")

    print(f"\n  COUNTERWEIGHT — features that argued for approval")
    print(f"  {'-' * 64}")
    helping = sorted(e.contributions, key=lambda c: c.shap_value)[:3]
    for c in helping:
        if c.shap_value < 0:
            print(f"  {c.feature:<28} {c.display_value:<26} {c.shap_value:>+7.4f}")
