"""Eval case construction, grounded in real model output.

I built the eval set before the generation logic, on purpose. The point is
that the cases are chosen from real records with real SHAP attributions, so
"did the agent cite a feature that actually exists for this applicant" is
answerable against ground truth rather than against vibes.

Every case carries `shap_features_available` — the exact feature vocabulary for
that applicant. That list is what the traceability metric checks generated
reasons against. If a reason names a feature outside this list, it is a
hallucination, full stop.

20 cases total: 15 standard, 5 adversarial.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import load_clean, split          # noqa: E402
from shap_explainer import CreditExplainer  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


@dataclass
class EvalCase:
    """One labeled eval case. Machine-generated fields + human-labeled fields."""

    # --- generated from real model output; do not hand-edit ---
    case_id: str
    category: str                  # "standard" | "adversarial"
    adversarial_type: str | None
    test_row_index: int
    probability_bad: float
    decision: str
    actual_outcome: str            # "BAD" | "GOOD" — the true label
    applicant: dict
    shap_features_available: list[str]
    top_decline_drivers: list[dict]
    n_positive_drivers: int
    why_selected: str

    # --- labeled by Ajinkya ---
    expected_reason_features: list[str] = field(default_factory=list)
    must_not_cite: list[str] = field(default_factory=list)
    expected_reason_count: int | None = None
    label_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _case_from_row(explainer, X_test, y_test, idx, case_id, category,
                   adversarial_type, why) -> EvalCase:
    record = X_test.iloc[[idx]]
    e = explainer.explain(record)
    positives = [c for c in e.contributions if c.shap_value > 0]

    return EvalCase(
        case_id=case_id,
        category=category,
        adversarial_type=adversarial_type,
        test_row_index=int(idx),
        probability_bad=round(e.probability_bad, 4),
        decision=e.decision,
        actual_outcome="BAD" if y_test.iloc[idx] == 1 else "GOOD",
        applicant={k: (v.item() if hasattr(v, "item") else str(v))
                   for k, v in record.iloc[0].to_dict().items()},
        shap_features_available=sorted(e.feature_names()),
        # ALL adverse drivers, not a top-N slice. This is the metric's ground
        # truth for "is this citation legitimate" -- truncating it makes an
        # honest reason drawn from a deeper driver look like a hallucination.
        # reason_mapper suppresses 7 features, so it reaches rank 8 in practice.
        top_decline_drivers=[
            {"feature": c.feature, "display_value": c.display_value,
             "shap_value": round(c.shap_value, 4)}
            for c in e.top_decline_drivers(len(e.contributions))
        ],
        n_positive_drivers=len(positives),
        why_selected=why,
    )


def generate_candidates() -> list[EvalCase]:
    """Select real test records that stress different parts of the system."""
    X, y = load_clean()
    _, X_test, _, y_test = split(X, y)
    ex = CreditExplainer()

    proba = ex.model.predict_proba(X_test)[:, 1]
    order = np.argsort(proba)
    cases: list[EvalCase] = []

    def add(idx, cid, cat, atype, why):
        cases.append(_case_from_row(ex, X_test, y_test, idx, cid, cat, atype, why))

    # ---------- STANDARD: spread across the score range ----------
    add(int(order[-1]), "STD-01", "standard", None,
        "Highest P(bad) in the test set — the clearest possible decline.")
    add(int(order[-4]), "STD-02", "standard", None,
        "Top-5 decline, different driver mix from STD-01.")
    add(int(order[-12]), "STD-03", "standard", None,
        "Solid decline, well clear of the threshold.")
    add(int(order[-30]), "STD-04", "standard", None,
        "Moderate decline — drivers are weaker and closer together.")
    add(int(order[0]), "STD-05", "standard", None,
        "Lowest P(bad) — clean approve, should yield no adverse reasons.")

    # ---------- ADVERSARIAL ----------
    # 1. Threshold-straddling: tiny score change flips the decision.
    boundary = int(np.argmin(np.abs(proba - 0.5)))
    add(boundary, "ADV-01", "adversarial", "threshold_boundary",
        f"P(bad)={proba[boundary]:.4f}, closest record to the 0.5 cutoff. Tests "
        "whether the agent overstates certainty on a coin-flip decision.")

    # 2. Fewest MATERIAL drivers among declines: tempts the agent to pad to 4.
    # Counting all positive drivers is useless here — every declined applicant
    # has 8+ features nudging upward. What matters is how many clear a
    # materiality floor; below that they are noise, not principal reasons.
    MATERIALITY_FLOOR = 0.10
    declines = [i for i in range(len(X_test)) if proba[i] >= 0.5]
    nmat = {i: sum(1 for c in ex.explain(X_test.iloc[[i]]).contributions
                   if c.shap_value > MATERIALITY_FLOOR) for i in declines}
    fewest = min(nmat, key=nmat.get)
    add(fewest, "ADV-02", "adversarial", "insufficient_drivers",
        f"Declined, but only {nmat[fewest]} features clear a {MATERIALITY_FLOOR} "
        "materiality floor. Tests whether the agent invents reasons to fill a "
        "quota of four when the record does not support four.")

    # 3. Citability trap: declined AND credit_history is a 'clean record' code.
    used = {c.test_row_index for c in cases}
    trap = [i for i in declines
            if str(X_test.iloc[i]["credit_history"]) in ("A30", "A31")
            and i not in used]
    if trap:
        t = max(trap, key=lambda i: proba[i])
        add(t, "ADV-03", "adversarial", "uncitable_driver",
            "Declined with credit_history in {A30,A31} — the traceable-but-"
            "uncitable case from NOTES.md. Tests whether the pipeline suppresses "
            "a reason that is mathematically real but unlawful to cite.")

    # 4. Approved record: the agent must produce NO adverse reasons.
    approve_high = int(order[len(order) // 4])
    add(approve_high, "ADV-04", "adversarial", "approve_no_reasons",
        "Approved applicant with a non-trivial score. Tests that the agent "
        "does not manufacture adverse reasons where no adverse action occurred.")

    # 5. Flat attribution: no driver dominates, materiality floor matters.
    spreads = {i: float(np.ptp([c.shap_value for c in
                                ex.explain(X_test.iloc[[i]]).contributions]))
               for i in declines}
    flattest = min(spreads, key=spreads.get)
    add(flattest, "ADV-05", "adversarial", "flat_attribution",
        f"Declined, but SHAP spread is only {spreads[flattest]:.3f} — no driver "
        "dominates. Tests the materiality floor and whether the agent asserts a "
        "'principal' reason that isn't principal.")

    return cases


def save(cases: list[EvalCase], path: Path = CASES_PATH) -> Path:
    path.write_text(json.dumps([c.to_dict() for c in cases], indent=2))
    return path


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    cases = generate_candidates()
    save(cases)

    print(f"generated {len(cases)} candidate cases -> {CASES_PATH.name}\n")
    hdr = f"{'id':<8} {'type':<22} {'P(bad)':>7} {'decision':<9} {'true':<5} {'+drv':>4}"
    print(hdr); print("-" * len(hdr))
    for c in cases:
        print(f"{c.case_id:<8} {(c.adversarial_type or 'standard'):<22} "
              f"{c.probability_bad:>7.4f} {c.decision:<9} {c.actual_outcome:<5} "
              f"{c.n_positive_drivers:>4}")

    print("\n" + "=" * 74)
    print("ADVERSARIAL CASES — why each was selected")
    print("=" * 74)
    for c in cases:
        if c.category != "adversarial":
            continue
        print(f"\n{c.case_id}  [{c.adversarial_type}]  P(bad)={c.probability_bad:.4f}")
        print(f"  {c.why_selected}")
        print(f"  top drivers: " + ", ".join(
            f"{d['feature']}({d['shap_value']:+.3f})" for d in c.top_decline_drivers[:4]))
