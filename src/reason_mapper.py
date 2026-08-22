"""SHAP feature attributions -> ECOA adverse action reason codes.

ASSUMPTIONS (data.py / shap_explainer.py / NOTES.md not available to me --
reconcile against the real files before relying on this):
  - Feature names follow the UCI/Statlog German Credit schema implied by the
    original docstring's examples (checking_status, credit_history, etc.).
  - PROHIBITED_BASIS_COLUMNS assumed >= {age_years, personal_status_sex,
    foreign_worker}; swap in the real constant.
  - The A31/A32 "paid duly" raw codes for credit_history are guessed;
    confirm against data.py::CODE_LABELS.

Pure and deterministic: same Explanation in -> identical ReasonCode list out.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from data import PROHIBITED_BASIS_COLUMNS  # type: ignore
except ImportError:
    PROHIBITED_BASIS_COLUMNS = {"age_years", "personal_status_sex", "foreign_worker"}


# ---------------------------------------------------------------------------
# Piece 1: The ReasonCode container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasonCode:
    code: str          # our ECOA/Reg B reason code identifier
    text: str          # applicant-facing sentence
    feature: str       # MUST match the real Explanation feature name
    shap_value: float  # MUST match that feature's actual attribution
    rank: int          # 1 = principal reason


# ---------------------------------------------------------------------------
# Piece 2: The mapping table
# ---------------------------------------------------------------------------
# feature -> (code, applicant-facing text) or (None, None) if suppressed.
# Text is drawn from the real Reg B model-form adverse-action reason list
# (12 CFR Pt. 1002, App. C), NOT from FICO's numeric score-factor codes
# (e.g. FICO 038/018/014) -- those are a different citation basis (FCRA
# 609(f)/615(a)(2), used when the decision is based on a purchased credit
# score). Our Explanation is a SHAP breakdown of our own underwriting
# model, not a FICO score, so a FICO code wouldn't correspond to anything
# we actually produced -- citing one would be citing a number this letter
# has no basis for.
#
# `code` here is our own internal identifier (R01-R12), not a FICO code.
# Note several features share the same underlying Reg B reason text (e.g.
# R08/R11 are both "Excessive obligations in relation to income") but each
# still gets its own code so the validator can trace back to exactly which
# feature drove the citation.
#
# `None` means: no lawful Reg B reason exists for this feature, full stop --
# never cite it regardless of SHAP value. One-line reason on each.
_MAPPING_TABLE: dict[str, tuple[str | None, str | None]] = {
    "checking_status":              ("R01", "Insufficient checking account balance"),
    "duration_months":              ("R02", "Length of requested credit term"),
    "credit_history":               ("R03", "Insufficient credit file / limited credit experience"),
    "credit_amount":                ("R05", "Excessive obligations in relation to income"),
    "savings_status":               ("R06", "Insufficient savings account balance"),
    "employment_since":             ("R07", "Length of employment"),
    "installment_rate_pct_income":  ("R08", "Excessive obligations in relation to income"),
    "property_magnitude":           ("R10", "Value or type of collateral not sufficient"),
    "other_installment_plans":      ("R11", "Excessive obligations in relation to income"),
    "existing_credits_count":       ("R12", "Number of credit obligations with us / other creditors"),
    # Suppressed -- no lawful code maps to these, on any applicant:
    "purpose":               (None, None),  # stated loan purpose (car/furniture/etc.) isn't
                                             # collateral and has no Reg B reason of its own --
                                             # mapping it to "collateral not sufficient" would
                                             # invent a security-interest claim the feature
                                             # doesn't support
    "other_debtors":         (None, None),  # co-applicant/guarantor presence isn't a DTI signal;
                                             # "excessive obligations" was a fabricated basis --
                                             # no honest Reg B reason maps to this feature
    "residence_since_years": (None, None),  # time-at-address isn't a Reg B model-form reason
    "housing":               (None, None),  # too close to a protected-class proxy to phrase safely
    "job":                   (None, None),  # occupation category risks proxying protected traits
    "num_dependents":        (None, None),  # family status -- unlawful basis under Reg B
    "telephone":             (None, None),  # not a Reg B model-form reason; no credit-relevance
}


def _phrase(contribution, template: str) -> str:
    """Numeric features get a value clause; categorical ones already have
    display_value text ready to go. No invented numbers -- only the
    applicant's own raw_value/display_value already on the contribution."""
    if isinstance(contribution.raw_value, (int, float)) and not isinstance(
        contribution.raw_value, bool
    ):
        return f"{template} (value: {contribution.display_value})"
    return f"{template}: {contribution.display_value}"


# ---------------------------------------------------------------------------
# Piece 3: is_citable()
# ---------------------------------------------------------------------------
MATERIALITY_FLOOR = 0.05  # |shap_value| below this is noise, not a reason

# The "paid on time" problem (NOTES.md 0:45): credit_history = A31/A32
# ("paid duly") pushes toward DECLINE and is mathematically real, but
# unusable as an adverse-action reason. DECISION: suppress it, then fall
# through to the next-biggest driver. Not re-mapped (that would cite a
# reason the SHAP value doesn't support -- fabrication), not a refuse-and-
# escalate-the-whole-decision (that throws away other, valid drivers this
# applicant may have). Per-driver suppression is the precise fix.
_UNCITABLE_VALUES = {
    "credit_history": {"A31", "A32"},
}


def is_citable(contribution) -> bool:
    """May I legally put this driver in a letter?"""
    feature = contribution.feature

    assert feature not in PROHIBITED_BASIS_COLUMNS, (
        f"'{feature}' is a prohibited basis column and should never reach "
        "the reason mapper -- this should be unreachable."
    )

    if abs(contribution.shap_value) < MATERIALITY_FLOOR:
        return False
    if contribution.direction != "toward_decline":
        return False

    code, _text = _MAPPING_TABLE.get(feature, (None, None))
    if code is None:
        return False

    if contribution.raw_value in _UNCITABLE_VALUES.get(feature, set()):
        return False

    return True


# ---------------------------------------------------------------------------
# Piece 4: map_reasons()
# ---------------------------------------------------------------------------
MAX_REASONS = 4


def map_reasons(explanation, max_reasons: int = MAX_REASONS) -> list[ReasonCode]:
    """Assembly line: Explanation -> up to `max_reasons` ReasonCodes.

    Edge cases decided:
      - APPROVE returns []. Reg B/FCRA adverse-action obligations attach to
        adverse actions; there's no comparable requirement (and real
        litigation risk) in volunteering "what would've hurt you" on an
        approval. That'd be a separate, explicitly-named function if wanted.
      - Fewer than max_reasons (including zero) is a legitimate outcome when
        suppression removes enough drivers -- not a bug. Callers must handle
        the empty case (e.g. escalate to human underwriter review).
    """
    if explanation.decision != "DECLINE":
        return []

    # .top_decline_drivers() already restricts to toward-decline, sorted
    # biggest-first; pull all of them so fall-through has room to work.
    drivers = explanation.top_decline_drivers(len(explanation.contributions))

    reasons: list[ReasonCode] = []
    for contribution in drivers:
        if len(reasons) >= max_reasons:
            break
        if not is_citable(contribution):
            continue  # materiality floor or citability rule rejected it

        code, template = _MAPPING_TABLE.get(contribution.feature, (None, None))
        assert code is not None  # is_citable already confirmed this exists
        reasons.append(
            ReasonCode(
                code=code,
                text=_phrase(contribution, template),
                feature=contribution.feature,
                shap_value=contribution.shap_value,
                rank=len(reasons) + 1,
            )
        )

    return reasons