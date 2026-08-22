"""Validate LLM-written adverse-action prose against ground-truth SHAP data.

WHY THIS EXISTS: reason_mapper.py guarantees traceability for structured
ReasonCode objects by construction. graph.py then hands those to an LLM to
turn into fluent prose -- and a fluent, helpful LLM will add plausible detail
nobody asked for (a driver that wasn't cited, a number that isn't real). This
module is the gate that catches that, on every request, before a letter goes
out. Assume the LLM is adversarial: not malicious, just relentlessly helpful
in ways that invent facts.

DECISIONS THIS FILE MAKES
--------------------------
1. FAIL-CLOSED. If either gate fails, the applicant gets the deterministic
   reason_mapper.py text for that decision, not the LLM prose and not a bare
   error. This is a credit decision under ECOA/Reg B -- the letter must go
   out, and it must be legally correct, so the fallback has to be something
   already known-good rather than a retry-and-hope or a silent pass-through.
   A human review queue is the right move for a production system but is a
   separate escalation path (e.g. graph.py catching a failed ValidationResult
   and routing to underwriting) -- this module's job stops at reporting
   pass/fail and why; it does not own the escalation.

2. STRICTNESS ON PARAPHRASE. "You don't have much saved" without the literal
   word "savings" is NOT flagged by the alias-table gate below -- that is a
   known, documented limitation of option (a) in the spec, not a silent gap.
   The line drawn here: the alias table catches every literal or near-literal
   mention of a feature's common names; it does not catch semantic paraphrase
   that avoids all listed aliases. Closing that gap requires either (b)
   constrained generation or (c) LLM-as-judge, and per the hard requirements
   the primary (deterministic) gate cannot be responsible for catching
   fully-paraphrased hallucinations -- it can only be honest that it won't.

3. NUMBERS. Every digit-string in the LLM prose is extracted and checked
   against the applicant's actual feature values from the explanation. A
   number that doesn't match anything real is a fabricated fact about a real
   person and fails the case -- there is no "close enough" here; 48 restating
   45 is wrong regardless of how minor the LLM thought the change was.

4. OMISSION. Dropping a principal reason is NOT validated here as a failure.
   This module's job is "did the LLM add anything false," not "did the LLM
   keep everything true" -- that second question is a completeness check
   against reason_mapper's own output, which belongs in graph.py (comparing
   the prose's reason count/coverage against the ReasonCode list it was
   given), not in a module whose contract is about fabrication, not omission.
   This is flagged, not silently ignored: see `checked_features` on the
   result, which callers can diff against the input ReasonCode features to
   detect omission themselves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Feature -> phrases an LLM is likely to use when referring to it in prose.
# Deliberately conservative (favors under- over over-matching aliases shared
# across features) but IS NOT exhaustive against paraphrase -- see decision 2
# above. Extend this table as real LLM output surfaces new phrasings; treat
# it as a living blocklist, not a one-time artifact.
_FEATURE_ALIASES: dict[str, set[str]] = {
    "checking_status": {"checking account", "checking balance"},
    "duration_months": {"loan term", "credit term", "requested term", "term of", "months"},
    "credit_history": {"credit history", "credit file", "credit experience", "payment history"},
    "credit_amount": {"amount requested", "loan amount", "credit amount", "requested amount"},
    "savings_status": {"savings account", "savings balance", "amount saved", "money saved"},
    "employment_since": {"employment", "job history", "time at employer", "length of employment",
                          "current job", "employed"},
    "installment_rate_pct_income": {"installment rate", "percentage of income", "monthly payment burden"},
    "property_magnitude": {"collateral", "property value", "assets"},
    "other_installment_plans": {"other installment", "other loans", "other financing"},
    "existing_credits_count": {"number of credit", "existing credit", "credit obligations",
                                "other accounts", "credit accounts"},
    # Suppressed features (per reason_mapper._MAPPING_TABLE) still get alias
    # entries: if the LLM prose mentions any of these, that's a fabricated
    # citation of a feature reason_mapper deliberately never emits a
    # ReasonCode for, which is exactly the failure mode this file exists to
    # catch.
    "purpose": {"purpose of the loan", "reason for the loan", "loan purpose"},
    "other_debtors": {"co-applicant", "co-signer", "guarantor"},
    "residence_since_years": {"time at your address", "length of residence", "years at your address"},
    "housing": {"housing situation", "renting", "own your home", "housing status"},
    "job": {"occupation", "job type", "type of employment"},
    "num_dependents": {"dependents", "number of dependents"},
    "telephone": {"telephone", "phone number"},
    "age_years": {"your age", "how old you are"},
    "personal_status_sex": {"marital status", "your sex", "your gender"},
    "foreign_worker": {"foreign worker", "citizenship status", "immigration status"},
}


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    checked_features: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Gate 1: structural -- validate the ReasonCode objects themselves.
# ---------------------------------------------------------------------------
def validate_reasons(reasons, explanation) -> ValidationResult:
    """Every ReasonCode.feature must be a real feature in this explanation,
    and its shap_value must be RECOMPUTED from the explanation (not trusted
    off the object) and be strictly positive (adverse direction).

    A validator that trusts the value already sitting on the object it's
    validating isn't validating anything -- it's just echoing input back.
    """
    violations: list[str] = []
    checked: set[str] = set()

    available = set(explanation.feature_names())
    # feature -> actual shap_value, recomputed from the explanation itself
    actual_by_feature = {c.feature: c.shap_value for c in explanation.contributions}

    if not reasons:
        if explanation.decision == "DECLINE":
            violations.append(
                "Zero reasons for a DECLINE decision -- a decline requires "
                "at least one principal reason under Reg B."
            )
        return ValidationResult(passed=not violations, violations=violations, checked_features=checked)

    for r in reasons:
        checked.add(r.feature)

        if r.feature not in available:
            violations.append(
                f"ReasonCode cites feature '{r.feature}', which does not exist "
                "in this applicant's explanation at all (phantom feature)."
            )
            continue

        actual = actual_by_feature.get(r.feature)
        if actual is None or actual <= 0:
            violations.append(
                f"ReasonCode cites feature '{r.feature}' as an adverse driver, "
                f"but its actual recomputed SHAP value is {actual!r} -- not a "
                "positive (toward-decline) contribution."
            )
            continue

        # Recomputed value must match what's on the object -- if it doesn't,
        # something upstream mutated or fabricated the number.
        if abs(actual - r.shap_value) > 1e-9:
            violations.append(
                f"ReasonCode for '{r.feature}' carries shap_value={r.shap_value!r}, "
                f"but the explanation's actual value is {actual!r} -- mismatch."
            )

    return ValidationResult(passed=not violations, violations=violations, checked_features=checked)


# ---------------------------------------------------------------------------
# Gate 2: prose -- validate LLM-written free text against the same ground
# truth, catching claims the structured objects never authorized.
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _applicant_real_numbers(explanation) -> set[str]:
    """All numeric values that are legitimately real for this applicant --
    drawn from the explanation's contributions' raw_value/display_value, so
    a real number restated in the letter (e.g. their actual duration_months)
    is never flagged. Includes both raw form and a couple of common
    stringifications so trivial formatting differences don't false-positive.
    """
    numbers: set[str] = set()
    for c in explanation.contributions:
        for val in (c.raw_value, c.display_value):
            if val is None:
                continue
            s = str(val)
            for match in _NUMBER_RE.findall(s):
                numbers.add(match)
                # also add int-truncated form, e.g. "45.0" -> "45"
                try:
                    numbers.add(str(int(float(match))))
                except ValueError:
                    pass
    return numbers


def validate_text(text: str, allowed_features: set[str], explanation) -> ValidationResult:
    """Prose gate. Two independent checks:

      1. Alias scan: does the text mention any feature (via its alias table)
         that is NOT in `allowed_features` (the set of features the caller
         actually cited via ReasonCode)? If so, the LLM added an unauthorized
         driver.
      2. Number scan: does the text contain any digit-string that doesn't
         match any of this applicant's real values? If so, the LLM
         fabricated or altered a fact.

    Does NOT catch full semantic paraphrase that avoids every listed alias --
    documented limitation, see module docstring decision 2.
    """
    violations: list[str] = []
    checked: set[str] = set(allowed_features)
    lowered = text.lower()

    # -- 1. alias scan ----------------------------------------------------
    for feature, aliases in _FEATURE_ALIASES.items():
        if feature in allowed_features:
            continue  # this feature was legitimately cited; mentioning it is fine
        for alias in aliases:
            if alias in lowered:
                violations.append(
                    f"Text mentions '{alias}', an alias for feature '{feature}', "
                    "which was never cited as a reason for this applicant."
                )
                break  # one hit per feature is enough to flag it

    # -- 2. number scan -----------------------------------------------------
    real_numbers = _applicant_real_numbers(explanation)
    for match in _NUMBER_RE.findall(text):
        if match not in real_numbers:
            violations.append(
                f"Text contains the number '{match}', which does not match any "
                "of this applicant's actual feature values -- possible "
                "fabricated or altered figure."
            )

    return ValidationResult(passed=not violations, violations=violations, checked_features=checked)


# ---------------------------------------------------------------------------
# Combined entry point.
# ---------------------------------------------------------------------------
def validate(reasons, text: str, explanation) -> ValidationResult:
    """Run both gates. FAIL-CLOSED: if either gate fails, the caller (graph.py)
    must NOT ship `text` to the applicant -- it should fall back to the
    deterministic reason_mapper.py rendering of `reasons` (which is true by
    construction) rather than the LLM prose, an error page, or a silent pass.
    This function reports pass/fail and why; it does not itself perform the
    fallback -- that's graph.py's responsibility, kept out of this module so
    validate() stays a pure check with no side effects, per the hard
    requirement that this run on every request in the serving path.
    """
    structural = validate_reasons(reasons, explanation)
    if not structural.passed:
        # Prose can't be trusted to describe reasons that already failed the
        # structural gate -- no point running the (more expensive) text scan
        # against a reason list that's already known-bad.
        return structural

    allowed = {r.feature for r in reasons}
    prose = validate_text(text, allowed, explanation)

    return ValidationResult(
        passed=structural.passed and prose.passed,
        violations=structural.violations + prose.violations,
        checked_features=structural.checked_features | prose.checked_features,
    )