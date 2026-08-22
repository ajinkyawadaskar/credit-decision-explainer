"""SHAP feature attributions -> ECOA adverse action reason codes.

===============================================================================
    OWNER: Ajinkya.  DO NOT let Claude implement this module.
    This is the interpretive core of the project and the part an interviewer
    will push hardest on. Every design decision below is a judgment call that
    you need to be able to defend out loud.
===============================================================================

WHAT THIS MODULE IS FOR
-----------------------
`shap_explainer.py` produces a mathematically exact, signed attribution per
feature. That is a *statistical* artifact. It is not a reason. This module is
the translation layer that turns "feature `checking_status` contributed +0.79
log-odds toward the positive class" into "Insufficient checking account
balance" — something you could lawfully put in a letter to an applicant.

Everything downstream depends on this being a PURE, DETERMINISTIC function.
No LLM call belongs in this file. The LLM (graph.py) writes prose; this module
decides which reasons are permitted to exist in the first place. Keeping that
boundary sharp is what makes the validator meaningful.


INPUT CONTRACT
--------------
Consumes `Explanation` from `shap_explainer.py`:

    e.probability_bad   float
    e.decision          "APPROVE" | "DECLINE"
    e.threshold         float
    e.base_value        float, log-odds
    e.contributions     list[FeatureContribution], sorted by |shap_value| desc

    FeatureContribution:
        .feature        str    e.g. "checking_status"
        .raw_value      object e.g. "A11" or 45
        .display_value  str    e.g. "< 0 DM"
        .shap_value     float  >0 pushes toward DECLINE, <0 toward APPROVE
        .direction      "toward_decline" | "toward_approve"

    e.top_decline_drivers(n) -> the n largest POSITIVE contributions, already
                                filtered so nothing that argued for approval
                                can ever appear. Returns FEWER than n if the
                                applicant has fewer than n adverse drivers.

The 17 features available are listed in `data.py::COLUMNS` minus
`PROHIBITED_BASIS_COLUMNS`. Human-readable value labels are in
`data.py::CODE_LABELS`, and `data.py::describe_code()` expands them.


WHAT YOU MUST DECIDE AND BUILD
------------------------------

1. THE MAPPING TABLE.  feature -> ECOA reason code + applicant-facing text.
   Regulation B, Appendix C provides the standard model-form reason list
   ("Insufficient income", "Length of employment", "Insufficient credit file",
   "Excessive obligations in relation to income", "Limited credit experience",
   "No deposit account with us", "Value or type of collateral not sufficient",
   etc.). Which of our 17 features maps to which code — and whether some
   features have NO lawful code and must be suppressed entirely — is your call
   and your defense.

2. THE CITABILITY PROBLEM.  **Read the 0:45 entry in NOTES.md before starting.**
   Measured on this dataset:

       credit_history = A31 "all credits at this bank paid duly"
           -> 57.1% actual bad rate, mean SHAP +0.61 toward DECLINE

   The model genuinely wants to decline people *because they have no
   delinquencies*. That attribution is real, exact, and fully SHAP-traceable —
   and it is unusable as an adverse action reason. "You were denied because you
   paid your credits on time" would not survive regulatory review.

   Meanwhile A30 ("no credits taken") at 62.5% bad rate IS defensible: it maps
   to the standard "Insufficient credit file" / "Limited credit experience".

   So traceability is NECESSARY BUT NOT SUFFICIENT. You need a notion of which
   (feature, value, direction) combinations are CITABLE, separate from which
   are merely TRUE. Decide: suppress uncitable drivers and fall through to the
   next one? Re-map them to a broader code? Refuse to generate and escalate?
   Any of those is defensible. Silently citing them is not.

3. HOW MANY REASONS.  Convention (Reg B model forms; FCRA 615(a) for score-
   based factors) is up to FOUR principal reasons. Decide what happens when
   suppression leaves you with fewer than four, or with zero.

4. DIRECTION HANDLING.  A DECLINE cites positive-SHAP features. What does an
   APPROVE return — nothing, or the counterfactual "what would have hurt you"?
   Decide and document; it changes the API shape.

5. MATERIALITY FLOOR.  A feature contributing +0.003 is noise, not a reason.
   Is there a minimum |SHAP| below which you refuse to cite? Justify the number
   you pick — an interviewer will ask where it came from.

6. NUMERIC vs CATEGORICAL PHRASING.  `duration_months = 45` and
   `checking_status = "A11"` need different sentence shapes. Categorical values
   have `display_value` prefilled; numerics need thresholds or comparisons to
   population norms to read naturally ("term of 45 months exceeds...").


HARD REQUIREMENTS
-----------------
- Pure and deterministic. Same Explanation in -> identical ReasonCode list out,
  every time. No randomness, no network, no LLM, no clock.
- Every returned reason MUST carry the `feature` string it came from and the
  `shap_value` that justified it. `validator.py` re-derives traceability from
  these fields; if they are absent or wrong, the validator cannot do its job.
- Never emit a reason sourced from a feature in
  `data.py::PROHIBITED_BASIS_COLUMNS`. Those are dropped before training, so
  this should be unreachable — assert it anyway. Defense in depth is the point
  of the project.
- No invented numbers in applicant-facing text.


SUGGESTED SURFACE (change it if you have a better shape)
--------------------------------------------------------
    @dataclass(frozen=True)
    class ReasonCode:
        code: str            # your ECOA code identifier
        text: str            # applicant-facing sentence
        feature: str         # MUST match a real Explanation feature name
        shap_value: float    # MUST match that feature's actual attribution
        rank: int            # 1 = principal reason

    def map_reasons(explanation, max_reasons: int = 4) -> list[ReasonCode]: ...
    def is_citable(contribution) -> bool: ...


TEST WITH
---------
    src/shap_explainer.py already prints a worked DECLINE (test row 96):
    credit_amount 11816 (+0.8089), checking_status "< 0 DM" (+0.7904),
    credit_history "no credits taken" (+0.5205), duration_months 45 (+0.5082).
    That record is a good first target — note driver #3 is exactly the
    citability case from NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# TODO(Ajinkya): implement. See module docstring for the full spec.
raise NotImplementedError(
    "reason_mapper.py is owned by Ajinkya and has not been implemented yet. "
    "See the module docstring for the spec."
)
