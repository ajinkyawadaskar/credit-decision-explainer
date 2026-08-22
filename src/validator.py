"""Deterministic SHAP-traceability enforcement.

===============================================================================
    OWNER: Ajinkya.  DO NOT let Claude implement this module.
    This is the single most important file in the repo. Everything else is
    plumbing around it.
===============================================================================

WHY THIS EXISTS (read this before writing a line)
-------------------------------------------------
`reason_mapper.py` already guarantees traceability for the STRUCTURED objects:
it only ever emits a ReasonCode built from a real FeatureContribution, so its
`feature` and `shap_value` fields are true by construction.

Then `graph.py` hands those reasons to an LLM to turn into applicant-facing
prose. **That is the moment the guarantee can break.** An LLM asked to write a
fluent denial letter will helpfully add things nobody asked for:

    given:   "Length of requested credit term: 45"
    written: "your 45-month term, combined with your limited employment
              history and recent credit inquiries, ..."

Employment history was never a driver for this applicant. Credit inquiries
aren't even a feature in this dataset. The output is fluent, plausible,
and contains two fabricated adverse action reasons. A human reviewer would
not catch it; the applicant certainly could not.

This module is what catches it. **Assume the LLM is adversarial** — not
malicious, just relentlessly helpful in ways that invent facts.


THE HARD PART
-------------
Validating structured objects is trivial set membership and worth doing as a
cheap first gate. But the real surface is FREE TEXT, and that is genuinely
hard. You must decide how to detect a feature claim inside a sentence.

Options, roughly in increasing order of effort and robustness:

  a) Keyword / alias table. Build feature -> {phrases} ("employment", "job
     history", "time at employer" -> employment_since). Scan the prose for
     any alias belonging to a feature NOT in this applicant's cited set.
     Deterministic, fast, no API call, no circularity. Misses paraphrase.

  b) Constrained generation instead of detection. Don't let the LLM write
     free prose at all — have it select from or lightly rephrase fixed
     sentence templates, so there's no room to add a claim. Sidesteps the
     detection problem entirely at the cost of fluency.

  c) LLM-as-judge. Ask a second model whether the text contains claims not in
     the source. Handles paraphrase, but it is stochastic, costs an API call,
     and you are now trusting an LLM to catch an LLM. Cannot be your only
     gate.

Whatever you pick, **the primary gate must be deterministic.** A stochastic
validator cannot make a guarantee, and the guarantee is the product.


WHAT TO BUILD
-------------
    @dataclass(frozen=True)
    class ValidationResult:
        passed: bool
        violations: list[str]     # human-readable, one per problem found
        checked_features: set[str]

    def validate_reasons(reasons, explanation) -> ValidationResult: ...
        # structural gate: every ReasonCode.feature must be in
        # explanation.feature_names(), and its shap_value must match that
        # feature's ACTUAL attribution and be > 0. Recompute; don't trust
        # the value carried on the object.

    def validate_text(text, allowed_features, explanation) -> ValidationResult: ...
        # prose gate: no feature claim outside `allowed_features`.

    def validate(...)  # combined entry point graph.py calls


DECISIONS YOU OWN
-----------------
1. FAIL-CLOSED OR FAIL-OPEN? If validation fails, does the applicant get the
   deterministic reason_mapper text (safe, blunt), an error, or a human
   review queue? For a credit decision the answer should be obvious, but say
   it out loud and put it in the README.

2. STRICTNESS ON PARAPHRASE. "You don't have much saved" doesn't name
   savings_status literally. Is that a violation? Where's the line between
   readable prose and an unverifiable claim?

3. NUMBERS. If the letter says "45 months", that number must equal the
   applicant's actual `duration_months`. An LLM restating 45 as "about four
   years" is fine; restating it as 48 is a fabricated fact about a real
   person. Consider validating every digit string in the output against the
   applicant's real values.

4. OMISSION. Is dropping a reason a violation, or just less helpful? Reg B
   requires the principal reasons — silently losing reason #1 is arguably
   worse than adding a spurious #5.


HARD REQUIREMENTS
-----------------
- Pure, deterministic, no network in the primary gate.
- Recompute traceability from `explanation`; never trust fields on the
  objects being validated. A validator that trusts its input validates
  nothing.
- Return WHY it failed, not just that it failed. `evals/metrics.py` reports
  these violations and the README quotes them.
- Must be fast enough to run on every request — it is in the serving path.


TEST AGAINST
------------
Hand it deliberately bad input and confirm it fails:
  - a ReasonCode whose feature isn't in the explanation
  - a ReasonCode whose shap_value is negative (argued for approval)
  - prose naming "employment history" for an applicant where
    employment_since was never cited
  - prose containing a number that isn't any of the applicant's values
  - empty reasons on a DECLINE

A validator that has never rejected anything has not been tested.
"""
from __future__ import annotations

from dataclasses import dataclass

# TODO(Ajinkya): implement. See module docstring for the full spec.
raise NotImplementedError(
    "src/validator.py is owned by Ajinkya and has not been implemented yet. "
    "See the module docstring for the spec."
)
