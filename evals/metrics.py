"""DeepEval metrics for the reason-mapper pipeline.

Scope locked per CLAUDE.md: one custom metric (TraceabilityMetric) + two
DeepEval built-ins, both explicitly wired to Gemini (DeepEval defaults to
OpenAI and will silently reach for OPENAI_API_KEY / the wrong provider if you
don't pass `model=` on every built-in).

-------------------------------------------------------------------------------
Decisions this file makes (see inline comments for the "why" at each site):
-------------------------------------------------------------------------------
  - THRESHOLD = 1.0. Any single untraceable reason fails the case. This is a
    regulated adverse-action letter, not a chat answer graded on vibes: a
    letter with 3 honest reasons and 1 fabricated one is a compliance
    violation, full stop, not "80% good." Softening this threshold would be
    grading the metric on how *few* hallucinations you got away with instead
    of whether the letter is defensible. If the project ever wants a
    lenient/diagnostic view, that's a second metric with a different name and
    a different threshold -- not a lowered bar on this one.

  - DIRECTION IS SCORED SEPARATELY FROM EXISTENCE. A reason whose feature name
    doesn't exist in the explanation at all ("phantom feature") and a reason
    whose feature exists but argued FOR approval ("wrong-direction citation")
    are different failure modes with different real-world severity --
    wrong-direction is arguably worse, because it *looks* traceable to a
    reviewer skimming the SHAP output ("yes, checking_status is in the
    explanation") while still being a lie about what that feature did. Both
    fail the case at threshold=1.0, but self.reason reports them distinctly
    so a human reading the eval output can tell which failure mode occurred.

  - EXTRACTION prefers structured data over prose. `measure()` looks for
    `test_case.metadata["generated_reasons"]` -- a list of dicts (or
    reason_mapper.ReasonCode instances) each carrying `feature` and
    `shap_value` -- and only falls back to naive parsing of
    `actual_output` (assumed JSON) if metadata isn't populated. Regexing
    feature names out of English prose was explicitly rejected: it would
    measure the regex, not the model.

  - EMPTY OUTPUT SCORES 0, NOT 1. Zero reasons is vacuously "100% traceable"
    under a naive fraction (0/0), which is exactly the failure mode called
    out in the spec: an agent that says nothing must not look like an agent
    that got everything right. Empty output on a DECLINE case is scored as a
    hard failure with score=0.0 and a distinct reason string, unless the case
    is legitimately an APPROVE case (adversarial_type == "approve_no_reasons"
    style cases), where empty output is the correct answer and is scored 1.0.
"""
from __future__ import annotations

import json
import os
from typing import Any

from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, FaithfulnessMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Shared Gemini judge for the two LLM-judged built-ins. Constructed once and
# imported by both metrics so there's a single place to change the model.
# ---------------------------------------------------------------------------
def get_judge() -> GeminiModel:
    return GeminiModel(
        model=os.environ["GEMINI_MODEL"],
        api_key=os.environ["GOOGLE_API_KEY"],
    )


# ---------------------------------------------------------------------------
# 1. TraceabilityMetric -- the custom, deterministic metric.
# ---------------------------------------------------------------------------
class TraceabilityMetric(BaseMetric):
    """Of the reasons generated for this applicant, what fraction name a
    feature that (a) actually appears in that applicant's SHAP explanation
    and (b) contributed in the adverse (toward-decline) direction?

    This is set membership against ground truth, not an LLM judgment. Ground
    truth comes from the EvalCase fields (threaded through via
    `test_case.metadata`):

        metadata["shap_features_available"]  -- list[str], ALL feature names
                                                  present in the explanation,
                                                  regardless of direction
        metadata["top_decline_drivers"]       -- list[dict] with "feature"
                                                  and "shap_value" keys;
                                                  already restricted to
                                                  shap_value > 0 by
                                                  Explanation.top_decline_drivers()
        metadata["generated_reasons"]         -- list[dict] (or ReasonCode-like
                                                  objects) with "feature" and
                                                  "shap_value" keys; the
                                                  agent's actual output,
                                                  structured

        metadata["decision"]                  -- "APPROVE" | "DECLINE", used
                                                  only to score empty output
                                                  correctly (see class
                                                  docstring: empty is right
                                                  on APPROVE, wrong on DECLINE)

    Falls back to parsing `test_case.actual_output` as a JSON list of
    {"feature": ..., "shap_value": ...} dicts if `generated_reasons` isn't in
    metadata. This fallback exists for callers who haven't wired the
    structured objects through yet; it is NOT the recommended path -- prefer
    populating metadata["generated_reasons"] with the real ReasonCode data.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""

    # -- extraction -----------------------------------------------------
    @staticmethod
    def _extract_generated_reasons(test_case: LLMTestCase) -> list[dict]:
        meta = test_case.metadata or {}
        reasons = meta.get("generated_reasons")
        if reasons is not None:
            out = []
            for r in reasons:
                if isinstance(r, dict):
                    out.append({"feature": r.get("feature"), "shap_value": r.get("shap_value")})
                else:  # duck-type a ReasonCode-like object
                    out.append({"feature": getattr(r, "feature", None),
                                "shap_value": getattr(r, "shap_value", None)})
            return out

        # Fallback: naive JSON parse of actual_output. Measures the regex/
        # parser, not the model -- avoid relying on this path.
        try:
            parsed = json.loads(test_case.actual_output)
            return [{"feature": r.get("feature"), "shap_value": r.get("shap_value")}
                    for r in parsed]
        except Exception:
            return []

    # -- core logic -------------------------------------------------------
    def measure(self, test_case: LLMTestCase) -> float:
        meta = test_case.metadata or {}
        available: set[str] = set(meta.get("shap_features_available", []))
        decline_drivers = meta.get("top_decline_drivers", [])
        adverse_features: set[str] = {
            d["feature"] for d in decline_drivers if d.get("shap_value", 0) > 0
        }
        decision = meta.get("decision", "DECLINE")

        generated = self._extract_generated_reasons(test_case)

        # Empty output: correct on APPROVE, a hard failure on DECLINE.
        if not generated:
            if decision != "DECLINE":
                self.score, self.success = 1.0, True
                self.reason = "No reasons generated; decision was not DECLINE, so this is correct."
            else:
                self.score, self.success = 0.0, False
                self.reason = ("Zero reasons generated for a DECLINE case. Silence is not "
                                "traceability -- scored as a hard failure, not a vacuous 1.0.")
            return self.score

        phantom = []      # feature name not in the explanation at all
        wrong_direction = []  # feature exists, but not an adverse (toward-decline) driver
        traceable = []

        for r in generated:
            feat = r.get("feature")
            if feat not in available:
                phantom.append(feat)
            elif feat not in adverse_features:
                wrong_direction.append(feat)
            else:
                traceable.append(feat)

        self.score = len(traceable) / len(generated)
        self.success = self.score >= self.threshold

        parts = [f"{len(traceable)}/{len(generated)} reasons traceable to an adverse SHAP driver."]
        if phantom:
            parts.append(f"Phantom features (not in explanation at all): {phantom}.")
        if wrong_direction:
            parts.append(f"Wrong-direction citations (feature exists but didn't argue toward "
                          f"decline): {wrong_direction}.")
        self.reason = " ".join(parts)
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Traceability"


# ---------------------------------------------------------------------------
# 2 & 3. Built-ins, both explicitly wired to Gemini.
# ---------------------------------------------------------------------------
def build_answer_relevancy_metric(threshold: float = 0.7) -> AnswerRelevancyMetric:
    """Does the generated reason text actually answer 'why was this
    applicant declined,' rather than drifting into unrelated boilerplate?"""
    return AnswerRelevancyMetric(model=get_judge(), threshold=threshold)


def build_faithfulness_metric(threshold: float = 0.7) -> FaithfulnessMetric:
    """Soft, LLM-judged mirror of TraceabilityMetric: does the reason text
    stay faithful to the SHAP drivers supplied as retrieval_context (the
    real top_decline_drivers for this applicant), or does it hallucinate
    claims the context doesn't support?

    Feed the real drivers in as retrieval_context when constructing the
    LLMTestCase, e.g.:

        retrieval_context=[
            f"{d['feature']}={d['display_value']} (SHAP {d['shap_value']:+.3f})"
            for d in case['top_decline_drivers']
        ]

    Worth contrasting directly with TraceabilityMetric in the README: this
    one costs an API call and is fuzzy (an LLM judging text against text);
    TraceabilityMetric is free, deterministic, and exact (set membership
    against ground truth). In production, trust TraceabilityMetric as the
    gate and treat this as a secondary signal for phrasing quality --
    faithfulness-by-LLM should never be the thing standing between a
    hallucinated reason and a real applicant.
    """
    return FaithfulnessMetric(model=get_judge(), threshold=threshold)