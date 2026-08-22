"""Custom DeepEval metrics for the credit reason pipeline.

===============================================================================
    OWNER: Ajinkya.  DO NOT let Claude implement this module.
    The traceability metric IS the thesis of this project. If someone asks you
    one question about this repo, it will be "how do you know the model didn't
    make the reason up," and this file is the answer.
===============================================================================

WHAT TO BUILD: one custom metric + two built-ins (scope locked in CLAUDE.md).


-------------------------------------------------------------------------------
1. TraceabilityMetric  (custom — this is the one that matters)
-------------------------------------------------------------------------------
Question it answers: of the reasons the agent generated for this applicant,
what fraction name a feature that actually appears in that applicant's SHAP
attribution, with a contribution in the adverse direction?

This is NOT an LLM-judged metric. It is deterministic set membership. That is
the whole point — an LLM judging whether an LLM hallucinated is circular. You
have exact ground truth in `case["shap_features_available"]`, so use it.

Implementation shape (DeepEval `BaseMetric`):

    class TraceabilityMetric(BaseMetric):
        def __init__(self, threshold: float = 1.0): ...
        def measure(self, test_case: LLMTestCase) -> float: ...
        async def a_measure(self, test_case) -> float: ...
        def is_successful(self) -> bool: ...
        @property
        def __name__(self): return "Traceability"

    Required attributes DeepEval reads after measure(): self.score,
    self.success, self.reason (a human-readable explanation string).

Decisions you own:
  - THRESHOLD. Should this be 1.0 (any untraceable reason fails the case) or
    something softer? Argue it. Given the project's premise, anything below 1.0
    needs a very good justification.
  - DIRECTION. Is it enough that the feature exists in the explanation, or must
    its SHAP value also be positive (adverse)? Citing a feature that argued FOR
    approval as a denial reason is a distinct and arguably worse failure than
    naming a feature that doesn't exist. Consider scoring them separately.
  - EXTRACTION. How do you get the feature names back out of the agent's
    output? If reason_mapper returns structured ReasonCode objects with a
    `feature` field, this is trivial and exact. If you're regex-ing feature
    names out of English prose, the metric measures your regex, not the model.
    Strongly prefer threading the structured objects through.
  - EMPTY OUTPUT. An agent that returns zero reasons is trivially 100%
    traceable. Make sure that scores as a failure, not a win. This is the
    single most likely way to accidentally build a metric that rewards
    silence.

-------------------------------------------------------------------------------
2 & 3. Two built-ins
-------------------------------------------------------------------------------
Pick two from DeepEval and configure them with Gemini. Verified available and
importable in this env (deepeval 4.1.10):

    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.models import GeminiModel      # native, no wrapper needed

    judge = GeminiModel(model="gemini-2.0-flash", api_key=os.environ["GOOGLE_API_KEY"])
    AnswerRelevancyMetric(model=judge, threshold=0.7)

FaithfulnessMetric measures hallucination against `retrieval_context` — you can
feed the SHAP drivers in as context, which makes it a soft LLM-judged mirror of
your hard traceability check. Contrasting the two in the README is a strong
"Decisions & Tradeoffs" entry: one is exact and cheap, one is fuzzy and costs
an API call, and you should be able to say which you'd trust in production.

Other candidates if you prefer: GEval (define your own rubric — good for
"would this sentence be acceptable in an adverse action notice?"), BiasMetric,
ToxicityMetric.

-------------------------------------------------------------------------------
LLMTestCase fields available (verified in this env)
-------------------------------------------------------------------------------
    input             required  — the applicant / prompt
    actual_output     the generated reasons
    expected_output   your labeled expectation
    context           list[str]
    retrieval_context list[str]  — put the real SHAP drivers here
    metadata          dict       — good place for shap_features_available

-------------------------------------------------------------------------------
GOTCHA
-------------------------------------------------------------------------------
DeepEval defaults to OpenAI for every LLM-judged metric. If you forget to pass
`model=judge`, it will fail on a missing OPENAI_API_KEY, or silently bill the
wrong provider. Pass the Gemini judge explicitly to every built-in.
"""
from __future__ import annotations

# TODO(Ajinkya): implement. See module docstring for the full spec.
raise NotImplementedError(
    "evals/metrics.py is owned by Ajinkya and has not been implemented yet. "
    "See the module docstring for the spec."
)
