"""LangGraph pipeline: applicant record -> lawful, traceable adverse-action
notice.

    explain (shap_explainer) -> select (reason_mapper) -> render (Gemini)
        -> verify (validator) --pass--> END
                              --fail, retries left--> render (retry)
                              --fail, retries exhausted--> fallback -> END

**The LLM is an enhancement, never a dependency.** If Gemini is down, rate-
limited, produces garbage, or keeps reintroducing a suppressed feature, the
service still returns a lawful, traceable notice: the deterministic
reason_mapper text, verbatim. That fallback edge is the most important line
in this file.

DECISIONS THIS FILE MAKES
--------------------------
1. RETRY COUNT = 1 (one retry after the initial render, so at most 2 render
   calls total before falling back). Justification: this runs in the serving
   path on every DECLINE. Each retry is a real API call with real latency and
   real cost, and the validator is deterministic -- if the first retry, fed
   the actual violation, still fails, there's no reason to believe a third
   blind attempt succeeds where a corrected second one didn't. Two chances
   (original + one corrected attempt) is enough to absorb ordinary LLM
   flakiness without turning a bounded compliance check into an unbounded
   cost center. If eval data later shows retry #2 has a meaningfully higher
   pass rate than retry #1, this constant is the one line to change.

2. THE RETRY SEES THE VIOLATION. `render` is given the validator's
   `violations` list on retry (see `_build_prompt`), phrased as a correction
   instruction ("your previous draft mentioned X, which is not one of the
   listed reasons -- remove it and do not reintroduce it"), not just handed
   back as-is. Rationale: a validator complaint with zero context ("retry")
   gives the model no signal about what to fix, so it's likely to repeat the
   same mistake or invent a different one. Explicitly naming the rejected
   claim is more likely to fix that specific claim than staying silent,
   which is why the option is used here. Risk accepted: the model could treat
   the complaint as material to discuss ("the applicant's credit history...")
   rather than an instruction to avoid it -- this is why `verify` re-checks
   the retry output with the exact same deterministic gate as the first
   attempt; the corrected prompt is a nudge, not a guarantee.

3. FALLBACK IS SURFACED, NOT HIDDEN. `used_fallback` is a top-level field on
   the returned state, always populated (True/False), never inferred by the
   caller from absence of other fields. Argument for surfacing over hiding:
   this is the single most interesting thing the API knows about a given
   response -- "a human should know a machine had to override the language
   model to stay lawful here" is exactly the kind of fact that belongs in an
   audit trail, and it costs nothing to expose since the graph already tracks
   it in state for the eval harness's fallback-rate metric.

4. APPROVE SHORT-CIRCUITS BEFORE render. `select` returns `reasons=[]` for
   an APPROVE explanation (per reason_mapper.map_reasons). The graph checks
   this immediately after `select` and routes straight to a trivial terminal
   node that sets `final_text = ""` (no adverse reasons to render), skipping
   `render` and `verify` entirely. Rationale: this removes a whole class of
   failure (an LLM asked to write prose for zero reasons might invent
   reasons out of sheer helpfulness) and saves a network call on every
   approval, which is the majority of any real applicant pool.
"""
from __future__ import annotations

import os
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

MAX_ATTEMPTS = 2  # initial render + 1 retry, per decision 1 above


# ---------------------------------------------------------------------------
# Lazy imports. Mirrors api.py::_load_reason_mapper -- a missing
# GOOGLE_API_KEY (or a broken langchain_google_genai install) must degrade
# the render step, not crash graph construction or process startup. Only
# `render` needs the LLM; `explain`, `select`, and `verify` never touch the
# network and must keep working even if Gemini is entirely unavailable.
# ---------------------------------------------------------------------------
def _load_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,  # compliance document, not copywriting
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def _load_explainer():
    from shap_explainer import CreditExplainer

    return CreditExplainer()


def _load_reason_mapper():
    from reason_mapper import map_reasons

    return map_reasons


def _load_validator():
    from validator import validate

    return validate


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    applicant: pd.DataFrame
    explanation: object | None       # shap_explainer.Explanation
    reasons: list                    # list[reason_mapper.ReasonCode]
    draft_text: str | None
    validation: object | None        # validator.ValidationResult
    attempts: int
    used_fallback: bool
    final_text: str | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def explain(state: GraphState) -> GraphState:
    """X -> Explanation. No network call -- shap_explainer is a local model."""
    explainer = _load_explainer()
    explanation = explainer.explain(state["applicant"])
    return {**state, "explanation": explanation}


def select(state: GraphState) -> GraphState:
    """Explanation -> list[ReasonCode]. Pure, deterministic, no network."""
    map_reasons = _load_reason_mapper()
    reasons = map_reasons(state["explanation"])
    return {**state, "reasons": reasons}


def _deterministic_text(reasons) -> str:
    """The blunt, always-correct rendering: reason_mapper's own text, joined
    verbatim with no LLM involvement. This is what `fallback` emits and what
    APPROVE cases (no reasons) reduce to."""
    if not reasons:
        return ""
    return "\n".join(f"[{r.code}] {r.text}" for r in reasons)


def approve_passthrough(state: GraphState) -> GraphState:
    """Terminal node for APPROVE cases (reasons == []). No render, no verify
    -- see decision 4. There is nothing adverse to explain, so there is
    nothing for an LLM to embellish."""
    return {
        **state,
        "draft_text": None,
        "validation": None,
        "used_fallback": False,
        "final_text": "",
    }


def _build_prompt(reasons, violations: list[str] | None) -> str:
    """Narrowest possible context: ONLY the selected reasons' text, never the
    raw SHAP vector, the applicant record, or the probability. Anything put
    in the prompt, the model can and eventually will mention -- so nothing
    goes in that isn't already authorized to appear in the letter.
    """
    reason_lines = "\n".join(f"- {r.text}" for r in reasons)

    instructions = (
        "You are drafting the reasons section of a credit adverse-action "
        "notice. Below are the ONLY approved reasons for this decision. "
        "Rephrase them into clear, professional, applicant-facing prose. "
        "Do not add any reason, factor, or detail that is not listed below. "
        "Do not invent or restate any numbers other than ones that literally "
        "appear below. Do not speculate about what would have changed the "
        "outcome. Do not mention any topic not listed below, even in passing.\n\n"
        f"Approved reasons:\n{reason_lines}"
    )

    if violations:
        violation_lines = "\n".join(f"- {v}" for v in violations)
        instructions += (
            "\n\nYour previous draft was rejected for the following reason(s). "
            "Remove the offending content and do not reintroduce it in any "
            f"form:\n{violation_lines}"
        )

    return instructions


def render(state: GraphState) -> GraphState:
    """ReasonCodes (+ optional prior violations) -> prose. The only node
    that touches the network. If the LLM is unavailable for any reason, this
    fails soft: draft_text is left None and validation naturally fails,
    routing straight to fallback rather than raising out of the graph.
    """
    reasons = state["reasons"]
    prior_violations = (
        state["validation"].violations if state.get("validation") else None
    )
    prompt = _build_prompt(reasons, prior_violations)

    attempts = state.get("attempts", 0) + 1

    try:
        llm = _load_llm()
        response = llm.invoke(prompt)
        content = response.content
        # ChatGoogleGenerativeAI can return content as a list of parts
        # (e.g. [{"type": "text", "text": "..."}]) instead of a plain str.
        # Normalize to a single string before it ever reaches validate_text(),
        # which calls .lower() on draft_text and will raise AttributeError
        # on a list.
        if isinstance(content, list):
            draft_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            draft_text = content
    except Exception:
        # LLM down, rate-limited, missing key, whatever -- draft_text stays
        # None, verify() will fail it, and the graph proceeds to fallback.
        draft_text = None

    return {**state, "draft_text": draft_text, "attempts": attempts}


def verify(state: GraphState) -> GraphState:
    """prose -> ValidationResult. No network call -- validator.py is pure."""
    validate = _load_validator()
    draft_text = state.get("draft_text")

    if draft_text is None:
        # render() failed outright (LLM unavailable). Build a ValidationResult
        # directly rather than calling validate() on a None text.
        from validator import ValidationResult

        validation = ValidationResult(
            passed=False,
            violations=["No draft text produced (LLM call failed or unavailable)."],
            checked_features=set(),
        )
    else:
        validation = validate(state["reasons"], draft_text, state["explanation"])

    return {**state, "validation": validation}


def fallback(state: GraphState) -> GraphState:
    """The most important node in the graph. Emit reason_mapper's own text,
    verbatim, no LLM. Blunt and correct beats fluent and unverified."""
    return {
        **state,
        "used_fallback": True,
        "final_text": _deterministic_text(state["reasons"]),
    }


def accept(state: GraphState) -> GraphState:
    """Terminal node for a validation pass: ship the LLM prose as-is."""
    return {
        **state,
        "used_fallback": False,
        "final_text": state["draft_text"],
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def _route_after_select(state: GraphState) -> str:
    return "approve_passthrough" if not state["reasons"] else "render"


def _route_after_verify(state: GraphState) -> str:
    if state["validation"].passed:
        return "accept"
    if state.get("attempts", 0) < MAX_ATTEMPTS:
        return "render"
    return "fallback"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("explain", explain)
    graph.add_node("select", select)
    graph.add_node("approve_passthrough", approve_passthrough)
    graph.add_node("render", render)
    graph.add_node("verify", verify)
    graph.add_node("fallback", fallback)
    graph.add_node("accept", accept)

    graph.set_entry_point("explain")
    graph.add_edge("explain", "select")

    # Decision 4: APPROVE short-circuits before render/verify entirely.
    graph.add_conditional_edges(
        "select",
        _route_after_select,
        {"approve_passthrough": "approve_passthrough", "render": "render"},
    )
    graph.add_edge("approve_passthrough", END)

    graph.add_edge("render", "verify")

    # Bounded: MAX_ATTEMPTS caps how many times this can loop back to
    # render, so every path terminates. No edge in this graph can spin
    # forever -- a validation loop that can is a production incident.
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {"accept": "accept", "render": "render", "fallback": "fallback"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(applicant: pd.DataFrame) -> GraphState:
    app = build_graph()
    initial_state: GraphState = {
        "applicant": applicant,
        "explanation": None,
        "reasons": [],
        "draft_text": None,
        "validation": None,
        "attempts": 0,
        "used_fallback": False,
        "final_text": None,
    }
    return app.invoke(initial_state)