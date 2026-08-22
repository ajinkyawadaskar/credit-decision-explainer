"""LangGraph agent: SHAP explanation -> validated adverse action notice.

===============================================================================
    OWNER: Ajinkya.  DO NOT let Claude implement this module.
===============================================================================

THE ARCHITECTURAL POINT (this is the interview answer)
------------------------------------------------------
The LLM does NOT decide anything. By the time it runs, the decision is made
(model.py), the drivers are computed (shap_explainer.py), and the reasons are
SELECTED (reason_mapper.py, deterministic). The LLM's only job is to render
already-chosen reasons into readable prose, and its output is then checked by
a deterministic validator that can reject it.

    decide -> attribute -> SELECT (deterministic) -> render (LLM) -> VERIFY

That ordering is the whole design. Most "LLM + explainability" projects let
the model generate the reasons, which means nothing downstream can tell a real
reason from a fluent invention. Here the LLM is a rendering layer wrapped in a
deterministic sandwich, so the guarantee survives contact with it.

Say it that way and you have answered "why is this trustworthy" in one breath.


GRAPH SHAPE
-----------
Nodes (suggested; reshape if you prefer):

    explain      X -> Explanation                (calls shap_explainer)
    select       Explanation -> list[ReasonCode] (calls reason_mapper)
    render       ReasonCodes -> prose            (calls Gemini)
    verify       prose -> ValidationResult       (calls validator)

Edges:

    explain -> select -> render -> verify
    verify --conditional--> END              if passed
                       --> render            if failed and retries remain
                       --> fallback          if retries exhausted

The `fallback` path is the most important edge in the graph: emit the
deterministic reason_mapper text verbatim. It is blunt and it is correct.
**The LLM must be an enhancement, never a dependency** — if Gemini is down,
rate-limited, or producing garbage, the service still returns a lawful,
traceable notice.


STATE
-----
LangGraph state is a TypedDict threaded through nodes. Suggested keys:

    applicant: pd.DataFrame
    explanation: Explanation | None
    reasons: list[ReasonCode]
    draft_text: str | None
    validation: ValidationResult | None
    attempts: int
    used_fallback: bool
    final_text: str | None

Keep `attempts` and `used_fallback` in state, not in a closure — the eval
harness reports fallback rate, and that number belongs in the README. A high
fallback rate is not a failure; it is evidence the validator is working.


PROMPTING — where the risk lives
--------------------------------
Give the model ONLY the selected reasons. Do not hand it the full SHAP vector,
the raw applicant record, or the probability. Anything you put in the prompt,
it can and eventually will mention. The narrowest possible context is also the
smallest hallucination surface.

Instruct explicitly: rephrase these N reasons, add nothing, invent no numbers,
do not speculate about what would have changed the outcome. Then assume it
disobeys anyway and let `verify` catch it — the prompt is a preference, the
validator is the guarantee.

Temperature 0. This is a compliance document, not copywriting.

Model wiring (installed and verified in this env):

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,
                                 google_api_key=os.environ["GOOGLE_API_KEY"])

`.env.example` already declares GOOGLE_API_KEY and GEMINI_MODEL. Load with
python-dotenv. Never hardcode the key; never log the prompt with it attached.


DECISIONS YOU OWN
-----------------
1. RETRY COUNT. How many times do you re-render before falling back? Each
   retry costs latency and money on a request in the serving path. 1? 2?
   Justify it.

2. DOES THE RETRY SEE THE VIOLATION? Feeding the validator's complaint back
   ("you mentioned employment history, which is not a driver") usually helps.
   It also risks the model treating the complaint as new material to discuss.
   Try both.

3. WHAT THE API RETURNS ON FALLBACK. Does the caller learn that the LLM was
   overridden? Argument for yes: auditability, and it makes the mechanism
   visible in the demo. Argument for no: it's an implementation detail. I'd
   surface it — it is the most interesting thing your API knows.

4. APPROVE PATH. reason_mapper returns [] for approvals. Does the graph
   short-circuit before `render`, or run through with empty reasons? Short-
   circuiting is cheaper and removes a whole class of failure.


HARD REQUIREMENTS
-----------------
- The graph must never emit text that failed validation. Fallback instead.
- No network call in `select` or `verify` — only `render` touches the LLM.
- Every path must terminate. Bound the retries; a validation loop that can
  spin forever is a production incident.
- The API imports this lazily (see api.py::_load_reason_mapper for the
  pattern) so a missing GOOGLE_API_KEY degrades rather than crashes startup.


TEST AGAINST
------------
    evals/cases.json — 10 grounded cases, 5 adversarial. ADV-03 is the one to
    watch: credit_history=A31 is suppressed by reason_mapper, so if the LLM
    reintroduces "your credit history" into the prose, `verify` must reject
    it. That single case exercises the entire architecture.
"""
from __future__ import annotations

# TODO(Ajinkya): implement. See module docstring for the full spec.
raise NotImplementedError(
    "src/graph.py is owned by Ajinkya and has not been implemented yet. "
    "See the module docstring for the spec."
)
