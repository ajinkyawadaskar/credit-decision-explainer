# Architecture

## Request path

```mermaid
flowchart TD
    A[Applicant JSON<br/>17 features] --> B{Pydantic<br/>validation}
    B -->|invalid| B1[422<br/>strict A-code Literals<br/>prohibited-basis fields unsubmittable]
    B -->|valid| C[XGBoost<br/>P bad risk]
    C --> D[SHAP TreeExplainer<br/>exact per-feature attribution]
    D --> E{P bad >= 0.5?}
    E -->|no| F[APPROVE<br/>no adverse action, no reasons]
    E -->|yes| G[reason_mapper<br/>DETERMINISTIC]

    G --> G1[drop approve-direction drivers]
    G1 --> G2[drop below materiality floor 0.05]
    G2 --> G3[drop uncitable:<br/>7 features + A31/A32 values]
    G3 --> G4[take top 4 -> ReasonCode<br/>carries feature + shap_value]

    G4 --> H[graph.py render<br/>Gemini, temp 0<br/>sees ONLY selected reasons]
    H --> I{validator.py<br/>DETERMINISTIC}
    I -->|passes| J[Adverse action notice]
    I -->|fails, retries left| H
    I -->|retries exhausted| K[FALLBACK<br/>deterministic reason text verbatim]
    K --> J

    style G fill:#1f6f43,color:#fff
    style I fill:#1f6f43,color:#fff
    style H fill:#8a5a00,color:#fff
    style K fill:#7a2020,color:#fff
```

**Green = deterministic. Orange = stochastic. The orange node is wrapped on both sides.**

## Why this ordering

The LLM never decides anything. By the time it runs:

- the decision is made (`model.py`)
- the drivers are computed (`shap_explainer.py`)
- the reasons are **selected** (`reason_mapper.py`, pure function)

Its only job is rendering already-chosen reasons into prose, and a deterministic
validator can reject that prose. Most "LLM + explainability" projects let the
model generate the reasons — which means nothing downstream can distinguish a
real reason from a fluent invention.

```
decide -> attribute -> SELECT (deterministic) -> render (LLM) -> VERIFY (deterministic)
```

## Where the guarantee could break, and what holds it

| Failure | Guard |
|---|---|
| Model trained on a prohibited basis | Dropped in `data.py` before training |
| Prohibited basis submitted via API | Absent from schema; `extra="forbid"` -> 422 |
| Prohibited basis reaches the mapper | `assert` in `is_citable()` (defence in depth) |
| Reason cites a feature with no SHAP support | `reason_mapper` builds only from real contributions |
| Reason cites a feature that argued for approval | `direction != toward_decline` filter |
| Reason is true but unlawful to cite | `_UNCITABLE_VALUES` (the A31 case) |
| **LLM invents a reason while writing prose** | **`validator.py` — the reason this project exists** |
| LLM unavailable or repeatedly invalid | Fallback to deterministic text |

## Module ownership

| Module | Purpose | Author |
|---|---|---|
| `data.py` | Load, drop prohibited bases, type, split | Claude |
| `model.py` | XGBoost train/eval/persist | Claude |
| `shap_explainer.py` | TreeSHAP attribution + additivity check | Claude |
| `schemas.py`, `api.py` | Pydantic models, FastAPI service | Claude |
| `evals/dataset.py` | Grounded eval case generation | Claude |
| **`reason_mapper.py`** | **SHAP -> ECOA reason codes** | **Ajinkya** |
| **`validator.py`** | **Traceability enforcement** | **Ajinkya** |
| **`graph.py`** | **LangGraph node wiring** | **Ajinkya** |
| **`evals/metrics.py`** | **Custom DeepEval metrics** | **Ajinkya** |

The four interpretive modules — the ones carrying every judgment call a
reviewer would question — were written by hand, deliberately.
