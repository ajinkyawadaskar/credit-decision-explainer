"""FastAPI service: credit decision + SHAP-traceable adverse action reasons.

Endpoints:
    GET  /health    liveness + what the reason pipeline is currently capable of
    POST /decision  score one applicant, return decision + SHAP drivers + reasons

Design note — graceful degradation. `reason_mapper.py` and `graph.py` are owned
by Ajinkya and may not be implemented yet. The API does NOT 500 in that case:
it returns the decision and the real SHAP drivers, and reports
`reasons_status="not_implemented"`. That keeps the service demoable and
deployable while the interpretive layer is still being written, and it means
`/health` tells you the truth about what is wired up.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import NUMERIC_COLUMNS  # noqa: E402
from schemas import (  # noqa: E402
    ApplicantRequest,
    DecisionResponse,
    DriverOut,
    HealthResponse,
    ReasonOut,
)
from shap_explainer import CreditExplainer  # noqa: E402

MODEL_VERSION = "xgb-german-credit-0.1.0"

_state: dict = {"explainer": None, "reason_pipeline": "not_implemented"}


def _load_reason_mapper():
    """Import reason_mapper only if it's actually implemented.

    The stub raises NotImplementedError at import time by design, so this
    returns None until Ajinkya fills it in — no special-casing needed later.
    """
    try:
        from reason_mapper import map_reasons  # noqa: F401

        return map_reasons
    except (NotImplementedError, ImportError):
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per request.
    _state["explainer"] = CreditExplainer()
    _state["map_reasons"] = _load_reason_mapper()
    _state["reason_pipeline"] = (
        "ok" if _state["map_reasons"] is not None else "not_implemented"
    )
    yield
    _state.clear()


app = FastAPI(
    title="Credit Decision Explainer",
    description=(
        "XGBoost credit decisions with SHAP-traceable ECOA adverse action reasons. "
        "Every generated reason must trace to a real SHAP feature for that applicant."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Never leak a stack trace to a credit API caller."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": type(exc).__name__},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    ex = _state.get("explainer")
    loaded = ex is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
        reason_pipeline=_state.get("reason_pipeline", "unknown"),
        n_features=len(ex.feature_names) if loaded else 0,
    )


@app.post("/decision", response_model=DecisionResponse, tags=["credit"])
def decision(applicant: ApplicantRequest) -> DecisionResponse:
    ex = _state.get("explainer")
    if ex is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Rebuild the exact dtypes the model was trained on. Category dtype matters:
    # XGBoost native categoricals will reject plain object columns.
    row = applicant.model_dump()
    frame = pd.DataFrame([row])[ex.feature_names]
    for col in frame.columns:
        if col not in NUMERIC_COLUMNS:
            frame[col] = frame[col].astype("category")

    try:
        e = ex.explain(frame)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not score applicant: {type(exc).__name__}",
        ) from exc

    drivers = [
        DriverOut(
            feature=c.feature,
            value=str(c.raw_value),
            display_value=c.display_value,
            shap_value=round(c.shap_value, 4),
            direction=c.direction,
        )
        for c in e.top_decline_drivers(4)
    ]

    reasons = None
    reasons_status = _state.get("reason_pipeline", "not_implemented")
    map_reasons = _state.get("map_reasons")
    if map_reasons is not None and e.decision == "DECLINE":
        mapped = map_reasons(e)
        reasons = [
            ReasonOut(
                code=r.code, text=r.text, feature=r.feature,
                shap_value=round(r.shap_value, 4), rank=r.rank,
            )
            for r in mapped
        ]

    return DecisionResponse(
        decision=e.decision,
        probability_bad=round(e.probability_bad, 4),
        threshold=e.threshold,
        base_value=round(e.base_value, 4),
        top_drivers=drivers,
        reasons=reasons,
        reasons_status=reasons_status,
        model_version=MODEL_VERSION,
    )
