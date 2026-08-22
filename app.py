"""Streamlit demo UI for the Credit Decision Explainer.

Deliberately shows the mechanism, not just the answer: every reason is
displayed next to the SHAP feature and value it traces to, and suppressed
drivers are shown explicitly so the citability rule is visible rather than
implied. That transparency IS the demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import CODE_LABELS, NUMERIC_COLUMNS, load_clean, split  # noqa: E402
from shap_explainer import CreditExplainer  # noqa: E402

st.set_page_config(page_title="Credit Decision Explainer", page_icon="🏦", layout="wide")


@st.cache_resource
def get_explainer():
    return CreditExplainer()


@st.cache_resource
def get_mapper():
    """reason_mapper may be an unimplemented stub — degrade, don't crash."""
    try:
        from reason_mapper import map_reasons

        return map_reasons
    except (NotImplementedError, ImportError):
        return None


@st.cache_data
def get_samples():
    X, y = load_clean()
    _, X_test, _, y_test = split(X, y)
    return X_test, y_test


NUMERIC_BOUNDS = {
    "duration_months": (4, 72, 24),
    "credit_amount": (250, 20000, 3500),
    "installment_rate_pct_income": (1, 4, 3),
    "residence_since_years": (1, 4, 2),
    "existing_credits_count": (1, 4, 1),
    "num_dependents": (1, 2, 1),
}

PRETTY = {
    "checking_status": "Checking account status",
    "duration_months": "Loan term (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Amount requested (DM)",
    "savings_status": "Savings account",
    "employment_since": "Employed for",
    "installment_rate_pct_income": "Installment rate (% of income)",
    "other_debtors": "Other debtors / guarantors",
    "residence_since_years": "Years at address",
    "property_magnitude": "Property owned",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits_count": "Existing credits at this bank",
    "job": "Employment type",
    "num_dependents": "Dependents",
    "telephone": "Telephone",
}

ex = get_explainer()
map_reasons = get_mapper()
X_test, y_test = get_samples()
FEATURES = ex.feature_names

st.title("🏦 Credit Decision Explainer")
st.caption(
    "XGBoost decision → SHAP attribution → ECOA adverse action reasons. "
    "Every reason must trace to a real SHAP feature for this applicant."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Applicant")

    preset = st.selectbox(
        "Load a real test record",
        ["— custom —"] + [f"row {i}" for i in range(0, len(X_test), 7)],
        help="Records from the held-out test set, so the decision is on unseen data.",
    )
    if preset != "— custom —":
        base = X_test.iloc[int(preset.split()[1])].to_dict()
    else:
        base = X_test.iloc[0].to_dict()

    # Kept at the top: with 17 inputs below, a bottom button means scrolling
    # the whole form before you can score anything.
    go = st.button("Score applicant", type="primary", use_container_width=True)
    st.divider()

    values = {}
    for f in FEATURES:
        label = PRETTY.get(f, f)
        if f in NUMERIC_COLUMNS:
            lo, hi, _ = NUMERIC_BOUNDS.get(f, (0, 100, 1))
            cur = int(base[f])
            values[f] = st.slider(label, lo, max(hi, cur), min(max(cur, lo), max(hi, cur)))
        else:
            opts = list(CODE_LABELS.get(f, {}).keys()) or sorted(X_test[f].unique())
            cur = str(base[f])
            idx = opts.index(cur) if cur in opts else 0
            values[f] = st.selectbox(
                label, opts, index=idx,
                format_func=lambda c, ff=f: CODE_LABELS.get(ff, {}).get(c, c),
            )


# ---------------------------------------------------------------- main
if go or "scored" in st.session_state:
    st.session_state["scored"] = True

    frame = pd.DataFrame([values])[FEATURES]
    for col in frame.columns:
        if col not in NUMERIC_COLUMNS:
            frame[col] = frame[col].astype("category")

    e = ex.explain(frame)
    declined = e.decision == "DECLINE"

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", e.decision)
    c2.metric("P(bad risk)", f"{e.probability_bad:.1%}",
              delta=f"{e.probability_bad - e.threshold:+.1%} vs threshold",
              delta_color="inverse")
    c3.metric("Threshold", f"{e.threshold:.0%}")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("SHAP attribution")
        st.caption("Positive pushes toward decline. Exact — verified additive.")
        contribs = sorted(e.contributions, key=lambda c: c.shap_value, reverse=True)
        chart_df = pd.DataFrame(
            {"feature": [PRETTY.get(c.feature, c.feature) for c in contribs],
             "SHAP": [c.shap_value for c in contribs]}
        ).set_index("feature")
        st.bar_chart(chart_df, horizontal=True, height=460)
        st.caption(
            f"base {e.base_value:+.3f} + Σshap "
            f"{sum(c.shap_value for c in e.contributions):+.3f} = "
            f"{e.base_value + sum(c.shap_value for c in e.contributions):+.3f} log-odds"
        )

    with right:
        st.subheader("Adverse action reasons")

        if not declined:
            st.success("**APPROVED** — no adverse action, so no reasons are generated.")
            st.caption(
                "ECOA reason-disclosure obligations attach to adverse actions. "
                "Volunteering 'what would have hurt you' on an approval is out of scope."
            )
        elif map_reasons is None:
            st.warning("`reason_mapper.py` is not implemented — showing raw drivers only.")
            for c in e.top_decline_drivers(4):
                st.write(f"- **{PRETTY.get(c.feature, c.feature)}**: {c.display_value} "
                         f"(SHAP {c.shap_value:+.3f})")
        else:
            reasons = map_reasons(e)
            st.caption(f"{len(reasons)} principal reason(s), each traced to a real SHAP driver.")
            for r in reasons:
                with st.container(border=True):
                    st.markdown(f"**{r.rank}. {r.text}**")
                    st.caption(
                        f"`{r.code}` · traces to **{r.feature}** · SHAP `{r.shap_value:+.4f}`"
                    )

            cited = {r.feature for r in reasons}
            dropped = [c for c in e.top_decline_drivers(len(e.contributions))
                       if c.feature not in cited][:5]
            if dropped:
                with st.expander(f"Adverse drivers NOT cited ({len(dropped)} shown) — why"):
                    st.caption(
                        "A driver can be real and still be unusable: below the materiality "
                        "floor, no lawful Reg B reason maps to it, or the value itself is "
                        "uncitable (e.g. 'paid all credits duly' pushing toward decline)."
                    )
                    for c in dropped:
                        st.write(f"- **{PRETTY.get(c.feature, c.feature)}** = "
                                 f"{c.display_value} (SHAP {c.shap_value:+.3f})")
else:
    st.info("👈 Pick a test record or build an applicant, then **Score applicant**.")
    st.markdown(
        "**How it works** — XGBoost scores the applicant, SHAP decomposes that score "
        "into exact per-feature contributions, and a *deterministic* mapper selects which "
        "contributions may lawfully be cited as adverse action reasons. An LLM only "
        "renders the final prose, and a validator can reject it."
    )
