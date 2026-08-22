# Credit Decision Explainer Agent

**An XGBoost credit model whose adverse action reasons are provably traceable to
real SHAP attributions — enforced by a deterministic validator, not by trusting
the language model.**

<!-- TODO(Ajinkya): 30-second demo GIF goes HERE, above the fold. -->
<!-- Record: POST a declining applicant -> reasons appear -> point at the SHAP -->
<!-- value each one traces to. Then show a validation rejection if you can. -->

![Demo](docs/demo.gif)

**Live API:** [https://web-production-7b272.up.railway.app](https://web-production-7b272.up.railway.app/health) · [Architecture](docs/architecture.md) · [What didn't work](#what-didnt-work)

---

## The problem

When a lender denies credit, ECOA and FCRA require them to state the *specific
principal reasons*. Vague or wrong reasons are a compliance failure.

The obvious build — feed model output to an LLM, ask for reasons — fails in a
way that is very hard to catch. LLMs write fluent, plausible, well-structured
denial reasons that are **not true of this applicant**:

> given: `Length of requested credit term: 45`
> written: *"your 45-month term, combined with your **limited employment history**
> and **recent credit inquiries**, ..."*

Employment history wasn't a driver for that applicant. Credit inquiries aren't
even a feature in this dataset. Two fabricated adverse action reasons, in a
legal document, invisible to the recipient.

**This project makes that structurally impossible rather than unlikely.**

## The approach

```
decide -> attribute -> SELECT (deterministic) -> render (LLM) -> VERIFY (deterministic)
```

The LLM decides nothing. Reasons are *selected* by a pure function from real
SHAP values; the LLM only *renders* them; a deterministic validator can reject
its output; and when it does, the system falls back to the deterministic text.
The LLM is an enhancement, never a dependency.

See [docs/architecture.md](docs/architecture.md) for the full diagram.

## Results

### Model — measured

| Metric | Value |
|---|---|
| ROC AUC (held-out, n=200) | **0.8046** |
| ROC AUC across 5 splits | **0.8089 ± 0.0220**, range [0.7725, 0.8399] |
| PR AUC | 0.6827 (base rate 0.300) |
| Accuracy @ threshold 0.5 | 0.7800 |
| **Recall on the decline class** | **0.52** |

Single-split numbers on 1,000 rows are unreliable, so the distribution is
reported alongside the point estimate. A 0.01 "improvement" here is noise.

Decline-class recall of 0.52 is reported prominently because it is the number a
credit risk team would actually care about, and it says this model is not
deployable as-is without threshold work driven by cost asymmetry.

### Reason generation — deterministic layer, measured

| Check | Result |
|---|---|
| SHAP additivity (base + Σshap == model log-odds) | **PASS, 200/200 rows** |
| Feature coverage in the mapping table | **17/17** |
| Features suppressed as having no lawful reason | 7 |
| Declines producing zero reasons | **0 / 46** |
| **Reasons citing a feature absent from that applicant's SHAP output** | **0** |
| **Reasons citing a feature that argued toward approval** | **0** |
| Deterministic across repeated calls | yes |

### Traceability metric — validated against known-bad input

`TraceabilityMetric` is deterministic set membership at threshold 1.0 — one
untraceable reason fails the whole case.

| Input | Score |
|---|---|
| 10/10 honest cases | **1.00 (pass)** |
| Phantom feature (not in explanation) | 0.00 fail |
| Prohibited-basis feature | 0.00 fail |
| Wrong-direction citation | 0.00 fail |
| Empty output on a DECLINE | **0.00 fail** |
| Mixed (1 real, 1 fabricated) | 0.50 fail |

Empty output scoring 0.00 matters: an agent that says nothing is trivially
"100% traceable", and a naive metric would reward silence.

### End-to-end with the LLM — pending

Measured over the 10 labelled cases (5 adversarial), Gemini 3.6 Flash:

| Metric | Value |
|---|---|
| **Traceability == 1.00** | **10 / 10 (100%)** |
| Mean traceability | 1.000 |
| Validator rejections | 0 / 10 |
| Fallback to deterministic text | 0 / 10 (0%) |
| Reasons lost to prose merging | **1 / 8 declines** |
| p50 / p95 latency | **13.0s / 33.7s** |
| Answer relevancy (Gemini judge) | 1.00 on n=1 — **quota-blocked** |
| Faithfulness (Gemini judge) | **not measured — quota-blocked** |

Every generated reason traced to a real adverse SHAP driver for that applicant,
including ADV-03, where `credit_history` is suppressed as uncitable and the
model never reintroduced it.

**Three honest caveats on that 100%:**

1. **Zero rejections means the validator's reject path was never exercised by
   real model output.** It is verified against 12 hand-built adversarial inputs,
   not against a live hallucination. A clean run is evidence the LLM behaved,
   not proof the guard would catch it.
2. **The prose gate is closed-world.** It detects misattribution to features
   that exist in the dataset. An invented concept mapping to *no* feature —
   "recent credit inquiries", "prior bankruptcy", "debt-to-income ratio" —
   has no alias to match and passes. 5 of 8 such probes were missed. This is
   the main known gap.
3. **Omission is not validated.** In ADV-02 the model merged two reasons that
   share the same text into one bullet, emitting 3 for 4 selected reasons.
   Every reason shown was true; one simply vanished. Under Reg B, losing a
   principal reason is arguably worse than adding a spurious one.

**4. The two LLM-judged built-ins are not meaningfully measured.** The Gemini
free tier allows 20 `generateContent` requests; the graph's own render call plus
two judged metrics per case exhausts it after roughly one case. Answer relevancy
returned 1.00 on a single case — a sample of one is not a result — and
faithfulness never completed. Both are wired correctly and fail on quota, not on
code. This is a billing limit, not a finding.

That run also demonstrated the telemetry problem worth fixing: with the judges
enabled, **5 of 8 declines fell back to deterministic text** — not because the
validator rejected anything, but because the graph's own LLM calls were being
rate-limited. `used_fallback=True` currently cannot distinguish "the safety net
caught a hallucination" from "the API is throttled." The clean 0% fallback
figure above comes from the run without judges, where no rate limiting occurred.

**Latency is too high for a synchronous credit decision** at p95 33.7s. The
deterministic path is ~30ms; effectively all of it is the LLM. Serving this for
real means rendering asynchronously and returning the deterministic notice
immediately.

## Decisions & tradeoffs

**1. Dropped three features as ECOA prohibited bases — before training, not after.**
German Credit ships `personal_status_sex`, `foreign_worker`, and `age_years`.
Filtering them at the *explanation* layer would be worse than useless: the model
would still decide on them, we'd just be hiding it. Rejected that; dropped them
at load time. Cost in AUC: none measurable (0.8046 on 17 features).
*Limitation:* this addresses disparate **treatment**, not disparate **impact** —
proxies like housing and job may still correlate with protected class. A proper
fairness audit is out of scope and logged in `IDEAS.md`.

**2. Native categorical features instead of one-hot encoding.**
One-hot shatters a single concept across dummy columns, so `checking_status`
becomes four SHAP values that must be re-aggregated before they mean anything.
Native categoricals keep one feature = one SHAP value = one reason code. This
was an interpretability decision, not a performance one — accuracy was
comparable either way.

**3. Reason *selection* is deterministic; only *rendering* is delegated to the LLM.**
The rejected alternative — let the LLM read SHAP output and produce reasons — is
the industry-standard build and is unverifiable by construction. Splitting
selection from rendering is what makes the validator meaningful: there is a
ground truth to check against.

**4. Suppress uncitable reasons rather than remap them.**
`purpose` is what a loan is *for*, not collateral. Mapping it to "value or type
of collateral not sufficient" would read plausibly and invent a security-interest
claim the feature doesn't support. A wrong-but-plausible reason code is worse
than none, because the applicant cannot falsify it. Seven features are suppressed
outright.

**5. Fail closed, to deterministic text.**
When the validator rejects the LLM's prose and retries are exhausted, the
applicant receives the blunt deterministic reason text. Lower quality prose,
same legal correctness. A service that returns nothing when Gemini is down is
not a credit system.

## What didn't work

Full running log in [NOTES.md](NOTES.md). The ones worth repeating:

**The model wanted to deny people for paying their bills on time.**
`credit_history = A31` ("all credits at this bank paid duly") pushed *toward*
decline, mean SHAP +0.61 — and the data backs it up: that group defaulted at
**57.1%**, versus **17.1%** for "critical account / other credits existing."

The attribution was exact and the reason was unusable. You cannot mail someone
*"you were denied because you have no delinquencies."*

This forced the central design insight: **SHAP-traceable is necessary but not
sufficient.** A reason can be mathematically perfect and legally uncitable. The
mapper needed a notion of *citable* separate from *true*.

**My eval harness would have reported a hallucination that never happened.**
Case ground truth stored only the top 6 adverse drivers, but the mapper
suppresses 7 features and reaches rank 8 to fill four slots. Four test records
would have scored an honest reason as fabricated.

The worst class of eval bug: not one that misses a failure, but one that
*invents* one. The natural fix — loosening the threshold below 1.0 — would have
destroyed the core guarantee to solve a problem that didn't exist.

**Graceful degradation hid a completely dead module.**
`api.py` degrades to `reasons_status="not_implemented"` when the reason pipeline
won't import. Correct for production. During development, a syntax error made
the mapper unimportable and the API served happily — a dead pipeline looked like
a healthy service.

**Three version pins written from memory were wrong**, including pandas by a
major version. Lockfiles get generated, never hand-written.

**xgboost wouldn't import at all** — the macOS arm64 wheel doesn't bundle the
OpenMP runtime and this machine had no Homebrew. Anaconda's `libomp.dylib`
looked like the fix and was x86_64. scikit-learn's wheel ships an arm64 one.

## Run it

```bash
git clone <repo> && cd credit-decision-explainer
uv venv --python 3.11 && uv pip install -r requirements.txt
cp .env.example .env        # add your GOOGLE_API_KEY
```

```bash
.venv/bin/python src/model.py            # train, prints real AUC
.venv/bin/python src/shap_explainer.py   # worked example, additivity check
.venv/bin/uvicorn api:app --app-dir src --port 8000
```

Or hit the deployed service directly:

```bash
curl -s https://web-production-7b272.up.railway.app/health
```

```bash
curl -s -X POST https://web-production-7b272.up.railway.app/decision -H 'Content-Type: application/json' -d '{
 "checking_status":"A11","duration_months":45,"credit_history":"A30","purpose":"A49",
 "credit_amount":11816,"savings_status":"A61","employment_since":"A75",
 "installment_rate_pct_income":2,"other_debtors":"A101","residence_since_years":4,
 "property_magnitude":"A123","other_installment_plans":"A143","housing":"A151",
 "existing_credits_count":2,"job":"A173","num_dependents":1,"telephone":"A191"}'
```

Returns `DECLINE`, `P(bad)=0.9353`, and four reasons each carrying the feature
and SHAP value that justify it.

## Stack

Python 3.11 · XGBoost 3.2 · SHAP 0.51 · FastAPI · Pydantic v2 · LangGraph ·
Gemini 2.0 Flash · DeepEval 4.1 · Railway

Data: [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
(1,000 applications, 20 attributes, 17 used).
