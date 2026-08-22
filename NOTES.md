# NOTES — dead ends, surprises, and decisions as they happened

Raw log. Becomes the "What Didn't Work" section of the README.

## 0:00 — Environment

- **Only Python 3.9 on the machine** (anaconda 3.9.12, system 3.9.6). Modern
  langgraph/deepeval want 3.10+. Rather than pin old releases and risk a wall
  at hour 4, installed `uv` and pinned Python 3.11.16. Cost ~4 min.
- `gh` CLI not installed — public repo (DoD #2) needs it or the browser.
- `railway` CLI not installed — `npm i -g @railway/cli` needed before deploy.
- Git identity was unset globally; would have produced misattributed commits.
- **DeepEval's built-in metrics default to OpenAI.** We chose Gemini, so the two
  built-ins need DeepEval's `GeminiModel` or a `DeepEvalBaseLLM` wrapper.
  Caught during planning rather than at 6:30.
- Found two identical copies of this project (one under `Documents/.../Portfolio`
  with a trailing space in the directory name). Working copy is
  `~/Projects/Portfolio/credit-decision-explainer`; the other is stale.

## 0:00 — DEAD END: xgboost wouldn't import (libomp missing)

`import xgboost` died with `Library not loaded: @rpath/libomp.dylib`. The
macOS arm64 xgboost 3.2.0 wheel does **not** bundle the OpenMP runtime, and
this machine has no Homebrew (`brew not found`, no `/opt/homebrew`), so the
standard `brew install libomp` fix was unavailable.

Anaconda had a `libomp.dylib` — but `file` showed it was **x86_64**, useless
for an arm64 interpreter. That's the trap: it looks like a fix and isn't.

Fix that worked: scikit-learn's macOS wheel bundles its own **arm64**
`libomp.dylib` at `sklearn/.dylibs/libomp.dylib`. xgboost's rpath already
searches the interpreter's `lib/` directory, so copying it there resolved the
load with no Homebrew and no sudo. Cost ~6 min.

**This is a local-dev-only fix.** Railway builds on Linux, where the manylinux
xgboost wheel links `libgomp` normally — so this must NOT become a deploy step.

## 0:00 — DECISION: dropped 3 features as ECOA prohibited bases

German Credit ships `personal_status_sex`, `foreign_worker`, and `age_years`.
All three are prohibited bases under ECOA / Reg B (12 CFR 1002.6). For a
project whose entire premise is generating *lawful* adverse action reasons,
training on them is indefensible — a SHAP attribution pointing at "sex" is a
reason we could never cite to an applicant.

Dropped all three. Model trains on 17 features instead of 20. Cost in accuracy
appears to be roughly nil (see AUC below), which is itself the interesting
finding for the Tradeoffs section.

## 0:00 — Model baseline (measured, seed=42, stratified 800/200)

- ROC AUC **0.8046**, PR AUC 0.6827 (base rate 0.300), accuracy 0.7800 @ 0.5
- Across 5 seeds: AUC **0.8089 +/- 0.0220**, range [0.7725, 0.8399]
- Recall on the decline class is only **0.52** at threshold 0.5. On 1000 rows
  that spread is wide; the threshold is a live decision, not a default.

## 0:45 — SHAP working, additivity verified

Used XGBoost's built-in `pred_contribs=True` rather than `shap.TreeExplainer`.
Same exact TreeSHAP algorithm, but it consumes pandas `category` dtypes
natively — the generic explainer path does not reliably handle the native
categorical features we deliberately chose in step 1. Avoided a likely dead end
rather than hitting one.

Added `verify_additivity()`: checks `base_value + sum(SHAP) == model log-odds`
for every row. **PASS on all 200 test rows.** This is SHAP's core mathematical
guarantee and it is the foundation the traceability validator stands on — if
additivity fails, every generated reason is built on sand. Worth asserting in
CI rather than trusting.

Worked example — test row 96, P(bad) = 0.9353, DECLINE:

| feature | value | SHAP |
|---|---|---|
| credit_amount | 11816 | +0.8089 |
| checking_status | < 0 DM | +0.7904 |
| credit_history | no credits taken / all paid duly | +0.5205 |
| duration_months | 45 | +0.5082 |

base -0.8483 + sum 3.5190 = 2.6707 log-odds -> sigmoid -> 0.9353. Checks out.

## 0:45 — SURPRISE: the credit_history encoding is inverted, and it's a compliance problem

`credit_history = A30` ("no credits taken / all paid duly") pushed this
applicant *toward decline*. That looked like a bug. It isn't — the model is
faithfully learning the data:

| code | meaning | n | actual bad rate | mean SHAP |
|---|---|---|---|---|
| A30 | no credits taken / all paid duly | 40 | **62.5%** | +0.6621 |
| A31 | all credits at this bank paid duly | 49 | **57.1%** | +0.6111 |
| A32 | existing credits paid duly till now | 530 | 31.9% | +0.1198 |
| A33 | delay in paying off in the past | 88 | 31.8% | +0.0613 |
| A34 | critical account / other credits existing | 293 | **17.1%** | -0.6350 |

Two separate things here:

**A30 is defensible.** "No credit history" genuinely is high risk — the thin-file
problem. And it maps cleanly to a real, lawful ECOA reason code: *"insufficient
credit file"* / *"limited credit experience"*. No issue.

**A31 is NOT defensible as a reason.** The model wants to decline people because
they *paid all their credits at this bank on time*. Handing that applicant an
adverse action notice reading "you were denied because you have no delinquencies"
is absurd on its face and would not survive regulatory review — even though it is
a *truthful* description of what the model did.

Also note A34 ("critical account") is the **lowest**-risk group at 17.1%. This is
a known quirk of the German Credit coding, most likely survivorship/selection in
how the sample was drawn.

**Implication for `reason_mapper.py` (Ajinkya's module):** SHAP traceability is
necessary but NOT sufficient. A reason can be perfectly traceable to a real SHAP
feature and still be an unusable adverse action reason. The mapper needs a notion
of which (feature, direction) pairs are *citable*, separate from which are merely
*true*. Flagged before the module was written, not after.
