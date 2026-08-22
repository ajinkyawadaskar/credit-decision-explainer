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
## Eval scaffolding — DeepEval/Gemini risk retired

Flagged in planning that DeepEval defaults to OpenAI for LLM-judged metrics
while we chose Gemini. Checked it early instead of at eval time: deepeval
4.1.10 ships a **native `deepeval.models.GeminiModel`**, so no `DeepEvalBaseLLM`
wrapper is needed. The remaining trap is that the default is still OpenAI —
every built-in metric must be passed `model=judge` explicitly or it will fail
on a missing OPENAI_API_KEY. Written into the metrics.py spec so it can't be
forgotten.

Versions pinned in this env: deepeval 4.1.10, langgraph 1.2.11,
langchain-google-genai 4.3.5, google-genai 2.19.0.

## Eval cases — two of my own generated cases were wrong, caught before labeling

Generated 10 candidates from real test records, then checked whether each case
actually tests what its name claims. Two did not:

1. **ADV-02 "insufficient_drivers" was measuring nothing.** I selected the
   declined applicant with the fewest positive SHAP drivers — but the minimum
   across all declines was **8**, because nearly every feature nudges the score
   a little. A case meant to test "does the agent pad to four reasons when the
   record only supports two" was handing it eight. Fixed by counting drivers
   above a 0.10 materiality floor instead; the new record has **3**, which
   actually creates the squeeze.

2. **ADV-03 duplicated STD-01.** Both selected test row 96 (P(bad)=0.9353) —
   the citability trap picked the highest-probability qualifying record, which
   was already the highest-probability record overall. 10 cases, 9 unique
   records. Fixed by excluding already-used rows; ADV-03 is now row 61,
   P(bad)=0.8778, `credit_history=A31` with SHAP **+0.8236** as the #2 driver.
   That is a much sharper trap than the original anyway.

Lesson worth keeping: an eval case named after a failure mode does not
necessarily *exercise* that failure mode. Both defects would have silently
inflated the final numbers.
## FastAPI — graceful degradation instead of a hard dependency

`reason_mapper.py` and `graph.py` are Ajinkya's and don't exist yet, but the
API needed to be demoable and deployable today. Rather than block, `/decision`
returns the decision plus real SHAP drivers and reports
`reasons_status="not_implemented"`; `/health` surfaces the same fact. The stub
raises `NotImplementedError` at import time by design, so the loader is just a
try/except — when the module lands, it starts working with no other change.

## FastAPI — prohibited-basis enforcement moved into the type system

`ApplicantRequest` sets `extra="forbid"` and simply has no `age_years`,
`personal_status_sex`, or `foreign_worker` field. Submitting one returns 422
`extra_forbidden`. So the ECOA constraint from step 1 is now enforced in three
independent places: dropped at training, absent from the request schema, and
asserted in the reason mapper spec. Defense in depth, and it demos well.

## FastAPI — scare that wasn't a bug

Hand-typed a payload "for test row 96" and got P(bad)=0.8051 against the batch
path's 0.9353. Looked like a serving/training skew bug — the classic one where
the API rebuilds dtypes differently than training did. It wasn't: I had
invented four field values (purpose, employment_since, housing,
existing_credits_count), so it was a different applicant.

Pulled the real row and re-ran: API and batch agree to 4 decimals on P(bad) and
produce an identical top-4 driver list. Replaced the OpenAPI example with the
verbatim record so the docs example is reproducible rather than composed.

Real lesson: I nearly wrote "verified API matches batch" on the strength of a
payload I typed from memory. Serving/training skew is a genuine risk with
XGBoost native categoricals, so the check was worth doing properly — the
dtype-restoration loop in `api.py::decision` is what makes it pass.
## Deploy prep — the model artifact was gitignored

`.gitignore` had `models/*.json`, which is the normal instinct (don't commit
build output). But Railway builds from the repo and there is no training step
at deploy time, so the container would have booted with no model and crashed on
startup. Caught by listing what git actually tracks before deploying rather
than after. The model is 326KB — now deliberately tracked, with a comment in
.gitignore explaining why so nobody "fixes" it later.

## Deploy prep — three of my version pins were wrong

Hand-wrote requirements.txt from memory and got pandas (guessed 2.3.4, actually
**3.0.5**), numpy (2.3.5 vs 2.4.6), and python-dotenv wrong. A pandas major
version is not a small miss. Regenerated the file programmatically from
`importlib.metadata` so the pins are the versions actually tested, not the
versions I believed were installed. Never hand-write a lockfile.

## Deploy prep — clean-room test, and what it does NOT prove

Copied only git-tracked files to a scratch dir, built a fresh venv from
requirements.txt alone, and booted with railway.json's exact startCommand.
Result: `/health` 200, and `/decision` returned **P(bad)=0.9353** with the same
four drivers in the same order as the dev environment. So there is no hidden
dependency on the dev venv.

Two honest caveats:
1. The clean room is still macOS arm64 on the same uv interpreter, which has
   the hand-copied libomp from step 1. It therefore does **not** validate the
   libomp story. Railway runs Linux, where the manylinux xgboost wheel links
   libgomp normally — that is reasoning, not evidence, until the deploy runs.
2. First attempt "failed" with an empty log and I nearly went debugging the
   app. The app was fine; my test harness backgrounded uvicorn in a subshell
   that never wrote its log. Worth remembering that a failed smoke test can be
   the harness, and the fix is to run it in the foreground and read the error
   before theorising.
## reason_mapper.py landed (Ajinkya) — and the near-loss that argues for committing early

Module complete: 17/17 feature coverage, 7 features suppressed as having no
lawful Reg B reason, per-driver suppression for the A31/A32 "paid on time"
case, materiality floor 0.05, max 4 reasons.

Verified across the full 200-row test set:
  - 46 declines, reason-count distribution {3: 1, 4: 45}, zero declines
    produced zero reasons
  - deterministic across repeat calls
  - **0 reasons cited a feature that wasn't in that applicant's SHAP
    explanation or that pushed toward approval** — the core invariant

Three review rounds found: (1) three feature names that didn't exist in the
data, silently suppressing R08/R12 rather than crashing, because is_citable()
looked up with .get(); (2) `purpose` mapped to "collateral not sufficient" and
`other_debtors` to "excessive obligations" — both fabricated bases, since a
stated loan purpose is not a security interest; (3) a hard `_MAPPING_TABLE[...]`
index that would KeyError if it ever disagreed with the .get() in is_citable().

Ajinkya's fix for (2) was better than the suggested remap: suppress outright,
on the reasoning that mapping purpose to a collateral code "would invent a
security-interest claim the feature doesn't support." A wrong-but-plausible
reason code is worse than none, because the applicant cannot falsify it.

**The near-loss:** mid-review an edit landed that fixed one bug while reverting
another and introducing a literal tab into a space-indented file. The module
stopped parsing entirely. Because `api.py` degrades gracefully on
NotImplementedError/ImportError, a dead reason_mapper would have shown up as
`reasons_status="not_implemented"` rather than a crash — silent. The revert was
only recoverable because the working version had been reviewed in-session; git
had nothing but the original stub.

Lesson: graceful degradation is right for deploys and dangerous during
development, and "commit after each working milestone" exists precisely so a
stale-copy overwrite costs minutes instead of an afternoon.
## metrics.py landed — and my eval harness was silently framing it

TraceabilityMetric is deterministic set membership, threshold 1.0 (any
untraceable reason fails the case), and it separates two distinct failure
modes in `self.reason`: **phantom** features (named but absent from the
explanation entirely) vs **wrong-direction** citations (feature exists but
argued toward approval). That distinction matters — citing a feature that
helped the applicant as a denial reason is arguably worse than naming one that
doesn't exist, because it's a lie about real evidence.

It also handles the trap flagged in the spec: zero reasons on a DECLINE scores
**0.0, not a vacuous 1.0**. An agent that says nothing must not win.

Verified: 10/10 honest cases score 1.00; all five deliberately-bad inputs
(phantom feature, prohibited-basis feature, wrong-direction, empty-on-decline,
mixed) correctly fail.

### The bug this surfaced was mine

`evals/dataset.py` stored `top_decline_drivers(6)` as each case's ground truth.
But reason_mapper suppresses 7 of 17 features, so to fill four slots it
routinely reaches deeper — **rank 8** at worst. Four test records
(14, 133, 154, 168) would have had a perfectly honest reason scored as a
hallucination.

None of the 10 current cases happened to hit it, so the harness looked green.
It would have started lying the moment we labeled more cases.

This is the worst class of eval bug: not a metric that misses a failure, but a
metric that **invents one**. A false hallucination report makes you distrust a
system that is actually working, and the natural response — loosening the
threshold below 1.0 — would have destroyed the project's core guarantee to fix
a problem that never existed.

Fixed by storing all adverse drivers rather than a top-N slice. Re-verified:
0 false positives across all 46 declines in the test set.
## Streamlit UI — built inside the 30-minute cap

Deliberately shows the mechanism rather than just the verdict: each reason is
rendered next to the SHAP feature and value it traces to, and an expander lists
the adverse drivers that were **not** cited with the reason why. The suppression
rule is the interesting part of the system, so hiding it would waste the demo.

Two things surfaced while driving it in a real browser:

1. **The Score button was below 17 form inputs.** Functionally fine, terrible
   for a 30-second GIF — you'd scroll the entire form before anything happened.
   Moved it directly under the record picker so the demo is two clicks. Only
   noticed by actually using the UI; reading the code would never have shown it.

2. **Verified on test row 14** — the record that would have been a false
   hallucination before the eval ground-truth fix. It cites
   `installment_rate_pct_income` at SHAP +0.0871, a driver ranked 7th by
   magnitude. Good end-to-end confirmation that the fix holds in the real path.

Arithmetic check visible in the UI: base -0.848 + Σshap +1.264 = +0.416
log-odds -> sigmoid -> 60.3%, matching the displayed P(bad).
