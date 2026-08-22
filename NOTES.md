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

