# CLAUDE.md — Credit Decision Explainer Agent

## What this is
XGBoost credit model + SHAP attribution + LangGraph agent that generates
FCRA/ECOA-compliant adverse action reasons, with a deterministic validator
that rejects any generated reason not traceable to a real SHAP feature.

## Owner
Ajinkya Wadaskar. Portfolio project P1. **Shipping TODAY.**

---

## HARD SCOPE LOCK — do not exceed

**IN:**
- Binary credit decision (approve / decline)
- SHAP local attribution per record
- LangGraph agent: SHAP values → ECOA-style adverse action reasons
- Deterministic traceability validator
- FastAPI: `POST /decision`, `GET /health`
- DeepEval harness: 20 labeled cases, 5 of them adversarial
- Minimal Streamlit UI (30 min hard cap)
- Deployed public URL

**OUT:**
- User auth, database, multi-model comparison, fine-tuning, RAG,
  chat interface, model monitoring, multi-tenancy, Docker (cut for
  today's timeline), CI/CD beyond a single eval job, and anything
  else not explicitly listed under IN.

If you think of a good addition: **DO NOT BUILD IT.** Add it to
`IDEAS.md` and keep going.

---

## Division of labor — CRITICAL

I must be able to defend this code in a technical interview. This
constraint overrides speed, including on a same-day ship.

**YOU WRITE:**
- Data loading, cleaning, train/test split
- XGBoost training
- SHAP explainer setup
- FastAPI scaffolding and Pydantic models
- Streamlit UI
- Deploy config
- Test boilerplate
- README prose (drawn from my NOTES.md)
- Repo hygiene, .gitignore, .env.example

**I WRITE.** Leave a stub with a clear TODO and a docstring spec of
what the module must do, then **STOP and tell me**:

1. `src/reason_mapper.py` — SHAP values → ECOA reason codes
2. `src/validator.py` — SHAP-traceability enforcement
3. `src/graph.py` — LangGraph node wiring
4. `evals/metrics.py` — custom DeepEval metric definitions

**Never fill in these four.** If I ask you to — including if I ask
because we're running late — remind me why and refuse. A project I
can't defend under follow-up questions is worth less than no project.

---

## Rules

- Build the eval set **before** the generation logic.
- No invented numbers anywhere. Use `___` placeholders until measured.
- 30-minute cap on UI. 2-hour cap on deployment — if you exceed it,
  switch platforms rather than debugging.
- Every generated reason must trace to a real SHAP feature. That
  constraint is the entire point of the project.
- Commit after each working milestone with a real message.
- Log every dead end and surprise to `NOTES.md` **as it happens** —
  that file becomes the "What Didn't Work" section, and I'll forget
  the good failures if we wait until the end.

---

## Working mode

- Two-line preamble before each step, then execute.
- After every step, run the code and show me **actual output**, never
  a description of what it should produce.
- Track elapsed time against the schedule and **tell me when we're
  slipping.** Do not quietly absorb delay.
- While I'm writing my four modules, work on whatever is unblocked —
  eval scaffolding, FastAPI skeleton, README structure. Don't idle.

---

## Today's schedule

| Elapsed | What |
|---|---|
| 0:00 | Repo skeleton, data loaded, XGBoost trained, AUC reported |
| 0:45 | SHAP working — top-4 negative contributors on one worked example |
| 1:30 | **STOP** → hand me `reason_mapper.py` stub + spec |
| 3:00 | **STOP** → hand me `validator.py` stub + spec |
| 4:00 | **STOP** → hand me `graph.py` stub + spec |
| 5:00 | Generate 10 candidate eval cases for my review (I label 20 total) |
| 6:00 | FastAPI + Pydantic + error handling |
| 6:30 | I run evals; we record real numbers |
| 7:00 | Deploy, confirm `/health` |
| 8:00 | Streamlit UI |
| 8:30 | Architecture diagram |
| 9:15 | README, ship checklist |

**Fallback if behind at hour 5:** cut the Streamlit UI entirely and
ship the API with a `curl` example in the README. A live API with real
eval numbers beats a pretty UI without them.

---

## Definition of Done — all 8 required

1. Live URL responding
2. Public repo, secrets scrubbed, `.env.example` committed
3. 30-second demo GIF at the **top** of the README
4. Architecture diagram
5. "Decisions & Tradeoffs" section — 3–5 real choices and what was rejected
6. Measured eval numbers published (no placeholders remaining)
7. "What Didn't Work" section
8. I can explain the whole thing for 20 minutes with no notes
