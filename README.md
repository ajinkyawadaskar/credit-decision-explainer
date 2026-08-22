# Credit Decision Explainer

A loan model that explains its own rejections, plus a checker that catches the AI when it starts making things up.

![Demo](docs/demo.gif)

Live: [web-production-7b272.up.railway.app](https://web-production-7b272.up.railway.app/health) | [How it's wired](docs/architecture.md) | [Things that broke](#things-that-broke)

---

## The problem I wanted to solve

If a bank rejects your loan application, they legally have to tell you why. Specific reasons, not hand-waving. In the US that's ECOA and FCRA, and the letter is called an adverse action notice.

So the obvious build is: take the model's output, hand it to an LLM, ask for a nice letter. I tried a version of that early on and it's genuinely scary how well it fails.

Here's what I mean. The system knew one thing: the applicant asked for a 45-month term. What came back was:

> "your 45-month term, combined with your limited employment history and recent credit inquiries..."

Employment history had nothing to do with this person's rejection. Credit inquiries aren't even in the dataset. The model invented two reasons out of thin air, wrote them in confident banking English, and dropped them into a legal document. Nobody receiving that letter could possibly know.

That's the thing I built this to prevent. Not "make it less likely" — actually prevent.

## How it works

Five steps. The LLM only shows up at step four, and step five checks its homework.

1. Score the applicant with XGBoost. How likely are they to default?
2. Break that score apart with SHAP, so I know which facts pushed it up and by how much.
3. Decide which of those facts I'm legally allowed to cite. This is plain Python. No AI anywhere near it.
4. Hand that short list to Gemini and ask it to write sentences.
5. Check every sentence back against step 3. If the LLM added anything, throw it out and ship the plain version instead.

The whole trick is that the LLM never decides anything. By the time it runs, the decision is made and the reasons are locked. It's a wording layer. If it goes off-script, step 5 catches it.

Most projects I've seen that mix LLMs with explainability let the model generate the reasons directly. That sounds fine until you realise there's nothing left to check the answer against. Splitting "pick the reasons" from "write the reasons" is what makes verification possible at all.

Quick note on SHAP if you haven't used it: the model spits out one number, like "89% chance of default." Useless for a letter. SHAP breaks that number into per-fact contributions — overdrawn checking account pushed it up 0.79, the loan size pushed it up 0.81, and so on. Those contributions are the raw material for the reasons.

## Numbers

### The model

ROC AUC on held-out data is **0.8046**. Across five different train/test splits it's **0.8089 ± 0.0220**, ranging from 0.7725 to 0.8399. PR AUC is 0.6827, against a base rate of 0.300. Accuracy at the default cutoff is 0.7800.

I report the spread because on 1,000 rows a single split is close to meaningless. Reshuffle which rows land in the test set and the score moves by 0.07. If someone tunes this and beats me by 0.01, that's noise, not an improvement.

The number I'd actually worry about: **it only catches 52% of the risky applicants.** Accuracy of 78% looks respectable right up until you notice that 70% of applicants are good anyway, so predicting "approve" for everyone gets you 70%. Fixing the 52% means picking the cutoff based on what a missed default costs versus a wrongly rejected customer, and that's a business call I don't have the inputs for. So the threshold sits in the code as an explicit constant rather than a silently tuned one.

### The explanation layer

| Check | Result |
|---|---|
| SHAP maths verified against model output | 200 / 200 applicants |
| Features with an explicit decision about them | 17 / 17 |
| Features blocked as never citable | 7 |
| Rejections where nobody got a reason | 0 / 46 |
| Reasons citing something that wasn't a factor | **0** |
| Reasons citing something that actually helped them | **0** |
| Same input, same output, every time | yes |

Those two zeros are the entire point.

### End to end, with the LLM in the loop

Ten test cases, five of them built specifically to trip it up, running Gemini 3.6 Flash:

- Every reason legitimate: **10 / 10**
- Checker rejected the LLM: 0 times
- Fell back to plain text: 0 times
- Reasons quietly lost when the LLM merged bullets: **1 of 8 rejections**
- Speed: 13s typical, 34s worst
- Two LLM-judged quality scores: never got them, hit the free tier ceiling

The nastiest test case passed. That one's an applicant whose second-biggest apparent factor is something I deliberately block, and the LLM never tried to smuggle it back in.

### Now the caveats on that 10/10

I don't want to oversell this, because a clean sweep is exactly when you should be suspicious.

**The checker has never actually caught a real lie.** I fed it 12 fake bad inputs by hand and it caught all 12. But across the real runs the LLM behaved itself, so the rejection path has never fired in anger. That tells you the LLM was well-behaved on ten cases. It doesn't tell you the checker works when it matters.

**The checker only knows about things that exist.** It spots the LLM misusing a real feature. It does not spot the LLM inventing something new. I probed it with eight made-up phrases. It caught "employment history", "job type" and "housing situation". It sailed straight past "recent credit inquiries", "bankruptcy filing", "collection account", "debt-to-income ratio" and "payment history" — which, annoyingly, are exactly the phrases an LLM reaches for. This is the biggest hole in the thing and I know it.

**Nothing checks whether reasons go missing.** One case picked four reasons and the LLM wrote three, quietly folding two similar ones together. Everything it printed was true. One just disappeared. Legally, dropping a real reason is arguably worse than adding a fake one, and my traceability metric happily gave it a perfect score, because everything present was fine.

**Two of the three metrics never ran.** Google's free tier is 20 requests. One eval run eats that almost immediately. Both metrics are wired up correctly and fail on billing, not code, which is unsatisfying but true.

That failed run did teach me something though. With the extra metrics running, 5 of 8 cases fell back to plain text. Looks alarming. Looks like the checker rejected most of the LLM's output. It didn't — Google was throttling us, and the fallback logic can't currently tell "safety net caught a lie" apart from "API is rate limited". Same code, same test cases, 0% fallback one run and 62% the next, purely depending on quota. That's a logging gap I'd fix first if I picked this up again.

**And it's too slow.** 34 seconds worst case, essentially all of it waiting on Gemini. The non-AI path runs in about 30 milliseconds. If this were real, you'd return the plain-text notice immediately and improve the wording afterwards.

## Choices I made

**I deleted three columns before training.** The dataset has sex, age and nationality in it. All three are illegal to base a credit decision on. I could have trained on them and just filtered them out of the explanation, but then the model is still deciding on them and I'm only hiding the evidence. So they're gone before the model ever sees them. Cost me nothing measurable — 0.8046 without them.

Worth being clear about the limit though: this stops the model using those facts directly. It does nothing about proxies. Housing type or job category could still be quietly standing in for them. Catching that needs a real fairness audit and I didn't do one.

**I didn't one-hot encode.** Standard practice would turn "checking account status" into four separate binary columns, which then gives me four separate SHAP values I'd have to glue back together before they mean anything. Keeping categories whole means one fact produces one number produces one reason. Accuracy came out the same either way, so this was purely about keeping the explanations clean.

**Plain code picks the reasons, the LLM only words them.** Covered above, but it's the main architectural decision so it belongs in this list.

**I blocked bad mappings instead of forcing them.** "Loan purpose" is what you want the money for. It is not collateral. I could have mapped it to "collateral not sufficient" and it would read perfectly naturally, and it would be a fabricated claim about a security interest that doesn't exist. A wrong-but-believable reason is worse than no reason, because the person reading it has no way to push back. Seven features are blocked outright on that basis.

**When the checker fires, send the boring version.** Retries run out, applicant gets blunt plain text. Worse writing, identical legal accuracy. A credit system that returns nothing because an AI vendor is having a bad day isn't a credit system.

## Things that broke

Full log is in [NOTES.md](NOTES.md). These are the ones that taught me something.

### The model wanted to reject people for paying their bills on time

This looked like a sign error. It wasn't.

Applicants flagged as "all credits paid back properly" were being pushed toward rejection. And the data genuinely supports it — that group defaulted **57.1%** of the time, versus **17.1%** for people with a troubled credit history. It's a known oddity of this dataset, probably something about who ends up applying in the first place.

So the maths was correct and the reason was unusable. You cannot post someone a letter that says "you were denied because you have no missed payments."

That's where the central idea came from. Tracing a reason back to real evidence isn't sufficient. It also has to be a reason you're allowed to give. There's a difference between true and citable, and the system needed to understand it.

### My own tests nearly reported a hallucination that never happened

Each test case stored the applicant's top 6 contributing factors as ground truth. But because seven features are blocked, the reason picker sometimes digs down to the 8th to fill its four slots. Four applicants would have had a completely honest reason flagged as fabricated.

Worst kind of testing bug. Not one that misses a problem — one that manufactures one. And the instinctive fix, relaxing the standard from 100%, would have gutted the core guarantee to solve something that was never broken.

### A safety feature hid a totally dead component

The API is built to survive the reason pipeline being unavailable. It reports the status and keeps serving. Correct behaviour for production.

Then during development a typo made that pipeline unimportable, and the API carried on cheerfully. A completely dead component looked like a healthy service for a while. Failing gracefully is right for users and genuinely dangerous for whoever's building the thing.

### Assorted time sinks

Three package versions I typed from memory were wrong, one of them by an entire major version. Generate your lockfile, don't write it.

Every Gemini model name I tried had been retired. Two 404s before I found a live one. Google's own "list available models" endpoint cheerfully listed both of the dead ones, so the only thing that actually tells you the truth is making a real call.

XGBoost wouldn't even import. It needs a maths runtime that the Mac wheel doesn't bundle, and this machine has no package manager to go fetch it. There was a copy sitting right there in Anaconda, built for the wrong chip. Ended up borrowing the correct one out of scikit-learn's install, which does ship it.

## Try it

Against the live one:

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

Comes back DECLINE, 93.53% risk, four reasons, each carrying the exact feature and number behind it.

Locally:

```bash
git clone https://github.com/ajinkyawadaskar/credit-decision-explainer
cd credit-decision-explainer
uv venv --python 3.11 && uv pip install -r requirements.txt
cp .env.example .env        # drop your Google API key in
```

```bash
.venv/bin/python src/model.py            # trains, prints the real numbers
.venv/bin/python src/shap_explainer.py   # one worked example
.venv/bin/streamlit run app.py           # the demo in the gif above
```

## A note on how I built this

I used AI assistance for the scaffolding — data loading, model training, the
API layer, the Streamlit page, deploy config. I wrote the four files that make
the actual decisions by hand, on purpose: the reason mapper, the validator, the
graph wiring, and the traceability metric.

That split was deliberate and I set it before I started. Every judgment call in
this thing lives in those four files, and I wanted to be the one who made them.
More on that in [docs/architecture.md](docs/architecture.md).

## Stack

Python 3.11, XGBoost, SHAP, FastAPI, LangGraph, Gemini 3.6 Flash, DeepEval, Streamlit, Railway.

Data is [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — 1,000 real applications, 20 facts each, 17 of which I use.
