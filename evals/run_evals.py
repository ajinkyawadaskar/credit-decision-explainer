"""Run the graph over evals/cases.json and report measured numbers.

Reports the deterministic TraceabilityMetric on every case. The Gemini-judged
built-ins are opt-in via --judge because they cost API calls per case.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from deepeval.test_case import LLMTestCase  # noqa: E402

from data import load_clean, split  # noqa: E402
from metrics import TraceabilityMetric  # noqa: E402
import graph as G  # noqa: E402


def main(judge: bool):
    X, y = load_clean()
    _, X_test, _, _ = split(X, y)
    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    app = G.build_graph()
    trace = TraceabilityMetric()

    rows, lat = [], []
    for c in cases:
        row = X_test.iloc[[c["test_row_index"]]]
        t0 = time.time()
        try:
            out = app.invoke({"applicant": row, "attempts": 0, "used_fallback": False})
            err = None
        except Exception as e:
            out, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        lat.append(dt)

        reasons = out.get("reasons", []) or []
        v = out.get("validation")
        tc = LLMTestCase(
            input=c["case_id"],
            actual_output=out.get("final_text") or "",
            retrieval_context=[f"{d['feature']}: {d['shap_value']}"
                               for d in c["top_decline_drivers"]],
            metadata={
                "shap_features_available": c["shap_features_available"],
                "top_decline_drivers": c["top_decline_drivers"],
                "decision": c["decision"],
                "generated_reasons": [{"feature": r.feature, "shap_value": r.shap_value}
                                      for r in reasons],
            },
        )
        score = trace.measure(tc)
        rows.append({
            "case_id": c["case_id"],
            "type": c["adversarial_type"] or "standard",
            "decision": c["decision"],
            "n_reasons": len(reasons),
            "attempts": out.get("attempts", 0),
            "fallback": bool(out.get("used_fallback")),
            "validation_passed": getattr(v, "passed", None),
            "violations": list(getattr(v, "violations", []) or []),
            "traceability": score,
            "latency_s": round(dt, 2),
            "error": err,
            "text": (out.get("final_text") or "")[:400],
        })

    hdr = f"{'case':<8}{'type':<21}{'dec':<9}{'n':>2} {'try':>4} {'fb':>6} {'valid':>6} {'trace':>7} {'sec':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['case_id']:<8}{r['type']:<21}{r['decision']:<9}{r['n_reasons']:>2} "
              f"{r['attempts']:>4} {str(r['fallback']):>6} {str(r['validation_passed']):>6} "
              f"{r['traceability']:>7.2f} {r['latency_s']:>6.2f}")
        if r["error"]:
            print(f"         ERROR {r['error'][:100]}")

    n = len(rows)
    fb = sum(r["fallback"] for r in rows)
    passed = sum(1 for r in rows if r["validation_passed"] is True)
    rejected = sum(1 for r in rows if r["validation_passed"] is False)
    perfect = sum(1 for r in rows if r["traceability"] == 1.0)
    print(f"\ncases                     {n}")
    print(f"traceability == 1.00      {perfect}/{n}  ({perfect/n:.0%})")
    print(f"mean traceability         {statistics.mean(r['traceability'] for r in rows):.3f}")
    print(f"validator passed          {passed}/{n}")
    print(f"validator rejected        {rejected}/{n}")
    print(f"fell back to deterministic {fb}/{n}  ({fb/n:.0%})")
    print(f"latency p50 / p95         {statistics.median(lat):.2f}s / "
          f"{sorted(lat)[max(0,int(0.95*len(lat))-1)]:.2f}s")

    viol = [v for r in rows for v in r["violations"]]
    if viol:
        print(f"\nvalidator violations ({len(viol)}):")
        for v in viol[:12]:
            print(f"  - {v[:120]}")

    (ROOT / "evals" / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote evals/results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run Gemini-judged built-ins")
    main(ap.parse_args().judge)
