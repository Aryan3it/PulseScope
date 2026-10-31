"""End-to-end orchestration: `python -m pulsescope.pipeline [stage ...]`."""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import pandas as pd

from . import config as C

STAGES = ["simulate", "etl", "analytics", "segment", "churn", "experiments", "rca", "rvalidate", "nosql",
          "report"]


def _segments():
    p = C.OUTPUTS / "segments.parquet"
    return pd.read_parquet(p) if p.exists() else None


def r_crosscheck() -> dict:
    subprocess.run(["Rscript", str(C.ROOT / "r" / "ab_validation.R"), str(C.OUTPUTS)], check=True)
    r = json.loads((C.OUTPUTS / "r_validation.json").read_text())
    py = json.loads((C.OUTPUTS / "experiments.json").read_text())
    rows = []
    for exp in ("one_click_checkout", "reco_widget_v2"):
        pairs = {
            "srm_chi2": (py[exp]["srm"]["chi2"], r[exp]["srm"]["chi2"]),
            "conversion_diff": (py[exp]["conversion"]["diff"], r[exp]["conversion"]["diff"]),
            "conversion_p": (py[exp]["conversion"]["p_value"], r[exp]["conversion"]["p_value"]),
            "revenue_diff_welch": (py[exp]["revenue_per_user"]["raw"]["diff"], r[exp]["revenue_welch"]["diff"]),
            "revenue_p_welch": (py[exp]["revenue_per_user"]["raw"]["p_value"], r[exp]["revenue_welch"]["p_value"]),
            "revenue_diff_cuped_vs_ancova": (py[exp]["revenue_per_user"]["cuped"]["diff"],
                                             r[exp]["revenue_ancova"]["estimate"]),
        }
        for k, (a, b) in pairs.items():
            rows.append({"experiment": exp, "quantity": k, "python": a, "r": b,
                         "abs_diff": abs(a - b), "rel_diff": abs(a - b) / max(abs(a), 1e-12)})
    df = pd.DataFrame(rows)
    df.to_csv(C.OUTPUTS / "python_vs_r.csv", index=False)
    print(df.to_string(index=False))
    return {"max_rel_diff": float(df.rel_diff.max()), "n_checks": len(df)}


def main(stages: list[str], users: int, automl_seconds: int):
    timings = {}
    for s in stages:
        t0 = time.time()
        print(f"\n=== {s} ===", flush=True)
        if s == "simulate":
            from . import simulate
            simulate.main(users)
        elif s == "etl":
            from . import etl
            etl.run()
        elif s == "analytics":
            from . import analytics
            analytics.run()
        elif s == "segment":
            from . import segmentation
            segmentation.run()
        elif s == "churn":
            from . import churn
            churn.run(_segments(), automl_seconds)
        elif s == "experiments":
            from . import experiments
            experiments.run(_segments())
        elif s == "rca":
            from . import rca
            print(json.dumps(rca.run()["evaluation"], indent=2))
        elif s == "rvalidate":
            r_crosscheck()
        elif s == "nosql":
            from . import nosql
            nosql.run()
        elif s == "report":
            from . import report
            report.build()
        timings[s] = round(time.time() - t0, 1)
    prev = C.OUTPUTS / "timings.json"
    allt = json.loads(prev.read_text()) if prev.exists() else {}
    allt.update(timings)
    prev.write_text(json.dumps(allt, indent=2))
    print("\nstage timings (s):", timings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=STAGES)
    ap.add_argument("--users", type=int, default=100_000)
    ap.add_argument("--automl-seconds", type=int, default=180)
    a = ap.parse_args()
    main(a.stages, a.users, a.automl_seconds)
