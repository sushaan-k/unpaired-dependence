#!/usr/bin/env python3
"""One summary of the saving over paired-only estimation across the test data sets (descriptive, computed after
every test had been scored).

Test data sets: the external test (OverCITE-seq; its prespecified target, a recovered fraction of 0.3) and the nine
data sets of the generality benchmark (their primary target, half of the best recovery any method reached; a data
set in which no method recovered dependence is not informative). Each data set enters with its primary saving and
its own bootstrap distribution; the geometric mean over informative data sets has an interval that combines the
r-th bootstrap resample of every data set, as hypothesis G1 of the benchmark did. The external test's bootstrap
resamples are regenerated with the frozen scoring code and seed (run_external.curves, BOOT_SEED), and the
regenerated interval is checked against the stored one.

    python savings_summary.py      # writes results/savings_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(EXT / "semipaired" / "external"))
sys.path.insert(1, str(EXT / "semipaired"))
import run_external as rx  # noqa: E402

GEN = EXT / "generality" / "results"
LABELS = {"hao": "Blood, 8 donors", "stephenson": "Blood, COVID-19", "bmmc_cite": "Bone marrow",
          "colon": "Colon", "bmmc_multiome": "Bone marrow multiome", "scala_m1": "Motor cortex Patch-seq",
          "gouwens_visp": "Visual cortex Patch-seq", "banc": "Fly CNS, female", "malecns": "Fly CNS, male"}


def external_draws():
    """Stored factor and interval of the external test, and its 2,000 bootstrap log-savings regenerated with the
    frozen scoring code (paired-only envelope over the plan's arm, bjs2/mapped, at the prespecified target)."""
    info = json.loads((rx.HERE / "results/predict_info.json").read_text())
    parts = [rx.od.load(p, info["genes"]) for p in ("test_adaptation", "test_scoring")]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "library", "y", "part", "target", "condition")}
    pop = rx.pm.keys_of(cat)
    stats = rx.tg.group_stats(cat["counts"], cat["library"], cat["y"], pop, cat["part"])
    z = np.load(rx.HERE / "results/predictions.npz")
    preds = {k: z[k] for k in z.files}
    mean_pred, msq = {}, {}
    for key, C in preds.items():
        arm = "/".join(key.split("/")[:2])
        B = key.split("/")[2]
        c = key.split("/")[4]
        mean_pred.setdefault(arm, {}).setdefault(B, {}).setdefault(c, []).append(C)
    for arm in mean_pred:
        msq[arm] = {}
        for B in mean_pred[arm]:
            msq[arm][B] = {c: float(np.mean([np.sum(C * C) for C in v])) for c, v in mean_pred[arm][B].items()}
            mean_pred[arm][B] = {c: np.mean(v, 0) for c, v in mean_pred[arm][B].items()}
    budget_arms = {a: d for a, d in mean_pred.items() if "0" not in d}
    orfs = sorted({r["target"] for r in stats})
    rng = np.random.default_rng(rx.BOOT_SEED)
    member = {t: [i for i, r in enumerate(stats) if r["target"] == t] for t in orfs}
    arms = [a for a in rx.PAIRED_ONLY if a in budget_arms]
    logs = []
    for _ in range(rx.BOOT):
        w = np.zeros(len(stats))
        for t in rng.choice(orfs, len(orfs), replace=True):
            w[member[t]] += 1
        rf = rx.curves(stats, w, budget_arms, msq)[0]
        logs.append(np.log(rx.needed(rx.envelope(rf, arms), rx.TARGET)[0] / rx.needed(rf["bjs2/mapped"], rx.TARGET)[0]))
    stored = json.loads((rx.HERE / "results/overcite_plan_estimator.json").read_text())["savings"]["paired_only"]["0.3"]
    ci = np.quantile(np.exp(logs), [.025, .975])
    assert np.allclose(ci, stored["ci"], rtol=1e-9), (ci, stored["ci"])
    return {"factor": stored["factor"], "ci": stored["ci"], "comparator_relation": stored["comparator_relation"],
            "proposed_relation": stored["proposed_relation"], "log_boot": list(map(float, logs))}


def main():
    summ = json.loads((GEN / "summary_amend1.json").read_text())
    informative = summ["informative"]
    rows = []
    ext = external_draws()
    rows.append({"dataset": "overcite", "label": "T cells, ORF screen (external test)", "modality": "RNA-protein",
                 "informative": True, "target": "0.3 (prespecified)", **ext})
    modality = {"hao": "RNA-protein", "stephenson": "RNA-protein", "bmmc_cite": "RNA-protein", "colon": "RNA-protein",
                "bmmc_multiome": "RNA-chromatin", "scala_m1": "RNA-electrophysiology",
                "gouwens_visp": "RNA-electrophysiology", "banc": "brain-nerve cord wiring",
                "malecns": "brain-nerve cord wiring"}
    for n in LABELS:
        a = json.loads((GEN / f"{n}_amend1.json").read_text())
        s = a["savings"]["paired_only"]["0.5"]
        rows.append({"dataset": n, "label": LABELS[n], "modality": modality[n], "informative": n in informative,
                     "target": "half of the best recovery", "factor": s["factor"], "ci": s["ci"],
                     "comparator_relation": s["comparator_relation"], "proposed_relation": s["proposed_relation"],
                     "log_boot": s["log_boot"]})
    inf = [r for r in rows if r["informative"]]
    L = np.array([r["log_boot"] for r in inf])
    gm = float(np.exp(np.mean([np.log(r["factor"]) for r in inf])))
    lo, hi = np.exp(np.quantile(L.mean(0), [.025, .975]))
    out = {"note": "descriptive summary after all tests were scored; each data set at its primary target",
           "n_test": len(rows), "n_informative": len(inf),
           "geometric_mean": gm, "ci": [float(lo), float(hi)],
           "min": float(min(r["factor"] for r in inf)), "max": float(max(r["factor"] for r in inf)),
           "n_lower_bound_above_one": int(sum(r["ci"][0] > 1 for r in inf)),
           "rows": [{k: v for k, v in r.items() if k != "log_boot"} for r in rows]}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "savings_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in out["rows"]:
        print(f"{r['label']:38s} {r['factor']:.2f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f}) "
              f"{r['comparator_relation']}{r['proposed_relation']} informative={r['informative']}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
