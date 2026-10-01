#!/usr/bin/env python3
"""Post hoc (added after unsealing, in response to review): is the failure score more than channel identity?

Reads result files only. Three analyses on the 1,476 bone-marrow pairs of
analysis and channel:

1. Channel identity as a predictor: the development failure rate of the channel's
   family (frozen, P1, learning-curve), or of family x analysis kind (all cells
   versus within types), fixed from the development pairs; and, as an upper
   bound, the same rates computed in bone marrow itself (in-sample).
2. Discrimination within channel families: AUC of the frozen score within each
   family, and within the P1 within-type pairs that dominate the high-coverage
   failures.
3. Leave-one-family-out: the failure model refitted on development pairs without
   one family (same predictors and ridge), then applied to bone marrow.

Intervals: 2,000 bootstrap resamples of donors. Writes results/posthoc_diagnostic.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fit_failure_model import MODELS, RIDGE, features
from stats import auc, cluster_bootstrap, logistic_fit, logistic_score

HERE = Path(__file__).resolve().parent
BOOT, SEED = 2000, 20261014
HIGH = 0.4


def family(name):
    return name if name in ("frozen", "P1", "colon") else name.rstrip("0123456789")


def kind(analysis):
    return "total" if analysis == "total" else "within"


def score(model, rows):
    X = np.array([[features(r, c) for c in model["predictors"]] for r in rows])
    return logistic_score((model["intercept"], np.array(model["slopes"]), np.array(model["mean"]),
                           np.array(model["sd"])), X)


def fit(rows, name):
    cols = MODELS[name]
    X = np.array([[features(r, c) for c in cols] for r in rows])
    y = np.array([r["E_signal"] >= 1 for r in rows])
    b0, b, mu, sd = logistic_fit(X, y, ridge=RIDGE)
    return {"predictors": cols, "intercept": b0, "slopes": b.tolist(), "mean": mu.tolist(), "sd": sd.tolist()}


def interval(y, s, clusters, seed, other=None):
    if other is None:
        f = lambda i: auc(y[i], s[i])
    else:
        f = lambda i: auc(y[i], s[i]) - auc(y[i], other[i])
    b = cluster_bootstrap(clusters, f, BOOT, seed)
    b = b[np.isfinite(b)]
    point = auc(y, s) - (0 if other is None else auc(y, other))
    return {"value": float(point), "ci": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]}


def main():
    dev = json.loads((HERE / "results/dev_units2.json").read_text())
    tc = json.loads((HERE / "results/dev_typecheck.json").read_text())
    drows = [r for r in dev["pairs"] if r["dim_S"] > 0]
    for r in drows:
        r["type_agreement"] = tc[f'{r["cohort"]}/{r["donor"]}|{r["channel"]}']["type_agreement"]
        r["fam"] = family(r["channel"])
    frozen_model = json.loads((HERE / "results/failure_model.json").read_text())["models"]
    diag = json.loads((HERE / "results/bmmc_diagnostics.json").read_text())
    errors = json.loads((HERE / "results/bmmc_errors.json").read_text())
    rows = [r for r in diag if r["dim_S"] > 0 and r["type_agreement"] is not None]
    for r in rows:
        r["E_signal"] = errors[f'{r["tag"]}/{r["channel"]}/signal']
        r["fam"] = family(r["channel"])
    y = np.array([r["E_signal"] >= 1 for r in rows])
    donors = np.array([r["donor"] for r in rows])
    hi = np.array([r["coverage_signal"] >= HIGH for r in rows])

    def dev_rate(key):
        groups = {}
        for r in drows:
            groups.setdefault(key(r), []).append(r["E_signal"] >= 1)
        overall = {}
        for r in drows:
            overall.setdefault(kind(r["analysis"]), []).append(r["E_signal"] >= 1)
        return ({k: float(np.mean(v)) for k, v in groups.items()}, {k: float(np.mean(v)) for k, v in overall.items()})

    fam_rate, fallback = dev_rate(lambda r: r["fam"])
    fk_rate, _ = dev_rate(lambda r: (r["fam"], kind(r["analysis"])))
    s = {"primary": score(frozen_model["primary"], rows), "label_free": score(frozen_model["label_free"], rows),
         "coverage": -np.array([r["coverage_signal"] for r in rows]),
         "identity_family": np.array([fam_rate.get(r["fam"], fallback[kind(r["analysis"])]) for r in rows]),
         "identity_family_analysis": np.array([fk_rate.get((r["fam"], kind(r["analysis"])),
                                                           fallback[kind(r["analysis"])]) for r in rows])}
    cells = {}
    for r, f in zip(rows, y):
        cells.setdefault((r["fam"], kind(r["analysis"])), []).append(f)
    s["identity_in_sample"] = np.array([np.mean(cells[(r["fam"], kind(r["analysis"]))]) for r in rows])
    out = {"development_rates": {"family": fam_rate, "family_analysis": {f"{a}/{b}": v for (a, b), v in fk_rate.items()},
                                 "fallback": fallback},
           "all_pairs": {}, "high_coverage": {}, "within_family": {}, "leave_family_out": {}}
    for k, v in s.items():
        out["all_pairs"][k] = interval(y, v, donors, SEED + len(out["all_pairs"]))
        out["high_coverage"][k] = interval(y[hi], v[hi], donors[hi], SEED + 100 + len(out["high_coverage"]))
    for k in ("identity_family_analysis", "identity_in_sample"):
        out["all_pairs"][f"primary_minus_{k}"] = interval(y, s["primary"], donors, SEED + 50, s[k])
        out["high_coverage"][f"primary_minus_{k}"] = interval(y[hi], s["primary"][hi], donors[hi], SEED + 150, s[k][hi])
    for fam in ("frozen", "P1", "colon", "lc", "perm"):
        m = np.array([r["fam"] == fam for r in rows])
        if y[m].sum() >= 5 and (~y[m]).sum() >= 5:
            out["within_family"][fam] = {k: interval(y[m], s[k][m], donors[m], SEED + 200)
                                         for k in ("primary", "label_free", "coverage")}
            out["within_family"][fam]["pairs"] = int(m.sum())
            out["within_family"][fam]["failures"] = int(y[m].sum())
    m = np.array([r["fam"] == "P1" and r["analysis"] != "total" for r in rows]) & hi
    out["within_family"]["P1_within_high_coverage"] = {
        "pairs": int(m.sum()), "failures": int(y[m].sum()),
        **{k: interval(y[m], s[k][m], donors[m], SEED + 300) for k in ("primary", "label_free", "coverage")}}
    for fam in ("P1", "frozen", "lc"):
        train = [r for r in drows if r["fam"] != fam]
        models = {name: fit(train, name) for name in MODELS}
        sp = score(models["primary"], rows)
        sl = score(models["label_free"], rows)
        m = np.array([r["fam"] == fam for r in rows])
        res = {"development_pairs": len(train), "all_pairs": interval(y, sp, donors, SEED + 400),
               "high_coverage": interval(y[hi], sp[hi], donors[hi], SEED + 401),
               "high_coverage_label_free": interval(y[hi], sl[hi], donors[hi], SEED + 402),
               "slopes": dict(zip(MODELS["primary"], models["primary"]["slopes"]))}
        if y[m].sum() >= 5 and (~y[m]).sum() >= 5:
            res["held_out_family"] = interval(y[m], sp[m], donors[m], SEED + 403)
        out["leave_family_out"][fam] = res
    out["pairs"], out["failures"] = len(rows), int(y.sum())
    out["high_coverage_pairs"], out["high_coverage_failures"] = int(hi.sum()), int(y[hi].sum())
    (HERE / "results/posthoc_diagnostic.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
