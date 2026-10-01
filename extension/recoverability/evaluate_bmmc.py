#!/usr/bin/env python3
"""Bone-marrow test, step 3: open the sealed scoring file and evaluate (PLAN.md, "Endpoints").

Refuses to run unless the result files match results/bmmc_manifest.json and the
code matches results/freeze.json. Recomputes every prediction from the
adaptation file and checks it against its recorded digest, then scores it
against the scoring half. Writes results/bmmc_evaluation.json and results/bmmc_errors.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "cross_study"
sys.path.insert(0, str(CS))

from data import cross_correlation, type_centre  # noqa: E402

import reference as ref  # noqa: E402
from bmmc import load  # noqa: E402
from compat import psd_power  # noqa: E402
from freeze import check_freeze  # noqa: E402
from noise import lognorm  # noqa: E402
from predict_bmmc import (ANALYSES, BUDGETS, RESULT_FILES, STRATEGIES, analysis_inputs, arm_predictions,  # noqa: E402
                          channel_predictions, digest, load_channels, sample_inputs)
from stats import auc, cluster_bootstrap, logistic_score, spearman  # noqa: E402

BOOT = 2000
BOOT_SEED = 20261004
HIGH_COVERAGE = 0.4
MIN_CLASS = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest():
    manifest = json.loads((HERE / "results/bmmc_manifest.json").read_text())
    for name in RESULT_FILES:
        assert sha(HERE / "results" / name) == manifest[name], name
    check_freeze()
    return manifest


def rel(C, T):
    return float(np.linalg.norm(C - T) / np.linalg.norm(T))


def targets(score, units):
    out = {}
    for batch, unit in units.items():
        m = score["batch"] == batch
        x, y = lognorm(score["counts"][m], score["library"][m]), score["y"][m]
        for analysis, key in ANALYSES:
            if key is None:
                out[f"{batch}/{analysis}"] = cross_correlation(x, y)
            else:
                labels, types = score["labels"][key][m], unit["types"][key]
                cells = np.isin(labels, types)
                out[f"{batch}/{analysis}"] = cross_correlation(type_centre(x[cells], labels[cells], types),
                                                               type_centre(y[cells], labels[cells], types))
        cells = np.isin(score["labels"]["coarse"][m], unit["types"]["coarse"])
        out[f"{batch}/total_kept"] = cross_correlation(x[cells], y[cells])
    return out


def score_predictions(units, chans, T):
    """Recompute every prediction, check its digest and return errors keyed by tag/arm or tag/channel/estimator."""
    digests = json.loads((HERE / "results/bmmc_digests.json").read_text())
    rec = load("adaptation")
    errors, checked = {}, 0
    for batch, unit in units.items():
        s = sample_inputs(rec, batch, unit)
        for analysis, key in ANALYSES:
            tag = f"{batch}/{analysis}"
            a = analysis_inputs(s, key, None if key is None else unit["types"][key])
            arms, _ = arm_predictions(a, s, key)
            for arm, C in arms.items():
                assert digest(C) == digests[f"{tag}/{arm}"], f"{tag}/{arm}"
                errors[f"{tag}/{arm}"] = rel(C, T[tag])
                checked += 1
            for name, ch in chans.items():
                for k, C in channel_predictions(a, ch).items():
                    assert digest(C) == digests[f"{tag}/{name}/{k}"], f"{tag}/{name}/{k}"
                    errors[f"{tag}/{name}/{k}"] = rel(C, T[tag])
                    checked += 1
        print("scored", batch, flush=True)
    assert checked == len(digests)
    return errors, rec


def model_score(model, rows):
    cols = []
    for r in rows:
        v = []
        for c in model["predictors"]:
            v.append(np.log(r["F_signal"] + 1e-3) if c == "log_F_signal" else r[c])
        cols.append(v)
    return logistic_score((model["intercept"], np.array(model["slopes"]), np.array(model["mean"]),
                           np.array(model["sd"])), np.array(cols, float))


def auc_compare(y, a, b, clusters, seed):
    """AUC of score a, of score b, their difference and its donor-clustered bootstrap 95% interval."""
    diff = cluster_bootstrap(clusters, lambda i: auc(y[i], a[i]) - auc(y[i], b[i]), BOOT, seed)
    diff = diff[np.isfinite(diff)]
    return {"auc_a": auc(y, a), "auc_b": auc(y, b), "difference": auc(y, a) - auc(y, b),
            "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]}


def failure_analysis(diag, errors):
    model = json.loads((HERE / "results/failure_model.json").read_text())["models"]
    for r in diag:
        r["E_signal"] = errors[f'{r["tag"]}/{r["channel"]}/signal']
        r["E_measured"] = errors[f'{r["tag"]}/{r["channel"]}/measured']
    fams = sorted({r["family"] for r in diag})
    out = {"abstention": {f: float(np.mean([r["dim_S"] == 0 for r in diag if r["family"] == f])) for f in fams}}
    rows = [r for r in diag if r["dim_S"] > 0 and r["type_agreement"] is not None]
    y = np.array([r["E_signal"] >= 1 for r in rows])
    clusters = np.array([r["donor"] for r in rows])
    scores = {"primary": model_score(model["primary"], rows), "label_free": model_score(model["label_free"], rows),
              "coverage": -np.array([r["coverage_signal"] for r in rows]),
              "composition": -np.array([r["between_share"] for r in rows]),
              "cells": -np.log([r["cells"] for r in rows]), "n_train": -np.log([r["n_train"] for r in rows]),
              "shift": np.array([r["shift"] for r in rows]),
              "compatibility": np.array([r["F_signal"] for r in rows]),
              "type_disagreement": -np.array([r["type_agreement"] for r in rows])}
    E = np.array([r["E_signal"] for r in rows])
    out.update({"pairs": len(rows), "failures": int(y.sum()),
                "failure_rate_by_family_analysis": {f"{f}/{an}": float(np.mean([r["E_signal"] >= 1 for r in rows
                                                                               if r["family"] == f and r["analysis"] == an]))
                                                    for f in fams for an in ("total", "within_coarse", "within_fine")
                                                    if any(r["family"] == f and r["analysis"] == an for r in rows)},
                "auc": {k: auc(y, v) for k, v in scores.items()},
                "spearman_with_E": {k: spearman(v, E) for k, v in scores.items()}})
    comparisons = {}
    for i, other in enumerate(("coverage", "composition", "cells", "n_train", "shift")):
        comparisons[f"primary_vs_{other}"] = auc_compare(y, scores["primary"], scores[other], clusters, BOOT_SEED + i)
    comparisons["label_free_vs_coverage"] = auc_compare(y, scores["label_free"], scores["coverage"], clusters,
                                                        BOOT_SEED + 10)
    out["comparisons"] = comparisons
    hi = np.array([r["coverage_signal"] >= HIGH_COVERAGE for r in rows])
    yh = y[hi]
    evaluable = yh.sum() >= MIN_CLASS and (~yh).sum() >= MIN_CLASS
    out["high_coverage"] = {"pairs": int(hi.sum()), "failures": int(yh.sum()), "evaluable": bool(evaluable),
                            "families": {f: int(sum(1 for r, h in zip(rows, hi) if h and r["family"] == f)) for f in fams}}
    if evaluable:
        ch = clusters[hi]
        for i, k in enumerate(("type_disagreement", "compatibility", "primary", "label_free")):
            v = scores[k][hi]
            boot = cluster_bootstrap(ch, lambda j: auc(yh[j], v[j]), BOOT, BOOT_SEED + 20 + i)
            boot = boot[np.isfinite(boot)]
            out["high_coverage"][k] = {"auc": auc(yh, v), "ci": [float(np.quantile(boot, 0.025)),
                                                                 float(np.quantile(boot, 0.975))]}
    ranking = {k: [0.0, 0] for k in ("primary", "label_free", "coverage", "shift", "n_train", "compatibility",
                                     "type_disagreement")}
    tags = sorted({r["tag"] for r in rows})
    index = {t: [i for i, r in enumerate(rows) if r["tag"] == t] for t in tags}
    for t in tags:
        idx = index[t]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if abs(E[i] - E[j]) < 0.05:
                    continue
                worse = i if E[i] > E[j] else j
                better = j if worse == i else i
                for k in ranking:
                    d = scores[k][worse] - scores[k][better]
                    ranking[k][0] += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
                    ranking[k][1] += 1
    out["channel_ranking"] = {k: {"concordance": v[0] / v[1] if v[1] else None, "pairs": v[1]}
                              for k, v in ranking.items()}
    return out, rows


def estimator_summary(units, errors):
    out = {}
    for analysis, key in ANALYSES:
        tags = [f"{b}/{analysis}" for b in units]
        row = {}
        for name in ("frozen", "P1", "colon"):
            for k in ("signal", "measured"):
                v = [errors[f"{t}/{name}/{k}"] for t in tags]
                row[f"{name}/{k}"] = {"mean": float(np.mean(v)), "below_1": int(np.sum(np.array(v) < 1)), "values": v}
        for arm in ("benchmark", "label_between"):
            if f"{tags[0]}/{arm}" in errors:
                v = [errors[f"{t}/{arm}"] for t in tags]
                row[arm] = {"mean": float(np.mean(v)), "values": v}
        out[analysis] = row
    return out


def bounds_summary(T):
    z = np.load(HERE / "results/bmmc_bounds.npz")
    out = {}
    for key in sorted({k.rsplit("/", 1)[0] for k in z.files}):
        lo, hi, centre = z[f"{key}/lower"], z[f"{key}/upper"], z[f"{key}/centre"]
        t = T[key]
        signed = (lo > 0) | (hi < 0)
        correct = np.where(lo > 0, t > 0, t < 0)
        out[key] = {"inside": float(np.mean((t >= lo) & (t <= hi))), "sign_determined": float(signed.mean()),
                    "sign_correct": float(correct[signed].mean()) if signed.any() else None,
                    "mean_halfwidth": float(np.mean(hi - lo) / 2), "mean_abs_target": float(np.mean(np.abs(t))),
                    "centre_error": rel(centre, t)}
    return out


def compatibility_summary():
    rows = json.loads((HERE / "results/bmmc_bootstrap.json").read_text())["rows"]
    out = {}
    for name in ("frozen", "P1", "colon"):
        for analysis, _ in ANALYSES:
            for stat in ("F_measured", "F_signal"):
                r = [x for x in rows if x["channel"] == name and x["statistic"] == stat and x["tag"].endswith("/" + analysis)]
                out[f"{name}/{analysis}/{stat}"] = {
                    "median": float(np.median([x["point"] for x in r])),
                    "range": [float(min(x["point"] for x in r)), float(max(x["point"] for x in r))],
                    "above_1": int(sum(x["point"] > 1 for x in r)),
                    "lower_bound_above_1": int(sum(x["basic_lower"] > 1 for x in r)), "analyses": len(r)}
    return out


def reference_analysis(units, rec, T):
    z = np.load(HERE / "results/bmmc_selections.npz")
    offsets = z["offsets"]
    res = {}
    for batch, unit in units.items():
        m = rec["batch"] == batch
        x0, y0 = lognorm(rec["counts"][m], rec["library"][m]), rec["y"][m]
        coarse = rec["labels"]["coarse"][m]
        ctypes = unit["types"]["coarse"]
        cells = np.isin(coarse, ctypes)
        xk, yk, lk = x0[cells], y0[cells], coarse[cells]
        s = ref.summaries(xk, yk, lk, ctypes)
        Cbet = ref.between_estimate(s)
        Tw, Tt = T[f"{batch}/within_coarse"], T[f"{batch}/total_kept"]
        res[batch] = {"between_only_total": rel(Cbet, Tt)}
        rows = np.flatnonzero(z["batch"] == batch)
        acc = {}
        for i in rows:
            idx = z["index"][offsets[i]:offsets[i + 1]]
            key = (int(z["budget"][i]), str(z["strategy"][i]))
            if key[1] == "paired_only":
                acc.setdefault(key, {"total": []})["total"].append(rel(ref.paired_only_total(xk, yk, idx, s), Tt))
            else:
                Cw = ref.within_estimate(xk, yk, lk, idx, s)
                d = acc.setdefault(key, {"within": [], "total": []})
                d["within"].append(rel(Cw, Tw))
                d["total"].append(rel(Cbet + ref.within_to_total(Cw, s), Tt))
        res[batch]["budgets"] = {f"{N}/{st}": {k: float(np.mean(v)) for k, v in d.items()} for (N, st), d in acc.items()}
        print("reference", batch, flush=True)
    batches = list(units)
    donors = np.array([units[b]["donor"] for b in batches])
    summary = {}
    for N in BUDGETS:
        row = {}
        for st in STRATEGIES:
            row[st] = {"within": float(np.mean([res[b]["budgets"][f"{N}/{st}"]["within"] for b in batches])),
                       "hybrid_total": float(np.mean([res[b]["budgets"][f"{N}/{st}"]["total"] for b in batches]))}
        row["paired_only_total"] = float(np.mean([res[b]["budgets"][f"{N}/paired_only"]["total"] for b in batches]))
        for other in ("random", "balanced"):
            d = np.array([res[b]["budgets"][f"{N}/guided"]["within"] - res[b]["budgets"][f"{N}/{other}"]["within"]
                          for b in batches])
            row[f"guided_minus_{other}"] = {"mean": float(d.mean()), "samples_lower": int(np.sum(d < 0))}
        summary[str(N)] = row
    tests = {}
    for i, other in enumerate(("random", "balanced")):
        d = np.array([np.mean([res[b]["budgets"][f"{N}/guided"]["within"] - res[b]["budgets"][f"{N}/{other}"]["within"]
                               for N in BUDGETS]) for b in batches])
        boot = cluster_bootstrap(donors, lambda j: float(np.mean(d[j])), BOOT, BOOT_SEED + 40 + i)
        tests[f"guided_vs_{other}"] = {
            "mean_difference": float(d.mean()), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "lower_at_every_budget": bool(all(summary[str(N)][f"guided_minus_{other}"]["mean"] < 0 for N in BUDGETS))}
    between_only = float(np.mean([res[b]["between_only_total"] for b in batches]))
    return {"per_sample": res, "summary": summary, "tests": tests, "between_only_total": between_only}


def verdict(ev):
    c = ev["failure"]["comparisons"]
    return {
        "H1_primary_beats_coverage": c["primary_vs_coverage"]["ci"][0] > 0,
        "H2_primary_beats_controls": {k: c[f"primary_vs_{k}"]["ci"][0] > 0
                                      for k in ("composition", "cells", "n_train", "shift")},
        "H3_high_coverage_failures_recognized": (ev["failure"]["high_coverage"]["primary"]["ci"][0] > 0.5)
        if ev["failure"]["high_coverage"]["evaluable"] else None,
        "H6_guided_beats_random": ev["reference"]["tests"]["guided_vs_random"]["ci"][1] < 0
        and ev["reference"]["tests"]["guided_vs_random"]["lower_at_every_budget"],
        "H6_guided_beats_balanced": ev["reference"]["tests"]["guided_vs_balanced"]["ci"][1] < 0
        and ev["reference"]["tests"]["guided_vs_balanced"]["lower_at_every_budget"]}


def main():
    manifest = check_manifest()
    units = json.loads((HERE / "results/bmmc_units.json").read_text())
    chans = load_channels()
    score = load("scoring")                 # SHA-256 checked against bmmc_seal.json
    T = targets(score, units)
    errors, rec = score_predictions(units, chans, T)
    diag = json.loads((HERE / "results/bmmc_diagnostics.json").read_text())
    failure, _ = failure_analysis(diag, errors)
    ev = {"manifest": manifest, "failure": failure, "estimators": estimator_summary(units, errors),
          "bounds": bounds_summary(T), "compatibility": compatibility_summary(),
          "reference": reference_analysis(units, rec, T)}
    ev["verdict"] = verdict(ev)
    (HERE / "results/bmmc_errors.json").write_text(json.dumps(errors, indent=0) + "\n")
    (HERE / "results/bmmc_evaluation.json").write_text(json.dumps(ev, indent=1) + "\n")
    print(json.dumps({"verdict": ev["verdict"], "auc": failure["auc"], "comparisons": failure["comparisons"],
                      "high_coverage": failure["high_coverage"], "reference_tests": ev["reference"]["tests"]}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
