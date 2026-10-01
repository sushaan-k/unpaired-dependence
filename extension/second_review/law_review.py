#!/usr/bin/env python3
"""H and I of PLAN.md: dataset-aware inference for the prospective test of the law, and an end-to-end pilot design.

    python law_review.py smoke            # I on a development data set of the law (gouwens_visp), 2 replicates
    python law_review.py inference        # checks results/freeze.json; H from the published results
    python law_review.py pilot NAME       # checks results/freeze.json; I for one prospective data set
    python law_review.py score            # results/law_review.json (H and I)

A replicate of I draws a random pilot from the reservoir, predicts from it the paired cells that block James-Stein
needs for the chosen accuracy (Proposition S10 with the pilot's block moments), draws the remaining cells so that the
pilot counts towards the total, fits block James-Stein to all of them and scores it against the truth cells.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(EXT / "predictability"))
import prun  # noqa: E402  (frozen: predictability/results/freeze.json)
import ptools as pt  # noqa: E402

TARGETS = (0.5, 0.3, 0.7)
PILOTS = (50, 100, 200)
REPLICATES = 100
BOOT = 2000
SEED_WITHIN, SEED_SLOPE, SEED_PILOT = 20261310, 20261311, 20261312
STUDY = {"sln111": "mouse lymphoid CITE-seq", "sln206": "mouse lymphoid CITE-seq",
         "fafb_vpn": "fly visual neurons", "banc_vpn": "fly visual neurons"}
OUT = HERE / "results" / "law"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ H: inference with data sets as units

def err(r):
    return float(np.log2(r["observed"] / r["law"]))


def spearman_exact(x, y):
    """Spearman correlation and one-sided exact permutation P over all orderings of x."""
    rho = float(spearmanr(x, y).correlation)
    perms = [float(spearmanr(np.array(p), y).correlation) for p in itertools.permutations(x)]
    return rho, float(np.mean(np.array(perms) >= rho - 1e-12))


def within_permutation(rows):
    """Spearman over problems; predictions permuted only within data sets (all combinations)."""
    names = sorted({r["dataset"] for r in rows})
    x = np.array([r["law"] for r in rows])
    y = np.array([r["observed"] for r in rows])
    rho = float(spearmanr(x, y).correlation)
    groups = [[i for i, r in enumerate(rows) if r["dataset"] == n] for n in names]
    null = []
    for combo in itertools.product(*[list(itertools.permutations(g)) for g in groups]):
        xp = x.copy()
        for g, p in zip(groups, combo):
            xp[g] = x[list(p)]
        null.append(float(spearmanr(xp, y).correlation))
    null = np.array(null)
    return {"rho": rho, "permutations": int(len(null)), "p_one_sided": float(np.mean(null >= rho - 1e-12))}


def fe_slope(rows):
    """Slope of log2 observed on log2 predicted with a fixed effect per data set (copies of a data set in a
    resample are separate data sets)."""
    names = sorted({r["_set"] for r in rows})
    X = np.column_stack([[np.log2(r["law"]) for r in rows]] + [[float(r["_set"] == n) for r in rows] for n in names])
    y = np.array([np.log2(r["observed"]) for r in rows])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[0])


def inference():
    allrows = prun.rows()
    out = {}
    for tg in TARGETS:
        rs = [r for r in allrows if r["target"] == tg]
        est = [r for r in rs if r["estimable"] and r["law"]]
        names = sorted({r["dataset"] for r in est})
        rec = {"estimable": len(est), "datasets": names, "per_dataset": {}, "leave_one_out": {}}
        for n in names:
            e = [err(r) for r in est if r["dataset"] == n]
            rec["per_dataset"][n] = {"problems": len(e), "median_abs_log2": float(np.median(np.abs(e))),
                                     "median_log2": float(np.median(e)), "study": STUDY.get(n, n)}
            rest = [abs(err(r)) for r in est if r["dataset"] != n]
            rec["leave_one_out"][n] = float(np.median(rest)) if rest else None
        mx = np.array([np.mean([np.log2(r["law"]) for r in est if r["dataset"] == n]) for n in names])
        my = np.array([np.mean([np.log2(r["observed"]) for r in est if r["dataset"] == n]) for n in names])
        if len(names) >= 3:
            rho, p = spearman_exact(mx, my)
            rec["dataset_means"] = {"rho": rho, "p_one_sided_exact": p, "datasets": len(names)}
        if len(est) >= 3:
            rec["within_dataset_permutation"] = within_permutation(est)
        rng = np.random.default_rng(SEED_SLOPE)
        for r in est:
            r["_set"] = r["dataset"]
        rec["slope_fixed_effects"] = {"slope": fe_slope(est)}
        bs, bm, bst = [], [], []
        studies = sorted({STUDY.get(n, n) for n in names})
        for k in range(BOOT):
            pick = rng.choice(names, len(names))
            sample = []
            for j, n in enumerate(pick):
                for r in est:
                    if r["dataset"] == n:
                        sample.append({**r, "_set": f"{n}#{j}"})
            if len({s["_set"] for s in sample}) and len({np.log2(s["law"]) for s in sample}) > 1:
                try:
                    bs.append(fe_slope(sample))
                except np.linalg.LinAlgError:
                    pass
            bm.append(float(np.median([abs(err(s)) for s in sample])))
            spick = rng.choice(studies, len(studies))
            ssample = [r for s_ in spick for r in est if STUDY.get(r["dataset"], r["dataset"]) == s_]
            bst.append(float(np.median([abs(err(s)) for s in ssample])))
        rec["slope_fixed_effects"]["ci"] = np.percentile(bs, [2.5, 97.5]).tolist() if bs else None
        rec["median_abs_log2"] = float(np.median([abs(err(r)) for r in est]))
        rec["median_abs_log2_ci_datasets"] = np.percentile(bm, [2.5, 97.5]).tolist()
        rec["median_abs_log2_ci_studies"] = np.percentile(bst, [2.5, 97.5]).tolist()
        rec["non_estimable"] = [{"dataset": r["dataset"], "problem": r["problem"],
                                 "reason": ("target below both curves at the smallest budget"
                                            if r["rel_js"] == "<=" and r["rel_blocks"] == "<=" else
                                            "a curve does not reach the target within the budgets"
                                            if ">" in (r["rel_js"], r["rel_blocks"]) else
                                            "one curve starts above the target"),
                                 "rel_js": r["rel_js"], "rel_blocks": r["rel_blocks"]}
                                for r in rs if not r["estimable"]]
        out[str(tg)] = rec
    return out


# ------------------------------------------------------------------ I: end-to-end pilot design

def pilot_problem(name, di, tag, P, curve, budgets, log):
    N = len(P.Xr)
    rec = {"reservoir": N, "targets": {}}
    full = prun.law_predictions(P.moments(P.Xr, P.Yr, ell=False), TARGETS)
    for a in TARGETS:
        need = pt.needed(curve, a, budgets)
        ra = {"observed_cells": need[0], "observed_relation": need[1], "pilots": {}}
        designs = {}
        for m0 in PILOTS:
            m = min(m0, int(0.2 * N))
            if m < 2:
                continue
            designs.setdefault(m, []).append(m0)
        for m, labels in designs.items():
            res = []
            for k in range(REPLICATES):
                rng = np.random.default_rng([SEED_PILOT, di, int(tag[1:]), int(round(100 * a)), m, k])
                idx = rng.choice(N, m, replace=False)
                law = prun.law_predictions(P.moments(P.Xr[idx], P.Yr[idx], ell=False), (a,))[str(a)]
                flagged = law is None or not np.isfinite(law["cells_blocks"])
                total = N if flagged else max(m, int(np.ceil(law["cells_blocks"])))
                if total > N:
                    flagged, total = True, N
                rest = np.setdiff1d(np.arange(N), idx)
                more = rng.choice(rest, total - m, replace=False) if total > m else np.array([], int)
                sel = np.concatenate([idx, more])
                A, _ = P.fit(P.Xr[sel], P.Yr[sel])
                res.append((P.rf(A), total, flagged))
            res = np.array(res, float)
            ra["pilots"][str(m)] = {"labels": labels, "achieved": res[:, 0].tolist(), "total": res[:, 1].tolist(),
                                    "flagged": res[:, 2].astype(bool).tolist()}
        law = full[str(a)]
        if law is not None:
            total = min(N, max(2, int(np.ceil(law["cells_blocks"]))))
            res = []
            for k in range(REPLICATES):
                rng = np.random.default_rng([SEED_PILOT, di, int(tag[1:]), int(round(100 * a)), 0, k])
                sel = rng.choice(N, total, replace=False)
                A, _ = P.fit(P.Xr[sel], P.Yr[sel])
                res.append(P.rf(A))
            ra["reservoir_law"] = {"total": total, "achieved": res}
        rec["targets"][str(a)] = ra
        log(f"{name} {tag} target {a}: observed {need[0]:.0f}{need[1]}; " + "; ".join(
            f"m={m}: median achieved {np.median(v['achieved']):.3f}, reached {np.mean(np.array(v['achieved']) >= a):.2f}"
            for m, v in ra["pilots"].items()))
    return rec


def pilot(name, log, data=None, published=None):
    d = prun.dataset(name) if data is None else data
    ev = published if published is not None else json.loads((prun.RES / f"{name}.json").read_text())["problems"]
    di = prun.NEW.index(name) if name in prun.NEW else 99
    out = {}
    for tag, P in prun.problems(d):
        P.score_setup()
        e = ev[tag]
        out[tag] = pilot_problem(name, di, tag, P, e["curves"]["bjs2"], e["budgets"], log)
    return out


def summarize_pilot():
    files = sorted(OUT.glob("pilot_*.json"))
    per = {}
    for p in files:
        per[p.stem[len("pilot_"):]] = json.loads(p.read_text())
    out = {}
    for a in map(str, TARGETS):
        rec = {}
        for m in map(str, PILOTS):
            ach, reach, near, ratio, flag, tot = [], [], [], [], [], []
            for name, probs in per.items():
                for tag, pr in probs.items():
                    ta = pr["targets"][a]
                    v = ta["pilots"].get(m) or next((w for w in ta["pilots"].values() if int(m) in w["labels"]), None)
                    if v is None:
                        continue
                    x = np.array(v["achieved"])
                    ach.append(float(np.median(x)))
                    reach.append(float(np.mean(x >= float(a))))
                    near.append(float(np.mean(x >= float(a) - 0.05)))
                    flag.append(float(np.mean(v["flagged"])))
                    tot.append(float(np.median(v["total"])))
                    if ta["observed_relation"] == "=":
                        ratio.append(float(np.median(np.array(v["total"]) / ta["observed_cells"])))
            rec[m] = {"problems": len(ach), "median_achieved": float(np.median(ach)),
                      "share_reaching_target": float(np.mean(reach)), "share_within_0.05": float(np.mean(near)),
                      "share_flagged": float(np.mean(flag)), "median_total": float(np.median(tot)),
                      "median_total_over_needed": float(np.median(ratio)) if ratio else None,
                      "problems_with_needed": len(ratio)}
        ref = [np.array(pr["targets"][a]["reservoir_law"]["achieved"]) for probs in per.values()
               for pr in probs.values() if "reservoir_law" in pr["targets"][a]]
        rec["reservoir_law"] = {"problems": len(ref), "median_achieved": float(np.median([np.median(x) for x in ref])),
                                "share_reaching_target": float(np.mean([np.mean(x >= float(a)) for x in ref]))}
        out[a] = rec
    return out


# ------------------------------------------------------------------ smoke, main

def smoke():
    d = pt.load(Path("/home/claude/cbio/rawdata/generality") / "gouwens_visp.npz", "gouwens_visp")
    global REPLICATES
    REPLICATES = 2
    fake = {}
    for tag, P in prun.problems(d):
        m, avail, budgets = prun.budgets_for(P)
        fake[tag] = {"curves": {"bjs2": list(np.linspace(0.1, 0.8, len(budgets)))}, "budgets": budgets}
    out = pilot("gouwens_visp", log, data=d, published=fake)
    path = HERE / "results_smoke" / "law_pilot_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "inference", "pilot", "score"))
    ap.add_argument("name", nargs="?")
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        if a.what == "smoke":
            smoke()
            return
        from validation_review import check_freeze
        check_freeze()
        OUT.mkdir(parents=True, exist_ok=True)
        if a.what == "inference":
            (OUT / "inference.json").write_text(json.dumps(inference(), indent=1) + "\n")
        elif a.what == "pilot":
            (OUT / f"pilot_{a.name}.json").write_text(json.dumps(pilot(a.name, log)) + "\n")
        else:
            res = {"inference": json.loads((OUT / "inference.json").read_text()), "pilot": summarize_pilot()}
            (HERE / "results" / "law_review.json").write_text(json.dumps(res, indent=1) + "\n")
            print(json.dumps(res["pilot"], indent=1))


if __name__ == "__main__":
    main()
