#!/usr/bin/env python3
"""Perturbation test, step 3: open the sealed scoring files and evaluate (PLAN.md, "Endpoint" and "Hypotheses").

Refuses to run unless the result files match results/manifest.json and the code
matches results/freeze.json. Recomputes every prediction from the adaptation
files and checks it against its recorded digest, then scores it against the
scoring cells. Writes results/evaluation.json and results/group_terms.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import pmethods as pm
from freeze import check_freeze
from pdata import load, sha
from predict import PAIRED, PRIMARY, RESULT_FILES, digest, run

sys.path.append(str(pm.HERE.parent / "recoverability"))
from stats import cluster_bootstrap  # noqa: E402

BOOT = 2000
SEED = 20261013
CONDITIONS = ("control", "ifng", "coculture")


def check_manifest():
    manifest = json.loads((pm.HERE / "results/manifest.json").read_text())
    for name in RESULT_FILES:
        assert sha(pm.HERE / "results" / name) == manifest[name], name
    check_freeze()
    return manifest


def scoring(part, keys):
    d = load(f"{part}_scoring")
    allkeys = pm.keys_of(d)
    out = {}
    for key in keys:
        idx = np.flatnonzero(allkeys == key)
        out[key] = {"idx": idx, "t": pm.scoring_targets(d["counts"][idx], d["library"][idx], d["y"][idx],
                                                         d["part"][idx])}
    return d, out


def terms(preds, part, key, t):
    row = {}
    for name in sorted({k.split("/", 2)[2] for k in preds if k.startswith(f"{part}/{key}/")}):
        C = preds[f"{part}/{key}/{name}"]
        num, den = pm.endpoint_terms(C, t)
        row[name] = {"num": num, "den": den,
                     "rel_error": float(np.linalg.norm(C - t["T"]) / np.linalg.norm(t["T"]))}
    return row


def quantiles(values):
    """2.5% and 97.5% quantiles of the finite bootstrap values (nan if there are none)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan")] * 2


def ratio(a, b):
    """Share of the paired comparator's recovery; undefined (nan) when the comparator recovers nothing."""
    return a / b if b > 0 else float("nan")


def recovered(rows, name):
    num = sum(r[name]["num"] for r in rows)
    den = sum(r[name]["den"] for r in rows)
    return num / den


def heldout_analysis(rows):
    targets = np.array([r["target"] for r in rows])
    groups = [r["terms"] for r in rows]
    wp = [n for n in groups[0] if n.startswith("within_pseudo")]
    wd = [n for n in groups[0] if n.startswith("within_derange")]
    up = [n for n in groups[0] if n.startswith("pseudo")]
    ud = [n for n in groups[0] if n.startswith("derange")]
    base = [n for n in groups[0] if n not in wp + wd + up + ud]

    def stat(idx):
        g = [groups[i] for i in idx]
        rf = {n: recovered(g, n) for n in base}
        ctl = {k: float(np.mean([recovered(g, n) for n in names])) for k, names in
               (("within_pseudo", wp), ("within_derange", wd), ("pseudo", up), ("derange", ud))}
        vals = [rf[n] for n in base] + [ctl[k] for k in ctl]
        vals += [rf[PRIMARY] - ctl["within_pseudo"], rf[PRIMARY] - ctl["within_derange"],
                 ratio(rf[PRIMARY], rf[PAIRED]), rf[PRIMARY] - rf["pf_measured"],
                 rf["pf_measured"] - ctl["pseudo"], rf["pf_measured"] - ctl["derange"]]
        return np.array(vals)

    labels = base + ["control_within_pseudo", "control_within_derange", "control_pseudo", "control_derange",
                     "primary_minus_within_pseudo", "primary_minus_within_derange", "share_of_paired",
                     "primary_minus_unstratified", "unstratified_minus_pseudo", "unstratified_minus_derange"]
    point = stat(np.arange(len(groups)))
    boot = cluster_bootstrap(targets, stat, BOOT, SEED)
    out = {"groups": len(groups), "targets": len(set(targets)),
           "denominator": float(sum(g["independence"]["den"] for g in groups)), "estimates": {}}
    for j, lab in enumerate(labels):
        out["estimates"][lab] = {"value": float(point[j]), "ci": quantiles(boot[:, j])}
    out["by_condition"] = {c: {n: recovered([groups[i] for i in range(len(groups)) if rows[i]["condition"] == c], n)
                               for n in base} for c in CONDITIONS}
    out["mean_rel_error"] = {n: float(np.mean([g[n]["rel_error"] for g in groups])) for n in base}
    out["arms"] = base
    return out


def nt_analysis(d, sc, rows, preds):
    """Recovered fraction in non-targeting cells; intervals from resampling guide sets within each condition."""
    groups = [r["terms"] for r in rows]
    names = [n for n in groups[0] if n in (PRIMARY, PAIRED, "pf_within_latent", "pf_measured", "transferred_correlation",
                                           "reference_regression", "benchmark", "pf_within_condition")]
    out = {"by_condition": {r["condition"]: {n: r["terms"][n]["num"] / r["terms"][n]["den"] for n in names}
                            for r in rows},
           "pooled": {n: recovered(groups, n) for n in names}}
    rng = np.random.default_rng(SEED + 1)
    clusters = {}
    for r in rows:
        guides = d["guide"][sc[r["key"]]["idx"]]
        ids = sorted(set(guides))
        clusters[r["key"]] = [np.flatnonzero(guides == u) for u in ids]
    reps = []
    for _ in range(BOOT):
        g = []
        for r in rows:
            idx, members = sc[r["key"]]["idx"], clusters[r["key"]]
            pick = rng.choice(len(members), len(members), replace=True)
            j = idx[np.concatenate([members[k] for k in pick])]
            t = pm.scoring_targets(d["counts"][j], d["library"][j], d["y"][j], d["part"][j])
            g.append({n: dict(zip(("num", "den"), pm.endpoint_terms(preds[f"nt/{r['key']}/{n}"], t))) for n in names})
        reps.append([recovered(g, n) for n in names] + [ratio(recovered(g, PRIMARY), recovered(g, PAIRED))])
    reps = np.array(reps)
    out["pooled_ci"] = {n: quantiles(reps[:, k]) for k, n in enumerate(names)}
    share = ratio(recovered(groups, PRIMARY), recovered(groups, PAIRED))
    out["share_of_paired"] = {"value": share, "ci": quantiles(reps[:, -1])}
    out["guide_sets"] = {r["condition"]: int(len(set(d["guide"][sc[r["key"]]["idx"]]))) for r in rows}
    return out


def main():
    manifest = check_manifest()
    digests = json.loads((pm.HERE / "results/digests.json").read_text())
    preds, diag = run()
    assert set(preds) == set(digests)
    for k, C in preds.items():
        assert digest(C) == digests[k], k
    print("all", len(preds), "predictions match their digests", flush=True)
    ev = {"manifest": manifest}
    for part in ("heldout", "nt"):
        drows = [r for r in diag if r["part"] == part]
        d, sc = scoring(part, [r["key"] for r in drows])
        rows = [dict(r, terms=terms(preds, part, r["key"], sc[r["key"]]["t"])) for r in drows]
        if part == "heldout":
            ev["heldout"] = heldout_analysis(rows)
        else:
            ev["nt"] = nt_analysis(d, sc, rows, preds)
        (pm.HERE / f"results/group_terms_{part}.json").write_text(json.dumps(
            [{k: v for k, v in r.items()} for r in rows], indent=0) + "\n")
        print("scored", part, flush=True)
    e = ev["heldout"]["estimates"]
    ev["verdict"] = {"H1_recovery": e[PRIMARY]["ci"][0] > 0,
                     "H2_beats_pseudo_populations": e["primary_minus_within_pseudo"]["ci"][0] > 0,
                     "H2_beats_derangements": e["primary_minus_within_derange"]["ci"][0] > 0,
                     "H3_nontargeting": ev["nt"]["pooled_ci"][PRIMARY][0] > 0,
                     "H4_most_of_paired": e["share_of_paired"]["ci"][0] > 0.5}
    (pm.HERE / "results/evaluation.json").write_text(json.dumps(ev, indent=1) + "\n")
    print(json.dumps({"verdict": ev["verdict"], "heldout": {k: v for k, v in e.items()},
                      "nt": {k: v for k, v in ev["nt"].items() if k != "by_condition"}}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
