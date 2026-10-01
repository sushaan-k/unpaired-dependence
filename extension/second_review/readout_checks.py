#!/usr/bin/env python3
"""F of PLAN.md: calibration of the Fisher-z test behind the readout's reproducible associations.

    python readout_checks.py smoke     # synthetic values with the real design (folds 0 and 17) -> results_smoke/
    python readout_checks.py run       # checks results/freeze.json -> results/readout_checks.json

In every scored unit (one cell type in one held-out sample, at least 20 cells in each hash sub-half) and sub-half,
protein vectors are permuted across cells, which keeps both marginal distributions and removes the pairing. The
readout's statistic, z = artanh(r) sqrt(n - 3), is then computed for every gene-protein pair.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(EXT / "validation"))
sys.path.insert(0, str(EXT / "validation" / "readout"))
import vrun  # noqa: E402
import brun  # noqa: E402
from validation_review import check_freeze  # noqa: E402

PERMUTATIONS = 20
SEED = 20261308
LEVELS = (0.1, 0.05, 0.01, 0.001, 0.0001)
BINS = ((20, 39), (40, 79), (80, 159), (160, 10 ** 9))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def zstat(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    den = np.sqrt(np.maximum((xc * xc).sum(0), 1e-12)[:, None] * np.maximum((yc * yc).sum(0), 1e-12)[None, :])
    r = (xc.T @ yc) / den
    return np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12)) * np.sqrt(len(x) - 3)


def run(synthetic=False, fold_ids=None):
    new = vrun.load_new(synthetic)
    part, half, res = vrun.roles(new)
    info = json.loads((EXT / "validation" / "results" / "predictions_info.json").read_text())["panels"]
    gn = {g: i for i, g in enumerate(new["x_names"])}
    yn = {p: i for i, p in enumerate(new["y_names"])}
    gidx = np.array([gn[g] for g in info["genes"]]) if not synthetic else np.arange(200)
    yidx = np.array([yn[p] for p in info["proteins_new"]]) if not synthetic else np.arange(111)
    crit = {lv: float(norm.isf(lv / 2)) for lv in LEVELS}
    count = {b: {lv: 0 for lv in LEVELS} for b in BINS}
    pairs = {b: 0 for b in BINS}
    both = {"same_sign_both_p01": 0, "pairs": 0}
    observed = {"reproducible": 0, "pairs": 0}
    units = 0
    for fd in vrun.folds(new):
        if fold_ids is not None and fd["fold"] not in fold_ids:
            continue
        tr = brun.truth(new, fd, half, gidx, yidx)
        r = np.flatnonzero(new["sample"] == fd["heldout"])
        x = vrun.rna(new, r, gidx)[0]
        y = vrun.prot(new, r, yidx)
        cond, hv = new["cond"][r], half[r]
        for ci, c in enumerate(vrun.CONDS):
            if not tr[c]["scored"]:
                continue
            units += 1
            observed["reproducible"] += int(tr[c]["rep"].sum())
            observed["pairs"] += int(tr[c]["rep"].size)
            zs = {}
            for h in (0, 1):
                m = np.flatnonzero((cond == c) & (hv == h))
                n = len(m)
                b = next(bb for bb in BINS if bb[0] <= n <= bb[1])
                zs[h] = []
                for k in range(PERMUTATIONS):
                    rng = np.random.default_rng([SEED, fd["fold"], ci, h, k])
                    z = zstat(x[m], y[m][rng.permutation(n)])
                    a = np.abs(z)
                    for lv in LEVELS:
                        count[b][lv] += int(np.sum(a >= crit[lv]))
                    pairs[b] += z.size
                    zs[h].append(z)
            for k in range(PERMUTATIONS):
                zA, zB = zs[0][k], zs[1][k]
                ok = (np.abs(zA) >= brun.Z_CRIT) & (np.abs(zB) >= brun.Z_CRIT) & (np.sign(zA) == np.sign(zB))
                both["same_sign_both_p01"] += int(ok.sum())
                both["pairs"] += int(ok.size)
        log(f"fold {fd['fold']}: {units} units so far")
    rates = {f"{lo}-{hi if hi < 10 ** 9 else 'more'} cells": {"pairs": pairs[(lo, hi)],
             **{f"P<={lv}": (count[(lo, hi)][lv] / pairs[(lo, hi)] if pairs[(lo, hi)] else None) for lv in LEVELS}}
             for lo, hi in BINS}
    total = sum(pairs.values())
    overall = {f"P<={lv}": sum(count[b][lv] for b in BINS) / total for lv in LEVELS}
    null_rate = both["same_sign_both_p01"] / both["pairs"]
    obs_rate = observed["reproducible"] / observed["pairs"]
    out = {"units": units, "permutations": PERMUTATIONS, "overall": overall, "by_cells_per_sub_half": rates,
           "replicability": {"null_rate": null_rate, "nominal_rate": 2 * 0.005 ** 2,
                             "observed_rate": obs_rate, "implied_false_share": null_rate / obs_rate if obs_rate else None,
                             **both, "observed": observed}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "run"))
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        if a.what == "smoke":
            out = run(synthetic=True, fold_ids=[0, 17])
            path = HERE / "results_smoke" / "readout_checks.json"
        else:
            check_freeze()
            out = run()
            path = HERE / "results" / "readout_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("units", "overall", "replicability")}, indent=1))


if __name__ == "__main__":
    main()
