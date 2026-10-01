#!/usr/bin/env python3
"""Replication, step 2: open the sealed scoring files and evaluate (PLAN.md, "Endpoint" and "Hypotheses").

Refuses to run unless the result files match results/manifest.json and the code
matches results/freeze.json; checks every stored prediction against its digest.
Writes results/evaluation.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "perturbation"))
sys.path.append(str(HERE.parent / "recoverability"))

import pmethods as pm  # noqa: E402
from freeze import check_freeze  # noqa: E402
from predict import DRAWS, RESULT_FILES, digest  # noqa: E402
from rdata import ENCODING, load, seal, sha  # noqa: E402
from stats import cluster_bootstrap  # noqa: E402

BOOT, SEED = 2000, 20261020
GENESETS = Path("/home/claude/cbio/rawdata/genesets")
IFN_FILES = {"HALLMARK_INTERFERON_GAMMA_RESPONSE.json": "845aabdb9ff04fb62bbb1bc8a527654c549075852c84f31d64f50d9e43134353",
             "HALLMARK_INTERFERON_ALPHA_RESPONSE.json": "2a3d93f3170bc9866ced1135bdde9870f18294f7ad3ae6cb3732dc3680483455"}
PRIMARY, PAIRED = "pf_within_measured", "paired_closed_form"
BASE = ("pf_within_measured", "pf_within_latent", "paired_closed_form", "reference_regression",
        "transferred_correlation", "benchmark", "replicate")


def check_manifest():
    manifest = json.loads((HERE / "results/manifest.json").read_text())
    for name in RESULT_FILES:
        assert sha(HERE / "results" / name) == manifest[name], name
    check_freeze()
    return manifest


def ifn_block(record):
    ifn = set()
    for name, dg in IFN_FILES.items():
        path = GENESETS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == dg
        d = json.loads(path.read_text())
        ifn |= set(d[next(iter(d))]["geneSymbols"])
    rows = [i for i, g in enumerate(record["genes"]) if g in ifn]
    cols = [j for j, p in enumerate(record["proteins"]) if any(g in ifn for g in ENCODING[p])]
    other_r = [i for i in range(len(record["genes"])) if i not in rows]
    other_c = [j for j in range(len(record["proteins"])) if j not in cols]
    return {"ifn": (rows, cols), "complement": (other_r, other_c)}, {
        "ifn_genes": [record["genes"][i] for i in rows], "ifn_proteins": [record["proteins"][j] for j in cols]}


def ratio(a, b):
    return a / b if b > 0 else float("nan")


def main():
    manifest = check_manifest()
    digests = json.loads((HERE / "results/digests.json").read_text())
    z = np.load(HERE / "results/predictions.npz")
    preds = {k: z[k] for k in z.files}
    assert set(preds) == set(digests)
    for k, C in preds.items():
        assert digest(C) == digests[k], k
    diag = json.loads((HERE / "results/diagnostics.json").read_text())
    record = seal()
    blocks, block_defs = ifn_block(record)
    ev = {"manifest": manifest, "blocks": block_defs}
    controls = {"pseudo": [f"within_pseudo{i:02d}_measured" for i in range(10)],
                "derange": [f"within_derange{i:02d}_measured" for i in range(10)]}
    random_arms = [f"abl_random{r:02d}_measured" for r in range(DRAWS)]
    for part in ("test", "nt"):
        sc = load(f"{part}_scoring")
        sk = pm.keys_of(sc)
        rows = []
        for r in [r for r in diag if r["part"] == part]:
            j = np.flatnonzero(sk == r["key"])
            t = pm.scoring_targets(sc["counts"][j], sc["library"][j], sc["y"][j], sc["part"][j])
            VS = preds[f"{part}/{r['key']}/S"]
            terms = {}
            for bname, blk in list(blocks.items()) + [("all", None), ("along_S", "S")]:
                def cut(M, blk=blk):
                    if blk is None:
                        return M
                    if blk == "S":
                        return VS.T @ M
                    return M[np.ix_(blk[0], blk[1])]
                if blk == "S" and VS.shape[1] == 0:
                    continue
                TA, TB, T = cut(t["TA"]), cut(t["TB"]), cut(t["T"])
                terms[bname] = {"den": float(np.sum(TA * TB)), "nA": float(np.sum(TA * TA)), "nB": float(np.sum(TB * TB))}
                for arm in list(BASE) + controls["pseudo"] + controls["derange"] + ["abl_exposure_measured",
                                                                                     "abl_strength_measured"] + random_arms:
                    M = cut(preds[f"{part}/{r['key']}/{arm}"])
                    terms[bname][arm] = 2 * float(np.sum(M * T)) - float(np.sum(M * M))
            guides = sc["guide"][j]
            rows.append(dict(r, terms=terms, guides=guides.tolist(), scoring_index=j.tolist()))

        def rf(rs, b, arm):
            rs = [x for x in rs if b in x["terms"]]
            den = sum(x["terms"][b]["den"] for x in rs)
            return sum(x["terms"][b][arm] for x in rs) / den if rs else float("nan")

        def summary(rs):
            out = {}
            for b in ("all", "ifn", "complement", "along_S"):
                out[b] = {arm: rf(rs, b, arm) for arm in BASE}
                out[b]["pseudo"] = float(np.mean([rf(rs, b, a) for a in controls["pseudo"]]))
                out[b]["derange"] = float(np.mean([rf(rs, b, a) for a in controls["derange"]]))
                out[b]["remove_exposure"] = rf(rs, b, "abl_exposure_measured")
                out[b]["remove_strength"] = rf(rs, b, "abl_strength_measured")
                out[b]["remove_random_mean"] = float(np.mean([rf(rs, b, a) for a in random_arms]))
                out[b]["share_paired"] = ratio(out[b][PRIMARY], out[b][PAIRED])
                out[b]["reliability"] = (sum(x["terms"][b]["den"] for x in rs if b in x["terms"])
                                         / np.sqrt(sum(x["terms"][b]["nA"] for x in rs if b in x["terms"])
                                                   * sum(x["terms"][b]["nB"] for x in rs if b in x["terms"])))
            return out

        res = {"groups": len(rows), "targets": len({r["target"] for r in rows}), "summary": summary(rows),
               "by_replicate": {rep: summary([r for r in rows if r["replicate"] == rep])
                                for rep in sorted({r["replicate"] for r in rows})}}
        if part == "test":
            clusters = np.array([r["target"] for r in rows])

            def stat(idx):
                rs = [rows[i] for i in idx]
                s = summary(rs)
                return np.array([s["all"][PRIMARY], s["all"][PRIMARY] - s["all"]["pseudo"],
                                 s["all"][PRIMARY] - s["all"]["derange"], s["ifn"]["share_paired"],
                                 s["all"]["remove_random_mean"] - s["all"]["remove_exposure"], s["all"][PAIRED],
                                 s["ifn"][PRIMARY], s["ifn"][PAIRED], s["all"]["share_paired"],
                                 s["ifn"]["remove_random_mean"] - s["ifn"]["remove_exposure"], s["along_S"][PRIMARY],
                                 s["along_S"]["share_paired"], s["ifn"]["transferred_correlation"],
                                 s["ifn"]["replicate"]])
        else:
            members = {}
            for i, r in enumerate(rows):
                g = np.array(r["guides"])
                members[i] = [np.flatnonzero(g == u) for u in sorted(set(g))]
            sc_all = sc

            def stat(_idx, rng=np.random.default_rng(SEED + 1)):
                rs = []
                for i, r in enumerate(rows):
                    pick = rng.choice(len(members[i]), len(members[i]), replace=True)
                    jj = np.array(r["scoring_index"])[np.concatenate([members[i][k] for k in pick])]
                    t = pm.scoring_targets(sc_all["counts"][jj], sc_all["library"][jj], sc_all["y"][jj],
                                           sc_all["part"][jj])
                    rr = {"terms": {"all": {"den": float(np.sum(t["TA"] * t["TB"]))}}}
                    for arm in (PRIMARY, PAIRED):
                        M = preds[f"nt/{r['key']}/{arm}"]
                        rr["terms"]["all"][arm] = 2 * float(np.sum(M * t["T"])) - float(np.sum(M * M))
                    rs.append(rr)
                return np.array([rf(rs, "all", PRIMARY), rf(rs, "all", PAIRED)])
            clusters = np.arange(len(rows))
        labels = (["primary", "primary_minus_pseudo", "primary_minus_derange", "ifn_share_paired",
                   "random_minus_exposure_removal", "paired", "ifn_primary", "ifn_paired", "share_paired",
                   "ifn_random_minus_exposure_removal", "along_S_primary", "along_S_share_paired", "ifn_transferred",
                   "ifn_replicate"] if part == "test" else ["primary", "paired"])
        boot = cluster_bootstrap(clusters, stat, BOOT, SEED)
        point = stat(np.arange(len(rows))) if part == "test" else np.array(
            [res["summary"]["all"][PRIMARY], res["summary"]["all"][PAIRED]])
        res["intervals"] = {}
        for k, lab in enumerate(labels):
            b = boot[:, k][np.isfinite(boot[:, k])]
            res["intervals"][lab] = {"value": float(point[k]),
                                     "ci": [float(np.quantile(b, .025)), float(np.quantile(b, .975))] if len(b) else None}
        ev[part] = res
        print("scored", part, flush=True)
    e, n = ev["test"]["intervals"], ev["nt"]["intervals"]
    ev["verdict"] = {"R1_recovery": e["primary"]["ci"][0] > 0,
                     "R2_beats_pseudo_populations": e["primary_minus_pseudo"]["ci"][0] > 0,
                     "R2_beats_derangements": e["primary_minus_derange"]["ci"][0] > 0,
                     "R3_ifn_share_above_half": e["ifn_share_paired"]["ci"][0] > 0.5,
                     "R4_nontargeting": n["primary"]["ci"][0] > 0,
                     "R5_exposure_targets_matter": e["random_minus_exposure_removal"]["ci"][0] > 0}
    (HERE / "results/evaluation.json").write_text(json.dumps(ev, indent=1, default=float) + "\n")
    print(json.dumps({"verdict": ev["verdict"], "test": ev["test"]["intervals"], "nt": n,
                      "by_replicate": {rep: {b: {k: round(v, 3) for k, v in s[b].items() if k in (PRIMARY, PAIRED, "transferred_correlation", "replicate", "remove_exposure", "remove_random_mean")}
                                             for b in ("all", "ifn")} for rep, s in ev["test"]["by_replicate"].items()}},
                     indent=1, default=float))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
