#!/usr/bin/env python3
"""Program-level recovery in the perturbation test (PLAN_programs.md, analysis A).

Prespecified before computation (freeze_posthoc.py) but after the scoring files
had been opened by evaluate.py for the aggregate endpoint, so not sealed.

Programs are defined without held-out outcomes: MSigDB Hallmark gene sets
(files and SHA-256 in GENESETS) intersected with the frozen gene panel, and, for
the interferon program, the antibodies whose encoding genes belong to the
Hallmark interferon-gamma or interferon-alpha response sets. Blocks of the
within-group gene-protein cross-correlation:

  ifn          interferon genes x interferon antibodies (primary)
  ifn_genes    interferon genes x all antibodies
  complement   all other genes x all other antibodies
  along_S      the RNA score along the primary channel's single supported
               direction (training fit) x all antibodies
  <program>    genes of each Hallmark program with at least 5 panel genes x all antibodies

For every block and arm: the recovered fraction over held-out groups (and
non-targeting cells), the share of the paired closed form's and of the
transferred correlation's recovery, and measurement repeatability, both as the
recovered fraction of an independent replicate (the group's adaptation half, its
own paired cross-correlation) and as the split-half reliability of the scoring
target. Support of a program, from unpaired adaptation RNA only: the mean over
its genes of the squared correlation with the RNA score along S.
Writes results/programs.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import pmethods as pm
from pdata import ENCODING, lognorm, load, seal
from predict import groups

sys.path.append(str(pm.HERE.parent / "recoverability"))
from stats import cluster_bootstrap  # noqa: E402

GENESETS = Path("/home/claude/cbio/rawdata/genesets")
GS_SHA256 = {
    "HALLMARK_INTERFERON_GAMMA_RESPONSE.json": "845aabdb9ff04fb62bbb1bc8a527654c549075852c84f31d64f50d9e43134353",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE.json": "2a3d93f3170bc9866ced1135bdde9870f18294f7ad3ae6cb3732dc3680483455",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB.json": "ade1ab9b67e047ceb70f8b8a06c65c6631e17f06b10c1b0283f962be848d2954",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.json": "b4f4872e065400c48c6b16f191c59078a3b5bc549c740b3dbbf89a016bf126dd",
    "HALLMARK_G2M_CHECKPOINT.json": "a246146297b9fd928560d727724311e4cf4f8ec76158acb2704606fe1a5950b3",
    "HALLMARK_E2F_TARGETS.json": "8007be6789c45bd70192940322d69b137bddfe8e2a7b2f93ef33773ed981f1c1",
    "HALLMARK_HYPOXIA.json": "ff32259a58f8ffc8266bbc783f6f40bfb18b9dc3719df00fce0e8f8cfc79e29d",
    "HALLMARK_MYC_TARGETS_V1.json": "445c9cde22413cb1695127198ab21a73ed075488dd60b78414095d1253c4103c",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION.json": "c4836d2a970fdd1ce70afa2b41407395f0e77099677f1c25c222e050afb7b246"}
IFN_SETS = ("HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_INTERFERON_ALPHA_RESPONSE")
MIN_GENES = 5
ARMS = ("pf_within_measured", "paired_closed_form", "transferred_correlation", "reference_regression", "benchmark",
        "replicate")
BOOT, SEED = 2000, 20261016


def gene_sets():
    out = {}
    for name, digest in GS_SHA256.items():
        path = GENESETS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        d = json.loads(path.read_text())
        key = next(iter(d))
        out[key] = set(d[key]["geneSymbols"])
    return out


def define_blocks(record, sets, v):
    genes, proteins = record["genes"], record["proteins"]
    ifn = set().union(*(sets[s] for s in IFN_SETS))
    rows = [i for i, g in enumerate(genes) if g in ifn]
    cols = [j for j, p in enumerate(proteins) if any(g in ifn for g in ENCODING[p])]
    other_r = [i for i in range(len(genes)) if i not in rows]
    other_c = [j for j in range(len(proteins)) if j not in cols]
    allc = list(range(len(proteins)))
    blocks = {"ifn": (rows, cols), "ifn_genes": (rows, allc), "complement": (other_r, other_c), "along_S": ("S", allc)}
    for name in sorted(sets):
        if name in IFN_SETS:
            continue
        r = [i for i, g in enumerate(genes) if g in sets[name]]
        if len(r) >= MIN_GENES:
            blocks[name.replace("HALLMARK_", "").lower()] = (r, allc)
    defs = {k: {"genes": [genes[i] for i in r] if r != "S" else "supported direction",
                "proteins": [proteins[j] for j in c]} for k, (r, c) in blocks.items()}
    return blocks, defs


def ratio(a, b):
    return a / b if b > 0 else float("nan")


def restrict(M, block, v):
    rows, cols = block
    if rows == "S":
        return (v @ M)[None, cols]
    return M[np.ix_(rows, cols)]


def main():
    record = seal()
    sets = gene_sets()
    z = np.load(pm.HERE / "results/channels.npz")
    VS = z["pf_within/VS"]
    assert VS.shape[1] == 1, "the primary channel was frozen with one supported direction"
    v = VS[:, 0]
    blocks, defs = define_blocks(record, sets, v)
    preds = np.load(pm.HERE / "results/predictions.npz")
    out = {"gene_sets_sha256": GS_SHA256, "blocks": defs}
    for part in ("heldout", "nt"):
        ad, sc = load(f"{part}_adaptation"), load(f"{part}_scoring")
        sk = pm.keys_of(sc)
        rows = []
        for key, idx in groups(ad):
            j = np.flatnonzero(sk == key)
            t = pm.scoring_targets(sc["counts"][j], sc["library"][j], sc["y"][j], sc["part"][j])
            x_ad = lognorm(ad["counts"][idx], ad["library"][idx])
            C = {arm: preds[f"{part}/{key}/{arm}"] for arm in ARMS if arm != "replicate"}
            xs, _ = pm.standardize(x_ad)
            ys, _ = pm.standardize(ad["y"][idx])
            C["replicate"] = xs.T @ ys / len(idx)
            Rx = pm.shrink(xs.T @ xs / len(idx))
            score_var = float(v @ Rx @ v)
            corr2 = (Rx @ v) ** 2 / (np.diag(Rx) * score_var)
            row = {"key": key, "target": key.split("|")[0], "condition": key.split("|")[1], "terms": {}, "support": {}}
            for b, blk in blocks.items():
                TA, TB, T = (restrict(t[k], blk, v) for k in ("TA", "TB", "T"))
                row["terms"][b] = {"den": float(np.sum(TA * TB)), "nA": float(np.sum(TA * TA)),
                                   "nB": float(np.sum(TB * TB))}
                for arm, M in C.items():
                    Mb = restrict(M, blk, v)
                    row["terms"][b][arm] = 2 * float(np.sum(Mb * T)) - float(np.sum(Mb * Mb))
                if blk[0] != "S":
                    row["support"][b] = float(np.mean(corr2[blk[0]]))
            rows.append(row)
        print(part, len(rows), "groups", flush=True)

        def rf(rs, b, arm):
            return sum(r["terms"][b][arm] for r in rs) / sum(r["terms"][b]["den"] for r in rs)

        def reliability(rs, b):
            return sum(r["terms"][b]["den"] for r in rs) / np.sqrt(sum(r["terms"][b]["nA"] for r in rs)
                                                                    * sum(r["terms"][b]["nB"] for r in rs))

        res = {}
        for b in blocks:
            res[b] = {"rf": {arm: rf(rows, b, arm) for arm in ARMS}, "reliability": reliability(rows, b),
                      "support": float(np.median([r["support"][b] for r in rows])) if b in rows[0]["support"] else None}
            res[b]["share_paired"] = ratio(res[b]["rf"]["pf_within_measured"], res[b]["rf"]["paired_closed_form"])
            res[b]["share_transferred"] = ratio(res[b]["rf"]["pf_within_measured"],
                                                res[b]["rf"]["transferred_correlation"])
            res[b]["by_condition"] = {c: {arm: rf([r for r in rows if r["condition"] == c], b, arm)
                                          for arm in ("pf_within_measured", "paired_closed_form",
                                                      "transferred_correlation", "replicate")}
                                      for c in ("control", "ifng", "coculture")
                                      if any(r["condition"] == c for r in rows)}
        if part == "heldout":
            targets = np.array([r["target"] for r in rows])
            labels, stats = [], []
            for b in blocks:
                for arm in ARMS:
                    labels.append(f"{b}/rf/{arm}")
                labels += [f"{b}/share_paired", f"{b}/share_transferred", f"{b}/reliability"]
            labels += ["ifn_minus_complement/pf", "along_S_minus_complement/pf"]

            def stat(idx):
                rs = [rows[i] for i in idx]
                vals = []
                for b in blocks:
                    r_ = {arm: rf(rs, b, arm) for arm in ARMS}
                    vals += [r_[arm] for arm in ARMS]
                    vals += [ratio(r_["pf_within_measured"], r_["paired_closed_form"]),
                             ratio(r_["pf_within_measured"], r_["transferred_correlation"]), reliability(rs, b)]
                vals += [rf(rs, "ifn", "pf_within_measured") - rf(rs, "complement", "pf_within_measured"),
                         rf(rs, "along_S", "pf_within_measured") - rf(rs, "complement", "pf_within_measured")]
                return np.array(vals)

            boot = cluster_bootstrap(targets, stat, BOOT, SEED)
            point = stat(np.arange(len(rows)))
            ci = {}
            for k, lab in enumerate(labels):
                bk = boot[:, k][np.isfinite(boot[:, k])]
                ci[lab] = {"value": float(point[k]),
                           "ci": [float(np.quantile(bk, .025)), float(np.quantile(bk, .975))] if len(bk) else [np.nan] * 2}
            res["intervals"] = ci
            progs = [b for b in blocks if b not in ("ifn", "complement", "along_S", "ifn_genes")]
            sup = np.array([res[b]["support"] for b in progs])
            rec = np.array([res[b]["rf"]["pf_within_measured"] for b in progs])
            ranks = lambda a: np.argsort(np.argsort(a))
            res["support_vs_recovery"] = {"programs": progs, "support": sup.tolist(), "recovery": rec.tolist(),
                                          "spearman": float(np.corrcoef(ranks(sup), ranks(rec))[0, 1])}
            e = ci
            res["verdict"] = {
                "P1_ifn_share_above_half": e["ifn/share_paired"]["ci"][0] > 0.5,
                "P2_along_S_share_above_half": e["along_S/share_paired"]["ci"][0] > 0.5,
                "P3_ifn_above_complement": e["ifn_minus_complement/pf"]["ci"][0] > 0}
        out[part] = res
    (pm.HERE / "results/programs.json").write_text(json.dumps(out, indent=1) + "\n")
    h = out["heldout"]
    print(json.dumps({"verdict": h["verdict"], "support_vs_recovery": h["support_vs_recovery"],
                      "blocks": {b: {"rf": {k: round(x, 3) for k, x in h[b]["rf"].items()},
                                     "reliability": round(h[b]["reliability"], 3), "support": h[b]["support"],
                                     "share_paired": round(h[b]["share_paired"], 3)} for b in blocks}}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
