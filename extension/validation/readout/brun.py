#!/usr/bin/env python3
"""Biological readout of the validation test (PLAN.md in this folder): do the paired cells that the atlas saves call
within-cell-type gene-protein associations that replicate in independent held-out cells?

    python brun.py smoke   # synthetic values, real design (two folds, budgets 25-100, two draws); integrity check
                           # against the validation's smoke predictions (results_smoke/)
    python brun.py run     # checks results/freeze.json; regenerates per-draw predictions with the validation's frozen
                           # code path, checks them against its hashed means, scores E1-E3 (results/readout.json)

Predictions come from vrun.py (frozen by ../results/freeze.json), called exactly as vrun.predict calls it; the
paired-only methods, the references and the draws are the validation's. Only the scoring is new.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
VAL = HERE.parent
EXT = VAL.parent
sys.path.insert(0, str(VAL))
import vrun  # noqa: E402  (frozen in ../results/freeze.json)

drun, pc, es, rs, grun = vrun.drun, vrun.pc, vrun.es, vrun.rs, vrun.grun
CONDS, PAIRED_ONLY, PO_ARMS = vrun.CONDS, vrun.PAIRED_ONLY, vrun.PO_ARMS
SEED, MIN_PERCOND = vrun.SEED, vrun.MIN_PERCOND
REFS = ("atlas", "other", "own")
ARMS = PO_ARMS + REFS
BUDGETS = (25, 50, 100, 150, 200, 400, 800)
DRAWS = 5
Z_CRIT = 2.5758293035489004          # two-sided p = 0.01
MIN_SUB = 20                         # held-out cells of the type in each sub-half for a scored unit
TOP_K = 100
BOOT, BOOT_SEED = 2000, 20261204
HYP = {"B1": ("E1", (50, 100, 150)), "B2": ("E2", (50, 100, 150)), "B3": ("E3", (100, 150))}
ENDPOINTS = ("E1", "E2", "E3", "E1_cognate")
TOL_P, TOL_MSQ = 1e-5, 1e-6
COGNATE = [("anti-human-CD11b-TotalSeqC", "ITGAM"), ("anti-human-CD8-TotalSeqC", "CD8A"),
           ("anti-human-CD14-TotalSeqC", "CD14"), ("anti-human-CD16-TotalSeqC", "FCGR3A"),
           ("anti-human-CD31-TotalSeqC", "PECAM1"), ("anti-human-CD161-TotalSeqC", "KLRB1"),
           ("anti-human-KLRG1-(MAFA)-TotalSeqC", "KLRG1"), ("anti-human-HLA-DR-TotalSeqC", "HLA-DRA"),
           ("anti-human-CD314-(NKG2D)-TotalSeqC", "KLRK1"), ("anti-human-CD79b-(IgBeta)-TotalSeqC", "CD79B"),
           ("anti-human-CD122-(IL-2Rbeta)-TotalSeqC", "IL2RB"), ("anti-human-CD83-TotalSeqC", "CD83"),
           ("anti-human-CD124-(IL-4Ralpha)-TotalSeqC", "IL4R"), ("anti-human-CD127-(IL-7Ralpha)-TotalSeqC", "IL7R"),
           ("anti-human-CD94-TotalSeqC", "KLRD1"), ("anti-human-CD85j-(ILT2)-TotalSeqC", "LILRB1")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ truth and per-draw scores

def cognate_index(new, genes, proteins, strict=True):
    """The panel antibodies whose target gene is a panel gene, as (gene index, protein index) arrays. The synthetic
    smoke panel is chosen from synthetic values, so its genes differ; it uses the first 16 (gene, protein) pairs."""
    t_of = {str(n): str(t) for n, t in zip(new["y_names"], new["y_target"])}
    gi = {g: i for i, g in enumerate(genes)}
    found = [(p, t_of[p]) for p in proteins if t_of.get(p, "") in gi]
    if strict:
        assert found == COGNATE, f"cognate pairs differ from the plan: {found}"
    elif not found:
        return np.arange(16), np.arange(16)
    return (np.array([gi[t] for _, t in found]), np.array([proteins.index(p) for p, _ in found]))


def truth(new, fd, half, gidx, yidx):
    """Held-out correlations of the held-out sample per cell type (vrun.heldout) and its reproducible associations."""
    st = vrun.heldout(new, fd, half, gidx, yidx)
    r = np.flatnonzero(new["sample"] == fd["heldout"])
    cond, hv = new["cond"][r], half[r]
    out = {}
    for c in CONDS:
        nA, nB = int(np.sum((cond == c) & (hv == 0))), int(np.sum((cond == c) & (hv == 1)))
        T = st[c]
        rec = {"T": T["T"], "TA": T["TA"], "TB": T["TB"], "nA": nA, "nB": nB, "scored": min(nA, nB) >= MIN_SUB}
        if rec["scored"]:
            zA = np.arctanh(np.clip(T["TA"], -1 + 1e-12, 1 - 1e-12)) * np.sqrt(nA - 3)
            zB = np.arctanh(np.clip(T["TB"], -1 + 1e-12, 1 - 1e-12)) * np.sqrt(nB - 3)
            rep = (np.abs(zA) >= Z_CRIT) & (np.abs(zB) >= Z_CRIT) & (np.sign(T["TA"]) == np.sign(T["TB"]))
        else:
            rep = np.zeros(T["T"].shape, bool)
        rec["rep"], rec["dir"] = rep, np.sign(T["TA"])
        out[c] = rec
    return out


def unit_scores(Q, tr, cog):
    """E1 credit, E2 expected hits among the TOP_K largest |Q|, E3 numerator and E1 credit on cognate pairs."""
    s = np.sign(Q)
    credit = np.where(s == 0, 0.5, (s == tr["dir"]).astype(float)) * tr["rep"]
    a, cr = np.abs(Q).ravel(), credit.ravel()
    v = np.partition(a, a.size - TOP_K)[a.size - TOP_K]
    gt, eq = a > v, a == v
    e2 = float(cr[gt].sum()) + (TOP_K - int(gt.sum())) * float(cr[eq].mean())
    gi, pj = cog
    qc = Q[gi, pj]
    e3 = 2.0 * float(qc @ tr["T"][gi, pj]) - float(qc @ qc)
    return np.array([float(credit.sum()), e2, e3, float(credit[gi, pj].sum())])


def unit_constants(tr, cog):
    gi, pj = cog
    n_rep = float(tr["rep"].sum())
    return {"den": np.array([n_rep, float(TOP_K), float(tr["TA"][gi, pj] @ tr["TB"][gi, pj]),
                             float(tr["rep"][gi, pj].sum())]),
            "random_e2": TOP_K * n_rep / tr["rep"].size * 0.5, "perfect_e2": float(min(TOP_K, n_rep)),
            "pairs": tr["rep"].size}


# ------------------------------------------------------------------ run

def run(synthetic=False, fold_ids=None, budgets=BUDGETS, draws=DRAWS, stored_dir=VAL / "results", out_dir=HERE / "results",
        log=print):
    new, atlas = vrun.load_new(synthetic), vrun.load_atlas(synthetic)
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    proteins = [str(new["y_names"][j]) for j in yidx]
    cog = cognate_index(new, genes, proteins, strict=not synthetic)
    A_pool, A, n_atlas = vrun.atlas_context(atlas, part_a, gidx_a, yidx_a)
    stored = np.load(stored_dir / "predictions.npz")
    stored_budgets = set(json.loads((stored_dir / "predictions_info.json").read_text())["folds"]["0"]["budgets"])
    log(f"atlas reference: {n_atlas} cells; cognate pairs {len(cog[0])}")
    fds = [fd for fd in vrun.folds(new) if fold_ids is None or fd["fold"] in fold_ids]
    nF, nA, nB, nC = len(fds), len(ARMS), len(budgets), len(CONDS)
    num = np.zeros((nF, nA, nB, 4))
    den = np.zeros((nF, 4))
    num_t = np.zeros((nF, nA, nB, nC, 2))
    den_t = np.zeros((nF, nC, 2))
    base = np.zeros((nF, 2))
    units = {c: 0 for c in CONDS}
    reps = {c: 0 for c in CONDS}
    check = {"max_abs_P": 0.0, "max_rel_msq": 0.0, "compared": 0}
    other_cache = {}
    t0 = time.time()
    for fi, fd in enumerate(fds):
        f = fd["fold"]
        if fd["pool"] not in other_cache:
            other_cache[fd["pool"]] = vrun.other_context(new, fd, part, gidx, yidx)
        O_pool, O, n_other = other_cache[fd["pool"]]
        W_pool, W, n_own = vrun.own_context(new, fd, part, res, gidx, yidx)
        rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & res)
        xr, _, _ = vrun.rna(new, rres, gidx)
        yr = vrun.prot(new, rres, yidx)
        popr, condr = vrun.popkey(new, rres), new["cond"][rres]
        tr = truth(new, fd, half, gidx, yidx)
        for ci, c in enumerate(CONDS):
            if tr[c]["scored"]:
                k = unit_constants(tr[c], cog)
                den[fi] += k["den"]
                den_t[fi, ci] += k["den"][:2]
                base[fi] += (k["random_e2"], k["perfect_e2"])
                units[c] += 1
                reps[c] += int(k["den"][0])
        for bi, B in enumerate(budgets):
            if B > len(rres):
                continue
            mean_q, msq = {}, {}
            for dr in range(draws):
                rng = np.random.default_rng([SEED, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                x, y, pp, cc = xr[pick], yr[pick], popr[pick], condr[pick]
                preds = {}
                hc = pc.contrasts(x, y, pp)
                if hc is not None:
                    Xh, Yh, ih = hc
                    ch = cc[ih]
                    # exactly as vrun.predict: one generator per method, used by the pooled fit, then per type
                    for a in PAIRED_ONLY:
                        crng = np.random.default_rng([SEED + 1, f, B, dr])
                        Pp = rs.denoise(a, Xh, Yh, A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
                        for c in CONDS:
                            preds[(f"{a}/pooled", c)] = Pp
                            m = ch == c
                            preds[(f"{a}/percond", c)] = (rs.denoise(a, Xh[m], Yh[m], A.Rx, A.Ry, A.n_x, A.n_y,
                                                                     A.Uz, A.blocks, crng)
                                                          if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
                    for arm, ctx in (("atlas", A), ("other", O), ("own", W)):
                        Pe = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
                        for c in CONDS:
                            preds[(arm, c)] = drun.mapped(ctx, Pe, c)
                for (arm, c), Q in preds.items():
                    ai, ci = ARMS.index(arm), CONDS.index(c)
                    if B in stored_budgets:
                        mean_q[(arm, c)] = mean_q.get((arm, c), 0.0) + Q / draws
                        msq[(arm, c)] = msq.get((arm, c), 0.0) + float(np.sum(Q * Q)) / draws
                    if tr[c]["scored"]:
                        sc = unit_scores(Q, tr[c], cog) / draws
                        num[fi, ai, bi] += sc
                        num_t[fi, ai, bi, ci] += sc[:2]
            for (arm, c), Qm in mean_q.items():
                k = f"P/{f}/{arm}/{B}/{c}"
                P = stored[k].astype(np.float64)
                check["max_abs_P"] = max(check["max_abs_P"], float(np.max(np.abs(Qm - P))))
                ms = float(stored[f"msq/{f}/{arm}/{B}/{c}"])
                check["max_rel_msq"] = max(check["max_rel_msq"], abs(msq[(arm, c)] - ms) / max(ms, 1e-300))
                check["compared"] += 1
            if check["max_abs_P"] > TOL_P or check["max_rel_msq"] > TOL_MSQ:
                raise SystemExit(f"integrity check failed at fold {f}, budget {B}: {check}")
        log(f"fold {f} ({fd['heldout']}) done [{time.time() - t0:.0f}s]; integrity {check}")
    return score(fds, budgets, num, den, num_t, den_t, base, units, reps, check, genes, proteins, cog, log)


# ------------------------------------------------------------------ scoring

def pooled(w, num, den):
    return np.einsum("f,fabk->abk", w, num) / np.maximum(w @ den, 1e-300)[None, None, :]


def with_envelope(P):
    """Adds the paired-only envelope (best of the eight methods at each budget, per endpoint) as the last arm."""
    env = np.nanmax(P[:len(PO_ARMS)], axis=0)
    return np.concatenate([P, env[None]], axis=0)


def needed(values, target, budgets):
    return grun.needed(values, target, budgets)


def score(fds, budgets, num, den, num_t, den_t, base, units, reps, check, genes, proteins, cog, log):
    names = list(ARMS) + ["paired_only"]
    ia, ip = names.index("atlas"), names.index("paired_only")
    w1 = np.ones(len(fds))
    P = with_envelope(pooled(w1, num, den))
    donors = sorted({fd["donor"] for fd in fds})
    folds_of = {d: [i for i, fd in enumerate(fds) if fd["donor"] == d] for d in donors}
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(BOOT):
        w = np.zeros(len(fds))
        for d in rng.choice(donors, size=len(donors), replace=True):
            w[folds_of[d]] += 1.0
        boots.append(with_envelope(pooled(w, num, den)))
    boots = np.stack(boots)
    out = {"budgets": list(budgets), "arms": names, "endpoints": list(ENDPOINTS), "folds": len(fds),
           "donors": len(donors), "integrity": check,
           "cognate": [[genes[g], proteins[p]] for g, p in zip(*cog)],
           "units_by_type": units, "reproducible_by_type": reps,
           "pairs_per_unit": len(genes) * len(proteins),
           "random_E2": float(base[:, 0].sum() / den[:, 1].sum()), "perfect_E2": float(base[:, 1].sum() / den[:, 1].sum()),
           "values": {}, "ci": {}, "hypotheses": {}, "differences": {}, "by_type": {}, "needed": {}}
    for k, e in enumerate(ENDPOINTS):
        out["values"][e] = {a: [float(v) for v in P[i, :, k]] for i, a in enumerate(names)}
        out["ci"][e] = {a: [[float(np.percentile(boots[:, i, j, k], 2.5)), float(np.percentile(boots[:, i, j, k], 97.5))]
                            for j in range(len(budgets))] for i, a in enumerate(names)}
    for h, (e, bs) in HYP.items():
        k = ENDPOINTS.index(e)
        rec = {}
        bs = [B for B in bs if B in budgets]       # the smoke run has budgets 25-100 only
        if not bs:
            continue
        for B in bs:
            j = budgets.index(B)
            d = P[ia, j, k] - P[ip, j, k]
            db = boots[:, ia, j, k] - boots[:, ip, j, k]
            rec[str(B)] = {"atlas": float(P[ia, j, k]), "paired_only": float(P[ip, j, k]), "difference": float(d),
                           "ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}
        out["differences"][h] = rec
        out["hypotheses"][h] = bool(all(r["ci"][0] > 0 for r in rec.values()))
    # per cell type (E1, E2)
    for ci, c in enumerate(CONDS):
        Pt = with_envelope(np.einsum("fabk->abk", num_t[:, :, :, ci]) / np.maximum(den_t[:, ci].sum(0), 1e-300))
        out["by_type"][c] = {e: {a: [float(v) for v in Pt[i, :, k]] for i, a in enumerate(names)}
                             for k, e in enumerate(("E1", "E2"))}
    # paired cells needed: E1 midpoint between chance and the best arm; E2 half of the best arm
    for k, e, tgt in ((0, "E1", lambda best: 0.5 + (best - 0.5) / 2), (1, "E2", lambda best: best / 2)):
        best = float(np.nanmax(P[len(PO_ARMS):, :, k]))
        target = tgt(best)
        n = {a: needed(P[i, :, k], target, budgets) for i, a in enumerate(names) if i >= len(PO_ARMS)}
        bn = []
        for b in boots:
            bn.append((needed(b[ip, :, k], target, budgets)[0], needed(b[ia, :, k], target, budgets)[0]))
        ratios = [p / a for p, a in bn]
        out["needed"][e] = {"best": best, "target": target,
                            "cells": {a: {"cells": v[0], "relation": v[1]} for a, v in n.items()},
                            "saving_atlas": {"ratio": n["paired_only"][0] / n["atlas"][0],
                                             "ci": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                                             "numerator_relation": n["paired_only"][1],
                                             "denominator_relation": n["atlas"][1]}}
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    log(json.dumps(out["hypotheses"]))
    for h, rec in out["differences"].items():
        log(f"{h}: " + "; ".join(f"B={B} atlas {r['atlas']:.3f} po {r['paired_only']:.3f} diff {r['difference']:.3f} "
                                 f"({r['ci'][0]:.3f}-{r['ci'][1]:.3f})" for B, r in rec.items()))
    return out


def check_freeze():
    rec = json.loads((HERE / "results" / "freeze.json").read_text())
    for rel, digest in rec["files"].items():
        assert sha(EXT / rel) == digest, f"{rel} changed since the freeze"
    for name, (path, digest) in rec["data"].items():
        assert sha(path) == digest, f"{name} changed since the freeze"
    vrun.check_freeze()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    out = HERE / ("results_smoke" if cmd == "smoke" else "results")
    out.mkdir(exist_ok=True)
    lp = out / f"{cmd}.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(lp, "a") as fh:
            fh.write(line + "\n")

    with threadpool_limits(limits=2):
        if cmd == "smoke":
            r = run(synthetic=True, fold_ids={0, 17}, budgets=(25, 50, 100), draws=2,
                    stored_dir=VAL / "results_smoke", log=log)
            r["synthetic"] = True
            (out / "readout.json").write_text(json.dumps(r, indent=1))
        elif cmd == "run":
            check_freeze()
            r = run(log=log)
            r["synthetic"] = False
            (out / "readout.json").write_text(json.dumps(r, indent=1))
        else:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
