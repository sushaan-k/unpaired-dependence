#!/usr/bin/env python3
"""Two analyses of the validation test asked for in review, specified after the test and its readout had been
scored (PLAN.md in this folder):

A. semi-paired alternatives given the same paired cells, atlas reference, panel and tuning information as the
   estimator (SemiCCA and reference ridge regression in the benchmark's forms; Champollion on the first draw);
B. calibration of the readout: significance threshold, false discovery rate, majority-sign baseline, weighting of
   samples and cell types, intervals of the headline differences and multiplicity.

    python rrun.py smoke              # synthetic values with the real design (folds 0 and 17, budgets 25-100, two
                                      # draws; Champollion on one fold): the whole pipeline, with the integrity check
                                      # against the validation's smoke predictions (results_smoke/)
    python rrun.py timing             # seconds per Champollion fit on paired draws of fold 0 (no held-out cell)
    python rrun.py run --part 0|1     # checks results/freeze.json; every arm and draw of the even or odd folds
    python rrun.py champ --part 0|1   # checks results/freeze.json; Champollion on the first draw of each budget
    python rrun.py score              # results/review.json

Predictions are regenerated with the validation's frozen code path exactly as ../readout/brun.py regenerates them
(same folds, panels, references, draws and seeds) and checked against the validation's hashed means. The
alternatives are the benchmark's implementations, imported unchanged (extension/semipaired: rs.denoise 'semicca',
'ridge_p', 'ridge_u' in the pooled, mapped and per-cell-type forms of run_study.run_fold and grun; Champollion
through champ_tune.run_champ and champ_worker.py with the external test's settings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
VAL = HERE.parent
EXT = VAL.parent
sys.path.insert(0, str(VAL))
sys.path.insert(0, str(VAL / "readout"))
sys.path.insert(0, str(EXT / "semipaired"))
import vrun  # noqa: E402  (frozen: ../results/freeze.json)
import brun  # noqa: E402  (frozen: ../readout/results/freeze.json)
import champ_tune as ct  # noqa: E402  (Champollion's environment and call, as in the external test)

drun, pc, es, rs, grun = vrun.drun, vrun.pc, vrun.es, vrun.rs, vrun.grun
CONDS, PAIRED_ONLY, PO_ARMS = vrun.CONDS, vrun.PAIRED_ONLY, vrun.PO_ARMS
SEED, MIN_PERCOND, TOP_K = vrun.SEED, vrun.MIN_PERCOND, brun.TOP_K
BUDGETS, V_BUDGETS, DRAWS = brun.BUDGETS, vrun.BUDGETS, brun.DRAWS
METHODS = ("semicca", "ridge_p", "ridge_u")
FORMS = ("pooled", "mapped", "percond")
COMP = tuple(f"{m}/{f}" for m in METHODS for f in FORMS)
ARMS = tuple(brun.ARMS) + COMP                 # eight paired-only arms, atlas, other, own, then the alternatives
FAMILIES = {"paired_only": tuple(PO_ARMS), "semicca": COMP[:3], "regression": COMP[3:]}
CHAMP_CHOICE = EXT / "semipaired" / "logs" / "champ_choice.json"
CHAMP_BUDGETS = V_BUDGETS
CHAMP_DRAWS = 1
CHAMP_POOL, CHAMP_POOL_SEED = 2000, 20261042   # champ_tune.condition_pools defaults
TRUTHS = ("p01", "p05", "p001", "fdr05_unit", "fdr05_all")
ZCRIT = {"p01": brun.Z_CRIT, "p05": float(norm.isf(0.025)), "p001": float(norm.isf(0.0005))}
FDR_Q = 0.05
WEIGHTINGS = ("pooled", "unit", "type", "sample")
BOOT, BOOT_SEED = brun.BOOT, brun.BOOT_SEED   # the readout's resamples, reproduced
PBOOT, PBOOT_SEED = 20000, 20261206           # one-sided P-values and simultaneous intervals
CBOOT, CBOOT_SEED = 2000, 20261207            # the alternatives and Champollion
TOL_P, TOL_MSQ = brun.TOL_P, brun.TOL_MSQ
PLANNED = [("B1", "E1", B) for B in (50, 100, 150)] + [("B2", "E2", B) for B in (50, 100, 150)] \
    + [("B3", "E3", B) for B in (100, 150)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ setup

def setup(synthetic=False):
    new, atlas = vrun.load_new(synthetic), vrun.load_atlas(synthetic)
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    proteins = [str(new["y_names"][j]) for j in yidx]
    cog = brun.cognate_index(new, genes, proteins, strict=not synthetic)
    A_pool, A, n_atlas = vrun.atlas_context(atlas, part_a, gidx_a, yidx_a)
    S = SimpleNamespace(new=new, atlas=atlas, part=part, half=half, res=res, part_a=part_a, genes=genes,
                        gidx=gidx, gidx_a=gidx_a, yidx=yidx, yidx_a=yidx_a, proteins=proteins, cog=cog,
                        A_pool=A_pool, A=A, n_atlas=n_atlas, synthetic=synthetic)
    S.pools = champ_pools(S)
    # memory: keep only the panel genes' counts (vrun.rna reads the same values; library sizes are stored apart)
    for k in ("x_data", "x_indices", "x_indptr"):
        new.pop(k, None)
    new["x_counts"] = new["x_counts"][:, gidx].tocsr()
    S.gidx = np.arange(len(gidx))
    S.atlas = None
    return S


def champ_pools(S):
    """Per cell type, up to 2,000 RNA-half and 2,000 protein-half atlas cells of the atlas pool's populations,
    centred on their population means, in pooled-SD units (champ_tune.condition_pools, as in development)."""
    rows = np.arange(len(S.atlas["cell"]))
    x, _, _ = vrun.rna(S.atlas, rows, S.gidx_a)
    y = vrun.prot(S.atlas, rows, S.yidx_a)
    pop = vrun.popkey(S.atlas, rows)
    pools = ct.condition_pools(pop, S.part_a, x, y, S.A_pool, SimpleNamespace(conds=list(CONDS)), n=CHAMP_POOL,
                               seed=CHAMP_POOL_SEED)
    return {c: (pools[c][0], pools[c][1]) for c in CONDS}


def champ_settings():
    ch = json.loads(CHAMP_CHOICE.read_text())
    c_by = {int(k): float(v) for k, v in ch["c_by_budget"].items()}
    c_by[800] = c_by[400]                     # 400 took the value chosen at 200; 800 takes the same
    return float(ch["epsilon"]), int(ch["max_iter"]), c_by


# ------------------------------------------------------------------ truth sets

def replicability(tr):
    """Per pair: m = the smaller |z| of the two sub-halves if their signs agree (else 0), and the replicability
    P-value 2 (1 - Phi(m)) (1 if the signs disagree): both sub-halves significant at level P with the same sign
    exactly when that P-value is at most P."""
    zA = np.arctanh(np.clip(tr["TA"], -1 + 1e-12, 1 - 1e-12)) * np.sqrt(tr["nA"] - 3)
    zB = np.arctanh(np.clip(tr["TB"], -1 + 1e-12, 1 - 1e-12)) * np.sqrt(tr["nB"] - 3)
    same = np.sign(tr["TA"]) == np.sign(tr["TB"])
    m = np.where(same, np.minimum(np.abs(zA), np.abs(zB)), 0.0)
    return m, np.where(same, 2 * norm.sf(m), 1.0)


def bh_threshold(p, q):
    """Largest P-value that Benjamini-Hochberg rejects at level q among the P-values p (-1 if none)."""
    ps = np.sort(np.asarray(p).ravel())
    ok = np.flatnonzero(ps <= q * np.arange(1, ps.size + 1) / ps.size)
    return float(ps[ok[-1]]) if ok.size else -1.0


def truth_arrays(S, fd, gthr):
    tr = brun.truth(S.new, fd, S.half, S.gidx, S.yidx)
    nC, nS = len(CONDS), len(TRUTHS)
    npairs = len(S.gidx) * len(S.yidx)
    reps = np.zeros((nC, nS, npairs), bool)
    den = np.zeros((nC, nS, 3))            # reproducible associations; positive; negative
    den3 = np.zeros((nC, 2))               # <TA, TB> over cognate entries; reproducible cognate pairs (primary)
    scored = np.array([bool(tr[c]["scored"]) for c in CONDS])
    for ci, c in enumerate(CONDS):
        if not scored[ci]:
            continue
        m, p = replicability(tr[c])
        sets = {"p01": m >= ZCRIT["p01"], "p05": m >= ZCRIT["p05"], "p001": m >= ZCRIT["p001"],
                "fdr05_unit": p <= bh_threshold(p, FDR_Q), "fdr05_all": p <= gthr}
        assert np.array_equal(sets["p01"], tr[c]["rep"]), "primary truth differs from the readout's"
        d = tr[c]["dir"]
        for si, s in enumerate(TRUTHS):
            r = sets[s]
            reps[ci, si] = r.ravel()
            den[ci, si] = (r.sum(), (r & (d > 0)).sum(), (r & (d < 0)).sum())
        den3[ci] = brun.unit_constants(tr[c], S.cog)["den"][2:]
    return tr, reps, den, den3, scored


def global_threshold(S, fds, log=print):
    """Benjamini-Hochberg over every pair of every scored unit of every fold (the fdr05_all truth set)."""
    ps = []
    for fd in fds:
        tr = brun.truth(S.new, fd, S.half, S.gidx, S.yidx)
        for c in CONDS:
            if tr[c]["scored"]:
                ps.append(replicability(tr[c])[1].ravel())
    p = np.concatenate(ps)
    t = bh_threshold(p, FDR_Q)
    log(f"global BH threshold at q={FDR_Q}: P <= {t:.3g} ({int(np.sum(p <= t))} of {p.size} pairs)")
    return t


# ------------------------------------------------------------------ predictions and per-draw scores

def predictions(S, x, y, pp, cc, O, W, f, B, dr):
    """Every arm's per-cell-type prediction from one paired draw."""
    A = S.A
    preds = {}
    hc = pc.contrasts(x, y, pp)
    if hc is None:
        return preds
    Xh, Yh, ih = hc
    ch = cc[ih]
    # the validation's arms, exactly as vrun.predict and brun.run: one generator per method, pooled fit then per type
    for a in PAIRED_ONLY:
        crng = np.random.default_rng([SEED + 1, f, B, dr])
        Pp = rs.denoise(a, Xh, Yh, A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
        for c in CONDS:
            preds[(f"{a}/pooled", c)] = Pp
            m = ch == c
            preds[(f"{a}/percond", c)] = (rs.denoise(a, Xh[m], Yh[m], A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
                                          if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
    for arm, ctx in (("atlas", A), ("other", O), ("own", W)):
        Pe = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
        for c in CONDS:
            preds[(arm, c)] = drun.mapped(ctx, Pe, c)
    # the alternatives in the benchmark's three forms (run_study.run_fold, grun.predict): pooled, through the
    # atlas's condition map, and refitted on each cell type's contrasts with the atlas's summaries of that type
    for meth in METHODS:
        crng = np.random.default_rng([SEED + 1, f, B, dr])
        Pp = rs.denoise(meth, Xh, Yh, A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
        for c in CONDS:
            preds[(f"{meth}/pooled", c)] = Pp
            preds[(f"{meth}/mapped", c)] = drun.mapped(A, Pp, c)
        for c in CONDS:
            m = ch == c
            d = A.cond[c]
            preds[(f"{meth}/percond", c)] = (rs.denoise(meth, Xh[m], Yh[m], d["Rx"], d["Ry"], d["n_x"], d["n_y"],
                                                        d["Uz"], A.blocks, crng)
                                             if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
    return preds


def unit_scores(Q, tr, reps, cog):
    """brun.unit_scores for every truth set: E1 credit and E2 expected hits per truth set; E3 numerator and E1
    credit on cognate pairs (primary truth)."""
    s = np.sign(Q)
    agree = np.where(s == 0, 0.5, (s == tr["dir"]).astype(float))
    a, ag = np.abs(Q).ravel(), agree.ravel()
    v = np.partition(a, a.size - TOP_K)[a.size - TOP_K]
    gt, eq = a > v, a == v
    fill = TOP_K - int(gt.sum())
    e12 = np.zeros((len(reps), 2))
    for si, r in enumerate(reps):
        cr = ag * r
        e12[si] = (float(cr.sum()), float(cr[gt].sum()) + fill * float(cr[eq].mean()))
    gi, pj = cog
    qc = Q[gi, pj]
    e3 = 2.0 * float(qc @ tr["T"][gi, pj]) - float(qc @ qc)
    cog_credit = float((agree * reps[0].reshape(Q.shape))[gi, pj].sum())
    return e12, np.array([e3, cog_credit])


def fold_inputs(S, fd, O_cache):
    new = S.new
    if fd["pool"] not in O_cache:
        O_cache[fd["pool"]] = vrun.other_context(new, fd, S.part, S.gidx, S.yidx)
    W = vrun.own_context(new, fd, S.part, S.res, S.gidx, S.yidx)[1]
    rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & S.res)
    xr, _, _ = vrun.rna(new, rres, S.gidx)
    yr = vrun.prot(new, rres, S.yidx)
    return O_cache[fd["pool"]][1], W, rres, xr, yr, vrun.popkey(new, rres), new["cond"][rres]


def fold_run(S, fd, draws, budgets, stored, stored_budgets, gthr, O_cache, log):
    f = fd["fold"]
    t0 = time.time()
    O, W, rres, xr, yr, popr, condr = fold_inputs(S, fd, O_cache)
    tr, reps, den, den3, scored = truth_arrays(S, fd, gthr)
    nC, nA, nB, nS = len(CONDS), len(ARMS), len(budgets), len(TRUTHS)
    u_num = np.zeros((nC, nA, nB, nS, 2))
    u_e3 = np.zeros((nC, nA, nB, 2))
    rf_num = np.full((nA, nB), np.nan)
    rf_den = sum(float(np.sum(tr[c]["TA"] * tr[c]["TB"])) for c in CONDS)
    check = {"max_abs_P": 0.0, "max_rel_msq": 0.0, "compared": 0}
    for bi, B in enumerate(budgets):
        if B > len(rres):
            continue
        sQ, sq = {}, {}
        for dr in range(draws):
            rng = np.random.default_rng([SEED, f, B, dr])
            pick = rng.choice(len(rres), size=B, replace=False)
            preds = predictions(S, xr[pick], yr[pick], popr[pick], condr[pick], O, W, f, B, dr)
            for (arm, c), Q in preds.items():
                ai, ci = ARMS.index(arm), CONDS.index(c)
                sQ[(arm, c)] = sQ.get((arm, c), 0.0) + Q / draws
                sq[(arm, c)] = sq.get((arm, c), 0.0) + float(np.sum(Q * Q)) / draws
                if scored[ci]:
                    e12, e3 = unit_scores(Q, tr[c], reps[ci], S.cog)
                    u_num[ci, ai, bi] += e12 / draws
                    u_e3[ci, ai, bi] += e3 / draws
        if B in stored_budgets:
            for arm in brun.ARMS:
                for c in CONDS:
                    k = f"{f}/{arm}/{B}/{c}"
                    P = stored[f"P/{k}"].astype(np.float64)
                    check["max_abs_P"] = max(check["max_abs_P"], float(np.max(np.abs(sQ[(arm, c)] - P))))
                    ms = float(stored[f"msq/{k}"])
                    check["max_rel_msq"] = max(check["max_rel_msq"], abs(sq[(arm, c)] - ms) / max(ms, 1e-300))
                    check["compared"] += 1
            if check["max_abs_P"] > TOL_P or check["max_rel_msq"] > TOL_MSQ:
                raise SystemExit(f"integrity check failed at fold {f}, budget {B}: {check}")
        # recovered-fraction terms of vrun.fold_terms, with each mean prediction rounded to float32 as stored
        for ai, arm in enumerate(ARMS):
            terms = [2 * float(np.sum(np.asarray(sQ[(arm, c)], np.float32) * tr[c]["T"])) - float(sq[(arm, c)])
                     for c in CONDS if (arm, c) in sQ]
            if terms:
                rf_num[ai, bi] = float(sum(terms))
    log(f"fold {f} ({fd['heldout']}) done [{time.time() - t0:.0f}s]; integrity {check}")
    return {"fold": np.array(f), "donor": np.array(str(fd["donor"])), "heldout": np.array(str(fd["heldout"])),
            "budgets": np.array(budgets), "arms": np.array(ARMS), "truths": np.array(TRUTHS),
            "u_num": u_num, "u_e3": u_e3, "u_den": den, "e3_den": den3, "scored": scored, "rf_num": rf_num,
            "rf_den": np.array(rf_den), "gthr": np.array(gthr), "seconds": np.array(time.time() - t0),
            "check": np.array([check["max_abs_P"], check["max_rel_msq"], check["compared"]])}


def champ_fold(S, fd, pools, draws, budgets, gthr, settings, tag, log):
    """Champollion fitted on the paired contrasts of each draw (lasso weight c_B / sqrt(rows)); transport between
    the atlas's RNA and protein cells of each cell type; the plan's cross-correlation is the prediction. The
    estimator with the atlas on the same draw is scored alongside it."""
    f = fd["fold"]
    eps, iters, c_by = settings
    t0 = time.time()
    new, A = S.new, S.A
    rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & S.res)
    xr, _, _ = vrun.rna(new, rres, S.gidx)
    yr = vrun.prot(new, rres, S.yidx)
    popr = vrun.popkey(new, rres)
    tr, reps, den, den3, scored = truth_arrays(S, fd, gthr)
    arms = ("champollion", "atlas_draw")
    nC, nB, nS = len(CONDS), len(budgets), len(TRUTHS)
    u_num = np.zeros((nC, 2, nB, nS, 2))
    u_e3 = np.zeros((nC, 2, nB, 2))
    rf_num = np.full((2, nB), np.nan)
    rf_den = sum(float(np.sum(tr[c]["TA"] * tr[c]["TB"])) for c in CONDS)
    fit_s = []
    for bi, B in enumerate(budgets):
        if B > len(rres):
            continue
        acc = np.zeros(2)
        for dr in range(draws):
            rng = np.random.default_rng([SEED, f, B, dr])
            pick = rng.choice(len(rres), size=B, replace=False)
            Xh, Yh, _ = pc.contrasts(xr[pick], yr[pick], popr[pick])
            Cs, secs = ct.run_champ(Xh, Yh, pools, list(CONDS), eps, c_by[B] / np.sqrt(len(Xh)), iters, dr, tag)
            fit_s.append(secs)
            Pe = es.two_sided_js(Xh, Yh, A.Uz, A.Ry, A.blocks)[0]
            for ai, preds in enumerate(({c: Cs[i] for i, c in enumerate(CONDS)},
                                        {c: drun.mapped(A, Pe, c) for c in CONDS})):
                for ci, c in enumerate(CONDS):
                    Q = preds[c]
                    acc[ai] += (2 * float(np.sum(Q * tr[c]["T"])) - float(np.sum(Q * Q))) / draws
                    if scored[ci]:
                        e12, e3 = unit_scores(Q, tr[c], reps[ci], S.cog)
                        u_num[ci, ai, bi] += e12 / draws
                        u_e3[ci, ai, bi] += e3 / draws
        rf_num[:, bi] = acc
    log(f"champollion fold {f} ({fd['heldout']}) done [{time.time() - t0:.0f}s; fits {np.mean(fit_s):.1f}s]")
    return {"fold": np.array(f), "donor": np.array(str(fd["donor"])), "budgets": np.array(budgets),
            "arms": np.array(arms), "truths": np.array(TRUTHS), "u_num": u_num, "u_e3": u_e3, "u_den": den,
            "e3_den": den3, "scored": scored, "rf_num": rf_num, "rf_den": np.array(rf_den), "gthr": np.array(gthr),
            "fit_seconds": np.array(fit_s), "seconds": np.array(time.time() - t0)}


# ------------------------------------------------------------------ runs

def run(S, fds, all_fds, draws, budgets, stored_dir, out_dir, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = np.load(stored_dir / "predictions.npz")
    stored_budgets = set(json.loads((stored_dir / "predictions_info.json").read_text())["folds"]["0"]["budgets"])
    gthr = global_threshold(S, all_fds, log)
    O_cache = {}
    for fd in fds:
        path = out_dir / f"fold{fd['fold']}.npz"
        if path.exists():
            continue
        rec = fold_run(S, fd, draws, budgets, stored, stored_budgets, gthr, O_cache, log)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **rec)
        tmp.rename(path)


def champ(S, fds, all_fds, draws, budgets, out_dir, tag, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = S.pools
    log("champollion pools: " + ", ".join(f"{c} {len(pools[c][0])}/{len(pools[c][1])}" for c in CONDS))
    gthr = global_threshold(S, all_fds, log)
    settings = champ_settings()
    for fd in fds:
        path = out_dir / f"fold{fd['fold']}.npz"
        if path.exists():
            continue
        rec = champ_fold(S, fd, pools, draws, budgets, gthr, settings, tag, log)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **rec)
        tmp.rename(path)


def timing(S, log):
    """Seconds per Champollion fit (fit on the paired contrasts, transport in five cell types) for the first draw
    of fold 0 at 100 and 800 paired cells. Reads no held-out cell."""
    pools = S.pools
    eps, iters, c_by = champ_settings()
    fd = vrun.folds(S.new)[0]
    rres = np.flatnonzero(np.isin(S.new["sample"], fd["paired"]) & S.res)
    xr, _, _ = vrun.rna(S.new, rres, S.gidx)
    yr = vrun.prot(S.new, rres, S.yidx)
    popr = vrun.popkey(S.new, rres)
    out = {}
    for B in (100, 800):
        rng = np.random.default_rng([SEED, 0, B, 0])
        pick = rng.choice(len(rres), size=B, replace=False)
        Xh, Yh, _ = pc.contrasts(xr[pick], yr[pick], popr[pick])
        t0 = time.time()
        _, secs = ct.run_champ(Xh, Yh, pools, list(CONDS), eps, c_by[B] / np.sqrt(len(Xh)), iters, 0, "timing")
        out[B] = {"total": time.time() - t0, "fit": secs, "rows": len(Xh)}
        log(f"B={B}: {out[B]}")
    return out


# ------------------------------------------------------------------ scoring

def load_folds(d):
    recs = {}
    for p in sorted(d.glob("fold*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p, allow_pickle=False)
        recs[int(z["fold"])] = {k: z[k] for k in z.files}
    return recs


def donor_weights(donors_f, n, seed):
    """Resamples of donors (a donor carries its held-out samples), as the readout drew them."""
    donors = sorted(set(donors_f))
    folds_of = {d: [i for i, x in enumerate(donors_f) if x == d] for d in donors}
    rng = np.random.default_rng(seed)
    W = np.zeros((n, len(donors_f)))
    for r in range(n):
        for d in rng.choice(donors, size=len(donors), replace=True):
            W[r, folds_of[d]] += 1.0
    return W


def weighted(W, num, den, how):
    """Aggregate per-unit numerators num and denominators den (folds x cell types x ...) under fold weights W
    (resamples x folds): pooled (units weighted by their denominators), every unit alike, every cell type alike
    (types pooled within), every held-out sample alike (units pooled within)."""
    F, C = num.shape[:2]
    rest = num.shape[2:]
    n2 = num.reshape(F, C, -1)
    d2 = np.broadcast_to(np.asarray(den, float).reshape(F, C, 1), n2.shape)
    if how == "pooled":
        out = (W @ n2.sum(1)) / np.maximum(W @ d2.sum(1), 1e-300)
    elif how == "unit":
        ok = d2 > 0
        r = np.where(ok, n2 / np.where(ok, d2, 1.0), 0.0)
        out = (W @ r.sum(1)) / np.maximum(W @ ok.sum(1), 1e-300)
    elif how == "type":
        vals, oks = [], []
        for c in range(C):
            dn = W @ d2[:, c]
            ok = dn > 0
            vals.append(np.where(ok, (W @ n2[:, c]) / np.where(ok, dn, 1.0), 0.0))
            oks.append(ok)
        out = np.sum(vals, axis=0) / np.maximum(np.sum(oks, axis=0), 1)
    elif how == "sample":
        dn = d2.sum(1)
        ok = dn > 0
        r = np.where(ok, n2.sum(1) / np.where(ok, dn, 1.0), 0.0)
        out = (W @ r) / np.maximum(W @ ok, 1e-300)
    else:
        raise ValueError(how)
    return out.reshape((W.shape[0],) + rest)


def envelope(X, arm_idx):
    """Adds, for a family of arms (indices on axis 1), their best value at each budget."""
    return np.nanmax(X[:, arm_idx], axis=1)


def pct(x, q):
    return [float(v) for v in np.percentile(x, q)]


def score(run_dir, champ_dir, out_path, stored_summary, readout_path, log, smoke=False):
    R = load_folds(run_dir)
    fs = sorted(R)
    donors_f = [str(R[f]["donor"]) for f in fs]
    budgets = [int(b) for b in R[fs[0]]["budgets"]]
    arms = [str(a) for a in R[fs[0]]["arms"]]
    assert tuple(arms) == ARMS
    U = np.stack([R[f]["u_num"] for f in fs])          # folds, types, arms, budgets, truths, (E1, E2)
    U3 = np.stack([R[f]["u_e3"] for f in fs])          # folds, types, arms, budgets, (E3 numerator, E1 cognate)
    D = np.stack([R[f]["u_den"] for f in fs])          # folds, types, truths, (reproducible, positive, negative)
    D3 = np.stack([R[f]["e3_den"] for f in fs])        # folds, types, (<TA,TB> cognate, reproducible cognate)
    SC = np.stack([R[f]["scored"] for f in fs]).astype(float)
    RFn = np.stack([R[f]["rf_num"] for f in fs])       # folds, arms, budgets
    RFd = np.array([float(R[f]["rf_den"]) for f in fs])
    one = np.ones((1, len(fs)))
    ia = arms.index("atlas")
    po = [arms.index(a) for a in PO_ARMS]
    out = {"folds": len(fs), "donors": len(set(donors_f)), "budgets": budgets, "arms": arms, "truths": list(TRUTHS),
           "integrity": {"max_abs_P": float(max(R[f]["check"][0] for f in fs)),
                         "max_rel_msq": float(max(R[f]["check"][1] for f in fs)),
                         "compared": int(sum(R[f]["check"][2] for f in fs))},
           "global_bh_threshold": float(R[fs[0]]["gthr"])}

    # ---- reproduction of the readout (primary truth, pooled) and of the validation's recovered fractions
    E = {"E1": weighted(one, U[..., 0, 0], D[:, :, 0, 0], "pooled")[0],
         "E2": weighted(one, U[..., 0, 1], TOP_K * SC, "pooled")[0],
         "E3": weighted(one, U3[..., 0], D3[:, :, 0], "pooled")[0],
         "E1_cognate": weighted(one, U3[..., 1], D3[:, :, 1], "pooled")[0]}
    rd = json.loads(readout_path.read_text())
    diff = 0.0
    assert [int(b) for b in rd["budgets"]] == budgets, "budgets differ from the readout's"
    for e, X in E.items():
        for i, a in enumerate(brun.ARMS):
            ref = np.array(rd["values"][e][a], float)
            dd = np.abs(X[i] - ref)
            diff = max(diff, float(np.max(dd[np.isfinite(dd)])) if np.isfinite(dd).any() else 0.0)
    out["reproduction"] = {"readout_max_abs_difference": diff}
    vi = [budgets.index(B) for B in V_BUDGETS if B in budgets]
    vb = [budgets[i] for i in vi]
    rf = (one @ RFn.reshape(len(fs), -1)).reshape(len(arms), len(budgets)) / RFd.sum()
    sm = json.loads(stored_summary.read_text())
    rdiff = max(float(np.max(np.abs(rf[arms.index(a), vi] - np.array(sm["rf"][a])[:len(vi)])))
                for a in brun.ARMS)
    out["reproduction"]["rf_max_abs_difference"] = rdiff
    log(f"reproduction: readout {diff:.2e}; recovered fraction {rdiff:.2e}")

    # ---- A. alternatives: recovered fraction, paired cells needed, readout endpoints
    target = float(sm["targets"]["0.5"]["target"])
    Wc = donor_weights(donors_f, CBOOT, CBOOT_SEED)
    rf_b = (Wc @ RFn.reshape(len(fs), -1)).reshape(CBOOT, len(arms), len(budgets)) / (Wc @ RFd)[:, None, None]
    fam_idx = {k: [arms.index(a) for a in v] for k, v in FAMILIES.items()}
    curves = {a: rf[i] for i, a in enumerate(arms)}
    bcurves = {a: rf_b[:, i] for i, a in enumerate(arms)}
    for k, idx in fam_idx.items():
        curves[k] = np.nanmax(rf[idx], axis=0)
        bcurves[k] = np.nanmax(rf_b[:, idx], axis=1)
    alt = {"target": target, "budgets": budgets, "rf": {a: [float(v) for v in c] for a, c in curves.items()},
           "rf_ci": {a: [pct(bcurves[a][:, j], [2.5, 97.5]) for j in range(len(budgets))] for a in curves},
           "cells": {}, "ratios": {}}
    keyarms = ["atlas", "other", "own"] + list(FAMILIES) + list(COMP)
    nd = {a: grun.needed(curves[a][vi], target, vb) for a in keyarms}
    bnd = {a: np.array([grun.needed(bcurves[a][r, vi], target, vb)[0] for r in range(CBOOT)]) for a in keyarms}
    alt["cells"] = {a: {"cells": v[0], "relation": v[1], "ci": pct(bnd[a], [2.5, 97.5])} for a, v in nd.items()}
    for name, (num_a, den_a) in {"K1_semicca_over_atlas": ("semicca", "atlas"),
                                 "K2_regression_over_atlas": ("regression", "atlas"),
                                 "paired_only_over_atlas": ("paired_only", "atlas"),
                                 "paired_only_over_semicca": ("paired_only", "semicca"),
                                 "paired_only_over_regression": ("paired_only", "regression"),
                                 "semicca_over_own": ("semicca", "own"),
                                 "regression_over_own": ("regression", "own")}.items():
        rb = bnd[num_a] / bnd[den_a]
        alt["ratios"][name] = {"ratio": nd[num_a][0] / nd[den_a][0], "ci": pct(rb, [2.5, 97.5]),
                               "numerator_relation": nd[num_a][1], "denominator_relation": nd[den_a][1]}
    alt["K1"] = bool(alt["ratios"]["K1_semicca_over_atlas"]["ci"][0] > 1)
    alt["K2"] = bool(alt["ratios"]["K2_regression_over_atlas"]["ci"][0] > 1)
    # readout endpoints (primary truth, pooled) of every arm and family, and the atlas's differences
    ends = {"E1": (U[..., 0, 0], D[:, :, 0, 0]), "E2": (U[..., 0, 1], TOP_K * SC), "E3": (U3[..., 0], D3[:, :, 0])}
    alt["readout"] = {}
    for e, (num, den) in ends.items():
        X = weighted(one, num, den, "pooled")[0]
        Xb = weighted(Wc, num, den, "pooled")
        vals = {a: X[i] for i, a in enumerate(arms)}
        bvals = {a: Xb[:, i] for i, a in enumerate(arms)}
        for k, idx in fam_idx.items():
            vals[k], bvals[k] = np.nanmax(X[idx], axis=0), np.nanmax(Xb[:, idx], axis=1)
        rec = {"values": {a: [float(v) for v in x] for a, x in vals.items()},
               "ci": {a: [pct(bvals[a][:, j], [2.5, 97.5]) for j in range(len(budgets))] for a in vals},
               "atlas_minus": {}}
        for k in FAMILIES:
            d = vals["atlas"] - vals[k]
            db = bvals["atlas"] - bvals[k]
            rec["atlas_minus"][k] = {str(B): {"difference": float(d[j]), "ci": pct(db[:, j], [2.5, 97.5])}
                                     for j, B in enumerate(budgets)}
        alt["readout"][e] = rec
    out["alternatives"] = alt
    log(f"alternatives: cells {', '.join(f'{a} {nd[a][1]}{nd[a][0]:.0f}' for a in keyarms)}; "
        f"K1 {alt['K1']} K2 {alt['K2']}")

    # ---- Champollion (first draw), with the estimator on the same draws
    Cr = load_folds(champ_dir) if champ_dir is not None and champ_dir.exists() else {}
    if Cr:
        cf = sorted(Cr)
        cdon = [str(Cr[f]["donor"]) for f in cf]
        cb = [int(b) for b in Cr[cf[0]]["budgets"]]
        CU = np.stack([Cr[f]["u_num"] for f in cf])
        CD = np.stack([Cr[f]["u_den"] for f in cf])
        CSC = np.stack([Cr[f]["scored"] for f in cf]).astype(float)
        CRn = np.stack([Cr[f]["rf_num"] for f in cf])
        CRd = np.array([float(Cr[f]["rf_den"]) for f in cf])
        c1 = np.ones((1, len(cf)))
        Wk = donor_weights(cdon, CBOOT, CBOOT_SEED)
        crf = (c1 @ CRn.reshape(len(cf), -1)).reshape(2, len(cb)) / CRd.sum()
        crf_b = (Wk @ CRn.reshape(len(cf), -1)).reshape(CBOOT, 2, len(cb)) / (Wk @ CRd)[:, None, None]
        cn = [grun.needed(crf[i], target, cb) for i in range(2)]
        cbn = [np.array([grun.needed(crf_b[r, i], target, cb)[0] for r in range(CBOOT)]) for i in range(2)]
        ch = {"folds": len(cf), "draws": CHAMP_DRAWS, "budgets": cb,
              "rf": {"champollion": [float(v) for v in crf[0]], "atlas_same_draws": [float(v) for v in crf[1]]},
              "rf_ci": {a: [pct(crf_b[:, i, j], [2.5, 97.5]) for j in range(len(cb))]
                        for i, a in enumerate(("champollion", "atlas_same_draws"))},
              "cells": {a: {"cells": cn[i][0], "relation": cn[i][1], "ci": pct(cbn[i], [2.5, 97.5])}
                        for i, a in enumerate(("champollion", "atlas_same_draws"))},
              "champollion_over_atlas": {"ratio": cn[0][0] / cn[1][0], "ci": pct(cbn[0] / cbn[1], [2.5, 97.5]),
                                         "numerator_relation": cn[0][1], "denominator_relation": cn[1][1]},
              "fit_seconds_mean": float(np.mean(np.concatenate([Cr[f]["fit_seconds"] for f in cf]))),
              "readout": {}}
        for e, (num, den) in {"E1": (CU[..., 0, 0], CD[:, :, 0, 0]), "E2": (CU[..., 0, 1], TOP_K * CSC)}.items():
            X = weighted(c1, num, den, "pooled")[0]
            Xb = weighted(Wk, num, den, "pooled")
            ch["readout"][e] = {"champollion": [float(v) for v in X[0]], "atlas_same_draws": [float(v) for v in X[1]],
                                "atlas_minus_champollion": {str(B): {"difference": float(X[1, j] - X[0, j]),
                                                                     "ci": pct(Xb[:, 1, j] - Xb[:, 0, j], [2.5, 97.5])}
                                                            for j, B in enumerate(cb)}}
        out["champollion"] = ch
        log(f"champollion: cells {cn[0][1]}{cn[0][0]:.0f} vs atlas (same draws) {cn[1][1]}{cn[1][0]:.0f}")

    # ---- B. calibration of the readout
    Wr = donor_weights(donors_f, BOOT, BOOT_SEED)          # the readout's resamples
    Wp = donor_weights(donors_f, PBOOT, PBOOT_SEED)
    NR = len(brun.ARMS)                                    # the readout's arms come first
    assert arms[:NR] == list(brun.ARMS)
    cal = {"counts": {}, "values": {}, "differences": {}, "majority": {}, "by_type": {}}
    types = list(CONDS)
    for si, s in enumerate(TRUTHS):
        n = D[:, :, si]
        cal["counts"][s] = {"reproducible": int(n[..., 0].sum()), "positive": int(n[..., 1].sum()),
                            "negative": int(n[..., 2].sum()),
                            "by_type": {c: int(n[:, ci, 0].sum()) for ci, c in enumerate(types)},
                            "share_by_type": {c: float(n[:, ci, 0].sum() / max(n[..., 0].sum(), 1))
                                              for ci, c in enumerate(types)}}
        cal["values"][s], cal["differences"][s], cal["majority"][s] = {}, {}, {}
        for how in WEIGHTINGS:
            cal["values"][s][how], cal["differences"][s][how] = {}, {}
            for e, (num, den) in {"E1": (U[:, :, :NR, :, si, 0], D[:, :, si, 0]),
                                  "E2": (U[:, :, :NR, :, si, 1], TOP_K * SC)}.items():
                X = weighted(one, num, den, how)[0]
                Xb = weighted(Wr, num, den, how)
                env, envb = np.nanmax(X[:len(po)], axis=0), np.nanmax(Xb[:, :len(po)], axis=1)
                ja = brun.ARMS.index("atlas")
                cal["values"][s][how][e] = {"atlas": [float(v) for v in X[ja]], "paired_only": [float(v) for v in env],
                                            "other": [float(v) for v in X[brun.ARMS.index("other")]],
                                            "own": [float(v) for v in X[brun.ARMS.index("own")]]}
                cal["differences"][s][how][e] = {str(B): {"atlas": float(X[ja, j]), "paired_only": float(env[j]),
                                                          "difference": float(X[ja, j] - env[j]),
                                                          "ci": pct(Xb[:, ja, j] - envb[:, j], [2.5, 97.5])}
                                                 for j, B in enumerate(budgets) if B in (50, 100, 150)}
            # majority-sign baselines: each unit's own majority sign (an oracle), and one sign for all units
            maj = np.maximum(D[:, :, si, 1], D[:, :, si, 2])
            mb = weighted(Wr, maj[:, :, None], D[:, :, si, 0], how)[:, 0]
            cal["majority"][s][how] = {"unit_majority": float(weighted(one, maj[:, :, None], D[:, :, si, 0], how)[0, 0]),
                                       "ci": pct(mb, [2.5, 97.5])}
        tot = D[:, :, si].sum((0, 1))
        cal["majority"][s]["overall_sign"] = float(max(tot[1], tot[2]) / max(tot[0], 1))
        cal["majority"][s]["positive_share"] = float(tot[1] / max(tot[0], 1))
        npairs = int(rd["pairs_per_unit"])
        cal["majority"][s]["random_E2"] = float(np.sum(SC * n[..., 0] / npairs * 0.5) / SC.sum())
        cal["majority"][s]["perfect_E2"] = float(np.sum(SC * np.minimum(TOP_K, n[..., 0])) / (TOP_K * SC.sum()))
    # per cell type (primary truth): atlas minus paired-only with intervals
    for ci, c in enumerate(types):
        rec = {}
        for e, (num, den) in {"E1": (U[:, ci:ci + 1, :NR, :, 0, 0], D[:, ci:ci + 1, 0, 0]),
                              "E2": (U[:, ci:ci + 1, :NR, :, 0, 1], TOP_K * SC[:, ci:ci + 1])}.items():
            X = weighted(one, num, den, "pooled")[0]
            Xb = weighted(Wr, num, den, "pooled")
            env, envb = np.nanmax(X[:len(po)], axis=0), np.nanmax(Xb[:, :len(po)], axis=1)
            ja = brun.ARMS.index("atlas")
            rec[e] = {str(B): {"atlas": float(X[ja, j]), "paired_only": float(env[j]),
                               "difference": float(X[ja, j] - env[j]), "ci": pct(Xb[:, ja, j] - envb[:, j], [2.5, 97.5])}
                      for j, B in enumerate(budgets) if B in (50, 100, 150)}
        cal["by_type"][c] = rec
    # planned comparisons (primary truth, pooled, as frozen): 95% intervals of the readout's resamples, one-sided
    # bootstrap P-values, Bonferroni-simultaneous intervals over the eight comparisons, Holm over B1-B3
    endpoint = {"E1": (U[:, :, :NR, :, 0, 0], D[:, :, 0, 0]), "E2": (U[:, :, :NR, :, 0, 1], TOP_K * SC),
                "E3": (U3[:, :, :NR, :, 0], D3[:, :, 0])}
    plan = []
    lvl = 0.05 / len(PLANNED)
    for h, e, B in PLANNED:
        if B not in budgets:
            continue
        j = budgets.index(B)
        num, den = endpoint[e]
        X = weighted(one, num, den, "pooled")[0]
        ja = brun.ARMS.index("atlas")
        d = float(X[ja, j] - np.nanmax(X[:len(po), j]))
        dr_ = weighted(Wr, num[:, :, :, j:j + 1], den, "pooled")[:, :, 0]
        dp_ = weighted(Wp, num[:, :, :, j:j + 1], den, "pooled")[:, :, 0]
        db = dr_[:, ja] - np.nanmax(dr_[:, :len(po)], axis=1)
        dpb = dp_[:, ja] - np.nanmax(dp_[:, :len(po)], axis=1)
        plan.append({"hypothesis": h, "endpoint": e, "budget": B, "difference": d, "ci95": pct(db, [2.5, 97.5]),
                     "p_one_sided": float((1 + np.sum(dpb <= 0)) / (PBOOT + 1)),
                     "simultaneous_ci": pct(dpb, [100 * lvl / 2, 100 * (1 - lvl / 2)])})
    hyp = {}
    for h in ("B1", "B2", "B3"):
        ps = [r["p_one_sided"] for r in plan if r["hypothesis"] == h]
        if ps:
            hyp[h] = {"p_intersection_union": max(ps)}
    order = sorted(hyp, key=lambda h: hyp[h]["p_intersection_union"])
    run_max = 0.0
    for i, h in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - i) * hyp[h]["p_intersection_union"]))
        hyp[h]["p_holm"] = run_max
    cal["planned"] = {"comparisons": plan, "hypotheses": hyp, "bonferroni_level": lvl,
                      "p_resolution": 1 / (PBOOT + 1)}
    out["calibration"] = cal
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    out["smoke"] = smoke
    out_path.write_text(json.dumps(out, indent=1))
    log("calibration: " + "; ".join(
        f"{s} E1@100 {100 * cal['differences'][s]['pooled']['E1']['100']['atlas']:.1f} vs "
        f"{100 * cal['differences'][s]['pooled']['E1']['100']['paired_only']:.1f}"
        for s in TRUTHS if "100" in cal["differences"][s]["pooled"]["E1"]))
    return out


# ------------------------------------------------------------------ freeze check and entry point

def check_freeze():
    rec = json.loads((HERE / "results" / "freeze.json").read_text())
    for rel, digest in rec["files"].items():
        assert sha(EXT / rel) == digest, f"{rel} changed since the freeze"
    for name, (path, digest) in rec["data"].items():
        assert sha(path) == digest, f"{name} changed since the freeze"
    brun.check_freeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("smoke", "timing", "run", "champ", "score"))
    ap.add_argument("--part", type=int, choices=(0, 1))
    args = ap.parse_args()
    out = HERE / ("results_smoke" if args.cmd == "smoke" else "results")
    out.mkdir(exist_ok=True)
    lp = out / f"{args.cmd}{'' if args.part is None else args.part}.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(lp, "a") as fh:
            fh.write(line + "\n")

    threads = 2 if args.part is None else 1
    with threadpool_limits(limits=threads):
        if args.cmd == "smoke":
            S = setup(synthetic=True)
            fds = [fd for fd in vrun.folds(S.new) if fd["fold"] in (0, 17)]
            run(S, fds, fds, 2, (25, 50, 100), VAL / "results_smoke", out / "folds", log)
            champ(S, fds[:1], fds, 1, (25, 50), out / "champ", "smoke", log)
            score(out / "folds", out / "champ", out / "review.json", VAL / "results_smoke" / "summary.json",
                  VAL / "readout" / "results_smoke" / "readout.json", log, smoke=True)
        elif args.cmd == "timing":
            json.dump(timing(setup(), log), open(out / "timing.json", "w"), indent=1)
        elif args.cmd in ("run", "champ"):
            check_freeze()
            S = setup()
            all_fds = vrun.folds(S.new)
            fds = [fd for fd in all_fds if args.part is None or fd["fold"] % 2 == args.part]
            if args.cmd == "run":
                run(S, fds, all_fds, DRAWS, BUDGETS, VAL / "results", out / "folds", log)
            else:
                champ(S, fds, all_fds, CHAMP_DRAWS, CHAMP_BUDGETS, out / "champ", f"review{args.part}", log)
        elif args.cmd == "score":
            check_freeze()
            score(out / "folds", out / "champ", out / "review.json", VAL / "results" / "summary.json",
                  VAL / "readout" / "results" / "readout.json", log)


if __name__ == "__main__":
    main()
