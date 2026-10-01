#!/usr/bin/env python3
"""B, C, D1, E and G of PLAN.md on the validation test: every paired draw regenerated and scored on its own.

    python validation_review.py smoke            # synthetic values with the real design (folds 0 and 17, budgets
                                                 # 25-100, two draws): the whole pipeline -> results_smoke/
    python validation_review.py run --part 0|1   # checks results/freeze.json; even or odd folds
                                                 # -> results/validation/fold<f>.npz
    python validation_review.py score            # results/validation_review.json

Draws, seeds, contrasts, references and the paired-only and reference-regression arms are those of the validation
test and of its review (vrun.predict; posthoc_review/rrun.predictions without SemiCCA). The regenerated means of the
validation's arms are checked against its hashed predictions. Scores are kept per draw.
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
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
VAL = EXT / "validation"
sys.path.insert(0, str(VAL))
sys.path.insert(0, str(VAL / "readout"))
import vrun  # noqa: E402  (frozen: validation/results/freeze.json)
import brun  # noqa: E402  (frozen: validation/readout/results/freeze.json)
import posthoc_robustness as pr  # noqa: E402  (the skewed atlases)

drun, pc, es, rs = vrun.drun, vrun.pc, vrun.es, vrun.rs
CONDS, PAIRED_ONLY, PO_ARMS, SEED = vrun.CONDS, vrun.PAIRED_ONLY, vrun.PO_ARMS, vrun.SEED
MIN_PERCOND, K_MAP = vrun.MIN_PERCOND, rs.K_MAP
BUDGETS, DRAWS, TOP_K, MIN_SUB, Z_CRIT = brun.BUDGETS, brun.DRAWS, brun.TOP_K, brun.MIN_SUB, brun.Z_CRIT
REGRESSION = tuple(f"{m}/{f}" for m in ("ridge_p", "ridge_u") for f in ("pooled", "mapped", "percond"))
SKEWED = ("few_B_mono", "few_T")
REFS = ("full",) + SKEWED
ARMS = (PO_ARMS + ("atlas", "other", "own") + REGRESSION
        + ("atlas/axes", "other/axes", "own/axes", "atlas/map_only")
        + tuple(f"{r}/reweighted" for r in REFS)
        + tuple(f"{r}/{m}" for r in SKEWED for m in ("mapped", "axes")))
TRUTHS = ("primary", "fine", "technical", "both")
RELATIONSHIPS = (
    ("R1", "CD14 Mono", ("ISG15", "IFI6", "IFI44L", "MX1", "OAS1", "OAS2", "HERC5", "XAF1", "EPSTI1", "IFITM3", "STAT2",
                         "IRF9"), "anti-human-CD169-(Sialoadhesin,-Siglec-1)-TotalSeqC", 1),
    ("R2", "CD14 Mono", ("HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1", "HLA-DMA",
                         "HLA-DMB"), "anti-human-HLA-DR-TotalSeqC", 1),
    ("R3", "B", ("HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1", "HLA-DMA",
                 "HLA-DMB"), "anti-human-HLA-DR-TotalSeqC", 1),
    ("R4", "CD8 T", ("GZMB", "GZMH", "PRF1", "GNLY", "NKG7", "FGFBP2", "CST7", "KLRD1", "ADGRG1"),
     "anti-human-CD57-Recombinant-TotalSeqC", 1),
    ("R5", "CD8 T", ("GZMB", "GZMH", "PRF1", "GNLY", "NKG7", "FGFBP2", "CST7", "KLRD1", "ADGRG1"),
     "anti-human-CD27-TotalSeqC", -1),
    ("R6", "NK", ("GZMB", "PRF1", "FGFBP2", "SPON2", "GNLY", "NKG7"), "anti-human-CD16-TotalSeqC", 1),
    ("R7", "NK", ("GZMB", "PRF1", "FGFBP2", "SPON2", "GNLY", "NKG7"), "anti-human-CD56-TotalSeqC", -1),
    ("R8", "CD4 T", ("LEF1",), "anti-human-CD45RA-TotalSeqC", 1))
PRIMARY_BUDGETS = (50, 100, 150)
BOOT = 2000
BOOT_B, BOOT_D1, BOOT_E, BOOT_G = 20261304, 20261305, 20261307, 20261309
OUT = HERE / "results" / "validation"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ references and reweighted condition maps

def summaries(ctx):
    """Per-type latent RNA covariances, scale factors and reliabilities of a reference, its pooled latent covariance
    and its own type weights (RNA cells of the type over all)."""
    n = {c: float(ctx.cond[c]["n_x"]) for c in ctx.cond}
    return SimpleNamespace(ctx=ctx, S=ctx.un.R["__all__"]["Rz"], Sz={c: ctx.cond[c]["Sz"] for c in ctx.cond},
                           rx={c: ctx.cond[c]["rx"] for c in ctx.cond}, ry={c: ctx.cond[c]["ry"] for c in ctx.cond},
                           s=dict(ctx.reliability), w0={c: v / sum(n.values()) for c, v in n.items()})


def reweighted_maps(ref, w):
    """Condition maps of run_study.Context.set_maps with the reference's pooled latent covariance moved to type
    weights w: S + sum_c (w_c - w0_c) S_c, in the reweighted pooled standard deviations. With w = w0 these are the
    published maps."""
    dw = {c: w.get(c, 0.0) - ref.w0[c] for c in ref.Sz}
    S = ref.S + sum(dw[c] * ref.Sz[c] for c in ref.Sz)
    dx = np.sqrt(sum(w.get(c, 0.0) * ref.rx[c] ** 2 for c in ref.Sz) / sum(ref.w0[c] * ref.rx[c] ** 2 for c in ref.Sz))
    dy = np.sqrt(sum(w.get(c, 0.0) * ref.ry[c] ** 2 for c in ref.Sz) / sum(ref.w0[c] * ref.ry[c] ** 2 for c in ref.Sz))
    U, lam = es.eigenbasis(S / np.outer(dx, dx))
    Uk = U[:, :K_MAP]
    Pk = Uk @ np.diag(1 / lam[:K_MAP]) @ Uk.T
    Qk = np.eye(len(lam)) - Uk @ Uk.T
    I = np.eye(len(lam))
    maps = {}
    for c in ref.Sz:
        s = ref.s[c]
        G = (ref.Sz[c] / np.outer(dx, dx)) @ Pk + Qk
        maps[c] = (I + s * (G - I), 1 + s * (ref.rx[c] / dx - 1), 1 + s * (ref.ry[c] / dy - 1))
    return maps


def apply_map(maps, P, c):
    if c not in maps:
        return P
    G, sx, sy = maps[c]
    return (1 / sx)[:, None] * (G @ P) * (1 / sy)[None, :]


# ------------------------------------------------------------------ setup

def setup(synthetic=False):
    new, atlas = vrun.load_new(synthetic), vrun.load_atlas(synthetic)
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    proteins = [str(new["y_names"][j]) for j in yidx]
    _, A, _ = vrun.atlas_context(atlas, part_a, gidx_a, yidx_a)
    refs = {"full": summaries(A)}
    variants = pr.variants(atlas, part_a)
    for v in SKEWED:
        _, ctx = pr.variant_context(atlas, np.flatnonzero(variants[v][1]), part_a, gidx_a, yidx_a)
        refs[v] = summaries(ctx)
        log(f"reference {v}: maps {sorted(ctx.maps)}")
    for v, ref in refs.items():                    # with the reference's own weights, the published maps
        G0 = reweighted_maps(ref, ref.w0)
        d = max(float(np.max(np.abs(G0[c][0] - ref.ctx.maps[c][0]))) for c in ref.Sz)
        log(f"reference {v}: own-weight maps reproduce the published maps (max abs difference {d:.1e})")
        assert d < 1e-6
    rel = []
    for name, c, gs, p, sign in RELATIONSHIPS:
        gi = [genes.index(g) for g in gs if g in genes]
        assert len(gi) == len(gs) or synthetic, f"{name}: genes missing from the panel"
        pj = proteins.index(p) if p in proteins else 0
        assert p in proteins or synthetic, f"{name}: protein missing from the panel"
        rel.append((name, c, np.array(gi if gi else [0], dtype=int), pj, sign))
    # technical covariates of every cell of the validation study: log library sizes and 10x lane
    lib_y = np.log(np.maximum(np.asarray(new["y"].sum(1), float), 1))
    # memory: keep only the panel genes' counts (vrun.rna reads the same values; library sizes are kept apart)
    for k in ("x_data", "x_indices", "x_indptr"):
        new.pop(k, None)
    new["x_counts"] = new["x_counts"][:, gidx].tocsr()
    S = SimpleNamespace(new=new, part=part, half=half, res=res, genes=genes, gidx=np.arange(len(gidx)), yidx=yidx,
                        proteins=proteins, A=A, refs=refs, rel=rel, lib_y=lib_y, synthetic=synthetic)
    return S


# ------------------------------------------------------------------ truths of a held-out sample

def truths(S, fd):
    """Per cell type: T, TA, TB, reproducible associations and direction under each truth variant, and whether the
    unit is scored (MIN_SUB cells in each sub-half)."""
    new = S.new
    r = np.flatnonzero(new["sample"] == fd["heldout"])
    x = vrun.rna(new, r, S.gidx)[0]
    y = vrun.prot(new, r, S.yidx)
    cond, hv, fine, lane = new["cond"][r], S.half[r], new["fine"][r], new["well"][r]
    tech = np.column_stack([np.log(np.maximum(new["x_library"][r].astype(float), 1)), S.lib_y[r]])
    out = {}
    for c in CONDS:
        out[c] = {}
        for v in TRUTHS:
            rec = {}
            for s, sel in (("T", np.ones(len(r), bool)), ("TA", hv == 0), ("TB", hv == 1)):
                m = np.flatnonzero((cond == c) & sel)
                keep, Z = design(v, m, fine, lane, tech)
                rec[s], rec[f"n{s}"], rec[f"k{s}"] = residual_corr(x[keep], y[keep], Z, len(S.gidx), len(S.yidx))
            scored = min(rec["nTA"], rec["nTB"]) >= MIN_SUB
            if scored:
                zA = np.arctanh(np.clip(rec["TA"], -1 + 1e-12, 1 - 1e-12)) * np.sqrt(rec["nTA"] - 3 - rec["kTA"])
                zB = np.arctanh(np.clip(rec["TB"], -1 + 1e-12, 1 - 1e-12)) * np.sqrt(rec["nTB"] - 3 - rec["kTB"])
                rep = (np.abs(zA) >= Z_CRIT) & (np.abs(zB) >= Z_CRIT) & (np.sign(rec["TA"]) == np.sign(rec["TB"]))
            else:
                rep = np.zeros(rec["T"].shape, bool)
            out[c][v] = {"T": rec["T"], "TA": rec["TA"], "TB": rec["TB"], "rep": rep, "dir": np.sign(rec["TA"]),
                         "scored": bool(scored), "den": float(np.sum(rec["TA"] * rec["TB"]))}
    return out


def design(v, m, fine, lane, tech):
    """Rows kept and covariates beyond the mean for truth variant v among held-out rows m."""
    if v == "primary":
        return m, None
    keep = m
    cols = []
    if v in ("fine", "both"):
        states, counts = np.unique(fine[m], return_counts=True)
        ok = set(states[counts >= 2])
        keep = m[np.isin(fine[m], list(ok))]
        st = sorted(ok)
        cols += [(fine[keep] == s_).astype(float) for s_ in st[1:]]
    if v in ("technical", "both"):
        cols += [tech[keep, 0], tech[keep, 1]]
        lanes = sorted(set(lane[keep]))
        cols += [(lane[keep] == l_).astype(float) for l_ in lanes[1:]]
    return keep, (np.column_stack(cols) if cols else None)


def residual_corr(x, y, Z, p, q):
    """Correlations of x and y after removing the mean and covariates Z (as vrun.heldout does for the mean); the
    number of rows and of covariate terms kept (rank of the centred Z)."""
    n = len(x)
    if n < 2:
        return np.zeros((p, q)), n, 0
    xc, yc = x - x.mean(0), y - y.mean(0)
    k = 0
    if Z is not None and n > 3:
        Zc = Z - Z.mean(0)
        Qz, Rz = np.linalg.qr(Zc)
        keep = np.abs(np.diag(Rz)) > 1e-8 * max(1.0, float(np.abs(Rz).max()))
        Qz = Qz[:, keep]
        k = int(keep.sum())
        xc, yc = xc - Qz @ (Qz.T @ xc), yc - Qz @ (Qz.T @ yc)
    den = np.sqrt(np.maximum((xc * xc).sum(0), 1e-12)[:, None] * np.maximum((yc * yc).sum(0), 1e-12)[None, :])
    return (xc.T @ yc) / den, n, k


# ------------------------------------------------------------------ predictions of one draw

def predictions(S, x, y, pp, cc, O, W, f, B, dr):
    """Every arm's per-type prediction from one paired draw (rrun.predictions without SemiCCA, plus the arms of
    B and C)."""
    preds = {}
    hc = pc.contrasts(x, y, pp)
    if hc is None:
        return preds
    Xh, Yh, ih = hc
    ch = cc[ih]
    A = S.A
    for a in PAIRED_ONLY:
        crng = np.random.default_rng([SEED + 1, f, B, dr])
        Pp = rs.denoise(a, Xh, Yh, A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
        for c in CONDS:
            preds[(f"{a}/pooled", c)] = Pp
            m = ch == c
            preds[(f"{a}/percond", c)] = (rs.denoise(a, Xh[m], Yh[m], A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
                                          if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
        if a == "js":
            for c in CONDS:
                preds[("atlas/map_only", c)] = drun.mapped(A, Pp, c)
    for arm, ctx in (("atlas", A), ("other", O), ("own", W)):
        Pe = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
        for c in CONDS:
            preds[(arm, c)] = drun.mapped(ctx, Pe, c)
            preds[(f"{arm}/axes", c)] = Pe
    for meth in ("ridge_p", "ridge_u"):
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
    # C: reweighting. The pooled contrast estimate weights each type by its contrast rows.
    rows = {c: float(np.sum(ch == c)) for c in CONDS}
    w = {c: rows[c] / max(sum(rows.values()), 1.0) for c in CONDS}
    for name in REFS:
        ref = S.refs[name]
        ctx = ref.ctx
        Pe = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
        rw = reweighted_maps(ref, w)
        for c in CONDS:
            preds[(f"{name}/reweighted", c)] = apply_map(rw, Pe, c)
            if name in SKEWED:
                preds[(f"{name}/mapped", c)] = drun.mapped(ctx, Pe, c)
                preds[(f"{name}/axes", c)] = Pe
    return preds


def unit_scores(Q, tr):
    """Recovered-fraction numerator, E1 credit and E2 (reproducible associations with the right direction among the
    TOP_K largest |Q|) for one prediction and one truth."""
    g = 2.0 * float(np.sum(Q * tr["T"])) - float(np.sum(Q * Q))
    if not tr["scored"]:
        return g, 0.0, 0.0
    s = np.sign(Q)
    credit = np.where(s == 0, 0.5, (s == tr["dir"]).astype(float)) * tr["rep"]
    a, cr = np.abs(Q).ravel(), credit.ravel()
    v = np.partition(a, a.size - TOP_K)[a.size - TOP_K]
    gt, eq = a > v, a == v
    e2 = float(cr[gt].sum()) + (TOP_K - int(gt.sum())) * float(cr[eq].mean())
    return g, float(credit.sum()), e2


# ------------------------------------------------------------------ one fold

def run_fold(S, fd, budgets, draws, stored, O_cache, log):
    f = fd["fold"]
    t0 = time.time()
    new = S.new
    if fd["pool"] not in O_cache:
        O_cache[fd["pool"]] = vrun.other_context(new, fd, S.part, S.gidx, S.yidx)[1]
    O = O_cache[fd["pool"]]
    W = vrun.own_context(new, fd, S.part, S.res, S.gidx, S.yidx)[1]
    rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & S.res)
    xr, _, _ = vrun.rna(new, rres, S.gidx)
    yr = vrun.prot(new, rres, S.yidx)
    popr, condr = vrun.popkey(new, rres), new["cond"][rres]
    tr = truths(S, fd)
    nA, nB, nC, nT, nR = len(ARMS), len(budgets), len(CONDS), len(TRUTHS), len(S.rel)
    score = np.full((nB, draws, nA, nC, nT, 3), np.nan)       # recovered-fraction numerator, E1 credit, E2
    prog = np.full((nB, draws, nA, nR), np.nan)                # predicted relationship values
    check = {"max_abs_mean": 0.0, "max_rel_msq": 0.0, "compared": 0}
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
                for ti, v in enumerate(TRUTHS):
                    score[bi, dr, ai, ci, ti] = unit_scores(Q, tr[c][v])
                for ri, (_, rc, gi, pj, _) in enumerate(S.rel):
                    if rc == c:
                        prog[bi, dr, ai, ri] = float(Q[gi, pj].mean())
                if arm in vrun.ARMS:
                    sQ[(arm, c)] = sQ.get((arm, c), 0.0) + Q / draws
                    sq[(arm, c)] = sq.get((arm, c), 0.0) + float(np.sum(Q * Q)) / draws
        if stored is not None and B in vrun.BUDGETS:
            for (arm, c), P in sQ.items():
                k = f"{f}/{arm}/{B}/{c}"
                check["max_abs_mean"] = max(check["max_abs_mean"],
                                            float(np.max(np.abs(P - stored[f"P/{k}"].astype(np.float64)))))
                ms = float(stored[f"msq/{k}"])
                check["max_rel_msq"] = max(check["max_rel_msq"], abs(sq[(arm, c)] - ms) / max(ms, 1e-300))
                check["compared"] += 1
            if check["max_abs_mean"] > brun.TOL_P or check["max_rel_msq"] > brun.TOL_MSQ:
                raise SystemExit(f"regenerated means differ from the validation's at fold {f}, budget {B}: {check}")
    truth_arr = {}
    for ti, v in enumerate(TRUTHS):
        truth_arr[f"den/{v}"] = np.array([tr[c][v]["den"] for c in CONDS])
        truth_arr[f"reps/{v}"] = np.array([float(tr[c][v]["rep"].sum()) for c in CONDS])
        truth_arr[f"scored/{v}"] = np.array([tr[c][v]["scored"] for c in CONDS])
    rel_truth = np.full((nR, 3), np.nan)
    for ri, (_, rc, gi, pj, _) in enumerate(S.rel):
        t = tr[rc]["primary"]
        if t["scored"]:
            rel_truth[ri] = (t["T"][gi, pj].mean(), t["TA"][gi, pj].mean(), t["TB"][gi, pj].mean())
    log(f"fold {f} ({fd['heldout']}) done [{time.time() - t0:.0f}s]; check {check}")
    return {"fold": np.array(f), "donor": np.array(str(fd["donor"])), "heldout": np.array(str(fd["heldout"])),
            "budgets": np.array(budgets), "score": score, "prog": prog, "rel_truth": rel_truth,
            "check": np.array([check["max_abs_mean"], check["max_rel_msq"], check["compared"]]), **truth_arr}


# ------------------------------------------------------------------ scoring

def needed(values, target, budgets):
    return vrun.grun.needed(values, target, budgets)


def load(out_dir):
    R = [dict(np.load(p)) for p in sorted(out_dir.glob("fold*.npz"))]
    return R


def boot_units(R, seed):
    donors = sorted({str(r["donor"]) for r in R})
    idx = {d: [i for i, r in enumerate(R) if str(r["donor"]) == d] for d in donors}
    rng = np.random.default_rng(seed)
    W = np.zeros((BOOT, len(R)))
    for k in range(BOOT):
        for d in rng.choice(donors, len(donors)):
            W[k, idx[d]] += 1
    return W


def curves(R, w, draw=None, truth="primary"):
    """Recovered fraction per arm and budget, pooled over folds with weights w (folds); draw: one draw index per fold
    (array) or None for the mean over draws."""
    ti = TRUTHS.index(truth)
    num = np.zeros((len(ARMS), len(R[0]["budgets"])))
    den = 0.0
    for i, r in enumerate(R):
        if w[i] == 0:
            continue
        s = r["score"][..., ti, 0]                               # budgets x draws x arms x types
        v = np.nanmean(s, 1) if draw is None else s[:, draw[i]]
        num += w[i] * np.nansum(v, -1).T
        den += w[i] * float(np.sum(r[f"den/{truth}"]))
    return num / den


def envelope(cv, arms):
    return np.nanmax(np.stack([cv[ARMS.index(a)] for a in arms]), 0)


def readout(R, w, truth, e):
    """E1 (e=1) or E2 (e=2) per arm and budget, pooled over scored units with fold weights w."""
    ti = TRUTHS.index(truth)
    num = np.zeros((len(ARMS), len(R[0]["budgets"])))
    den = 0.0
    for i, r in enumerate(R):
        if w[i] == 0:
            continue
        sc = r[f"scored/{truth}"].astype(bool)
        v = np.nanmean(r["score"][..., ti, e], 1)[..., sc]       # budgets x arms x scored types
        num += w[i] * v.sum(-1).T
        den += w[i] * (float(r[f"reps/{truth}"][sc].sum()) if e == 1 else TOP_K * int(sc.sum()))
    return num / den


def score(out_dir=OUT, target=None):
    R = load(out_dir)
    budgets = [int(b) for b in R[0]["budgets"]]
    if target is None:
        target = json.loads((VAL / "results" / "summary.json").read_text())["targets"]["0.5"]["target"]
    ones = np.ones(len(R))
    Wb = boot_units(R, BOOT_B)
    out = {"folds": len(R), "donors": len(set(str(r["donor"]) for r in R)), "budgets": budgets, "target": target,
           "integrity": {"max_abs_mean": float(max(r["check"][0] for r in R)),
                         "max_rel_msq": float(max(r["check"][1] for r in R)),
                         "compared": int(sum(r["check"][2] for r in R))}}
    cv = curves(R, ones)
    boots = [curves(R, w) for w in Wb]

    def summary(arm_curve):
        pt = arm_curve(cv)
        bs = np.array([arm_curve(b) for b in boots])
        n = needed(pt, target, budgets)
        nb = [needed(b, target, budgets)[0] for b in bs]
        return {"rf": pt.tolist(), "rf_ci": np.percentile(bs, [2.5, 97.5], 0).T.tolist(),
                "cells": n[0], "relation": n[1], "cells_ci": np.percentile(nb, [2.5, 97.5]).tolist()}

    po = lambda c: envelope(c, PO_ARMS)                          # noqa: E731
    reg = lambda c: envelope(c, REGRESSION)                      # noqa: E731
    arms_out = {a: summary(lambda c, a=a: c[ARMS.index(a)]) for a in ARMS}
    arms_out["paired_only"] = summary(po)
    arms_out["regression"] = summary(reg)
    out["arms"] = arms_out
    # savings over the paired-only envelope, with the same resamples
    sav = {}
    for a in ("atlas", "atlas/axes", "atlas/map_only", "other", "other/axes", "own", "own/axes", "regression")\
            + tuple(f"{r}/reweighted" for r in REFS)\
            + tuple(f"{r}/{m}" for r in SKEWED for m in ("mapped", "axes")):
        get = reg if a == "regression" else (lambda c, a=a: c[ARMS.index(a)])
        n_po, n_a = needed(po(cv), target, budgets), needed(get(cv), target, budgets)
        bs = [needed(po(b), target, budgets)[0] / needed(get(b), target, budgets)[0] for b in boots]
        sav[a] = {"ratio": n_po[0] / n_a[0], "numerator_relation": n_po[1], "denominator_relation": n_a[1],
                  "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
    out["savings_over_paired_only"] = sav
    # C: rescue criterion, the reweighted skewed atlas within the full atlas's interval
    full_ci = np.array(arms_out["atlas"]["rf_ci"])
    out["rescue"] = {v: {str(B): bool(full_ci[i, 0] <= arms_out[f"{v}/reweighted"]["rf"][i] <= full_ci[i, 1])
                         for i, B in enumerate(budgets)} for v in SKEWED}
    # D1: one draw per study
    out["single_draw"] = single_draw(R, budgets, target)
    # E: relationships
    out["relationships"] = relationships(R, budgets)
    # G: readout under truth variants
    out["truth_variants"] = truth_variants(R, budgets)
    (HERE / ("results" if out_dir == OUT else "results_smoke") / "validation_review.json").write_text(
        json.dumps(out, indent=1) + "\n")
    for a in ("paired_only", "atlas", "atlas/axes", "atlas/map_only", "regression", "few_B_mono/mapped",
              "few_B_mono/reweighted", "few_T/mapped", "few_T/reweighted"):
        print(f"{a:<24} cells {arms_out[a]['cells']:.0f}{arms_out[a]['relation']}  rf "
              + " ".join(f"{100 * v:.1f}" for v in arms_out[a]["rf"]))
    return out


def single_draw(R, budgets, target):
    ones = np.ones(len(R))
    nd = R[0]["score"].shape[1]
    per = {}
    for d in range(nd):
        cv = curves(R, ones, draw=np.full(len(R), d))
        n_at = needed(cv[ARMS.index("atlas")], target, budgets)
        per[d] = {"V1": needed(envelope(cv, PO_ARMS), target, budgets)[0] / n_at[0],
                  "K2": needed(envelope(cv, REGRESSION), target, budgets)[0] / n_at[0],
                  "atlas_cells": n_at[0]}
    Wb = boot_units(R, BOOT_D1)
    rng = np.random.default_rng(BOOT_D1 + 1)
    v1, k2 = [], []
    for w in Wb:
        cv = curves(R, w, draw=rng.integers(0, nd, len(R)))
        n_at = needed(cv[ARMS.index("atlas")], target, budgets)[0]
        v1.append(needed(envelope(cv, PO_ARMS), target, budgets)[0] / n_at)
        k2.append(needed(envelope(cv, REGRESSION), target, budgets)[0] / n_at)
    return {"per_draw": per, "V1": {"median": float(np.median(v1)), "ci": np.percentile(v1, [2.5, 97.5]).tolist(),
                                    "share_above_one": float(np.mean(np.array(v1) > 1))},
            "K2": {"median": float(np.median(k2)), "ci": np.percentile(k2, [2.5, 97.5]).tolist(),
                   "share_above_one": float(np.mean(np.array(k2) > 1))}}


def relationships(R, budgets):
    """Per relationship: recovered fraction, sign agreement, mean absolute error and donor-level differences."""
    out = {}
    donors = sorted({str(r["donor"]) for r in R})
    rng = np.random.default_rng(BOOT_E)
    for ri, (name, c, gs, p, sign) in enumerate(RELATIONSHIPS):
        rows = [(i, r) for i, r in enumerate(R) if np.isfinite(r["rel_truth"][ri, 0])]
        if not rows:
            out[name] = {"cell_type": c, "units": 0}
            continue
        t = np.array([r["rel_truth"][ri] for _, r in rows])             # units x (T, TA, TB)
        pred = np.array([r["prog"][:, :, :, ri] for _, r in rows])       # units x budgets x draws x arms
        agree = np.sign(t[:, 1]) == np.sign(t[:, 2])
        rec = {"cell_type": c, "genes": list(gs), "protein": p, "expected_sign": sign, "units": len(rows),
               "units_reliable_sign": int(agree.sum()), "truth_mean": float(t[:, 0].mean()),
               "truth_share_expected_sign": float(np.mean(np.sign(t[:, 0]) == sign)), "arms": {}}
        num = (2 * pred * t[:, 0, None, None, None] - pred ** 2)
        den = float(np.sum(t[:, 1] * t[:, 2]))
        err = np.abs(pred - t[:, 0, None, None, None])
        sgn = (np.sign(pred) == np.sign(t[:, 0])[:, None, None, None])[agree]
        per_arm = {}
        for ai, a in enumerate(ARMS):
            per_arm[a] = {"rf": (np.nanmean(num[:, :, :, ai], 2).sum(0) / den).tolist(),
                          "sign": np.nanmean(np.nanmean(sgn[:, :, :, ai], 2), 0).tolist(),
                          "mae": np.nanmean(np.nanmean(err[:, :, :, ai], 2), 0).tolist()}
        # families: best member per budget on the pooled error (favours the comparators)
        for fam, members in (("paired_only", PO_ARMS), ("regression", REGRESSION)):
            best = [min(members, key=lambda a: per_arm[a]["mae"][bi]) for bi in range(len(budgets))]
            per_arm[fam] = {k: [per_arm[best[bi]][k][bi] for bi in range(len(budgets))] for k in ("rf", "sign", "mae")}
            per_arm[fam]["members"] = best
        rec["arms"] = {a: per_arm[a] for a in ("atlas", "other", "own", "atlas/axes", "paired_only", "regression")}
        # donor level: mean absolute error per donor (over its units and draws), atlas minus comparator
        dn = [str(r["donor"]) for _, r in rows]
        by = {d: [k for k, x in enumerate(dn) if x == d] for d in donors if d in dn}
        diffs = {}
        for fam in ("paired_only", "regression"):
            members = per_arm[fam]["members"]
            dd = []
            for bi, B in enumerate(budgets):
                ai_a, ai_c = ARMS.index("atlas"), ARMS.index(members[bi])
                e_a = np.nanmean(err[:, bi, :, ai_a], 1)
                e_c = np.nanmean(err[:, bi, :, ai_c], 1)
                d = np.array([np.mean(e_a[ix] - e_c[ix]) for ix in by.values()])
                bs = []
                for _ in range(BOOT):
                    bs.append(np.mean(d[rng.integers(0, len(d), len(d))]))
                dd.append({"budget": B, "mean_difference": float(d.mean()), "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
                           "donors_atlas_better": int(np.sum(d < 0)), "donors": int(len(d)),
                           "per_donor": dict(zip(by.keys(), d.tolist()))})
            diffs[fam] = dd
        rec["donor_level"] = diffs
        out[name] = rec
    return out


def truth_variants(R, budgets):
    out = {}
    Wb = boot_units(R, BOOT_G)
    ones = np.ones(len(R))
    for v in TRUTHS:
        rec = {"units_scored": int(sum(r[f"scored/{v}"].sum() for r in R)),
               "reproducible": int(sum(r[f"reps/{v}"].sum() for r in R))}
        for name, fn in (("recovered_fraction", lambda w: curves(R, w, truth=v)),
                         ("E1", lambda w: readout(R, w, v, 1)), ("E2", lambda w: readout(R, w, v, 2))):
            pt = fn(ones)
            bs = [fn(w) for w in Wb]
            vals = {}
            for label, get in (("atlas", lambda c: c[ARMS.index("atlas")]), ("paired_only", lambda c: envelope(c, PO_ARMS)),
                               ("regression", lambda c: envelope(c, REGRESSION))):
                vals[label] = get(pt).tolist()
            diffs = {}
            for comp, get in (("paired_only", lambda c: envelope(c, PO_ARMS)),
                              ("regression", lambda c: envelope(c, REGRESSION))):
                d = pt[ARMS.index("atlas")] - get(pt)
                db = np.array([b[ARMS.index("atlas")] - get(b) for b in bs])
                diffs[comp] = {"difference": d.tolist(), "ci": np.percentile(db, [2.5, 97.5], 0).T.tolist()}
            rec[name] = {"values": vals, "atlas_minus": diffs}
        out[v] = rec
    return out


# ------------------------------------------------------------------ smoke, run

def smoke():
    S = setup(synthetic=True)
    fds = [fd for fd in vrun.folds(S.new) if fd["fold"] in (0, 17)]
    out_dir = HERE / "results_smoke" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_store = VAL / "results_smoke" / "predictions.npz"
    stored = np.load(smoke_store) if smoke_store.exists() else None
    O_cache = {}
    for fd in fds:
        rec = run_fold(S, fd, [25, 50, 100], 2, stored, O_cache, log)
        np.savez_compressed(out_dir / f"fold{fd['fold']:02d}.npz", **rec)
    score(out_dir, target=0.1)


def check_freeze():
    rec = json.loads((HERE / "results" / "freeze.json").read_text())
    bad = [f for f, dg in rec["files"].items() if sha(EXT / f) != dg]
    assert not bad, f"changed since the freeze: {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "run", "score"))
    ap.add_argument("--part", type=int, choices=(0, 1))
    a = ap.parse_args()
    with threadpool_limits(limits=2):          # as vrun.main: some comparators' tuning depends on it
        if a.what == "smoke":
            smoke()
        elif a.what == "run":
            check_freeze()
            S = setup()
            stored = np.load(VAL / "results" / "predictions.npz")
            OUT.mkdir(parents=True, exist_ok=True)
            O_cache = {}
            for fd in vrun.folds(S.new):
                if fd["fold"] % 2 != a.part or (OUT / f"fold{fd['fold']:02d}.npz").exists():
                    continue
                rec = run_fold(S, fd, list(BUDGETS), DRAWS, stored, O_cache, log)
                np.savez_compressed(OUT / f"fold{fd['fold']:02d}.npz", **rec)
        else:
            score()


if __name__ == "__main__":
    main()
