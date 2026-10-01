#!/usr/bin/env python3
"""Generality benchmark (PLAN.md): cross-fitted predictions, then scoring on held-out replication units.

    python grun.py run <dataset>      # predict, then evaluate
    python grun.py predict <dataset>  # predictions of every fold (training units only), hashed to
                                      # results/<dataset>_manifest.json
    python grun.py evaluate <dataset> # checks the hashes, then held-out statistics and scores, results/<dataset>.json
    python grun.py summary            # hypotheses G1-G3 across data sets, results/summary.json

Every unit (donor, patient, mouse, recording day, cell type) is held out in exactly one of three folds (every
third unit in SHA-256 order). In each fold, cells of the other units are training cells: a fixed barcode hash puts
25% of them in a paired calibration reservoir and the rest in an unpaired pool whose cells contribute X from one
assay half and Y from the other. The estimators, the condition map and the comparators are those of
extension/semipaired, unchanged; the self-tuning arm is variant V2 of semipaired/selftune.py (DEV_SELFTUNE.md).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SP = HERE.parent / "semipaired"
sys.path.insert(0, str(SP))

import common as cm  # noqa: E402
import run_study as rs  # noqa: E402
import selftune as st  # noqa: E402
import sp_estimators as es  # noqa: E402

pm = cm.pm
DATA = Path("/home/claude/cbio/rawdata/generality")
RES = HERE / "results"
PRIMARY = ("hao", "stephenson", "bmmc_cite", "colon", "bmmc_multiome", "scala_m1", "gouwens_visp", "banc", "malecns")
SECONDARY = ("banc_crossanimal",)
FOLDS = 3
BUDGETS = (25, 50, 100, 200, 400, 800)
DRAWS = 5
SEED = 20261060
PANEL_X, PANEL_Y = 200, 200
HVG_MIN_DET, HVG_BINS = 0.05, 20
MIN_TRAIN_HALF = 10               # training population: >= 10 cells in each assay half
MIN_PERCOND = 6                   # per-condition refits need >= 6 paired cells of the condition
COMPARATORS = ("js", "scose", "fcose", "lowrank", "semicca", "ridge_p", "ridge_u")
PAIRED_ONLY = [f"{d}/{m}" for d in ("js", "scose", "fcose", "lowrank") for m in ("pooled", "percond")]
SEMICCA = ["semicca/pooled", "semicca/percond"]
ORIGINAL = PAIRED_ONLY + SEMICCA + [f"{d}/{m}" for d in ("ridge_p", "ridge_u") for m in ("pooled", "percond")]
PROPOSED, SELFTUNED = "bjs2/mapped", "selftune/mapped"
TARGETS = (0.5, 0.25, 0.75)       # fractions of the recovered fraction with all training cells paired (primary 0.5)
BOOT, BOOT_SEED = 2000, 20261061


def h(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16 ** 12


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


# ------------------------------------------------------------------ data and roles

def load(name):
    rec = RES / "freeze.json"
    if name != "synthetic" and rec.exists():
        assert sha(DATA / f"{name}.npz") == json.loads(rec.read_text())["data"][name], f"{name}.npz changed since the freeze"
    z = np.load(DATA / f"{name}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if "x_data" in d:
        d["x_counts"] = sp.csr_matrix((d["x_data"], d["x_indices"], d["x_indptr"]), shape=tuple(d["x_shape"]))
    d["y_kind"] = str(d["y_kind"])
    for k in ("cell", "unit", "cond", "pop"):
        d[k] = d[k].astype(str)
    d["cell"] = np.array([f"{name}|{c}" for c in d["cell"]])
    return d


def roles(name, d):
    units = sorted(set(d["unit"]), key=lambda u: hashlib.sha256(f"generality-fold-v1|{name}|{u}".encode()).hexdigest())
    fold_of = {u: i % FOLDS for i, u in enumerate(units)}
    fold = np.array([fold_of[u] for u in d["unit"]])
    part = np.array([int(h(f"generality-part-v1|{c}") >= 0.5) for c in d["cell"]])
    half = np.array([int(h(f"generality-half-v1|{c}") >= 0.5) for c in d["cell"]])
    return fold, part, half


def hvg(values, detected, n):
    """Binned-dispersion rule of the external test's panel (semipaired/external/run_external.py panel()) applied to
    per-feature means and variances of log values; features detected in >= 5% of cells; ties by index."""
    mean, var = values
    cand = np.flatnonzero((detected >= HVG_MIN_DET) & (mean > 0) & (var > 0))
    if len(cand) <= n:
        return np.sort(cand)
    disp = np.log(var[cand] / mean[cand])
    edges = np.quantile(mean[cand], np.linspace(0, 1, HVG_BINS + 1))
    b = np.clip(np.searchsorted(edges, mean[cand], side="right") - 1, 0, HVG_BINS - 1)
    z = np.zeros(len(cand))
    for k in range(HVG_BINS):
        m = b == k
        if m.sum() > 1 and disp[m].std() > 0:
            z[m] = (disp[m] - disp[m].mean()) / disp[m].std()
    order = sorted(range(len(cand)), key=lambda j: (-z[j], cand[j]))
    return np.sort(cand[order[:n]])


def lognorm_sparse(C, library):
    X = sp.csr_matrix(C, dtype=np.float64).copy()
    X = sp.diags(1e4 / np.maximum(library, 1)) @ X
    X.data = np.log1p(X.data)
    return sp.csr_matrix(X)


def x_panel(d, rows):
    """X features of a fold, chosen from its training cells (both assay halves)."""
    if "x_counts" in d:
        L = lognorm_sparse(d["x_counts"][rows], d["x_library"][rows])
        det = np.bincount(L.indices, minlength=L.shape[1]) / L.shape[0]
        mean = np.asarray(L.mean(0)).ravel()
        var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
        # the dispersion rule needs positive means; log values of counts are non-negative
        return hvg((mean, var), det, PANEL_X)
    V = d["x_dense"][rows]
    det = (V > 0).mean(0)
    var = V.var(0)
    cand = np.flatnonzero((det >= HVG_MIN_DET) & (var > 0))
    return np.sort(cand[np.argsort(-var[cand], kind="stable")[:PANEL_X]])


def y_panel(d, rows):
    Y = d["y"][rows]
    if d["y_kind"] == "raw":
        return np.flatnonzero(Y.std(0) > 0)
    det = (Y > 0).mean(0)
    if d["y_kind"] == "clr":
        ok = np.flatnonzero(det >= 0.01)
        if len(ok) <= PANEL_Y:
            return ok
        L = np.log1p(Y[:, ok])
        return np.sort(ok[np.argsort(-L.var(0), kind="stable")[:PANEL_Y]])
    L = np.log1p(1e4 * Y / np.maximum(d["y_library"][rows], 1)[:, None])
    return hvg((L.mean(0), L.var(0)), det, PANEL_Y)


def features(d, rows, xp, yp):
    """x, y and the X counts of the panel (None without counts) for the given cells."""
    if "x_counts" in d:
        C = d["x_counts"][rows][:, xp].toarray().astype(np.float64)
        lib = d["x_library"][rows].astype(np.float64)
        x = np.log1p(1e4 * C / np.maximum(lib, 1)[:, None])
    else:
        C, lib = None, None
        x = d["x_dense"][rows][:, xp].astype(np.float64)
    Y = d["y"][rows][:, yp].astype(np.float64)
    if d["y_kind"] == "clr":
        y = np.log1p(Y)
        y = y - y.mean(1, keepdims=True)
    elif d["y_kind"] == "lognorm":
        y = np.log1p(1e4 * Y / np.maximum(d["y_library"][rows], 1)[:, None])
    else:
        y = Y
    return x, y, C, lib


# ------------------------------------------------------------------ one fold

class NoCounts:
    """Unpaired summaries without counts: the latent X correlation equals the measured one (no count splitting)."""

    def __enter__(self):
        self.saved = cm.split_halves
        cm.split_halves = lambda counts, library, seed=None: (counts, counts)
        return self

    def __exit__(self, *a):
        cm.split_halves = self.saved


def prepare_fold(d, f, fold, part):
    """Training cells of fold f with the fold's panels (chosen from its training cells)."""
    tr = np.flatnonzero(fold != f)
    xp, yp = x_panel(d, tr), y_panel(d, tr)
    x, y, C, lib = features(d, tr, xp, yp)
    pop = np.array([f"{p}|{c}" for p, c in zip(d["pop"][tr], d["cond"][tr])])
    train = {"cell": d["cell"][tr], "part": part[tr], "guide": d["unit"][tr],
             "counts": C if C is not None else x, "library": lib if lib is not None else np.ones(len(tr))}
    return train, x, y, pop, (xp, yp), C is not None


def outside_reservoir(prefix, ids):
    """Identifiers that the reservoir hash never selects (unpaired-only cells of the cross-animal analysis)."""
    out = []
    for c in ids:
        k = 0
        while cm.hm.reservoir([f"{prefix}|{c}|{k}"])[0]:
            k += 1
        out.append(f"{prefix}|{c}|{k}")
    return np.array(out)


def prepare_crossanimal(d, f, fold, part):
    """Secondary analysis: BANC reservoir neurons of the training cell types are the paired cells; the unpaired pool
    is FAFB (brain side only) and MANC (nerve-cord side only), not BANC."""
    rec = json.loads((RES / "freeze.json").read_text())
    for n in ("fafb_brain", "manc_vnc"):
        assert sha(DATA / f"{n}.npz") == rec["data"][n], f"{n}.npz changed since the freeze"
    fa, mv = np.load(DATA / "fafb_brain.npz"), np.load(DATA / "manc_vnc.npz")
    xnames = [n for n in d["x_names"] if n in set(fa["names"])]
    ynames = [n for n in d["y_names"] if n in set(mv["names"])]
    fx = sp.csr_matrix(fa["counts"][:, [list(fa["names"]).index(n) for n in xnames]])
    my = mv["counts"][:, [list(mv["names"]).index(n) for n in ynames]]
    # panels from the unpaired connectomes
    L = lognorm_sparse(fx, fa["library"])
    det = np.bincount(L.indices, minlength=L.shape[1]) / L.shape[0]
    mean = np.asarray(L.mean(0)).ravel()
    xsel = hvg((mean, np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2), det, PANEL_X)
    Ly = np.log1p(1e4 * my / np.maximum(mv["library"], 1)[:, None])
    ysel = hvg((Ly.mean(0), Ly.var(0)), (my > 0).mean(0), PANEL_Y)
    xp = np.array([list(d["x_names"]).index(xnames[j]) for j in xsel])
    yp = np.array([list(d["y_names"]).index(ynames[j]) for j in ysel])
    tr = np.flatnonzero(fold != f)
    tr = tr[cm.hm.reservoir(d["cell"][tr])]
    xb, yb, Cb, libb = features(d, tr, xp, yp)
    Cf = fx[:, xsel].toarray().astype(np.float64)
    xf = np.log1p(1e4 * Cf / np.maximum(fa["library"], 1)[:, None])
    ym = Ly[:, ysel]
    nb, nf, nm = len(tr), len(Cf), len(ym)
    x = np.vstack([xb, xf, np.zeros((nm, xb.shape[1]))])
    y = np.vstack([yb, np.zeros((nf, yb.shape[1])), ym])
    C = np.vstack([Cb, Cf, np.zeros((nm, Cb.shape[1]))])
    lib = np.concatenate([libb, fa["library"], np.ones(nm)])
    cells = np.concatenate([d["cell"][tr], outside_reservoir("fafb", fa["cell"]), outside_reservoir("manc", mv["cell"])])
    prt = np.concatenate([part[tr], np.zeros(nf, int), np.ones(nm, int)])
    cond = np.concatenate([d["cond"][tr], fa["cond"].astype(str), mv["cond"].astype(str)])
    pop = np.array([f"all|{c}" for c in cond])
    guide = np.concatenate([d["unit"][tr], np.array(["fafb"] * nf), np.array(["manc"] * nm)])
    train = {"cell": cells, "part": prt, "guide": guide, "counts": C, "library": lib}
    return train, x, y, pop, (xp, yp), True


def estimate_fold(name, f, train, x, y, pop, counts, log):
    keys = pm.eligible_training(pop, train["part"], MIN_TRAIN_HALF)
    ctx_mgr = None if counts else NoCounts()
    if ctx_mgr:
        ctx_mgr.__enter__()
    try:
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, keys)
        ctx = rs.build_context(train, x, y, pop, keys, pool, un)
    finally:
        if ctx_mgr:
            ctx_mgr.__exit__()
    conds = un.conds
    V, _ = es.eigenbasis(ctx.Ry)
    pblocks = es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)
    store, msq = {}, {}
    # fully paired reference: every training cell of the pool populations with both modalities, per condition
    both = np.ones(len(pop), bool) if name != "banc_crossanimal" else np.char.startswith(train["cell"].astype(str), "banc|")
    for c in conds:
        acc, n = 0.0, 0
        for k in [k for k in pool.keys if cm.cond_of(k) == c]:
            m = (pop == k) & both
            if m.sum() < 2:
                continue
            xs, _ = pm.standardize(x[m])
            ys, _ = pm.standardize(y[m])
            acc = acc + xs.T @ ys
            n += int(m.sum())
        P = acc / max(n, 1) if n else np.zeros((x.shape[1], y.shape[1]))
        store[f"{f}/paired_all/pc/0/{c}"] = P
        msq[f"{f}/paired_all/pc/0/{c}"] = float(np.sum(P * P))
    for c in conds:
        G, sx, sy = ctx.maps[c]
        store[f"{f}/map/{c}/G"], store[f"{f}/map/{c}/sx"], store[f"{f}/map/{c}/sy"] = G, sx, sy
    budgets = [B for B in BUDGETS if B <= len(X_res)]
    seconds = {}
    idx = PRIMARY.index(name) if name in PRIMARY else (len(PRIMARY) + SECONDARY.index(name) if name in SECONDARY else 99)
    for B in budgets:
        acc = {}
        for dr in range(DRAWS):
            rng = np.random.default_rng([SEED, idx, f, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            X, Y, cc = X_res[pick], Y_res[pick], cond_res[pick]
            for arm in ("bjs2", "selftune") + COMPARATORS:
                crng = np.random.default_rng([SEED + 1, f, B, dr])
                t0 = time.time()
                if arm == "selftune":
                    P = st.estimate(X, Y, ctx.Uz, V, "V2")
                else:
                    P = rs.denoise(arm, X, Y, ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y, ctx.Uz, ctx.blocks, crng)
                seconds[arm] = seconds.get(arm, 0.0) + time.time() - t0
                acc[f"{arm}/P"] = acc.get(f"{arm}/P", 0.0) + P / DRAWS
                for c in conds:
                    Pm = ctx.mapped(P, c)
                    acc[f"{arm}/mapped/{c}"] = acc.get(f"{arm}/mapped/{c}", 0.0) + float(np.sum(Pm * Pm)) / DRAWS
                    acc[f"{arm}/pooled/{c}"] = acc.get(f"{arm}/pooled/{c}", 0.0) + float(np.sum(P * P)) / DRAWS
                if arm in COMPARATORS:
                    for c in conds:
                        m = cc == c
                        dd = ctx.cond[c]
                        Q = (rs.denoise(arm, X[m], Y[m], dd["Rx"], dd["Ry"], dd["n_x"], dd["n_y"], dd["Uz"], ctx.blocks,
                                        crng) if m.sum() >= MIN_PERCOND else np.zeros_like(P))
                        acc[f"{arm}/percond/Q/{c}"] = acc.get(f"{arm}/percond/Q/{c}", 0.0) + Q / DRAWS
                        acc[f"{arm}/percond/{c}"] = acc.get(f"{arm}/percond/{c}", 0.0) + float(np.sum(Q * Q)) / DRAWS
            # ablation: the same shrinkage in bases estimated from the paired cells themselves
            Pb = es.block_js2(X, Y, np.linalg.eigh(X.T @ X / B)[1][:, ::-1], np.linalg.eigh(Y.T @ Y / B)[1][:, ::-1],
                              ctx.blocks, pblocks)[0]
            acc["abl_pairbasis/P"] = acc.get("abl_pairbasis/P", 0.0) + Pb / DRAWS
            for c in conds:
                acc[f"abl_pairbasis/pooled/{c}"] = acc.get(f"abl_pairbasis/pooled/{c}", 0.0) + float(np.sum(Pb * Pb)) / DRAWS
        for arm in ("bjs2", "selftune", "abl_pairbasis") + COMPARATORS:
            store[f"{f}/{arm}/P/{B}"] = acc[f"{arm}/P"]
            for c in conds:
                for mode in ("pooled", "mapped", "percond"):
                    k = f"{arm}/{mode}/{c}"
                    if k in acc:
                        msq[f"{f}/{arm}/{mode}/{B}/{c}"] = acc[k]
                if f"{arm}/percond/Q/{c}" in acc:
                    store[f"{f}/{arm}/percond/{B}/{c}"] = acc[f"{arm}/percond/Q/{c}"]
        log(f"fold {f} B={B} ({', '.join(f'{k} {1000 * v / DRAWS / (budgets.index(B) + 1):.0f}' for k, v in seconds.items())} ms/fit)")
    info = {"conditions": conds, "n_reservoir": int(len(X_res)), "n_pool_keys": len(pool.keys),
            "reservoir_by_condition": {c: int(np.sum(cond_res == c)) for c in conds}, "budgets": budgets,
            "reliability": ctx.reliability, "training_cells": int(len(pop)),
            "seconds_per_fit": {k: v / (len(budgets) * DRAWS) for k, v in seconds.items()}}
    return store, msq, info


# ------------------------------------------------------------------ held-out statistics

def heldout_stats(d, f, fold, half, xp, yp, conds):
    """Per condition and held-out unit: sums for T (all cells), TA and TB (hash sub-halves). Populations nested in
    units are centred within population here; populations equal to the condition ('all') are centred across units
    at scoring time, under the bootstrap weights."""
    te = np.flatnonzero(fold == f)
    x, y, _, _ = features(d, te, xp, yp)
    unit, cond, popl, hv = d["unit"][te], d["cond"][te], d["pop"][te], half[te]
    pooled_pop = bool(np.all(popl == "all"))
    stats = {}
    units = sorted(set(unit))
    for c in conds:
        rec = {"units": units, "pooled_pop": pooled_pop}
        for s, sel in (("T", np.ones(len(te), bool)), ("TA", hv == 0), ("TB", hv == 1)):
            n = np.zeros(len(units))
            sx = np.zeros((len(units), x.shape[1]))
            sy = np.zeros((len(units), y.shape[1]))
            sxx = np.zeros((len(units), x.shape[1]))
            syy = np.zeros((len(units), y.shape[1]))
            sxy = np.zeros((len(units), x.shape[1], y.shape[1]))
            for i, u in enumerate(units):
                m = (unit == u) & (cond == c) & sel
                if pooled_pop:
                    if m.sum() == 0:
                        continue
                    xi, yi = x[m], y[m]
                    n[i], sx[i], sy[i] = m.sum(), xi.sum(0), yi.sum(0)
                    sxx[i], syy[i], sxy[i] = (xi * xi).sum(0), (yi * yi).sum(0), xi.T @ yi
                else:
                    for p in sorted(set(popl[m])):
                        mm = m & (popl == p)
                        if mm.sum() < 2:
                            continue
                        xc = x[mm] - x[mm].mean(0)
                        yc = y[mm] - y[mm].mean(0)
                        n[i] += mm.sum()
                        sxx[i] += (xc * xc).sum(0)
                        syy[i] += (yc * yc).sum(0)
                        sxy[i] += xc.T @ yc
            rec[s] = {"n": n, "sx": sx, "sy": sy, "sxx": sxx, "syy": syy, "sxy": sxy.astype(np.float32)}
        stats[c] = rec
    return stats


def targets(rec, W):
    """Pooled within-population cross-correlations T, TA, TB of one condition for a batch of unit weights W
    (R x units); returns dict of R x p x q arrays (None where a subset is empty)."""
    out = {}
    for s in ("T", "TA", "TB"):
        a = rec[s]
        n = W @ a["n"]
        xy = np.einsum("ru,upq->rpq", W, a["sxy"])
        xx, yy = W @ a["sxx"], W @ a["syy"]
        if rec["pooled_pop"]:
            mx = (W @ a["sx"]) / np.maximum(n, 1)[:, None]
            my = (W @ a["sy"]) / np.maximum(n, 1)[:, None]
            xy = xy - n[:, None, None] * mx[:, :, None] * my[:, None, :]
            xx = xx - n[:, None] * mx * mx
            yy = yy - n[:, None] * my * my
        out[s] = xy / np.sqrt(np.maximum(xx, 1e-12)[:, :, None] * np.maximum(yy, 1e-12)[:, None, :])
        out[s][n < 2] = 0.0
    return out


# ------------------------------------------------------------------ scoring

def arm_list(store, f, conds):
    """(arm label, budget, function condition -> mean prediction, msq key) for every scored arm of a fold."""
    arms = []
    for key in store:
        parts = key.split("/")
        if parts[0] != str(f) or len(parts) != 4 or parts[2] != "P":
            continue
        base, B = parts[1], parts[3]
        arms.append((f"{base}/pooled", B))
        if base in ("bjs2", "selftune") + COMPARATORS:
            arms.append((f"{base}/mapped", B))
        if base in COMPARATORS:
            arms.append((f"{base}/percond", B))
    return arms


def prediction(store, f, arm, B, c):
    base, mode = arm.split("/")
    if mode == "percond":
        return store[f"{f}/{base}/percond/{B}/{c}"]
    P = store[f"{f}/{base}/P/{B}"]
    if mode == "pooled":
        return P
    G, sx, sy = store[f"{f}/map/{c}/G"], store[f"{f}/map/{c}/sx"], store[f"{f}/map/{c}/sy"]
    return (1 / sx)[:, None] * (G @ P) * (1 / sy)[None, :]


def needed(values, target, budgets):
    """Paired cells to reach target on the running maximum of a curve (log-linear interpolation), as in the external
    test; a curve starting above the target needs at most the smallest budget, one never reaching it more than the
    largest."""
    env = np.maximum.accumulate(np.nan_to_num(np.asarray(values, float), nan=-np.inf))
    b = np.asarray(budgets, float)
    if target <= env[0]:
        return float(b[0]), "<="
    if target > env[-1]:
        return float(b[-1]), ">"
    i = int(np.argmax(env >= target))
    lo, hi = env[i - 1], env[i]
    fr = (target - lo) / (hi - lo) if hi > lo else 1.0
    return float(np.exp(np.log(b[i - 1]) + fr * (np.log(b[i]) - np.log(b[i - 1])))), "="


def saving(rf, comp_arms, target, budgets, proposed=PROPOSED):
    """Cells the comparator envelope needs over cells the proposed arm needs; censoring is resolved against the
    proposed estimator (a comparator that never reaches the target counts as needing the largest budget; a
    proposed curve that never reaches it counts as needing the largest budget)."""
    env = np.nanmax(np.array([rf[a] for a in comp_arms]), axis=0)
    n_c, r_c = needed(env, target, budgets)
    n_p, r_p = needed(rf[proposed], target, budgets)
    return n_c / n_p, n_c, r_c, n_p, r_p


def score(name, stats_by_fold, store, msq_by_fold, budgets_by_fold, W_units=None):
    """Recovered-fraction curves of every arm for one weighting of the held-out units (W_units: dict fold -> R x
    units weights; None = all ones). Returns rf[arm] (R x common budgets), the reference's rf (R) and the budgets."""
    common = sorted(set.intersection(*[set(b) for b in budgets_by_fold.values()]))
    folds = sorted(stats_by_fold)
    R = 1 if W_units is None else next(iter(W_units.values())).shape[0]
    num, den = {}, np.zeros(R)
    for f in folds:
        conds = list(stats_by_fold[f])
        arms = [(a, B) for a, B in arm_list(store, f, conds) if int(B) in common]
        for c in conds:
            rec = stats_by_fold[f][c]
            W = np.ones((1, len(rec["units"]))) if W_units is None else W_units[f]
            T = targets(rec, W)
            den += np.einsum("rpq,rpq->r", T["TA"], T["TB"])
            A = np.stack([prediction(store, f, a, B, c).ravel() for a, B in arms]
                         + [store[f"{f}/paired_all/pc/0/{c}"].ravel()])
            sq = np.array([msq_by_fold[f][f"{f}/{a.split('/')[0]}/{a.split('/')[1]}/{B}/{c}"] for a, B in arms]
                          + [msq_by_fold[f][f"{f}/paired_all/pc/0/{c}"]])
            v = 2 * (A @ T["T"].reshape(R, -1).T) - sq[:, None]
            for (a, B), row in zip(arms + [("paired_all/pc", "0")], v):
                key = (a, int(B))
                num[key] = num.get(key, 0.0) + row
    rf = {}
    for (arm, B), v in num.items():
        rf.setdefault(arm, {})[B] = v / den
    curves = {a: np.array([d_[B] for B in common]).T for a, d_ in rf.items() if a != "paired_all/pc"}
    return curves, rf["paired_all/pc"][0], common


def source(name):
    return "banc" if name == "banc_crossanimal" else name


def predict(name, log):
    d = load(source(name))
    fold, part, half = roles(source(name), d)
    RES.mkdir(exist_ok=True)
    store, msq, infos = {}, {}, {}
    for f in range(FOLDS):
        prep = prepare_crossanimal if name == "banc_crossanimal" else prepare_fold
        train, x, y, pop, pn, counts = prep(d, f, fold, part)
        s_, m, info = estimate_fold(name, f, train, x, y, pop, counts, log)
        info["x_panel_index"] = [int(v) for v in pn[0]]
        info["y_panel_index"] = [int(v) for v in pn[1]]
        info["x_panel"] = [str(v) for v in d["x_names"][pn[0]]]
        info["y_panel"] = [str(v) for v in d["y_names"][pn[1]]]
        store.update(s_)
        msq[f] = m
        infos[f] = info
    pred = RES / f"{name}_predictions.npz"
    np.savez_compressed(pred, **{k: np.asarray(v, np.float64) for k, v in store.items()})
    (RES / f"{name}_msq.json").write_text(json.dumps({str(f): m for f, m in msq.items()}) + "\n")
    (RES / f"{name}_info.json").write_text(json.dumps({str(f): i for f, i in infos.items()}) + "\n")
    manifest = {"predictions": sha(pred), "msq": sha(RES / f"{name}_msq.json"), "info": sha(RES / f"{name}_info.json"),
                "plan": sha(HERE / "PLAN.md") if (HERE / "PLAN.md").exists() else None,
                "freeze": sha(RES / "freeze.json") if (RES / "freeze.json").exists() else None,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": "written after all predictions and before any held-out statistic was computed"}
    (RES / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log("predictions hashed")


def evaluate(name, log):
    """Scores from the hashed prediction files and the held-out units' cells."""
    man = json.loads((RES / f"{name}_manifest.json").read_text())
    for key, fn in (("predictions", f"{name}_predictions.npz"), ("msq", f"{name}_msq.json"), ("info", f"{name}_info.json")):
        assert sha(RES / fn) == man[key], f"{fn} does not match the manifest"
    z = np.load(RES / f"{name}_predictions.npz")
    store = {k: z[k] for k in z.files}
    msq = {int(f): m for f, m in json.loads((RES / f"{name}_msq.json").read_text()).items()}
    infos = {int(f): i for f, i in json.loads((RES / f"{name}_info.json").read_text()).items()}
    d = load(source(name))
    fold, part, half = roles(source(name), d)
    stats = {f: heldout_stats(d, f, fold, half, np.array(infos[f]["x_panel_index"]), np.array(infos[f]["y_panel_index"]),
                              infos[f]["conditions"]) for f in range(FOLDS)}
    budgets = {f: infos[f]["budgets"] for f in range(FOLDS)}
    curves, ref, common = score(name, stats, store, msq, budgets)
    ref = float(np.asarray(ref).ravel()[0])
    rf = {a: v[0].tolist() for a, v in curves.items()}
    # bootstrap over units (all folds): each resample draws units with replacement
    units = sorted(set(d["unit"]))
    rng = np.random.default_rng([BOOT_SEED, (PRIMARY + SECONDARY).index(name) if name in PRIMARY + SECONDARY else 99])
    draws = rng.integers(0, len(units), size=(BOOT, len(units)))
    counts = np.zeros((BOOT, len(units)))
    for r in range(BOOT):
        counts[r] = np.bincount(draws[r], minlength=len(units))
    uidx = {u: i for i, u in enumerate(units)}
    boot_rf, boot_ref = {a: [] for a in curves}, []
    for s_ in range(0, BOOT, 500):
        W = {f: counts[s_:s_ + 500][:, [uidx[u] for u in stats[f][next(iter(stats[f]))]["units"]]] for f in stats}
        cb, rb, _ = score(name, stats, store, msq, budgets, W)
        for a in cb:
            boot_rf[a].append(cb[a])
        boot_ref.append(rb)
        log(f"bootstrap {s_ + 500}/{BOOT}")
    boot_rf = {a: np.vstack(v) for a, v in boot_rf.items()}
    boot_ref = np.concatenate(boot_ref)
    out = {"dataset": name, "budgets": common, "draws": DRAWS, "units": len(units), "cells": int(len(d["cell"])),
           "rf": rf, "rf_ci": {a: np.quantile(v, [.025, .975], axis=0).T.tolist() for a, v in boot_rf.items()},
           "reference_rf_paired_all": float(ref), "reference_rf_ci": np.quantile(boot_ref, [.025, .975]).tolist(),
           "fold_info": {str(f): {k: v for k, v in infos[f].items() if not k.endswith("_index")} for f in infos},
           "savings": {}}
    groups = {"paired_only": PAIRED_ONLY, "semicca": SEMICCA, "original_forms": ORIGINAL,
              "all_comparators": [a for a in curves if not a.startswith(("bjs2", "selftune", "abl_"))]}
    for label, arms in groups.items():
        arms = [a for a in arms if a in curves]
        out["savings"][label] = {"arms": arms}
        for frac in TARGETS:
            tgt = frac * ref
            fac, n_c, r_c, n_p, r_p = saving(rf, arms, tgt, common)
            bs = np.array([saving({a: boot_rf[a][r] for a in boot_rf}, arms, frac * boot_ref[r], common)[0]
                           for r in range(BOOT)])
            out["savings"][label][str(frac)] = {"target_rf": tgt, "factor": fac, "comparator_cells": n_c,
                                                "comparator_relation": r_c, "proposed_cells": n_p,
                                                "proposed_relation": r_p,
                                                "ci": np.quantile(bs, [.025, .975]).tolist(),
                                                "log_boot": np.log(bs).tolist()}
    # self-tuning arm against the fixed estimator (cells the fixed estimator needs / cells the self-tuned needs)
    out["selftune_vs_fixed"] = {}
    for frac in TARGETS:
        n_f = needed(rf[PROPOSED], frac * ref, common)
        n_s = needed(rf[SELFTUNED], frac * ref, common)
        bs = np.array([needed(boot_rf[PROPOSED][r], frac * boot_ref[r], common)[0]
                       / needed(boot_rf[SELFTUNED][r], frac * boot_ref[r], common)[0] for r in range(BOOT)])
        out["selftune_vs_fixed"][str(frac)] = {"fixed_cells": n_f[0], "fixed_relation": n_f[1],
                                               "selftuned_cells": n_s[0], "selftuned_relation": n_s[1],
                                               "factor": n_f[0] / n_s[0], "ci": np.quantile(bs, [.025, .975]).tolist(),
                                               "log_boot": np.log(bs).tolist()}
    (RES / f"{name}.json").write_text(json.dumps(out) + "\n")
    log(f"done: proposed {', '.join(f'{100 * v:.1f}' for v in rf[PROPOSED])}; saving vs paired-only at 0.5: "
        f"{out['savings']['paired_only']['0.5']['factor']:.2f} {out['savings']['paired_only']['0.5']['ci']}")


def summary():
    """G1-G3 (PLAN.md): geometric means over the primary data sets with bootstrap intervals that combine the r-th
    resample of every data set."""
    res = {n: json.loads((RES / f"{n}.json").read_text()) for n in PRIMARY if (RES / f"{n}.json").exists()}
    out = {"datasets": list(res), "missing": [n for n in PRIMARY if n not in res]}
    for label, key in (("G1_paired_only", "paired_only"), ("G2_semicca", "semicca"), ("original_forms", "original_forms")):
        for frac in ("0.5", "0.25", "0.75"):
            logs = np.array([np.log(res[n]["savings"][key][frac]["factor"]) for n in res])
            boot = np.mean([res[n]["savings"][key][frac]["log_boot"] for n in res], axis=0)
            out[f"{label}/{frac}"] = {"geometric_mean": float(np.exp(logs.mean())),
                                      "ci": np.exp(np.quantile(boot, [.025, .975])).tolist(),
                                      "per_dataset": {n: res[n]["savings"][key][frac]["factor"] for n in res},
                                      "lower_bound_above_one": [n for n in res if res[n]["savings"][key][frac]["ci"][0] > 1]}
    for frac in ("0.5", "0.25", "0.75"):
        logs = np.array([np.log(res[n]["selftune_vs_fixed"][frac]["factor"]) for n in res])
        boot = np.mean([res[n]["selftune_vs_fixed"][frac]["log_boot"] for n in res], axis=0)
        out[f"G3_selftune_vs_fixed/{frac}"] = {"geometric_mean": float(np.exp(logs.mean())),
                                               "ci": np.exp(np.quantile(boot, [.025, .975])).tolist(),
                                               "per_dataset": {n: res[n]["selftune_vs_fixed"][frac]["factor"] for n in res}}
    g1 = out["G1_paired_only/0.5"]
    g2 = out["G2_semicca/0.5"]
    out["verdicts"] = {"G1_met": bool(g1["ci"][0] > 1 and not out["missing"]),
                       "G2_met": bool(g2["ci"][0] > 1 and not out["missing"])}
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or "geometric_mean" not in v}, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "geometric_mean" in v:
            print(k, round(v["geometric_mean"], 2), [round(c, 2) for c in v["ci"]])


def check_freeze():
    rec = json.loads((RES / "freeze.json").read_text())
    bad = [f for f, dgt in rec["files"].items() if sha(HERE / f) != dgt]
    assert not bad, f"changed since the freeze: {bad}"
    return rec


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {sys.argv[2] if len(sys.argv) > 2 else ''}: {msg}", flush=True)
    with threadpool_limits(limits=1):
        if stage in ("run", "predict", "evaluate") and sys.argv[2] != "synthetic":
            check_freeze()
        if stage in ("run", "predict"):
            predict(sys.argv[2], log)
        if stage in ("run", "evaluate"):
            evaluate(sys.argv[2], log)
        if stage == "summary":
            check_freeze()
            summary()
