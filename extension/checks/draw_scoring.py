#!/usr/bin/env python3
"""Check that every recovery curve averages the scores of single draws of paired cells.

At each budget a test draws paired cells D times and scores each draw's prediction C_d against the held-out truth T
with G(C) = 2<C, T> - ||C||^2. The curve should be the mean of G(C_d), what one study with one draw can expect.
Scoring the averaged prediction instead would add the spread of the draws,

    G(mean_d C_d) = mean_d G(C_d) + mean_d ||C_d - mean_d C_d||^2,

and so credit a study with paired cells from D draws. The tests store the mean prediction and msq, the mean of the
per-draw squared norms, and compute 2<mean C, T> - msq, which equals mean_d G(C_d) exactly. This script checks that
the stored msq is mean_d ||C_d||^2 and not ||mean_d C_d||^2, by rerunning each test's frozen prediction code,
watching every draw's prediction as it is accumulated, scoring it on its own, and comparing:

  1. the mean of the regenerated predictions and of their squared norms with the stored means and msq;
  2. the curve averaged over per-draw scores with the published curve (for a partial rerun, the mean per-draw score
     of each rerun unit with the unit's stored term);
  3. the published curve with the one that scoring the averaged prediction would have given.

    python draw_scoring.py identity                     # the identity above on random matrices
    python draw_scoring.py external                     # per-draw predictions are stored; nothing to regenerate
    python draw_scoring.py deployment [--folds 0 1]     # bone-marrow test (all 12 held-out batches by default)
    python draw_scoring.py validation [--folds 0 1]     # validation test (all 34 held-out samples by default)
    python draw_scoring.py validation --from-review     # the same from the per-draw scores of the second review's
                                                        # regeneration (second_review/validation_review.py)
    python draw_scoring.py benchmark NAME [--folds 0]   # one benchmark data set (all three folds by default)
    python draw_scoring.py law NAME                     # one data set of the prospective test of the law
    python draw_scoring.py report                       # results/draw_scoring.json and a table
    python draw_scoring.py verify                       # recompute every curve from the per-draw scores in
                                                        # results/ and compare it with the published curves;
                                                        # for a partial rerun, every rerun unit's term (needs
                                                        # no data)

Regenerated prediction files go to results/scratch/ and are compared with the frozen ones; the frozen results are
only read. The development comparison needs no check: run_study.run_fold scores each draw as it is drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
OUT = HERE / "results"
SCRATCH = OUT / "scratch"
TOL_CURVE = 1e-6          # recovered fraction; the validation and bone-marrow tests store float32 mean predictions


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain(C, T):
    return 2.0 * float(np.sum(C * T)) - float(np.sum(C * C))


# ------------------------------------------------------------------ the identity

def identity(draws=5, seed=0):
    rng = np.random.default_rng(seed)
    T = 0.1 * rng.standard_normal((40, 15))
    C = T + 0.2 * rng.standard_normal((draws, 40, 15))
    mean = C.mean(0)
    per_draw = float(np.mean([gain(c, T) for c in C]))
    msq = float(np.mean([np.sum(c * c) for c in C]))
    spread = float(np.mean([np.sum((c - mean) ** 2) for c in C]))
    stored = 2.0 * float(np.sum(mean * T)) - msq
    averaged = gain(mean, T)
    assert abs(stored - per_draw) < 1e-9 and abs(averaged - per_draw - spread) < 1e-9
    out = {"mean of per-draw scores": per_draw, "stored form 2<mean C,T> - msq": stored,
           "score of the averaged prediction": averaged, "spread of the draws": spread}
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ watching a frozen function

class Watch:
    """Calls fn(locals) whenever the frozen function func reaches the source line containing marker.

    once: names of locals identifying one event; a loop header is reached again on every iteration, and only the
    first arrival per identity is passed on."""

    def __init__(self, func, marker, fn, once=None):
        lines, start = inspect.getsourcelines(func)
        hits = [start + i for i, s in enumerate(lines) if marker in s]
        assert len(hits) == 1, f"{marker!r} found {len(hits)} times in {func.__name__}"
        self.code, self.line, self.fn, self.once, self.seen = func.__code__, hits[0], fn, once, set()

    def local(self, frame, event, arg):
        if event == "line" and frame.f_lineno == self.line:
            loc = frame.f_locals
            if self.once:
                key = tuple(loc[k] for k in self.once)
                if key in self.seen:
                    return self.local
                self.seen.add(key)
            self.fn(loc)
        return self.local


class Watching:
    def __init__(self, *watches):
        self.by_code = {}
        for w in watches:
            self.by_code.setdefault(w.code, []).append(w)

    def tracer(self, frame, event, arg):
        ws = self.by_code.get(frame.f_code)
        if not ws:
            return None
        if len(ws) == 1:
            return ws[0].local

        def local(fr, ev, a):
            for w in ws:
                w.local(fr, ev, a)
            return local
        return local

    def __enter__(self):
        sys.settrace(self.tracer)
        return self

    def __exit__(self, *exc):
        sys.settrace(None)


# ------------------------------------------------------------------ per-draw records

class Draws:
    """Per unit (held-out fold or problem), arm, budget and condition: the regenerated predictions' running sum and
    sum of squared norms; per unit, arm, budget and draw: the score summed over conditions."""

    def __init__(self):
        self.sum, self.sq, self.n, self.score = {}, {}, {}, {}

    def add(self, unit, arm, B, dr, c, Q, T):
        Q = np.asarray(Q, np.float64)
        k = (unit, arm, int(B), c)
        self.sum[k] = self.sum.get(k, 0.0) + Q
        self.sq[k] = self.sq.get(k, 0.0) + float(np.sum(Q * Q))
        self.n[k] = self.n.get(k, 0) + 1
        s = self.score.setdefault((unit, arm, int(B)), {})
        s[int(dr)] = s.get(int(dr), 0.0) + gain(Q, T)

    def mean(self, k):
        return self.sum[k] / self.n[k], self.sq[k] / self.n[k]


def compare_stored(rec, stored_mean, stored_msq):
    """Largest differences between the regenerated means and the stored ones. stored_mean(k) and stored_msq(k)
    return the stored mean prediction and msq for key k = (unit, arm, budget, condition)."""
    d_mean, d_msq, n = 0.0, 0.0, 0
    draws = set()
    for k in rec.sum:
        m, q = rec.mean(k)
        P, s = stored_mean(k), stored_msq(k)
        d_mean = max(d_mean, float(np.max(np.abs(m - P))))
        d_msq = max(d_msq, abs(q - s) / max(abs(s), 1e-300))
        draws.add(rec.n[k])
        n += 1
    return {"compared": n, "draws_per_key": sorted(draws), "max_abs_mean_difference": d_mean,
            "max_rel_msq_difference": d_msq}


def summarize(test, rec, den, published, budgets, stored_terms, extra=None):
    """Compare per-draw scores with the stored-form terms unit by unit and, when every unit was regenerated, the
    pooled per-draw curves with the published curves; the curves that scoring the averaged prediction would give
    come from stored_terms (unit, arm, budget -> (2<P,T> - msq, 2<P,T> - ||P||^2)) over all units."""
    regen = sorted({u for (u, _, _) in rec.score})
    all_units = sorted({u for (u, _, _) in stored_terms})
    complete = regen == all_units
    unit_diff = 0.0
    for k, s in rec.score.items():
        unit_diff = max(unit_diff, abs(float(np.mean(list(s.values()))) - stored_terms[k][0]) / den[k[0]])

    def pooled(value, units, arm):
        vals = []
        for B in budgets:
            ks = [(u, arm, B) for u in units]
            ok = all(k in stored_terms for k in ks) and all(value(k) is not None for k in ks)
            vals.append(sum(value(k) for k in ks) / sum(den[u] for u in units) if ok else None)
        return vals

    arms = sorted({a for (_, a, _) in stored_terms})
    stored_curves = {a: pooled(lambda k: stored_terms[k][0], all_units, a) for a in arms}
    avg_curves = {a: pooled(lambda k: stored_terms[k][1], all_units, a) for a in arms}
    per_draw = {}
    if complete:
        per_draw = {a: pooled(lambda k: float(np.mean(list(rec.score[k].values()))) if k in rec.score else None,
                              all_units, a) for a in sorted({a for (_, a, _) in rec.score})}
    d_draw = d_stored = 0.0
    compared, inflation = 0, []
    for a in arms:
        if a not in published:
            continue
        for i, p in enumerate(published[a]):
            if p is None or not np.isfinite(p) or stored_curves[a][i] is None:
                continue
            d_stored = max(d_stored, abs(stored_curves[a][i] - p))
            inflation.append(avg_curves[a][i] - p)
            if a in per_draw and per_draw[a][i] is not None:
                d_draw = max(d_draw, abs(per_draw[a][i] - p))
                compared += 1
    inflation = np.array(inflation)
    draws = sorted({len(s) for s in rec.score.values()})
    out = {"test": test, "units_regenerated": len(regen), "units": len(all_units), "budgets": budgets,
           "draws": draws, "curve_points_compared": compared,
           "max_abs_difference_per_draw_vs_published": d_draw if complete else None,
           "max_abs_difference_per_draw_vs_stored_unit_terms": unit_diff,
           "max_abs_difference_stored_form_vs_published": d_stored,
           "averaged_prediction_minus_published": {
               "min": float(inflation.min()), "median": float(np.median(inflation)), "max": float(inflation.max()),
               "points": int(inflation.size)} if inflation.size else None,
           "curves_from_per_draw_scores": per_draw, "curves_from_averaged_predictions": avg_curves,
           "published": {a: published[a] for a in arms if a in published},
           "per_draw_scores": {f"{u}|{a}|{B}": [s[d] for d in sorted(s)] for (u, a, B), s in rec.score.items()},
           "denominators": {str(u): den[u] for u in all_units},
           "stored_terms": {f"{u}|{a}|{B}": list(v) for (u, a, B), v in stored_terms.items()}}
    if extra:
        out.update(extra)
    out["passed"] = bool(unit_diff <= TOL_CURVE and d_stored <= TOL_CURVE and d_draw <= TOL_CURVE
                         and (inflation.size == 0 or inflation.min() >= -TOL_CURVE))
    return out


def write(name, out):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    brief = {k: v for k, v in out.items() if k not in ("curves_from_per_draw_scores", "curves_from_averaged_predictions",
                                                       "published", "per_draw_scores", "denominators",
                                                       "stored_terms")}
    log(json.dumps(brief, indent=1))


# ------------------------------------------------------------------ external test (per-draw predictions stored)

def external():
    sys.path.insert(0, str(EXT / "semipaired" / "external"))
    sys.path.insert(1, str(EXT / "semipaired"))
    import run_external as rx
    res = rx.HERE / "results"
    man = json.loads((res / "manifest.json").read_text())
    assert sha(res / "predictions.npz") == man["predictions.npz"]
    info = json.loads((res / "predict_info.json").read_text())
    parts = [rx.od.load(p, info["genes"]) for p in ("test_adaptation", "test_scoring")]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "library", "y", "part", "target", "condition")}
    stats = rx.tg.group_stats(cat["counts"], cat["library"], cat["y"], rx.pm.keys_of(cat), cat["part"])
    T = rx.tg.pooled(stats, np.ones(len(stats)))
    conds = sorted(T)
    den = {"all": sum(float(np.sum(T[c]["TA"] * T[c]["TB"])) for c in conds)}
    z = np.load(res / "predictions.npz")
    rec = Draws()
    for key in z.files:
        arm, mode, B, dr, c = key.split("/")
        if B == "0":
            continue
        rec.add("all", f"{arm}/{mode}", int(B), int(dr), c, z[key], T[c]["T"])
    published = json.loads((res / "overcite.json").read_text())
    budgets = [int(b) for b in published["budgets"]]
    terms = {}
    for (u, a, B), _ in rec.score.items():
        ip, msq, sqm = 0.0, 0.0, 0.0
        for c in conds:
            m, q = rec.mean((u, a, B, c))
            ip += float(np.sum(m * T[c]["T"]))
            msq += q
            sqm += float(np.sum(m * m))
        terms[(u, a, B)] = (2 * ip - msq, 2 * ip - sqm)
    out = summarize("external", rec, den, published["rf"], budgets, terms,
                    {"prediction_file_sha256_matches_manifest": True,
                     "draws": published["draws"], "note": "per-draw predictions are stored; scored directly"})
    write("external", out)


# ------------------------------------------------------------------ bone-marrow and validation tests

def fold_test(test, mod, fold_ids):
    """drun (bone marrow) and vrun (validation) share one layout: per fold, arm, budget and condition a stored mean
    prediction P/... and msq/..., and a predict() whose draw loop ends in 'for (arm, c), Q in preds.items():'."""
    res = mod.RES
    man = json.loads((res / "manifest.json").read_text())
    assert sha(res / "predictions.npz") == man["predictions"], "stored predictions differ from their manifest"
    stored = np.load(res / "predictions.npz")
    info = json.loads((res / "predictions_info.json").read_text())
    folds_info = info["folds"] if "folds" in info else info
    summary = json.loads((res / "summary.json").read_text())
    truth, den = {}, {}
    rec = Draws()

    def seen(loc):
        f = str(loc["f"])
        if f not in truth:
            fd = loc["fd"]
            if test == "validation":
                st = mod.heldout(loc["new"], fd, loc["half"], loc["gidx"], loc["yidx"])
            else:
                st = mod.heldout(loc["cite"], fd, loc["half"], loc["genes"], loc["yp"])
            truth[f] = {c: st[c]["T"] for c in mod.CONDS}
            den[f] = sum(float(np.sum(st[c]["TA"] * st[c]["TB"])) for c in mod.CONDS)
        for (arm, c), Q in loc["preds"].items():
            rec.add(f, arm, loc["B"], loc["dr"], c, Q, truth[f][c])

    scratch = SCRATCH / test
    scratch.mkdir(parents=True, exist_ok=True)
    w = Watch(mod.predict, "for (arm, c), Q in preds.items():", seen, once=("f", "B", "dr"))
    t0 = time.time()
    with threadpool_limits(limits=2), Watching(w):
        mod.predict(out_dir=scratch, fold_ids=fold_ids, log=log)
    seconds = time.time() - t0
    same_file = sha(scratch / "predictions.npz") == man["predictions"]
    check = compare_stored(rec, lambda k: stored[f"P/{k[0]}/{k[1]}/{k[2]}/{k[3]}"].astype(np.float64),
                           lambda k: float(stored[f"msq/{k[0]}/{k[1]}/{k[2]}/{k[3]}"]))
    # stored-form and averaged-prediction terms for every fold (truth for folds not regenerated)
    terms = stored_fold_terms(test, mod, stored, folds_info, info, truth, den)
    budgets = [int(b) for b in summary["budgets"]]
    out = summarize(test, rec, den, summary["rf"], budgets, terms,
                    {"regenerated_prediction_file_identical": bool(same_file), "stored_vs_regenerated": check,
                     "seconds": round(seconds)})
    shutil.rmtree(scratch, ignore_errors=True)
    tag = test if fold_ids is None else f"{test}_folds{'_'.join(map(str, fold_ids))}"
    write(tag, out)


def stored_fold_terms(test, mod, stored, folds_info, info, truth, den):
    need = [f for f in folds_info if f not in truth]
    if need:
        if test == "validation":
            new = mod.load_new()
            _, half, _ = mod.roles(new)
            yn = {n: i for i, n in enumerate(new["y_names"])}
            gn = {g: i for i, g in enumerate(new["x_names"])}
            gidx = np.array([gn[g] for g in info["panels"]["genes"]])
            yidx = np.array([yn[p] for p in info["panels"]["proteins_new"]])
            fds = {str(fd["fold"]): fd for fd in mod.folds(new)}
            for f in need:
                st = mod.heldout(new, fds[f], half, gidx, yidx)
                truth[f] = {c: st[c]["T"] for c in mod.CONDS}
                den[f] = sum(float(np.sum(st[c]["TA"] * st[c]["TB"])) for c in mod.CONDS)
        else:
            cite, mult = mod.load("bmmc_cite"), mod.load("bmmc_multiome")
            _, half, _ = mod.roles(cite)
            fds = {str(fd["fold"]): fd for fd in mod.folds(cite, mult)}
            for f in need:
                fi = folds_info[f]
                yp = np.array([list(cite["y_names"]).index(p) for p in fi["proteins"]])
                st = mod.heldout(cite, fds[f], half, fi["genes"], yp)
                truth[f] = {c: st[c]["T"] for c in mod.CONDS}
                den[f] = sum(float(np.sum(st[c]["TA"] * st[c]["TB"])) for c in mod.CONDS)
    terms = {}
    for f, fi in folds_info.items():
        for arm in mod.ARMS:
            for B in fi["budgets"]:
                ip = msq = sqm = 0.0
                found = False
                for c in mod.CONDS:
                    k = f"{f}/{arm}/{B}/{c}"
                    if f"P/{k}" not in stored.files:
                        continue
                    found = True
                    P = stored[f"P/{k}"]
                    ip += float(np.sum(P * truth[f][c]))
                    msq += float(stored[f"msq/{k}"])
                    sqm += float(np.sum(P.astype(np.float64) ** 2))
                if found:
                    terms[(f, arm, int(B))] = (2 * ip - msq, 2 * ip - sqm)
    return terms


def validation_from_review():
    """The validation's check from second_review/validation_review.py, which regenerated every draw of every fold with
    the validation's frozen code, checked the means and msq against the stored ones and kept each draw's score."""
    sys.path.insert(0, str(EXT / "validation"))
    import vrun
    res = vrun.RES
    stored = np.load(res / "predictions.npz")
    info = json.loads((res / "predictions_info.json").read_text())
    summary = json.loads((res / "summary.json").read_text())
    review = sorted((EXT / "second_review" / "results" / "validation").glob("fold*.npz"))
    sys.path.insert(0, str(EXT / "second_review"))
    import validation_review as vr
    rec, den, check = Draws(), {}, {"compared": 0, "max_abs_mean_difference": 0.0, "max_rel_msq_difference": 0.0}
    for p in review:
        z = np.load(p)
        f, budgets = str(int(z["fold"])), [int(b) for b in z["budgets"]]
        den[f] = float(np.sum(z["den/primary"]))
        sc = z["score"][..., vr.TRUTHS.index("primary"), 0]          # budgets x draws x arms x types
        for ai, arm in enumerate(vr.ARMS):
            if arm not in vrun.ARMS:
                continue
            for bi, B in enumerate(budgets):
                if B not in vrun.BUDGETS:
                    continue
                for dr in range(sc.shape[1]):
                    v = sc[bi, dr, ai]
                    if np.all(np.isnan(v)):
                        continue
                    rec.score.setdefault((f, arm, B), {})[dr] = float(np.nansum(v))
        c = z["check"]
        check["max_abs_mean_difference"] = max(check["max_abs_mean_difference"], float(c[0]))
        check["max_rel_msq_difference"] = max(check["max_rel_msq_difference"], float(c[1]))
        check["compared"] += int(c[2])
    check["draws_per_key"] = sorted({len(v) for v in rec.score.values()})
    terms = stored_fold_terms("validation", vrun, stored, info["folds"], info, {}, den)
    out = summarize("validation", rec, den, summary["rf"], [int(b) for b in summary["budgets"]], terms,
                    {"stored_vs_regenerated": check, "source": "second_review/validation_review.py"})
    write("validation", out)


# ------------------------------------------------------------------ benchmark

def benchmark(name, fold_ids):
    sys.path.insert(0, str(EXT / "generality"))
    import grun
    res = grun.RES
    man = json.loads((res / f"{name}_manifest.json").read_text())
    for key, fn in (("predictions", f"{name}_predictions.npz"), ("msq", f"{name}_msq.json"),
                    ("info", f"{name}_info.json")):
        assert sha(res / fn) == man[key], f"{fn} differs from its manifest"
    stored = np.load(res / f"{name}_predictions.npz")
    msq = {int(f): m for f, m in json.loads((res / f"{name}_msq.json").read_text()).items()}
    infos = {int(f): i for f, i in json.loads((res / f"{name}_info.json").read_text()).items()}
    published = json.loads((res / f"{name}.json").read_text())
    d = grun.load(grun.source(name))
    fold, part, half = grun.roles(grun.source(name), d)
    truth, den = {}, {}
    for f in range(grun.FOLDS):
        st = grun.heldout_stats(d, f, fold, half, np.array(infos[f]["x_panel_index"]),
                                np.array(infos[f]["y_panel_index"]), infos[f]["conditions"])
        truth[f], den[f] = {}, 0.0
        for c, rec_c in st.items():
            T = grun.targets(rec_c, np.ones((1, len(rec_c["units"]))))
            truth[f][c] = T["T"][0]
            den[f] += float(np.sum(T["TA"][0] * T["TB"][0]))
    mapped_arms = ("bjs2", "selftune") + grun.COMPARATORS
    rec = Draws()
    f_now = {}

    def pooled(loc):
        f, B, dr, arm, P, ctx = f_now["f"], loc["B"], loc["dr"], loc["arm"], loc["P"], loc["ctx"]
        for c in loc["conds"]:
            rec.add(f, f"{arm}/pooled", B, dr, c, P, truth[f][c])
            if arm in mapped_arms:
                rec.add(f, f"{arm}/mapped", B, dr, c, ctx.mapped(P, c), truth[f][c])

    def percond(loc):
        f = f_now["f"]
        rec.add(f, f"{loc['arm']}/percond", loc["B"], loc["dr"], loc["c"], loc["Q"], truth[f][loc["c"]])

    def pairbasis(loc):
        f = f_now["f"]
        for c in loc["conds"]:
            rec.add(f, "abl_pairbasis/pooled", loc["B"], loc["dr"], c, loc["Pb"], truth[f][c])

    fe = grun.estimate_fold
    watches = (Watch(fe, 'acc[f"{arm}/P"] = acc.get(f"{arm}/P", 0.0) + P / DRAWS', pooled),
               Watch(fe, 'acc[f"{arm}/percond/Q/{c}"] = acc.get(f"{arm}/percond/Q/{c}", 0.0) + Q / DRAWS', percond),
               Watch(fe, 'acc["abl_pairbasis/P"] = acc.get("abl_pairbasis/P", 0.0) + Pb / DRAWS', pairbasis))
    regen = list(range(grun.FOLDS)) if fold_ids is None else list(fold_ids)
    worst_store, worst_msq, n_keys = 0.0, 0.0, 0
    t0 = time.time()
    for f in regen:
        prep = grun.prepare_crossanimal if name == "banc_crossanimal" else grun.prepare_fold
        train, x, y, pop, pn, counts = prep(d, f, fold, part)
        f_now["f"] = f
        with threadpool_limits(limits=1), Watching(*watches):
            s_, m, _ = grun.estimate_fold(name, f, train, x, y, pop, counts, log)
        for k, v in s_.items():
            worst_store = max(worst_store, float(np.max(np.abs(np.asarray(v, np.float64) - stored[k]))))
            n_keys += 1
        for k, v in m.items():
            worst_msq = max(worst_msq, abs(v - msq[f][k]) / max(abs(msq[f][k]), 1e-300))
        log(f"{name} fold {f} regenerated [{time.time() - t0:.0f}s]")

    def stored_mean(k):
        f, arm, B, c = k
        base, mode = arm.split("/")
        return grun.prediction(stored, f, arm, B, c) if mode != "percond" else stored[f"{f}/{base}/percond/{B}/{c}"]

    def stored_msq(k):
        f, arm, B, c = k
        base, mode = arm.split("/")
        return msq[f][f"{f}/{base}/{mode}/{B}/{c}"]

    check = compare_stored(rec, stored_mean, stored_msq)
    check["rerun_fold_outputs"] = {"keys": n_keys, "max_abs_difference_means_and_maps": worst_store,
                                   "max_rel_difference_msq": worst_msq}
    terms = {}
    for f in range(grun.FOLDS):
        conds = list(truth[f])
        for arm, B in grun.arm_list(stored, f, conds):
            ip = ms = sqm = 0.0
            for c in conds:
                P = stored_mean((f, arm, B, c))
                ip += float(np.sum(P * truth[f][c]))
                ms += stored_msq((f, arm, B, c))
                sqm += float(np.sum(P * P))
            terms[(f, arm, int(B))] = (2 * ip - ms, 2 * ip - sqm)
    budgets = [int(b) for b in published["budgets"]]
    out = summarize(f"benchmark/{name}", rec, den, published["rf"], budgets, terms,
                    {"stored_vs_regenerated": check, "seconds": round(time.time() - t0)})
    tag = f"benchmark_{name}" if fold_ids is None else f"benchmark_{name}_folds{'_'.join(map(str, fold_ids))}"
    write(tag, out)


# ------------------------------------------------------------------ prospective test of the law

def law(name):
    sys.path.insert(0, str(EXT / "predictability"))
    import prun
    import ptools as pt
    res = prun.RES
    man = json.loads((res / f"{name}_manifest.json").read_text())
    for k, fn in (("estimates", f"{name}_estimates.npz"), ("msq", f"{name}_msq.json"),
                  ("predictions", f"{name}_predictions.json")):
        assert sha(res / fn) == man[k], f"{fn} differs from its manifest"
    stored = np.load(res / f"{name}_estimates.npz")
    msq = json.loads((res / f"{name}_msq.json").read_text())
    published = json.loads((res / f"{name}.json").read_text())["problems"]
    scratch = SCRATCH / f"law_{name}"
    scratch.mkdir(parents=True, exist_ok=True)
    for fn in ("freeze.json", "transfer_profile.json"):
        shutil.copy(res / fn, scratch / fn)
    truth, den = {}, {}
    rec = Draws()

    def seen(loc):
        tag = loc["tag"]
        if tag not in truth:
            Pt = pt.Problem(loc["d"], n_x=int(tag[1:]))
            assert np.array_equal(Pt.U, loc["P"].U)
            Pt.score_setup()
            truth[tag], den[tag] = Pt.T, float(Pt.den)
        for k, v in (("bjs2", loc["A"]), ("js", loc["A1"]), ("random", loc["Ar"])):
            rec.add(tag, k, loc["B"], loc["dr"], "all", v, truth[tag])

    w = Watch(prun.predict, 'for k, v in (("bjs2", A), ("js", A1), ("random", Ar)):', seen, once=("tag", "B", "dr"))
    t0 = time.time()
    prun.RES = scratch
    try:
        with threadpool_limits(limits=1), Watching(w):
            prun.predict(name, log)
    finally:
        prun.RES = res
    same = {fn: sha(scratch / fn) == sha(res / fn) for fn in (f"{name}_estimates.npz", f"{name}_msq.json",
                                                            f"{name}_predictions.json")}
    check = compare_stored(rec, lambda k: stored[f"{k[0]}/{k[1]}/{k[2]}"], lambda k: msq[f"{k[0]}/{k[1]}/{k[2]}"])
    # per problem, curves are not pooled: compare each problem's curve with its published one
    worst, worst_stored, inflation, points = 0.0, 0.0, [], 0
    per_draw_curves, avg_curves = {}, {}
    for tag, pr in published.items():
        b = pr["budgets"]
        for k in ("bjs2", "js", "random"):
            v = [float(np.mean(list(rec.score[(tag, k, B)].values()))) / den[tag] for B in b]
            s = [(2 * float(np.sum(stored[f"{tag}/{k}/{B}"] * truth[tag])) - msq[f"{tag}/{k}/{B}"]) / den[tag]
                 for B in b]
            a = [(2 * float(np.sum(stored[f"{tag}/{k}/{B}"] * truth[tag]))
                  - float(np.sum(stored[f"{tag}/{k}/{B}"] ** 2))) / den[tag] for B in b]
            p = pr["curves"][k]
            worst = max(worst, max(abs(x - y) for x, y in zip(v, p)))
            worst_stored = max(worst_stored, max(abs(x - y) for x, y in zip(s, p)))
            inflation += [x - y for x, y in zip(a, p)]
            points += len(b)
            per_draw_curves[f"{tag}/{k}"], avg_curves[f"{tag}/{k}"] = v, a
    inflation = np.array(inflation)
    out = {"test": f"law/{name}", "problems": sorted(published), "curve_points_compared": points,
           "max_abs_difference_per_draw_vs_published": worst,
           "max_abs_difference_stored_form_vs_published": worst_stored,
           "averaged_prediction_minus_published": {"min": float(inflation.min()), "median": float(np.median(inflation)),
                                                   "max": float(inflation.max()), "points": int(inflation.size)},
           "regenerated_files_identical": same, "stored_vs_regenerated": check, "seconds": round(time.time() - t0),
           "curves_from_per_draw_scores": per_draw_curves, "curves_from_averaged_predictions": avg_curves,
           "published": {f"{t}/{k}": published[t]["curves"][k] for t in published for k in ("bjs2", "js", "random")},
           "per_draw_scores": {f"{u}|{a}|{B}": [s[d] for d in sorted(s)] for (u, a, B), s in rec.score.items()},
           "denominators": den}
    out["passed"] = bool(worst <= TOL_CURVE and worst_stored <= TOL_CURVE and inflation.min() >= -TOL_CURVE)
    shutil.rmtree(scratch, ignore_errors=True)
    write(f"law_{name}", out)


# ------------------------------------------------------------------ report and data-free verification

def verify():
    """Recompute every curve from the per-draw scores and denominators in results/*.json and compare it with the
    published curves stored alongside them. A benchmark data set was rerun for the first of its three folds, so its
    per-draw scores cover one held-out unit of three: there the mean per-draw score of each rerun unit is compared with
    the unit's stored term, and the published curve with the curve pooled from the stored terms of all three units."""
    rows = []
    print(f"{'test':<28} {'units rerun':>11} {'curve points':>13} {'unit terms':>11}   largest difference")
    for p in sorted(OUT.glob("*.json")):
        if p.name == "draw_scoring.json":
            continue
        r = json.loads(p.read_text())
        worst, points, terms = 0.0, 0, 0
        if r["test"].startswith("law/"):
            for key, pub in r["published"].items():
                tag, k = key.split("/")
                den = r["denominators"][tag]
                b = [int(x.split("|")[2]) for x in r["per_draw_scores"] if x.startswith(f"{tag}|{k}|")]
                vals = {B: np.mean(r["per_draw_scores"][f"{tag}|{k}|{B}"]) / den for B in b}
                for B, v in sorted(vals.items()):
                    i = sorted(vals).index(B)
                    worst, points = max(worst, abs(v - pub[i])), points + 1
            rerun = f"{len(r['problems'])} of {len(r['problems'])}"
        else:
            den = r["denominators"]
            complete = r["units_regenerated"] == r["units"]
            # each unit's score at an arm and budget: its mean per-draw score, or for a partial rerun its stored term
            score = ({k: float(np.mean(s)) for k, s in r["per_draw_scores"].items()} if complete
                     else {k: t[0] for k, t in r["stored_terms"].items()})
            for a, pub in r["published"].items():
                for i, B in enumerate(r["budgets"]):
                    ks = [f"{u}|{a}|{B}" for u in den]
                    if pub[i] is None or not all(k in score for k in ks):
                        continue
                    v = sum(score[k] for k in ks) / sum(den.values())
                    worst, points = max(worst, abs(v - pub[i])), points + 1
            if not complete:
                for k, s in r["per_draw_scores"].items():
                    unit = k.split("|")[0]
                    worst, terms = max(worst, abs(float(np.mean(s)) - r["stored_terms"][k][0]) / den[unit]), terms + 1
            rerun = f"{r['units_regenerated']} of {r['units']}"
        rows.append((r["test"], points + terms, worst))
        print(f"{r['test']:<28} {rerun:>11} {points:>13} {terms or '':>11}   {worst:.2e}")
    assert all(w <= TOL_CURVE for _, n, w in rows if n), "a curve differs from the mean of its per-draw scores"
    return rows


def report():
    rows = []
    for p in sorted(OUT.glob("*.json")):
        if p.name == "draw_scoring.json":
            continue
        r = json.loads(p.read_text())
        chk = r.get("stored_vs_regenerated", {})
        infl = r.get("averaged_prediction_minus_published") or {}
        rows.append({"test": r["test"], "passed": r["passed"],
                     "units": f"{r.get('units_regenerated', len(r.get('problems', [])))}/"
                              f"{r.get('units', len(r.get('problems', [])))}",
                     "curve_points": r["curve_points_compared"],
                     "per_draw_vs_published": r.get("max_abs_difference_per_draw_vs_published"),
                     "per_draw_vs_stored_unit_terms": r.get("max_abs_difference_per_draw_vs_stored_unit_terms"),
                     "stored_form_vs_published": r["max_abs_difference_stored_form_vs_published"],
                     "mean_difference": chk.get("max_abs_mean_difference"),
                     "msq_rel_difference": chk.get("max_rel_msq_difference"),
                     "draws": chk.get("draws_per_key"),
                     "averaged_minus_published_median": infl.get("median"),
                     "averaged_minus_published_max": infl.get("max"),
                     "files_identical": r.get("regenerated_prediction_file_identical", r.get("regenerated_files_identical"))})
    (OUT / "draw_scoring.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                                                       "tests": rows}, indent=1) + "\n")
    for r in rows:
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("identity", "external", "deployment", "validation", "benchmark", "law", "report",
                                     "verify"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--from-review", action="store_true")
    a = ap.parse_args()
    if a.what == "identity":
        identity()
    elif a.what == "external":
        with threadpool_limits(limits=1):
            external()
    elif a.what == "deployment":
        sys.path.insert(0, str(EXT / "deployment"))
        import drun
        fold_test("deployment", drun, a.folds)
    elif a.what == "validation" and a.from_review:
        validation_from_review()
    elif a.what == "validation":
        sys.path.insert(0, str(EXT / "validation"))
        import vrun
        fold_test("validation", vrun, a.folds)
    elif a.what == "benchmark":
        benchmark(a.name, a.folds)
    elif a.what == "law":
        law(a.name)
    elif a.what == "report":
        report()
    else:
        verify()


if __name__ == "__main__":
    main()
