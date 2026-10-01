#!/usr/bin/env python3
"""External evaluation (PLAN.md): predictions before the held-out cells are opened, then scoring.

    python run_external.py panel      # the gene panel from training cells (results/panel.json; done before PLAN.md)
    python run_external.py predict    # every prediction from training cells only; hashed in results/manifest.json
    python run_external.py evaluate   # open the held-out files, check hashes, score, bootstrap, verdicts

`predict` refuses to run unless the files listed in FROZEN match results/freeze.json, and it never reads a
held-out file. `evaluate` checks the freeze and the prediction hashes before it reads the held-out files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import odata as od  # noqa: E402
import targets as tg  # noqa: E402
import common as cm  # noqa: E402
import run_study as rs  # noqa: E402
import champ_tune as ct  # noqa: E402

pm = cm.pm
BUDGETS = (25, 50, 100, 200, 400)
DRAWS = 20
SEED = 20261040
TARGET = 0.3                      # primary accuracy target (recovered fraction)
SECONDARY_TARGETS = (0.2, 0.4)
BOOT, BOOT_SEED = 2000, 20261045
PROPOSED = "bjs/mapped"
PAIRED_ONLY = [f"{d}/{m}" for d in ("js", "scose", "fcose", "lowrank") for m in ("pooled", "percond")]
SEMICCA = ["semicca/pooled", "semicca/percond"]
CHAMPOLLION = ["champollion/transport"]
ORIGINAL_FORMS = (PAIRED_ONLY + SEMICCA + CHAMPOLLION
                  + [f"{d}/{m}" for d in ("ridge_p", "ridge_u") for m in ("pooled", "percond")])
NONINFERIORITY = 0.8              # H3: the proposed estimator needs at most 1.25 times Champollion's paired cells
FROZEN = ["PLAN.md", "odata.py", "extract.py", "run_external.py", "targets.py", "freeze.py", "pilot.py",
          "pilot_champ.py", "results/pilot.json", "results/pilot_champ.json", "../run_study.py", "../common.py",
          "../sp_estimators.py", "../competitors.py", "../champ_worker.py", "../champ_tune.py",
          "../logs/champ_choice.json", "../../hybrid/hmethods.py", "../../hybrid/run.py",
          "../../perturbation/pmethods.py", "../../recoverability/noise.py", "results/seal.json",
          "results/panel.json"]
SCRATCH = Path("/tmp/claude-0/-home-claude/fa2f5e47-7e0a-544e-ba47-35b58380f98c/scratchpad")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_freeze():
    rec = json.loads((HERE / "results/freeze.json").read_text())
    bad = [f for f in FROZEN if rec["files"].get(f) != sha(HERE / f)]
    assert not bad, f"changed since the freeze: {bad}"
    return rec


def panel():
    d = od.load("train")
    X = d["sparse"].tocoo()
    n = X.shape[0]
    genes = d["all_genes"]
    v = np.log1p(1e4 * X.data / d["library"][X.row])
    det = np.bincount(X.col, minlength=len(genes)).astype(float)
    s1 = np.bincount(X.col, weights=v, minlength=len(genes))
    s2 = np.bincount(X.col, weights=v * v, minlength=len(genes))
    frac, mean = det / n, s1 / n
    var = s2 / n - mean ** 2
    index = {g: i for i, g in enumerate(genes)}
    seal = od.seal()
    orf_genes = set(seal["training_orfs"]) | set(seal["heldout_orfs"])     # transgenes, not endogenous expression
    encoding = []
    for names in od.ENCODING.values():
        for g in names:
            if (g in index and frac[index[g]] >= od.ENCODING_MIN_DETECTION and g not in encoding
                    and g not in orf_genes):
                encoding.append(g)
    ok = np.array([frac[i] >= od.HVG_MIN_DETECTION and not g.startswith(("MT-", "RPL", "RPS")) and g not in encoding
                   and g not in orf_genes for i, g in enumerate(genes)])
    cand = np.flatnonzero(ok)
    disp = np.log(var[cand] / mean[cand])
    edges = np.quantile(mean[cand], np.linspace(0, 1, od.HVG_BINS + 1))
    b = np.clip(np.searchsorted(edges, mean[cand], side="right") - 1, 0, od.HVG_BINS - 1)
    z = np.zeros(len(cand))
    for k in range(od.HVG_BINS):
        m = b == k
        if m.sum() > 1:
            z[m] = (disp[m] - disp[m].mean()) / disp[m].std()
    order = sorted(range(len(cand)), key=lambda j: (-z[j], genes[cand[j]]))
    hvg = [str(genes[cand[j]]) for j in order[:od.PANEL_SIZE - len(encoding)]]
    rec = {"encoding": [str(g) for g in encoding], "hvg": hvg, "genes": [str(g) for g in encoding] + hvg,
           "proteins": d["proteins"], "n_train": int(n), "excluded_orf_genes": sorted(orf_genes)}
    (HERE / "results/panel.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(len(rec["genes"]), "genes;", rec["encoding"])


def champollion(X, Y, pools, conds, gamma, choice, dr):
    """Champollion fitted on the B paired cells; transport between each condition's unpaired RNA and protein pool
    cells; the cross-correlation of each plan."""
    inp, outp = SCRATCH / "ext_champ_in.npz", SCRATCH / "ext_champ_out.npz"
    arrays = {"Xb": X, "Yb": Y, "epsilon": choice["epsilon"], "gamma": gamma, "max_iter": choice["max_iter"],
              "seed": dr}
    for i, c in enumerate(conds):
        arrays[f"Xg_{i:04d}"], arrays[f"Yg_{i:04d}"] = pools[c][0], pools[c][1]
    np.savez(inp, **arrays)
    subprocess.run([str(ct.VENV), str(HERE.parent / "champ_worker.py"), str(inp), str(outp)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    z = np.load(outp)
    return [z[f"C_{i:04d}"] for i in range(len(conds))], float(z["fit_seconds"])


def predict():
    freeze = check_freeze()
    t0 = time.time()
    genes = json.loads((HERE / "results/panel.json").read_text())["genes"]
    train = od.load("train", genes)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], od.MIN_TRAIN_HALF)
    pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, keys)
    assert np.all(pool.sd_x > 0) and np.all(pool.sd_y > 0), "a panel feature is constant in the pool"
    ctx = rs.build_context(train, x, y, pop, keys, pool, un)
    conds = un.conds
    use = np.isin(pop, pool.keys) & ~cm.hm.reservoir(train["cell"])
    pools = ct.condition_pools(pop[use], train["part"][use], x[use], y[use], pool, un)
    choice = json.loads((HERE.parent / "logs/champ_choice.json").read_text())
    preds = {}
    seconds = {"bjs2": 0.0, "champollion": 0.0, "champollion_fit": 0.0}
    # fully paired reference: every training cell of the pool populations, paired, per condition
    for c in conds:
        acc, n = 0.0, 0
        for k in [k for k in pool.keys if cm.cond_of(k) == c]:
            m = pop == k
            xs, _ = pm.standardize(x[m])
            ys, _ = pm.standardize(y[m])
            acc = acc + xs.T @ ys
            n += int(m.sum())
        preds[f"paired_all/all/0/0/{c}"] = acc / n
    for B in BUDGETS:
        for dr in range(DRAWS):
            rng = np.random.default_rng([SEED, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            X, Y, cc = X_res[pick], Y_res[pick], cond_res[pick]
            for name in rs.DENOISERS:
                crng = np.random.default_rng([SEED + 1, B, dr])
                t1 = time.time()
                P = rs.denoise(name, X, Y, ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y, ctx.Uz, ctx.blocks, crng)
                if name == "bjs2":
                    _ = [ctx.mapped(P, c) for c in conds]
                    seconds["bjs2"] += time.time() - t1
                for c in conds:
                    m = cc == c
                    d = ctx.cond[c]
                    preds[f"{name}/pooled/{B}/{dr}/{c}"] = P
                    preds[f"{name}/mapped/{B}/{dr}/{c}"] = ctx.mapped(P, c)
                    preds[f"{name}/percond/{B}/{dr}/{c}"] = (
                        rs.denoise(name, X[m], Y[m], d["Rx"], d["Ry"], d["n_x"], d["n_y"], d["Uz"], ctx.blocks, crng)
                        if m.sum() >= 6 else np.zeros_like(P))
            gamma = choice["gamma_by_budget"][str(B)]
            t1 = time.time()
            champ, fit_s = champollion(X, Y, pools, conds, gamma, choice, dr)
            seconds["champollion"] += time.time() - t1
            seconds["champollion_fit"] += fit_s
            for c, C in zip(conds, champ):
                preds[f"champollion/transport/{B}/{dr}/{c}"] = C
        print(f"[{time.time() - t0:.0f}s] budget {B}", flush=True)
    np.savez_compressed(HERE / "results/predictions.npz", **{k: v.astype(np.float64) for k, v in preds.items()})
    info = {"conditions": conds, "reliability": ctx.reliability, "n_reservoir": int(len(X_res)),
            "reservoir_by_condition": {c: int(np.sum(cond_res == c)) for c in conds},
            "n_pool_keys": len(pool.keys), "pool_keys": list(map(str, pool.keys)), "genes": genes,
            "champollion_pool_cells": {c: [int(len(pools[c][0])), int(len(pools[c][1]))] for c in conds},
            "seconds": round(time.time() - t0),
            "seconds_per_fit": {k: v / (len(BUDGETS) * DRAWS) for k, v in seconds.items()}}
    (HERE / "results/predict_info.json").write_text(json.dumps(info, indent=1) + "\n")
    manifest = {"predictions.npz": sha(HERE / "results/predictions.npz"),
                "predict_info.json": sha(HERE / "results/predict_info.json"),
                "freeze.json": sha(HERE / "results/freeze.json"), "frozen_at": freeze["frozen_at"],
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "note": "written before any held-out file was opened"}
    (HERE / "results/manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def needed(values, target, budgets=None):
    """Paired cells to reach target on the running maximum of a curve (log-linear interpolation); a curve that
    starts above the target needs at most the smallest budget, one that never reaches it more than the largest."""
    env = np.maximum.accumulate(np.nan_to_num(np.asarray(values, float), nan=-np.inf))
    b = np.asarray(BUDGETS if budgets is None else budgets, float)
    if target <= env[0]:
        return float(b[0]), "<="
    if target > env[-1]:
        return float(b[-1]), ">"
    i = int(np.argmax(env >= target))
    lo, hi = env[i - 1], env[i]
    f = (target - lo) / (hi - lo) if hi > lo else 1.0
    return float(np.exp(np.log(b[i - 1]) + f * (np.log(b[i]) - np.log(b[i - 1])))), "="


def curves(stats, weights, mean_pred, msq):
    """Recovered fraction and uncorrected relative loss of every arm and budget for one weighting of the held-out
    populations."""
    T = tg.pooled(stats, weights)
    conds = sorted(T)
    den = sum(float(np.sum(T[c]["TA"] * T[c]["TB"])) for c in conds)
    tn = sum(float(np.sum(T[c]["T"] ** 2)) for c in conds)
    rf, loss = {}, {}
    for arm, byb in mean_pred.items():
        rf[arm], loss[arm] = [], []
        for B in sorted(byb, key=int):
            ip = sum(float(np.sum(byb[B][c] * T[c]["T"])) for c in conds)
            sq = sum(msq[arm][B][c] for c in conds)
            rf[arm].append((2 * ip - sq) / den)
            loss[arm].append((sq - 2 * ip + tn) / tn)
    return rf, loss, den, tn


def envelope(rf, arms):
    return np.nanmax(np.array([rf[a] for a in arms]), axis=0)


def evaluate():
    freeze = check_freeze()
    man = json.loads((HERE / "results/manifest.json").read_text())
    for f in ("predictions.npz", "predict_info.json"):
        assert sha(HERE / "results" / f) == man[f], f
    info = json.loads((HERE / "results/predict_info.json").read_text())
    genes = info["genes"]
    parts = [od.load(p, genes) for p in ("test_adaptation", "test_scoring")]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "library", "y", "part", "target", "condition")}
    pop = pm.keys_of(cat)
    stats = tg.group_stats(cat["counts"], cat["library"], cat["y"], pop, cat["part"])
    z = np.load(HERE / "results/predictions.npz")
    out = score(stats, {k: z[k] for k in z.files}, freeze["frozen_at"])
    out["cells"] = int(len(pop))
    (HERE / "results/overcite.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdicts": out["verdicts"], "savings": {k: v[str(TARGET)] for k, v in out["savings"].items()}},
                     indent=1))


def score(stats, preds, frozen_at):
    """Endpoint, loss, bootstrap intervals, savings and verdicts from per-population statistics of the held-out
    cells and the prediction dictionary (keys arm/mode/budget/draw/condition)."""
    orfs = sorted({r["target"] for r in stats})
    mean_pred, msq = {}, {}
    for key, C in preds.items():
        arm = "/".join(key.split("/")[:2])
        B = key.split("/")[2]
        c = key.split("/")[4]
        mean_pred.setdefault(arm, {}).setdefault(B, {}).setdefault(c, []).append(C)
    for arm in mean_pred:
        msq[arm] = {}
        for B in mean_pred[arm]:
            msq[arm][B] = {c: float(np.mean([np.sum(C * C) for C in v])) for c, v in mean_pred[arm][B].items()}
            mean_pred[arm][B] = {c: np.mean(v, 0) for c, v in mean_pred[arm][B].items()}
    budget_arms = {a: d for a, d in mean_pred.items() if "0" not in d}
    refs = {a: d for a, d in mean_pred.items() if "0" in d}
    ones = np.ones(len(stats))
    rf, loss, den, tn = curves(stats, ones, budget_arms, msq)
    rf_ref, loss_ref, _, _ = curves(stats, ones, refs, msq)
    rng = np.random.default_rng(BOOT_SEED)
    member = {t: [i for i, r in enumerate(stats) if r["target"] == t] for t in orfs}
    boots = []
    for _ in range(BOOT):
        w = np.zeros(len(stats))
        for t in rng.choice(orfs, len(orfs), replace=True):
            w[member[t]] += 1
        boots.append(curves(stats, w, budget_arms, msq)[:2])
    groups = {"paired_only": PAIRED_ONLY, "semicca": SEMICCA, "champollion": CHAMPOLLION,
              "original_forms": ORIGINAL_FORMS, "all_comparators": [a for a in budget_arms if not a.startswith("bjs")]}
    out = {"dataset": "overcite", "budgets": list(BUDGETS), "draws": DRAWS, "target": TARGET,
           "held_out_orfs": orfs, "populations": [{k: r[k] for k in ("key", "n")} for r in stats],
           "denominator": den, "truth_norm2": tn,
           "rf": rf, "loss_rel": loss, "reference_rf": {a: v[0] for a, v in rf_ref.items()},
           "reference_loss_rel": {a: v[0] for a, v in loss_ref.items()},
           "rf_ci": {a: [[float(np.quantile([b[0][a][i] for b in boots], q)) for q in (.025, .975)]
                         for i in range(len(BUDGETS))] for a in rf},
           "freeze": frozen_at, "savings": {}}
    for label, arms in groups.items():
        arms = [a for a in arms if a in rf]
        out["savings"][label] = {"arms": arms}
        for tgt in (TARGET,) + SECONDARY_TARGETS:
            n_p, r_p = needed(rf[PROPOSED], tgt)
            n_e, r_e = needed(envelope(rf, arms), tgt)
            bs = []
            for b in boots:
                bs.append(needed(envelope(b[0], arms), tgt)[0] / needed(b[0][PROPOSED], tgt)[0])
            out["savings"][label][str(tgt)] = {
                "proposed_cells": n_p, "proposed_relation": r_p, "comparator_cells": n_e, "comparator_relation": r_e,
                "factor": n_e / n_p, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    diffs = []
    for i, B in enumerate(BUDGETS):
        best = max((a for a in rf if not a.startswith("bjs")), key=lambda a: rf[a][i])
        d = [b[0][PROPOSED][i] - b[0][best][i] for b in boots]
        diffs.append({"budget": B, "best": best, "proposed": rf[PROPOSED][i], "best_rf": rf[best][i],
                      "diff": rf[PROPOSED][i] - rf[best][i],
                      "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]})
    out["versus_best"] = diffs
    s = out["savings"]
    reached = s["paired_only"][str(TARGET)]["proposed_relation"] != ">"      # the proposed curve reaches the target
    out["verdicts"] = {
        "proposed_reaches_target": bool(reached),
        "H1_vs_paired_only": bool(reached and s["paired_only"][str(TARGET)]["ci"][0] > 1),
        "H2_vs_semicca": bool(reached and s["semicca"][str(TARGET)]["ci"][0] > 1),
        "H3_noninferior_to_champollion": bool(reached and s["champollion"][str(TARGET)]["ci"][0] > NONINFERIORITY)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("panel", "predict", "evaluate"))
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        {"panel": panel, "predict": predict, "evaluate": evaluate}[a.stage]()
