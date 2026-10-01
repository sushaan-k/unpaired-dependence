#!/usr/bin/env python3
"""Which perturbations make a program learnable (PLAN_programs.md, analysis B).

Prespecified before computation (freeze_posthoc.py) but evaluated on the held-out
groups whose scoring files evaluate.py had opened, so not sealed.

Exposure of a training target, from RNA only (training RNA halves; no protein,
no pairing): the sum over its condition populations of d' R_c d - tr(R_c^2) / n,
where d is the population's RNA mean minus its condition's cell-weighted mean in
pooled within-population SD units, R_c the pooled within-population RNA
covariance of the condition in the same units and n the population's RNA cells;
the subtracted term removes the expected contribution of sampling noise. It
measures how far a perturbation moves RNA along directions that vary within
conditions. Effect strength, the comparator, is |d|^2 - tr(R_c) / n.

Matched-count ablation: the primary channel (condition-centred, frozen rules)
is refitted on all training targets except the k with the highest exposure, and
except k random targets (50 draws, seed 20261017), for k = 5, 10 and 20, and
evaluated on the held-out groups and non-targeting cells (full cross-correlation
and the interferon block of programs.py). The same is done for the k targets of
highest effect strength. Finally, the channel is refitted without each training
target in turn. Writes results/design.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import pmethods as pm
from pdata import MIN_TRAIN_HALF, load, seal
from predict import groups
from programs import define_blocks, gene_sets

sys.path.append(str(pm.HERE.parent / "recoverability"))
from stats import cluster_bootstrap  # noqa: E402

REMOVE = (5, 10, 20)
DRAWS = 50
DRAW_SEED = 20261017
BOOT, BOOT_SEED = 2000, 20261018
PERMUTATIONS = 10000


def exposures(raw):
    sd_x, _ = pm.est.pooled_sd(raw)
    out, strength = {}, {}
    for c in sorted({r["patient"].split("|")[1] for r in raw}):
        rs = [r for r in raw if r["patient"].split("|")[1] == c]
        w = np.array([r["nx"] for r in rs], float)
        mean = sum(wi * r["mx"] for wi, r in zip(w, rs)) / w.sum()
        R = sum(wi * r["Sx"] for wi, r in zip(w, rs)) / w.sum() / np.outer(sd_x, sd_x)
        trR, trR2 = float(np.trace(R)), float(np.sum(R * R))
        for r in rs:
            t = r["patient"].split("|")[0]
            d = (r["mx"] - mean) / sd_x
            out[t] = out.get(t, 0.0) + float(d @ R @ d) - trR2 / r["nx"]
            strength[t] = strength.get(t, 0.0) + float(d @ d) - trR / r["nx"]
    return out, strength


def evaluation_groups():
    """Inputs (cached square roots) and scoring targets of every eligible held-out and non-targeting group."""
    out = []
    for part in ("heldout", "nt"):
        ad, sc = load(f"{part}_adaptation"), load(f"{part}_scoring")
        sk = pm.keys_of(sc)
        for key, idx in groups(ad):
            a = pm.group_inputs(ad["counts"][idx], ad["library"][idx], ad["y"][idx])
            j = np.flatnonzero(sk == key)
            t = pm.scoring_targets(sc["counts"][j], sc["library"][j], sc["y"][j], sc["part"][j])
            out.append({"part": part, "key": key, "target": key.split("|")[0], "Rx": a["Rx"], "Ry": a["Ry"],
                        "m": pm.Marginal(a["Rx"], a["Ry"]), "t": t})
    return out


def terms(ch, evals, blocks):
    """Per group and block: numerator of the recovered fraction."""
    out = np.zeros((len(blocks), len(evals)))
    for i, g in enumerate(evals):
        C = pm.transfer(g["Rx"], g["Ry"], ch["B"], g["m"])
        for b, (r, c) in enumerate(blocks):
            Cb, T = C[np.ix_(r, c)], g["t"]["T"][np.ix_(r, c)]
            out[b, i] = 2 * float(np.sum(Cb * T)) - float(np.sum(Cb * Cb))
    return out


def main():
    d = load("training")
    x, y = pm.features(d)
    pop = pm.keys_of(d)
    keys = pm.eligible_training(pop, d["part"], MIN_TRAIN_HALF)
    raw = pm.moments(x, y, pop, d["part"], keys)
    targets = sorted({r["patient"].split("|")[0] for r in raw})
    expo, strength = exposures(raw)
    record = seal()
    named, _ = define_blocks(record, gene_sets(), None)
    blocks = [(list(range(len(record["genes"]))), list(range(len(record["proteins"])))), named["ifn"]]
    evals = evaluation_groups()
    held = np.array([g["part"] == "heldout" for g in evals])
    den = np.array([[float(np.sum(g["t"]["TA"][np.ix_(r, c)] * g["t"]["TB"][np.ix_(r, c)])) for g in evals]
                    for r, c in blocks])
    print(f"{len(targets)} training targets, {held.sum()} held-out groups", flush=True)

    def fit_without(removed):
        gone = set(removed)
        return pm.fit_channel(pm.centre_within_condition([r for r in raw if r["patient"].split("|")[0] not in gone]))

    by_expo = sorted(targets, key=lambda t: (-expo[t], t))
    by_strength = sorted(targets, key=lambda t: (-strength[t], t))
    rng = np.random.default_rng(DRAW_SEED)
    names, removed = ["none"], [[]]
    for k in REMOVE:
        names += [f"{k}/exposure", f"{k}/strength"]
        removed += [by_expo[:k], by_strength[:k]]
        for r in range(DRAWS):
            names.append(f"{k}/random{r:02d}")
            removed.append(sorted(rng.choice(targets, size=k, replace=False)))
    NUM, dims = [], []
    for i, rem in enumerate(removed):
        ch = fit_without(rem)
        NUM.append(terms(ch, evals, blocks))
        dims.append(int(ch["VS"].shape[1]))
        if i % 25 == 0:
            print("fitted", i, "of", len(removed), flush=True)
    NUM = np.array(NUM)                               # sets x blocks x groups
    loo = []
    for i, t in enumerate(targets):
        loo.append(terms(fit_without([t]), evals, blocks)[0])
        if i % 25 == 0:
            print("leave-one-out", i, flush=True)
    LOO = np.array(loo)

    def rf(num, b, idx):
        return num[..., idx].sum(-1) / den[b, idx].sum()

    hidx, nidx = np.flatnonzero(held), np.flatnonzero(~held)
    gtargets = np.array([g["target"] for g in evals])[hidx]
    res = {"exposure": expo, "strength": strength, "removed": dict(zip(names, [list(map(str, r)) for r in removed])),
           "dim_S": dict(zip(names, dims)),
           "none": {"all": float(rf(NUM[0, 0], 0, hidx)), "ifn": float(rf(NUM[0, 1], 1, hidx)),
                    "nt_all": float(rf(NUM[0, 0], 0, nidx)), "nt_ifn": float(rf(NUM[0, 1], 1, nidx))},
           "ablation": {}}
    for k in REMOVE:
        i_e, i_s = names.index(f"{k}/exposure"), names.index(f"{k}/strength")
        i_r = [names.index(f"{k}/random{r:02d}") for r in range(DRAWS)]
        entry = {}
        for b, bname in enumerate(("all", "ifn")):
            rnd = rf(NUM[i_r, b], b, hidx)

            def stat(j, b=b, i_e=i_e, i_s=i_s, i_r=i_r):
                idx = hidx[j]
                e, s, r = rf(NUM[i_e, b], b, idx), rf(NUM[i_s, b], b, idx), rf(NUM[i_r, b], b, idx)
                return np.array([r.mean() - e, r.mean() - s])

            boot = cluster_bootstrap(gtargets, stat, BOOT, BOOT_SEED + 10 * k + b)
            q = lambda c: [float(np.quantile(boot[:, c], .025)), float(np.quantile(boot[:, c], .975))]
            e_val, s_val = float(rf(NUM[i_e, b], b, hidx)), float(rf(NUM[i_s, b], b, hidx))
            entry[bname] = {"remove_exposure": e_val, "remove_strength": s_val, "remove_random_mean": float(rnd.mean()),
                            "remove_random_q05": float(np.quantile(rnd, .05)), "remove_random_min": float(rnd.min()),
                            "random_values": rnd.tolist(),
                            "random_minus_exposure": {"value": float(rnd.mean() - e_val), "ci": q(0)},
                            "random_minus_strength": {"value": float(rnd.mean() - s_val), "ci": q(1)},
                            "random_below_exposure": int(np.sum(rnd <= e_val)),
                            "nt": {"remove_exposure": float(rf(NUM[i_e, b], b, nidx)),
                                   "remove_strength": float(rf(NUM[i_s, b], b, nidx)),
                                   "remove_random_mean": float(rf(NUM[i_r, b], b, nidx).mean())}}
        entry["dim_S"] = {"exposure": dims[i_e], "strength": dims[i_s],
                          "random_mean": float(np.mean([dims[i] for i in i_r])),
                          "random_zero": int(sum(dims[i] == 0 for i in i_r))}
        res["ablation"][str(k)] = entry
    drop = rf(NUM[0, 0], 0, hidx) - rf(LOO, 0, hidx)
    ev = np.array([expo[t] for t in targets])
    ranks = lambda a: np.argsort(np.argsort(a))
    sp = lambda a, b: float(np.corrcoef(ranks(a), ranks(b))[0, 1])
    prng = np.random.default_rng(BOOT_SEED)
    perm = np.array([sp(ev, prng.permutation(drop)) for _ in range(PERMUTATIONS)])
    res["leave_one_out"] = {"drop": dict(zip(targets, drop.tolist())), "spearman_exposure": sp(ev, drop),
                            "p_exposure": float((1 + np.sum(perm >= sp(ev, drop))) / (1 + PERMUTATIONS)),
                            "top_drop": [targets[i] for i in np.argsort(-drop)[:10]], "top_exposure": by_expo[:10]}
    v = {}
    for k in (5, 10):
        a = res["ablation"][str(k)]["all"]
        v[f"D1_remove{k}"] = a["random_minus_exposure"]["ci"][0] > 0 and a["remove_exposure"] < a["remove_random_q05"]
        f = res["ablation"][str(k)]["ifn"]
        v[f"D1_ifn_remove{k}"] = f["random_minus_exposure"]["ci"][0] > 0 and f["remove_exposure"] < f["remove_random_q05"]
    res["verdict"] = v
    (pm.HERE / "results/design.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"verdict": v, "none": res["none"],
                      "ablation": {k: {b: {kk: vv for kk, vv in e.items() if kk != "random_values"} if isinstance(e, dict)
                                       and "remove_exposure" in e else e for b, e in a.items()}
                                   for k, a in res["ablation"].items()},
                      "loo": {k: vv for k, vv in res["leave_one_out"].items() if k != "drop"}}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
