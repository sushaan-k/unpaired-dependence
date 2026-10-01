#!/usr/bin/env python3
"""Hybrid development comparison (PLAN.md).

    python run.py frangieh [--smoke]
    python run.py papalexi [--smoke]

Writes results/<dataset>.json (per-group endpoint terms for every arm, budget and
block, averaged over draws, plus fit summaries). summarize.py applies the
decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import hmethods as hm

pm = hm.pm
sys.path.append(str(hm.HERE.parent / "perturbation_replication"))
import pdata  # noqa: E402
import rdata  # noqa: E402

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
DRAW_SEED = 20261022
SETTINGS = {"frangieh": {"budgets": (50, 100, 200, 400, 800, 1600, 3200), "draws": 20},
            "papalexi": {"budgets": (50, 100, 200, 400, 800, 1600), "draws": 10}}
UNPAIRED = ("current", "splitup", "noise_aware")
PAIRED = ("paired_corr", "paired_js", "paired_lowrank", "paired_eot", "block_js", "combine_global", "hybrid",
          "hybrid_current", "hybrid_pc")


def gene_sets():
    out = {}
    for name, digest in GS_SHA256.items():
        path = GENESETS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        d = json.loads(path.read_text())
        key = next(iter(d))
        out[key] = set(d[key]["geneSymbols"])
    return out


def block_masks(genes, proteins, encoding):
    sets = gene_sets()
    p, q = len(genes), len(proteins)
    ifn = set().union(*(sets[s] for s in IFN_SETS))
    rows = [i for i, g in enumerate(genes) if g in ifn]
    cols = [j for j, pr in enumerate(proteins) if any(g in ifn for g in encoding[pr])]
    names, masks = [], []

    def add(name, r, c):
        m = np.zeros((p, q))
        m[np.ix_(r, c)] = 1.0
        names.append(name)
        masks.append(m)
    add("all", list(range(p)), list(range(q)))
    add("ifn", rows, cols)
    add("ifn_genes", rows, list(range(q)))
    add("complement", [i for i in range(p) if i not in rows], [j for j in range(q) if j not in cols])
    for name in sorted(sets):
        if name in IFN_SETS:
            continue
        r = [i for i, g in enumerate(genes) if g in sets[name]]
        if len(r) >= MIN_GENES:
            add(name.replace("HALLMARK_", "").lower(), r, list(range(q)))
    return names, np.array(masks), {"ifn_genes": [genes[i] for i in rows], "ifn_proteins": [proteins[j] for j in cols]}


def load_dataset(name):
    if name == "frangieh":
        mod, parts = pdata, {"train": "training", "test": "heldout", "nt": "nt"}
    else:
        mod, parts = rdata, {"train": "train", "test": "test", "nt": "nt"}
    train = mod.load(parts["train"])
    ev = {}
    for part in ("test", "nt"):
        ad, sc = mod.load(f"{parts[part]}_adaptation"), mod.load(f"{parts[part]}_scoring")
        akeys, skeys = pm.keys_of(ad), pm.keys_of(sc)
        groups = []
        for key in sorted(set(akeys)):
            ia = np.flatnonzero(akeys == key)
            if len(ia) < mod.MIN_ADAPT:
                continue
            isc = np.flatnonzero(skeys == key)
            a = pm.group_inputs(ad["counts"][ia], ad["library"][ia], ad["y"][ia])
            t = pm.scoring_targets(sc["counts"][isc], sc["library"][isc], sc["y"][isc], sc["part"][isc])
            groups.append({"key": key, "target": key.split("|")[0], "cond": key.split("|")[1], "Rx": a["Rx"],
                           "Ry": a["Ry"], "marginal": pm.Marginal(a["Rx"], a["Ry"]), "T": t["T"], "TA": t["TA"],
                           "TB": t["TB"], "n_adapt": int(len(ia)), "n_score": int(len(isc))})
        ev[part] = groups
    return mod, train, ev


def fold_fits(train, x, y, pop, keys, log):
    use = np.isin(pop, keys)
    res = hm.reservoir(train["cell"]) & use
    pl = use & ~res
    pool = hm.Pool(x[pl], y[pl], pop[pl], train["part"][pl], train["guide"][pl], keys)
    t0 = time.time()
    fits = {"current": hm.current(pool), "noise_aware": hm.noise_aware(pool), "splitup": hm.splitup(pool)}
    log(f"  unpaired fits {time.time() - t0:.0f}s: dims current {fits['current']['dim']}, "
        f"noise_aware {fits['noise_aware']['dim']} (threshold {fits['noise_aware']['threshold']:.4f}, "
        f"eigenvalues {np.round(fits['noise_aware']['eigenvalues'][:4], 4).tolist()})")
    rmask = res & np.isin(pop, pool.keys)
    X_res, Y_res = pool.centre_cells(x[rmask], y[rmask], pop[rmask])
    # context: all paired training cells, pooled within-population cross-correlation per condition
    paired_all = {}
    for c in pool.conds:
        acc, total = 0.0, 0
        for k in [k for k in pool.keys if hm.cond_of(k) == c]:
            m = pop == k
            xs, _ = pm.standardize(x[m])
            ys, _ = pm.standardize(y[m])
            acc = acc + xs.T @ ys
            total += int(m.sum())
        paired_all[c] = acc / total
    return pool, fits, X_res, Y_res, paired_all


def score(C, groups, masks, T, TAB):
    """Per-group, per-block numerator terms 2<C,T> - ||C||^2; C is (G,p,q) or a shared (p,q)."""
    if C.ndim == 2:
        C = np.broadcast_to(C, T.shape)
    return 2 * np.einsum("gpq,bpq->gb", C * T, masks) - np.einsum("gpq,bpq->gb", C * C, masks)


def run_fold(dataset, fold_id, groups, train, x, y, pop, keys, masks, settings, log, smoke=False):
    pool, fits, X_res, Y_res, paired_all = fold_fits(train, x, y, pop, keys, log)
    T = np.array([g["T"] for g in groups])
    den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in groups]), masks)
    out = {"den": den, "num": {}, "draw_rf": {}, "info": {}}
    base = {}
    for name in UNPAIRED:
        Bint = fits[name]["B"]
        base[name] = np.array([pm.transfer(g["Rx"], g["Ry"], Bint, g["marginal"]) for g in groups])
        out["num"][f"{name}"] = {0: score(base[name], groups, masks, T, None)}
    out["num"]["paired_all"] = {0: score(np.array([paired_all[g["cond"]] for g in groups]), groups, masks, T, None)}
    out["num"]["independence"] = {0: np.zeros_like(den)}
    train_na = pm.transfer(pool.Rx, pool.Ry, fits["noise_aware"]["B"], pool.marginal)
    U0, lx0 = hm.eigenbasis(pool.Rx)
    V, ly = hm.eigenbasis(pool.Ry)
    p = pool.Rx.shape[0]
    blocks = hm.eigen_blocks(p)
    M_global = U0.T @ train_na @ V
    k_na = fits["noise_aware"]["dim"]
    programs = {"hybrid": (fits["noise_aware"]["U"], fits["noise_aware"]["W"] @ fits["noise_aware"]["U"]),
                "hybrid_current": (fits["current"]["U"], fits["current"]["W"] @ fits["current"]["U"]),
                "hybrid_pc": (U0[:, :k_na], np.zeros((V.shape[0], k_na)))}
    RxU = {a: np.array([g["Rx"] @ U for g in groups]) for a, (U, _) in programs.items()}
    out["info"] = {"n_pool_keys": len(pool.keys), "n_reservoir": int(len(X_res)), "dim_noise_aware": k_na,
                   "dim_current": fits["current"]["dim"], "current_scale": fits["current"]["scale"],
                   "noise_aware_threshold": fits["noise_aware"]["threshold"],
                   "noise_aware_eigenvalues": fits["noise_aware"]["eigenvalues"],
                   "splitup_support": [c["support"] for c in fits["splitup"]["tuning"]],
                   "factors": {}, "lowrank_rank": {}}
    budgets = settings["budgets"][:2] if smoke else settings["budgets"]
    draws = 2 if smoke else settings["draws"]
    for B in budgets:
        if B > len(X_res):
            log(f"  budget {B} exceeds the reservoir ({len(X_res)}); skipped")
            continue
        acc = {a: np.zeros_like(den) for a in PAIRED}
        rf_draws = {a: [] for a in PAIRED}
        factors, ranks = [], []
        for dr in range(draws):
            rng = np.random.default_rng([DRAW_SEED, {"frangieh": 1, "papalexi": 2}[dataset], fold_id, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            X, Y = X_res[pick], Y_res[pick]
            C, var = hm.products(X, Y)
            Cjs = hm.james_stein(C, var)
            Clr, r = hm.low_rank(X, Y, rng)
            ranks.append(r)
            Beot = hm.eot_interaction(Cjs, pool)
            A0, v0 = hm.basis_coefficients(X, Y, U0, V)
            post_b, f_b = hm.block_js(A0, v0, np.zeros_like(A0), blocks)
            post_g, f_g = hm.block_js(A0, v0, M_global, [list(range(p))])
            preds = {"paired_corr": C, "paired_js": Cjs, "paired_lowrank": Clr,
                     "paired_eot": np.array([pm.transfer(g["Rx"], g["Ry"], Beot, g["marginal"]) for g in groups]),
                     "block_js": U0 @ post_b @ V.T,
                     "combine_global": base["noise_aware"] + U0 @ (post_g - M_global) @ V.T}
            fr = {"block_js": f_b, "combine_global": f_g}
            for a, (U, Wu) in programs.items():
                Wpost, Yres, f_p = hm.program_regression(X, Y, U, Wu)
                Ar, vr = hm.basis_coefficients(X, Yres, U0, V)
                post_r, f_r = hm.block_js(Ar, vr, np.zeros_like(Ar), blocks)
                preds[a] = RxU[a] @ Wpost.T + (U0 @ post_r @ V.T)[None]
                fr[a] = {"programs": f_p, "remainder": f_r}
            factors.append(fr)
            for a, Cp in preds.items():
                s_ = score(Cp, groups, masks, T, None)
                acc[a] += s_
                rf_draws[a].append(float(s_[:, 0].sum() / den[:, 0].sum()))
        for a in PAIRED:
            out["num"].setdefault(a, {})[B] = acc[a] / draws
            out["draw_rf"].setdefault(a, {})[B] = rf_draws[a]
        out["info"]["factors"][B] = factors[:3]
        out["info"]["lowrank_rank"][B] = ranks
        log(f"  budget {B}: " + ", ".join(f"{a} {np.mean(rf_draws[a]):.3f}" for a in PAIRED))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SETTINGS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    mod, train, ev = load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, block_defs = block_masks(train["genes"], train["proteins"], mod.ENCODING)
    log(f"{args.dataset}: {len(keys)} eligible training populations, test groups {len(ev['test'])}, "
        f"nt groups {len(ev['nt'])}, blocks {names}")
    result = {"dataset": args.dataset, "blocks": names, "block_defs": block_defs, "settings": SETTINGS[args.dataset],
              "smoke": args.smoke, "parts": {}}
    for part in ("test", "nt"):
        groups_all = ev[part]
        if args.dataset == "papalexi" and part == "test":
            folds = sorted({g["target"] for g in groups_all})
        else:
            folds = [None]
        if args.smoke:
            folds = folds[:2]
        rows = []
        for fi, held in enumerate(folds):
            gs = [g for g in groups_all if held is None or g["target"] == held]
            ks = [k for k in keys if k.split("|")[0] != held] if held is not None else keys
            log(f"{part} fold {held}: {len(gs)} groups, {len(ks)} training populations")
            out = run_fold(args.dataset, fi + (100 if part == "nt" else 0), gs, train, x, y, pop, ks, masks,
                           SETTINGS[args.dataset], log, args.smoke)
            for i, g in enumerate(gs):
                rows.append({"key": g["key"], "target": g["target"], "cond": g["cond"], "fold": held,
                             "n_adapt": g["n_adapt"], "n_score": g["n_score"], "den": out["den"][i].tolist(),
                             "num": {a: {str(B): v[i].tolist() for B, v in d.items()} for a, d in out["num"].items()}})
            result["parts"].setdefault(part, {"rows": [], "fold_info": [], "draw_rf": []})
            result["parts"][part]["fold_info"].append(dict(out["info"], fold=held))
            result["parts"][part]["draw_rf"].append({a: {str(B): v for B, v in d.items()} for a, d in out["draw_rf"].items()})
        result["parts"][part]["rows"] = rows
    name = f"{args.dataset}{'_smoke' if args.smoke else ''}.json"
    (hm.HERE / "results" / name).write_text(json.dumps(result, default=float) + "\n")
    log(f"wrote results/{name}")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
