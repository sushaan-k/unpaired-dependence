"""Development (training targets only): program-targeted design. Exploratory."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/cbio/cbio-agent-handoff/extension/perturbation")
exec(open("/home/claude/cbio/cbio-agent-handoff/extension/perturbation/dev_dryrun.py").read().split("preds, diag = predict.run(fits=fits)")[0])
import programs, design
from pdata import seal as pseal
from threadpoolctl import threadpool_limits
threadpool_limits(limits=2)
record = pseal(); sets = programs.gene_sets()
blocks, defs = programs.define_blocks(record, sets, None)
# dev-train raw moments
keys_tr = keys
raw_tr = raw
# evaluation groups from dev-test (files dict from dev_dryrun)
design.load = lambda p: files[p]
evals = design.evaluation_groups()
held = np.array([g["part"] == "heldout" for g in evals]); hidx = np.flatnonzero(held)
genes = record["genes"]
def rf_block(ch, blk, idx):
    num = den = 0.0
    for i in idx:
        g = evals[i]
        C = pm.transfer(g["Rx"], g["Ry"], ch["B"], g["m"])
        r, c = blk
        Cb = C[np.ix_(r, c)]; T = g["t"]["T"][np.ix_(r, c)]; TA = g["t"]["TA"][np.ix_(r, c)]; TB = g["t"]["TB"][np.ix_(r, c)]
        num += 2 * np.sum(Cb * T) - np.sum(Cb * Cb); den += np.sum(TA * TB)
    return num / den
allb = (list(range(len(genes))), list(range(20)))
sd_x, _ = pm.est.pooled_sd(raw_tr)
def prog_exposure(rows):
    out = {}
    for c in sorted({r["patient"].split("|")[1] for r in raw_tr}):
        rs = [r for r in raw_tr if r["patient"].split("|")[1] == c]
        w = np.array([r["nx"] for r in rs], float)
        mean = sum(wi * r["mx"] for wi, r in zip(w, rs)) / w.sum()
        R = (sum(wi * r["Sx"] for wi, r in zip(w, rs)) / w.sum() / np.outer(sd_x, sd_x))[np.ix_(rows, rows)]
        tr2 = float(np.sum(R * R))
        for r in rs:
            t = r["patient"].split("|")[0]
            d = ((r["mx"] - mean) / sd_x)[rows]
            out[t] = out.get(t, 0.0) + float(d @ R @ d) - tr2 / r["nx"]
    return out
targets = sorted({r["patient"].split("|")[0] for r in raw_tr})
rng = np.random.default_rng(1)
def fit(subset):
    s = set(subset); return pm.fit_channel(pm.centre_within_condition([r for r in raw_tr if r["patient"].split("|")[0] in s]))
progs = {"ifn": blocks["ifn_genes"], "tnfa": blocks["tnfa_signaling_via_nfkb"], "emt": blocks["epithelial_mesenchymal_transition"]}
expo = {p: prog_exposure(b[0]) for p, b in progs.items()}
full = fit(targets)
print("full", {p: round(rf_block(full, b, hidx), 3) for p, b in progs.items()}, "ifn block", round(rf_block(full, blocks["ifn"], hidx), 3), "all", round(rf_block(full, allb, hidx), 3))
for k in (10, 20, 40, 70):
    rand = [list(rng.choice(targets, k, replace=False)) for _ in range(15)]
    rres = [ {p: rf_block(ch, b, hidx) for p, b in list(progs.items()) + [("ifnblock", blocks["ifn"]), ("all", allb)]} for ch in map(fit, rand)]
    rm = {p: np.mean([r[p] for r in rres]) for p in rres[0]}
    rq = {p: np.quantile([r[p] for r in rres], .9) for p in rres[0]}
    line = f"k={k} random mean " + " ".join(f"{p}:{rm[p]:+.2f}(q90 {rq[p]:+.2f})" for p in rm)
    print(line, flush=True)
    for p in progs:
        top = sorted(targets, key=lambda t: -expo[p][t])[:k]
        ch = fit(top)
        res = {q: rf_block(ch, b, hidx) for q, b in list(progs.items()) + [("ifnblock", blocks["ifn"]), ("all", allb)]}
        print(f"   design[{p}] dimS {ch['VS'].shape[1]} " + " ".join(f"{q}:{v:+.2f}" for q, v in res.items()), "top", top[:6], flush=True)
