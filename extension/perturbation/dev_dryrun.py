"""Development: end-to-end run of predict.run and the evaluation on training targets only (code check before the freeze; paired interaction set to zero)."""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import pdata
import pmethods as pm
import predict
import evaluate
from pdata import CELL_SALT, h, load, seal, MIN_TRAIN_HALF

d = load("training")
x, y = pm.features(d)
pop = pm.keys_of(d)
targets = seal()["training_targets"]
order = sorted(targets, key=lambda t: h("perturb-dev-v1", t))
dev_test = set(order[::4])
tr = ~np.isin(d["target"], list(dev_test))
keys = pm.eligible_training(pop[tr], d["part"][tr], MIN_TRAIN_HALF)
raw = pm.moments(x[tr], y[tr], pop[tr], d["part"][tr], keys)
chans = {"pf_within": pm.fit_channel(pm.centre_within_condition(raw)), "pf": pm.fit_channel(raw)}
ctl = pm.control_channels(x[tr], y[tr], pop[tr], d["part"][tr], keys, d["condition"][tr], centre=True)
pm.CONTROL_DRAWS = 2
for k, v in list(ctl.items())[:2] + list(ctl.items())[10:12]:
    chans[f"within_{k}"] = v
for k, v in list(pm.control_channels(x[tr], y[tr], pop[tr], d["part"][tr], keys, d["condition"][tr]).items()):
    chans[k] = v
people = pm.paired_people(x[tr], y[tr], pop[tr], keys)
refs = pm.paired_references(people, closed_form=False)
cm = pm.condition_marginals(x[tr], y[tr], d["counts"][tr], d["library"][tr], pop[tr], d["part"][tr], keys)
p, q = x.shape[1], y.shape[1]
paired = {"B": np.zeros((p, q)), "W_ref": refs["reference_regression"]["W"], "corr": refs["transferred_correlation"]}
cond = {c: {k: cm[c][k] for k in ("Rx", "Ry", "Rz")} for c in cm}
for c in cond:
    cond[c]["marginal"] = pm.Marginal(cond[c]["Rx"], cond[c]["Ry"])
fits = ({k: {"W": v["W"], "B": v["B"], "VS": v["VS"]} for k, v in chans.items()}, paired, cond)


def subset(mask, part_override=None, target_override=None):
    out = {}
    for k, v in d.items():
        out[k] = v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v
    if part_override is not None:
        out["part"] = part_override
    if target_override is not None:
        out["target"] = np.array([target_override] * mask.sum())
    return out


te = ~tr
ad = subset(te & (d["part"] == 0), part_override=np.full(np.sum(te & (d["part"] == 0)), -1))
sc_mask = te & (d["part"] == 1)
sc_idx = np.flatnonzero(sc_mask)
keys_sc = pop[sc_idx]
sub = np.zeros(len(sc_idx), int)
for k in set(keys_sc):
    m = np.flatnonzero(keys_sc == k)
    m = m[np.argsort([h(CELL_SALT, d["cell"][sc_idx[i]]) for i in m])]
    sub[m[1::2]] = 1
sc = subset(sc_mask, part_override=sub)
# NT imitation: three dev-test targets relabelled NT
nt_t = sorted(dev_test)[:3]
ntm = np.isin(d["target"], nt_t)
nt_ad = subset(ntm & (d["part"] == 0), part_override=np.full(np.sum(ntm & (d["part"] == 0)), -1), target_override="NT")
nt_sc_idx = np.flatnonzero(ntm & (d["part"] == 1))
nt_sc = subset(ntm & (d["part"] == 1), part_override=np.arange(len(nt_sc_idx)) % 2, target_override="NT")
files = {"heldout_adaptation": ad, "heldout_scoring": sc, "nt_adaptation": nt_ad, "nt_scoring": nt_sc}
fake = lambda part: files[part]
predict.load = fake
evaluate.load = fake
preds, diag = predict.run(fits=fits)
print("predictions", len(preds), "groups", len(diag))
evaluate.BOOT = 50
ev = {}
for part in ("heldout", "nt"):
    drows = [r for r in diag if r["part"] == part]
    dd, scd = evaluate.scoring(part, [r["key"] for r in drows])
    rows = [dict(r, terms=evaluate.terms(preds, part, r["key"], scd[r["key"]]["t"])) for r in drows]
    ev[part] = evaluate.heldout_analysis(rows) if part == "heldout" else evaluate.nt_analysis(dd, scd, rows, preds)
e = ev["heldout"]["estimates"]
for k, v in e.items():
    print(f"{k:34s} {v['value']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
print("nt pooled", {k: round(v, 3) for k, v in ev["nt"]["pooled"].items()}, ev["nt"]["share_of_paired"], ev["nt"]["guide_sets"])
