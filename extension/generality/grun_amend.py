#!/usr/bin/env python3
"""Amendments 1 and 2 (DEVIATIONS.md): savings at targets relative to the largest recovered fraction any method
reached, from the hashed predictions of the frozen run; data sets in which no method recovers dependence (level's
bootstrap lower bound at or below zero) are not informative for savings (amendment 2).

    python grun_amend.py <dataset>     # results/<dataset>_amend1.json (checks the frozen savings first)
    python grun_amend.py summary       # results/summary_amend1.json (G1'-G3')
    python grun_amend.py law           # results/law_amend1.json (G4' with the amended observed savings)

The held-out statistics, bootstrap resamples, endpoint, envelopes, censoring and interval rules are those of
grun.py (imported unchanged). The script first recomputes every saving of the frozen rule (targets relative to the
fully paired reference) and requires it to equal results/<dataset>.json exactly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grun as gr  # noqa: E402

FRACS = (0.5, 0.25, 0.75)


def method_arms(curves):
    return [a for a in curves if not a.startswith("abl_")]


def level(rf_by_arm, arms):
    return float(max(np.max(rf_by_arm[a]) for a in arms))


def curves_and_boot(name, log):
    man = json.loads((gr.RES / f"{name}_manifest.json").read_text())
    for key, fn in (("predictions", f"{name}_predictions.npz"), ("msq", f"{name}_msq.json"), ("info", f"{name}_info.json")):
        assert gr.sha(gr.RES / fn) == man[key], f"{fn} does not match the manifest"
    z = np.load(gr.RES / f"{name}_predictions.npz")
    store = {k: z[k] for k in z.files}
    msq = {int(f): m for f, m in json.loads((gr.RES / f"{name}_msq.json").read_text()).items()}
    infos = {int(f): i for f, i in json.loads((gr.RES / f"{name}_info.json").read_text()).items()}
    d = gr.load(gr.source(name))
    fold, part, half = gr.roles(gr.source(name), d)
    stats = {f: gr.heldout_stats(d, f, fold, half, np.array(infos[f]["x_panel_index"]),
                                 np.array(infos[f]["y_panel_index"]), infos[f]["conditions"]) for f in range(gr.FOLDS)}
    budgets = {f: infos[f]["budgets"] for f in range(gr.FOLDS)}
    curves, ref, common = gr.score(name, stats, store, msq, budgets)
    ref = float(np.asarray(ref).ravel()[0])
    rf = {a: v[0] for a, v in curves.items()}
    units = sorted(set(d["unit"]))
    rng = np.random.default_rng([gr.BOOT_SEED, (gr.PRIMARY + gr.SECONDARY).index(name)
                                 if name in gr.PRIMARY + gr.SECONDARY else 99])
    draws = rng.integers(0, len(units), size=(gr.BOOT, len(units)))
    counts = np.zeros((gr.BOOT, len(units)))
    for r in range(gr.BOOT):
        counts[r] = np.bincount(draws[r], minlength=len(units))
    uidx = {u: i for i, u in enumerate(units)}
    boot_rf, boot_ref = {a: [] for a in curves}, []
    for s_ in range(0, gr.BOOT, 500):
        W = {f: counts[s_:s_ + 500][:, [uidx[u] for u in stats[f][next(iter(stats[f]))]["units"]]] for f in stats}
        cb, rb, _ = gr.score(name, stats, store, msq, budgets, W)
        for a in cb:
            boot_rf[a].append(cb[a])
        boot_ref.append(rb)
        log(f"bootstrap {s_ + 500}/{gr.BOOT}")
    return rf, ref, common, {a: np.vstack(v) for a, v in boot_rf.items()}, np.concatenate(boot_ref), len(units)


def savings(rf, boot_rf, common, targets, boot_targets, groups):
    out = {}
    for label, arms in groups.items():
        arms = [a for a in arms if a in rf]
        out[label] = {"arms": arms}
        for key, tgt in targets.items():
            fac, n_c, r_c, n_p, r_p = gr.saving(rf, arms, tgt, common)
            bs = np.array([gr.saving({a: boot_rf[a][r] for a in boot_rf}, arms, boot_targets[key][r], common)[0]
                           for r in range(gr.BOOT)])
            out[label][key] = {"target_rf": tgt, "factor": fac, "comparator_cells": n_c, "comparator_relation": r_c,
                               "proposed_cells": n_p, "proposed_relation": r_p,
                               "ci": np.quantile(bs, [.025, .975]).tolist(), "log_boot": np.log(bs).tolist()}
    return out


def selftune(rf, boot_rf, common, targets, boot_targets):
    out = {}
    for key, tgt in targets.items():
        n_f = gr.needed(rf[gr.PROPOSED], tgt, common)
        n_s = gr.needed(rf[gr.SELFTUNED], tgt, common)
        bs = np.array([gr.needed(boot_rf[gr.PROPOSED][r], boot_targets[key][r], common)[0]
                       / gr.needed(boot_rf[gr.SELFTUNED][r], boot_targets[key][r], common)[0] for r in range(gr.BOOT)])
        out[key] = {"fixed_cells": n_f[0], "fixed_relation": n_f[1], "selftuned_cells": n_s[0],
                    "selftuned_relation": n_s[1], "factor": n_f[0] / n_s[0],
                    "ci": np.quantile(bs, [.025, .975]).tolist(), "log_boot": np.log(bs).tolist()}
    return out


def run(name, log):
    rf, ref, common, boot_rf, boot_ref, n_units = curves_and_boot(name, log)
    groups = {"paired_only": gr.PAIRED_ONLY, "semicca": gr.SEMICCA, "original_forms": gr.ORIGINAL,
              "all_comparators": [a for a in rf if not a.startswith(("bjs2", "selftune", "abl_"))]}
    # check: the frozen rule reproduces results/<name>.json exactly
    frozen = json.loads((gr.RES / f"{name}.json").read_text())
    ft = {str(f): f * ref for f in gr.TARGETS}
    fb = {str(f): f * boot_ref for f in gr.TARGETS}
    chk = savings(rf, boot_rf, common, ft, fb, groups)
    worst = max(abs(chk[g][t]["factor"] - frozen["savings"][g][t]["factor"])
                + max(abs(a - b) for a, b in zip(chk[g][t]["ci"], frozen["savings"][g][t]["ci"]))
                for g in groups for t in ft)
    worst = max(worst, max(abs(float(rf[a][i]) - frozen["rf"][a][i]) for a in rf for i in range(len(common))))
    assert worst < 1e-9, f"frozen savings not reproduced ({worst})"
    arms = method_arms(rf)
    L = level(rf, arms)
    Lb = np.array([level({a: boot_rf[a][r] for a in arms}, arms) for r in range(gr.BOOT)])
    targets = {str(f): f * L for f in FRACS}
    boot_targets = {str(f): f * Lb for f in FRACS}
    out = {"dataset": name, "budgets": common, "units": n_units, "level": L,
           "level_arm": max(arms, key=lambda a: float(np.max(rf[a]))),
           "level_ci": np.quantile(Lb, [.025, .975]).tolist(), "reference_rf_paired_all": ref,
           "frozen_rule_reproduced_max_abs_difference": worst,
           "savings": savings(rf, boot_rf, common, targets, boot_targets, groups),
           "selftune_vs_fixed": selftune(rf, boot_rf, common, targets, boot_targets)}
    (gr.RES / f"{name}_amend1.json").write_text(json.dumps(out) + "\n")
    s = out["savings"]["paired_only"]["0.5"]
    log(f"amended: level {100 * L:.1f} ({out['level_arm']}); saving vs paired-only {s['factor']:.2f} "
        f"({s['ci'][0]:.2f}-{s['ci'][1]:.2f}); proposed {s['proposed_relation']}{s['proposed_cells']:.0f}, "
        f"paired-only {s['comparator_relation']}{s['comparator_cells']:.0f}")


def summary():
    """G1'-G3' over the primary data sets: with amendment 2 (data sets whose amended level has a bootstrap lower bound
    at or below zero are not informative and are left out) and without it (every data set's censored saving)."""
    res = {n: json.loads((gr.RES / f"{n}_amend1.json").read_text()) for n in gr.PRIMARY
           if (gr.RES / f"{n}_amend1.json").exists()}
    informative = [n for n in res if res[n]["level_ci"][0] > 0]
    out = {"datasets": list(res), "missing": [n for n in gr.PRIMARY if n not in res], "informative": informative,
           "not_informative": [n for n in res if n not in informative],
           "levels": {n: [res[n]["level"], res[n]["level_arm"]] + res[n]["level_ci"] for n in res}}
    for scope, names in (("informative", informative), ("all", list(res))):
        for label, key in (("G1_paired_only", "paired_only"), ("G2_semicca", "semicca"), ("original_forms", "original_forms")):
            for frac in ("0.5", "0.25", "0.75"):
                logs = np.array([np.log(res[n]["savings"][key][frac]["factor"]) for n in names])
                boot = np.mean([res[n]["savings"][key][frac]["log_boot"] for n in names], axis=0)
                out[f"{scope}/{label}/{frac}"] = {
                    "geometric_mean": float(np.exp(logs.mean())), "ci": np.exp(np.quantile(boot, [.025, .975])).tolist(),
                    "per_dataset": {n: res[n]["savings"][key][frac]["factor"] for n in names},
                    "lower_bound_above_one": [n for n in names if res[n]["savings"][key][frac]["ci"][0] > 1]}
        for frac in ("0.5", "0.25", "0.75"):
            logs = np.array([np.log(res[n]["selftune_vs_fixed"][frac]["factor"]) for n in names])
            boot = np.mean([res[n]["selftune_vs_fixed"][frac]["log_boot"] for n in names], axis=0)
            out[f"{scope}/G3_selftune_vs_fixed/{frac}"] = {
                "geometric_mean": float(np.exp(logs.mean())), "ci": np.exp(np.quantile(boot, [.025, .975])).tolist(),
                "per_dataset": {n: res[n]["selftune_vs_fixed"][frac]["factor"] for n in names}}
    complete = not out["missing"]
    out["verdicts"] = {"G1_amended_met": bool(out["informative/G1_paired_only/0.5"]["ci"][0] > 1 and complete),
                       "G2_amended_met": bool(out["informative/G2_semicca/0.5"]["ci"][0] > 1 and complete),
                       "G1_amended_all_met": bool(out["all/G1_paired_only/0.5"]["ci"][0] > 1 and complete),
                       "G2_amended_all_met": bool(out["all/G2_semicca/0.5"]["ci"][0] > 1 and complete)}
    (gr.RES / "summary_amend1.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        if isinstance(v, dict) and "geometric_mean" in v:
            print(k, round(v["geometric_mean"], 2), [round(c, 2) for c in v["ci"]])
    print(out["verdicts"], "missing", out["missing"], "not informative", out["not_informative"])


def law():
    """G4' with the amended observed savings; the predicted savings are those of glaw.py (training cells only)."""
    import glaw
    from scipy.stats import spearmanr
    pred = json.loads((gr.RES / "law_predicted.json").read_text())
    obs = {}
    for n in gr.PRIMARY:
        p = gr.RES / f"{n}_amend1.json"
        if p.exists():
            r = json.loads(p.read_text())
            if r["level_ci"][0] > 0:                           # amendment 2: informative data sets only
                obs[n] = r["savings"]["paired_only"]["0.5"]["factor"]
    for n in glaw.EARLIER:
        obs[n] = observed_earlier_amended(n)
    names = [n for n in obs if n in pred and np.isfinite(pred[n]["predicted_saving"])]
    xp = np.array([pred[n]["predicted_saving"] for n in names])
    yo = np.array([obs[n] for n in names])
    rho = float(spearmanr(xp, yo).correlation)
    rng = np.random.default_rng(glaw.PERM_SEED)
    null = np.array([spearmanr(xp, rng.permutation(yo)).correlation for _ in range(glaw.PERMUTATIONS)])
    pval = float((1 + np.sum(null >= rho)) / (1 + glaw.PERMUTATIONS))
    out = {"datasets": names, "predicted": dict(zip(names, xp.tolist())), "observed": dict(zip(names, yo.tolist())),
           "spearman": rho, "p_one_sided": pval, "permutations": glaw.PERMUTATIONS, "G4_amended_met": bool(pval < 0.05),
           "complete": len(names) == len(gr.PRIMARY) + len(glaw.EARLIER),
           "not_informative": [n for n in gr.PRIMARY if (gr.RES / f"{n}_amend1.json").exists() and n not in names]}
    (gr.RES / "law_amend1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def observed_earlier_amended(name):
    """Saving against the paired-only envelope at half the largest recovered fraction any method reached, from the
    stored development or external curves."""
    SPR = gr.SP / "results"
    if name == "overcite":
        o = json.loads((gr.SP / "external/results/overcite_plan_estimator.json").read_text())
        rf, budgets = o["rf"], o["budgets"]
    else:
        res = json.loads((SPR / f"{name}.json").read_text())
        budgets = res["budgets"]
        den = sum(r["den"][0] for r in res["rows"])
        rf = {}
        for a in sorted({a for r in res["rows"] for a in r["num"]}):
            if all(str(B) in res["rows"][0]["num"].get(a, {}) for B in budgets):
                rf[a] = [sum(r["num"][a][str(B)][0] for r in res["rows"]) / den for B in budgets]
    methods = [a for a in rf if not a.startswith(("abl", "dose", "paired_all", "independence"))]
    L = max(max(rf[a]) for a in methods)
    env = np.nanmax(np.array([rf[a] for a in gr.PAIRED_ONLY if a in rf]), axis=0)
    return gr.needed(env, 0.5 * L, budgets)[0] / gr.needed(rf["bjs2/mapped"], 0.5 * L, budgets)[0]


if __name__ == "__main__":
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {sys.argv[1]}: {msg}", flush=True)
    with threadpool_limits(limits=1):
        gr.check_freeze()
        if sys.argv[1] == "summary":
            summary()
        elif sys.argv[1] == "law":
            law()
        else:
            run(sys.argv[1], log)
