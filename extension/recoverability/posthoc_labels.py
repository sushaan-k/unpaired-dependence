#!/usr/bin/env python3
"""Post hoc (added after unsealing, in response to review): label-dependent results with assay-specific labels.

The deposited bone-marrow annotation was derived from both assays. Here each
recipient sample's coarse types are assigned twice and independently: from RNA
alone and from protein alone, each by a multinomial logistic regression trained on
the adaptation cells of the other donors (their deposited labels; the recipient
donor's cells are never used). RNA type means use RNA labels, protein type means
use protein labels, and type shares are the average of the two assays' shares.
Recomputed with these labels: the type-means estimator of total cross-correlation
(against the scoring cells), agreement with type means for every channel,
composition, and the frozen failure score (AUC over all pairs and high-coverage
pairs). Intervals: 2,000 bootstrap resamples of donors. Writes results/posthoc_labels.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cross_study"))

from data import standardize  # noqa: E402

from bmmc import load  # noqa: E402
from diagnostics import between_share  # noqa: E402
from evaluate_bmmc import model_score, rel, targets  # noqa: E402
from noise import lognorm  # noqa: E402
from predict_bmmc import load_channels  # noqa: E402
from stats import auc, cluster_bootstrap  # noqa: E402

BOOT, SEED = 2000, 20261015
HIGH = 0.4


def classify(train_x, train_y, test_x, n=20000, seed=SEED):
    """Multinomial logistic regression on (at most n, fixed-seed) standardized training cells."""
    if len(train_x) > n:
        keep = np.random.default_rng(seed).choice(len(train_x), n, replace=False)
        train_x, train_y = train_x[keep], train_y[keep]
    mu, sd = train_x.mean(0), train_x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    model = LogisticRegression(C=1.0, max_iter=3000)
    model.fit((train_x - mu) / sd, train_y)
    return model.predict((test_x - mu) / sd)


def split_between(x, y, lx, ly, types):
    """Between-type RNA covariance (RNA labels) and cross-correlation from assay-specific type means."""
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    px = np.array([np.mean(lx == t) for t in types])
    py = np.array([np.mean(ly == t) for t in types])
    pi = (px + py) / 2
    Sb = np.zeros((x.shape[1], x.shape[1]))
    Cb = np.zeros((x.shape[1], y.shape[1]))
    for k, t in enumerate(types):
        a = xs[lx == t].mean(0) if px[k] > 0 else np.zeros(x.shape[1])
        b = ys[ly == t].mean(0) if py[k] > 0 else np.zeros(y.shape[1])
        Sb += px[k] * np.outer(a, a)
        Cb += pi[k] * np.outer(a, b)
    return Sb, Cb


def main():
    units = json.loads((HERE / "results/bmmc_units.json").read_text())
    rec = load("adaptation")
    x_all = lognorm(rec["counts"], rec["library"])
    y_all = rec["y"]
    joint = rec["labels"]["coarse"]
    chans = load_channels()
    errors = json.loads((HERE / "results/bmmc_errors.json").read_text())
    T = targets(load("scoring"), units)
    per_sample, agreement, composition = {}, {}, {}
    for donor in sorted({u["donor"] for u in units.values()}):
        test_batches = [b for b, u in units.items() if u["donor"] == donor]
        train = ~np.isin(rec["batch"], test_batches)
        for b in test_batches:
            m = rec["batch"] == b
            types = units[b]["types"]["coarse"]
            lx = classify(x_all[train], joint[train], x_all[m])
            ly = classify(y_all[train], joint[train], y_all[m])
            Sb, Cb = split_between(x_all[m], y_all[m], lx, ly, types)
            per_sample[b] = {"rna_accuracy": float(np.mean(lx == joint[m])),
                             "protein_accuracy": float(np.mean(ly == joint[m])),
                             "assays_agree": float(np.mean(lx == ly)),
                             "type_means_error_assay_labels": rel(Cb, T[f"{b}/total"]),
                             "type_means_error_joint_labels": errors[f"{b}/total/label_between"],
                             "frozen_error": errors[f"{b}/total/frozen/signal"]}
            composition[b] = 0.5 * (between_share(x_all[m], lx, types) + between_share(y_all[m], ly, types))
            for name, ch in chans.items():
                if ch["VS"].shape[1]:
                    Cc = Sb @ (ch["W"] @ ch["VS"] @ ch["VS"].T).T
                    agreement[(b, name)] = float(np.sum(Cc * Cb) / (np.linalg.norm(Cc) * np.linalg.norm(Cb)))
            print(b, {k: round(v, 3) for k, v in per_sample[b].items()}, flush=True)
    samples = list(units)
    donors_s = np.array([units[b]["donor"] for b in samples])
    diff = np.array([per_sample[b]["type_means_error_assay_labels"] - per_sample[b]["type_means_error_joint_labels"]
                     for b in samples])
    gap = np.array([per_sample[b]["frozen_error"] - per_sample[b]["type_means_error_assay_labels"] for b in samples])

    def ci(v, seed):
        boot = cluster_bootstrap(donors_s, lambda i: float(np.mean(v[i])), BOOT, seed)
        return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

    summary = {k: float(np.mean([per_sample[b][k] for b in samples])) for k in per_sample[samples[0]]}
    summary["assay_minus_joint"] = {"mean": float(diff.mean()), "ci": ci(diff, SEED)}
    summary["frozen_minus_assay_label_type_means"] = {"mean": float(gap.mean()), "ci": ci(gap, SEED + 1),
                                                       "samples_type_means_better": int(np.sum(gap > 0))}
    model = json.loads((HERE / "results/failure_model.json").read_text())["models"]["primary"]
    diag = json.loads((HERE / "results/bmmc_diagnostics.json").read_text())
    rows = [r for r in diag if r["dim_S"] > 0 and r["type_agreement"] is not None]
    y = np.array([errors[f'{r["tag"]}/{r["channel"]}/signal'] >= 1 for r in rows])
    donors = np.array([r["donor"] for r in rows])
    hi = np.array([r["coverage_signal"] >= HIGH for r in rows])
    original = model_score(model, rows)
    new_rows = [dict(r, type_agreement=agreement[(r["batch"], r["channel"])],
                     between_share=composition[r["batch"]] if r["analysis"] == "total" else 0.0) for r in rows]
    relabelled = model_score(model, new_rows)
    disagreement = -np.array([r["type_agreement"] for r in new_rows])

    def auc_ci(yy, s, cl, seed, other=None):
        f = (lambda i: auc(yy[i], s[i])) if other is None else (lambda i: auc(yy[i], s[i]) - auc(yy[i], other[i]))
        b = cluster_bootstrap(cl, f, BOOT, seed)
        b = b[np.isfinite(b)]
        v = auc(yy, s) - (0 if other is None else auc(yy, other))
        return {"value": float(v), "ci": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]}

    failure = {"all_pairs": {"joint_labels": auc_ci(y, original, donors, SEED + 2),
                             "assay_labels": auc_ci(y, relabelled, donors, SEED + 3),
                             "difference": auc_ci(y, relabelled, donors, SEED + 4, original)},
               "high_coverage": {"joint_labels": auc_ci(y[hi], original[hi], donors[hi], SEED + 5),
                                 "assay_labels": auc_ci(y[hi], relabelled[hi], donors[hi], SEED + 6),
                                 "type_disagreement_assay_labels": auc_ci(y[hi], disagreement[hi], donors[hi],
                                                                          SEED + 7)}}
    out = {"per_sample": per_sample, "summary": summary, "failure": failure}
    (HERE / "results/posthoc_labels.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"summary": summary, "failure": failure}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
