#!/usr/bin/env python3
"""Fit the failure model on the development table and freeze it (PLAN.md, "Failure model").

Development pairs: recipient analyses of the untouched blood cohort and colon
crossed with the channels of dev_units2.py, excluding channels whose supported
subspace is empty (they predict zero and abstain). Outcome: failure, error of
closed-form transfer with latent RNA covariance >= 1. Predictors: coverage of
latent RNA variance by S, log compatibility statistic, agreement with the
recipient's own cell-type means, and between-type share (composition).
A second, label-free model uses coverage and log compatibility only.
Writes results/failure_model.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stats import auc, logistic_fit, logistic_score

HERE = Path(__file__).resolve().parent
MODELS = {"primary": ["coverage_signal", "log_F_signal", "type_agreement", "between_share"],
          "label_free": ["coverage_signal", "log_F_signal"]}
RIDGE = 1e-2


def features(row, name):
    if name == "log_F_signal":
        return float(np.log(row["F_signal"] + 1e-3))
    return float(row[name])


def main():
    d = json.loads((HERE / "results/dev_units2.json").read_text())
    tc = json.loads((HERE / "results/dev_typecheck.json").read_text())
    rows = [r for r in d["pairs"] if r["dim_S"] > 0]
    for r in rows:
        r["type_agreement"] = tc[f'{r["cohort"]}/{r["donor"]}|{r["channel"]}']["type_agreement"]
    y = np.array([r["E_signal"] >= 1 for r in rows])
    out = {"development_pairs": len(rows), "failures": int(y.sum()), "ridge": RIDGE, "models": {}}
    for name, cols in MODELS.items():
        X = np.array([[features(r, c) for c in cols] for r in rows])
        b0, b, mu, sd = logistic_fit(X, y, ridge=RIDGE)
        out["models"][name] = {"predictors": cols, "intercept": b0, "slopes": b.tolist(), "mean": mu.tolist(),
                               "sd": sd.tolist(),
                               "development_auc": auc(y, logistic_score((b0, b, mu, sd), X))}
    (HERE / "results/failure_model.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
