#!/usr/bin/env python3
"""Frozen feature rules of PLAN.md ("Features").

    python features.py candidates   # after extract_stephenson.py stats
    python features.py select       # after seal_hao.py (uses adaptation cells only)

Candidates: symbols present in all three studies with Stephenson mean > 0.05,
ranked by Stephenson variance/mean; the top 600 are carried into the Hao split
(so that the final top-200 can be drawn after the eligibility filter), plus the
nine panel genes. Final genes: top 200 candidates detected in >= 1% of the
adaptation cells of both recipient studies, plus the panel genes present in all
three studies.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from adt_matching import COLON, HAO, STEPHENSON, matched_targets

HERE = Path(__file__).resolve().parent
DATA = Path("/home/claude/cbio/rawdata/cross_study")
COLON_ADAPT = Path("/home/claude/cbio/rawdata/blood_colon/colon_adaptation.npz")
HAO_FEATURES = Path("/home/claude/cbio/rawdata/hao/GSM5008737_RNA_3P-features.tsv.gz")
PANEL = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
CARRY, KEEP, MIN_DETECTION, MIN_MEAN = 600, 200, 0.01, 0.05


def symbols():
    steph = np.load(DATA / "stephenson_gene_stats.npz")
    colon = set(np.load(COLON_ADAPT)["rna_names"].tolist())
    hao = {l.decode().split("\t")[0] for l in gzip.open(HAO_FEATURES)}
    return steph, colon, hao


def ranking():
    steph, colon, hao = symbols()
    names, mean, var = steph["genes"], steph["mean"], steph["var"]
    shared = np.array([g in colon and g in hao for g in names])
    ok = shared & (mean > MIN_MEAN)
    score = np.where(ok, var / np.maximum(mean, 1e-12), -np.inf)
    order = np.argsort(-score, kind="stable")
    ranked = [str(names[i]) for i in order if np.isfinite(score[i])]
    panel = [g for g in PANEL if g in set(names[shared])]
    return ranked, panel, {str(n): float(s) for n, s in zip(names, score) if np.isfinite(s)}


def candidates():
    ranked, panel, _ = ranking()
    genes = ranked[:CARRY] + [g for g in panel if g not in ranked[:CARRY]]
    (HERE / "results/gene_candidates.json").write_text(json.dumps(
        {"rule": f"top {CARRY} by Stephenson variance/mean (mean > {MIN_MEAN}) among symbols shared "
                 "by all three studies, plus panel genes", "genes": genes, "panel": panel}, indent=1) + "\n")
    print(len(genes), genes[:20])


def colon_detection():
    d = np.load(COLON_ADAPT, allow_pickle=False)
    rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
    rate = np.asarray((rna > 0).sum(axis=0)).ravel() / rna.shape[0]
    return dict(zip(d["rna_names"].tolist(), rate))


def select():
    ranked, panel, score = ranking()
    carried = json.loads((HERE / "results/gene_candidates.json").read_text())["genes"]
    colon = colon_detection()
    hao = np.load(DATA / "hao_adaptation_detection.npz")
    hao = dict(zip(hao["genes"].tolist(), hao["rate"]))
    eligible = [g for g in ranked if colon.get(g, 0) >= MIN_DETECTION and hao.get(g, 0) >= MIN_DETECTION]
    top = eligible[:KEEP]
    assert all(g in carried for g in top), "final genes must lie within the carried candidates"
    genes = top + [g for g in panel if g not in top]
    record = {"genes": genes, "panel_added": [g for g in panel if g not in top],
              "proteins": matched_targets(STEPHENSON, COLON, HAO),
              "eligible_within_carried": int(sum(g in carried for g in eligible)),
              "rank_of_last_selected": ranked.index(top[-1]) + 1,
              "detection": {g: [float(colon.get(g, 0)), float(hao.get(g, 0))] for g in genes},
              "dispersion": {g: score[g] for g in genes}}
    (HERE / "results/features.json").write_text(json.dumps(record, indent=1) + "\n")
    print(len(genes), "genes;", len(record["proteins"]), "proteins; last selected rank",
          record["rank_of_last_selected"], "panel added", record["panel_added"])


if __name__ == "__main__":
    {"candidates": candidates, "select": select}[sys.argv[1]]()
