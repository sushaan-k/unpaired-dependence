#!/usr/bin/env python3
"""Recipient analyses of the bone-marrow test from cell metadata only (PLAN.md, "Units").

Recomputes each cell's half from the deposited barcodes and the salt of
extract_bmmc.py (no count is read) and keeps, per sample and label level, the
cell types with at least 10 cells in each half. Writes results/bmmc_units.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from extract_bmmc import COARSE, RAW, SALT, column

HERE = Path(__file__).resolve().parent
MIN_CELLS = 10


def main():
    with h5py.File(RAW, "r") as f:
        obs = f["obs"]
        barcodes, batch = column(obs, "_index"), column(obs, "batch")
        donor, fine = column(obs, "DonorID").astype(str), column(obs, "cell_type")
    coarse_of = {t: c for c, ts in COARSE.items() for t in ts}
    coarse = np.array([coarse_of[t] for t in fine])
    half = np.zeros(len(barcodes), np.int8)
    for b in sorted(set(batch)):
        idx = np.flatnonzero(batch == b)
        keys = [hashlib.sha256(f"{SALT}|{b}|{barcodes[i]}".encode()).hexdigest() for i in idx]
        half[idx[np.argsort(keys)][len(idx) // 2:]] = 1
    seal = json.loads((HERE / "results/bmmc_seal.json").read_text())
    assert {b: [int(np.sum((batch == b) & (half == 0))), int(np.sum((batch == b) & (half == 1)))]
            for b in sorted(set(batch))} == seal["cells_per_batch"]
    units = {}
    for b in sorted(set(batch)):
        m = batch == b
        entry = {"donor": sorted(set(donor[m]))[0], "cells": [int(np.sum(m & (half == 0))), int(np.sum(m & (half == 1)))],
                 "types": {}}
        for level, labels in (("coarse", coarse), ("fine", fine)):
            kept = sorted(t for t in set(labels[m])
                          if np.sum(m & (labels == t) & (half == 0)) >= MIN_CELLS
                          and np.sum(m & (labels == t) & (half == 1)) >= MIN_CELLS)
            entry["types"][level] = kept
        units[b] = entry
    (HERE / "results/bmmc_units.json").write_text(json.dumps(units, indent=1) + "\n")
    for b, e in units.items():
        print(b, e["donor"], e["cells"], len(e["types"]["coarse"]), len(e["types"]["fine"]))


if __name__ == "__main__":
    main()
