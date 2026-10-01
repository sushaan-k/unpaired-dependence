#!/usr/bin/env python3
"""Split every colon donor's paired cells into adaptation and scoring halves.

The halves are written to separate files. Only the adaptation file is read by
feature selection, fitting, tuning and prediction; the scoring file's SHA-256
is recorded here and it is opened only by evaluate.py after the predictions
are hashed. Salt: blood-colon-v1 (new; not used by any earlier analysis).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

SOURCE = Path("/home/claude/cbio/rawdata/colon_paired.npz")
OUT = Path("/home/claude/cbio/rawdata/blood_colon")
SALT = "blood-colon-v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SOURCE, allow_pickle=False)
    rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
    donors, barcodes = d["CoLabs_patient"], d["barcodes"]
    half = np.zeros(len(donors), int)
    for donor in sorted(set(donors)):
        idx = np.flatnonzero(donors == donor)
        keys = [hashlib.sha256(f"{SALT}|{donor}|{barcodes[i]}".encode()).hexdigest() for i in idx]
        order = idx[np.argsort(keys)]
        half[order[len(order) // 2:]] = 1          # first half adaptation, second scoring
    meta = ("CoLabs_patient", "CoLabs_sample", "condition", "coarse_annotations_MK", "barcodes")
    record = {"salt": SALT, "files": {}}
    for name, flag in (("adaptation", 0), ("scoring", 1)):
        m = half == flag
        sub = rna[m].tocsr()
        path = OUT / f"colon_{name}.npz"
        np.savez_compressed(path, rna_data=sub.data, rna_indices=sub.indices, rna_indptr=sub.indptr,
                            rna_names=d["rna_names"], adt=d["adt"][m], adt_names=d["adt_names"],
                            **{k: d[k][m] for k in meta})
        record["files"][name] = {"cells": int(m.sum()),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record["cells_per_donor"] = {str(k): [int(np.sum((donors == k) & (half == 0))),
                                          int(np.sum((donors == k) & (half == 1)))]
                                 for k in sorted(set(donors))}
    Path(__file__).with_name("seal.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
