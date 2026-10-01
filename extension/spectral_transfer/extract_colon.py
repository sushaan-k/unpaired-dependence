#!/usr/bin/env python3
"""Extract paired (GEX_CITE) colon cells from the public Mennillo et al. count matrix.

Input: Figshare file 54027674 (article 21919356 v3), MD5 d0dbb0fc4c95fbd5ce8e72cf418aad21.
Output (not redistributed): a compressed npz with raw counts for all RNA genes and
all 177 ADTs of the 15,953 paired cells, plus donor, biopsy, condition and cell-type
labels. No cell is selected by expression.

    python extract_colon.py --h5ad /path/colon_counts.h5ad --out /path/colon_paired.npz
"""
import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


def column(node):
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]
    categories = node["categories"]
    categories = categories.asstr()[:] if categories.dtype.kind in "OSU" else categories[:]
    return categories[node["codes"][:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.h5ad, "r") as h:
        names = column(h["var/_index"])
        assay = column(h["var/assay"])
        obs = {k: column(h["obs"][k]) for k in ("CoLabs_patient", "CoLabs_sample", "condition",
                                                  "colon_biopsy", "LIBRARY.TYPE",
                                                  "coarse_annotations_MK", "fine_annotations_MK_V4")}
        barcodes = column(h["obs/_index"])
        rows = np.flatnonzero(obs["LIBRARY.TYPE"] == "GEX_CITE")
        counts = h["layers/counts"]
        assert counts.attrs["encoding-type"] == "csr_matrix"
        indptr = counts["indptr"][:]
        data, indices, pointer = [], [], [0]
        for row in rows:
            start, end = indptr[row], indptr[row + 1]
            data.append(counts["data"][start:end])
            indices.append(counts["indices"][start:end])
            pointer.append(pointer[-1] + end - start)
        matrix = sp.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(pointer)),
                               shape=(len(rows), len(names)))
    assert np.all(matrix.data >= 0) and np.all(matrix.data == np.floor(matrix.data))
    rna, adt = assay == "RNA", assay == "ADT"
    np.savez_compressed(
        args.out, rna_data=matrix[:, rna].tocsr().data, rna_indices=matrix[:, rna].tocsr().indices,
        rna_indptr=matrix[:, rna].tocsr().indptr, rna_names=names[rna].astype(str),
        adt=matrix[:, adt].toarray().astype(np.float32), adt_names=names[adt].astype(str),
        barcodes=barcodes[rows].astype(str), **{k.replace(".", "_"): v[rows].astype(str) for k, v in obs.items()})
    print("cells", len(rows), "rna genes", int(rna.sum()), "adts", int(adt.sum()),
          "nnz", matrix.nnz)


if __name__ == "__main__":
    main()
