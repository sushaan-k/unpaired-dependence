#!/usr/bin/env python3
"""Verify the second-order transfer extension.

Without raw data (default):
 1. unit tests (closed form, profile likelihood, cell coupling, pairing-free fits);
 2. recompute results/scale_summary.json from the saved per-donor results;
 3. recompute the nine-marker binary comparison for the 11 colon donors from the
    released arrays and compare with results/binary_released_losses.npz
    (--full recomputes all 106 recipients);
 4. regenerate the manuscript macros, figure, table and RESULTS.md and compare
    them byte for byte with revision/source and results/.
With --colon-data PATH (the Figshare count matrix extracted by extract_colon.py):
 5. rerun one leave-one-donor-out recipient end to end and compare.

    python verify_spectral.py [--full] [--colon-data /path/colon_paired.npz]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "revision" / "source"


def close(a, b, path="root", atol=1e-10):
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            close(a[k], b[k], f"{path}.{k}", atol)
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]", atol)
    elif isinstance(b, float):
        assert abs(a - b) <= atol * max(1.0, abs(b)), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--colon-data", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    report = {}

    import hashlib
    provenance = json.loads((HERE / "results/provenance.json").read_text())
    assert hashlib.sha256((HERE / "PLAN.md").read_bytes()).hexdigest() == provenance["plan_sha256"]
    for name, digest in provenance["result_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    report["provenance"] = {"plan_sha256": provenance["plan_sha256"],
                            "result_files_unchanged": len(provenance["result_sha256"])}
    subprocess.run([sys.executable, "-m", "unittest", "-q", "test_spectral.py"], cwd=HERE, env=env, check=True)
    report["unit_tests"] = "passed"

    saved = json.loads((HERE / "results/scale_summary.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        import analyze_scale
        recomputed = {"D1_leave_one_donor_out": analyze_scale.summarize(HERE / "results/d1_lodo.json",
                                                                       HERE / "results/d3_pairing_free.json"),
                      "D2_healthy_to_colitis": analyze_scale.summarize(HERE / "results/d2_hc_to_uc.json")}
        close(recomputed, saved, "scale_summary")
        report["scale_summary_recomputed"] = True

        import binary_released as br
        from gaussian_transfer import Marginal, first_order, transfer
        from scipy.special import rel_entr
        sys.path.insert(0, str(HERE.parent / "identified_set"))
        from identified import SUPPORT, kernel, scale as sinkhorn
        from run_identified import load_recipients
        data = load_recipients()
        release = HERE.parent.parent / "release" / "analysis"
        inputs = np.load(release / "assay_resolution/reconstruction_inputs.npz")
        colon = np.load(release / "colon_generalization/results/predictions.npz")
        marginals = np.concatenate([inputs["strict_marginals"], colon["marginals"]])
        stored = np.load(HERE / "results/binary_released_losses.npz")
        B, K = data["interaction"], kernel(data["interaction"])
        people = range(len(marginals)) if args.full else np.flatnonzero(data["cohorts"] == "Colon")
        worst = 0.0
        for p in people:
            r, c = marginals[p]
            m, n = r @ SUPPORT, c @ SUPPORT
            Sx = (SUPPORT - m).T @ ((SUPPORT - m) * r[:, None])
            Sy = (SUPPORT - n).T @ ((SUPPORT - n) * c[:, None])
            T_closed, _ = br.tables_from_cross(m, n, transfer(Marginal(Sx, Sy), B)[0])
            Q, _, _ = sinkhorn(K, br.pairwise_maxent(r), br.pairwise_maxent(c))
            for arm, T in (("closed_form", T_closed), ("pairwise_exact", br.tables_from_joint(Q))):
                value = np.mean(2 * rel_entr(data["truth"][p], T).sum((1, 2)))
                worst = max(worst, abs(value - stored["loss"][p, list(stored["arms"]).index(arm)]))
        assert worst < 1e-9, worst
        report["binary_losses_recomputed"] = {"people": len(people), "max_difference": worst}

        subprocess.run([sys.executable, "make_scale_assets.py", "--out", tmp, "--results-md",
                        str(Path(tmp) / "RESULTS.md")], cwd=HERE, env=env, check=True, stdout=subprocess.DEVNULL)
        for name in ("scale_macros.tex", "scale_figure.tex", "scale_table.tex"):
            assert (Path(tmp) / name).read_bytes() == (SOURCE / name).read_bytes(), name
        assert (Path(tmp) / "RESULTS.md").read_bytes() == (HERE / "results/RESULTS.md").read_bytes()
        report["manuscript_assets_identical"] = True

    if args.colon_data:
        import colon_scale
        colon_scale.DATA = args.colon_data
        rerun = colon_scale.run(200, ["XAUT1-HS3"])["donors"]["XAUT1-HS3"]["arms"]
        original = json.loads((HERE / "results/d1_lodo.json").read_text())["donors"]["XAUT1-HS3"]["arms"]
        close(rerun, original, "d1_rerun", atol=1e-6)
        report["colon_rerun_XAUT1-HS3"] = "matches"
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
