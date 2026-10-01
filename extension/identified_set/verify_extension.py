#!/usr/bin/env python3
"""Verify the identified-set extension without modifying the original release.

Checks, in order:
 1. every original release checksum (release/SHA256SUMS) is unchanged;
 2. the plan hash recorded in results/summary.json matches PLAN.md;
 3. unit tests of this extension;
 4. re-certification of saved star endpoints (colon donors by default,
    all 106 recipients with --full) from their saved parameters;
 5. recomputation of every summary in results/summary.json and
    results/s3_sensitivity.json from released arrays and saved endpoints;
 6. regeneration of the manuscript macros, figure and table, compared
    byte-for-byte with the copies in revision/source.

    python verify_extension.py [--full]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RELEASE = ROOT / "release"
SOURCE = ROOT / "revision" / "source"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, path="root", atol=1e-12):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            close(actual[key], expected[key], f"{path}.{key}", atol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected)):
            close(a, e, f"{path}[{index}]", atol)
    elif isinstance(expected, float):
        assert abs(actual - expected) <= atol, f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="re-certify all 17,172 endpoints")
    args = parser.parse_args()
    report = {}

    entries = (RELEASE / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        expected, name = line.split(maxsplit=1)
        assert sha256(RELEASE / name) == expected, f"release file changed: {name}"
    report["release_checksums_unchanged"] = len(entries)

    summary = json.loads((HERE / "results/summary.json").read_text())
    assert summary["plan_sha256"] == sha256(HERE / "PLAN.md"), "PLAN.md changed after analysis"
    report["plan_sha256"] = summary["plan_sha256"]

    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-m", "unittest", "-q", "test_identified.py"], cwd=HERE, env=env, check=True)
    report["unit_tests"] = "passed"

    from identified import certify, kernel
    from run_identified import load_recipients
    data = load_recipients()
    star = np.load(HERE / "results/star_endpoints.npz")
    K = kernel(data["interaction"])
    people = range(len(data["cohorts"])) if args.full else np.flatnonzero(data["cohorts"] == "Colon")
    worst = 0.0
    for p in people:
        mu, nu = data["rna_means"][p], data["protein_means"][p]
        for q in range(81):
            i, j = divmod(q, 9)
            for side in ("low", "high"):
                t = certify(K, mu, nu, i, j, star[f"z_{side}"][p, q], float(star["margin"]))
                worst = max(worst, abs(t - star[f"star_{side}"][p, q]))
    assert worst < 1e-9, worst
    report["endpoints_recertified"] = int(len(people) * 162)
    report["max_recertification_difference"] = worst

    import analyze_identified as analysis
    s3, _ = analysis.reversal_summary(data)
    close(s3, summary["S3_reversal_direction"], "S3")
    s124, _, extent = analysis.identified_summary(data, star)
    close(s124, summary["S1_S2_S4_identified_set"], "S1_S2_S4", atol=1e-10)
    close(extent, summary["endpoint_sources"], "endpoint_sources")
    import s3_sensitivity
    close(s3_sensitivity.compute(data), json.loads((HERE / "results/s3_sensitivity.json").read_text()),
          "S3_sensitivity")
    report["summaries_recomputed"] = "S1-S4 and S3 sensitivity match"
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run([sys.executable, "make_manuscript_assets.py", "--out", temporary,
                        "--results-md", str(Path(temporary) / "RESULTS.md")], cwd=HERE,
                       env=env, check=True, stdout=subprocess.DEVNULL)
        for name in ("identified_macros.tex", "identified_figure.tex", "identified_table.tex"):
            assert (Path(temporary) / name).read_bytes() == (SOURCE / name).read_bytes(), name
        assert (Path(temporary) / "RESULTS.md").read_bytes() == (HERE / "results/RESULTS.md").read_bytes()
    report["manuscript_assets_identical"] = True
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
