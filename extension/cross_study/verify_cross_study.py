#!/usr/bin/env python3
"""Verify the cross-study pairing-free extension.

Without raw data (default):
 1. unit tests;
 2. provenance: plan hash, seals, prediction manifest and reference-fit hash;
 3. recompute results/summary.json from scores.json and permutation_scores.json;
 4. regenerate the manuscript macros, figure, table and RESULTS.md and compare
    them byte for byte with revision/source and results/.
With --data (the extracted training and recipient files under rawdata/cross_study):
 5. refit pf_means from the Stephenson halves and compare its interaction;
 6. recompute every prediction for one Hao and one colon donor from the frozen
    reference fit and adaptation cells, and rescore them against the sealed
    scoring cells.

    python verify_cross_study.py [--data]
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
SOURCE = HERE.parent.parent / "revision" / "source"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    report = {}
    subprocess.run([sys.executable, "-m", "unittest", "-q", "test_cross_study.py"], cwd=HERE, env=env,
                   check=True)
    report["unit_tests"] = "passed"

    # PLAN.md grew by appending (amendment 1 before prediction, post hoc P1 and P2 after scoring); every
    # recorded version must be an exact prefix of the current file. (The P1 prefix was added to this list later;
    # the script had been written before P2 was appended and so checked only three of the four recorded
    # versions. PLAN.md and results/plan_sha256.txt are unchanged; see CORRECTION_2026-09-27.md for
    # the stated times of P1 and P2.)
    text = (HERE / "PLAN.md").read_text()
    recorded = [line.split()[0] for line in (HERE / "results/plan_sha256.txt").read_text().splitlines()]
    versions = {"original": text[:text.index("## Amendment 1") - 1],
                "amendment_1": text[:text.index("## Post hoc analysis P1") - 1],
                "post_hoc_p1": text[:text.index("## Post hoc analysis P2") - 1], "current": text}
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in versions.items()}
    assert set(hashes.values()) == set(recorded), (hashes, recorded)
    plan = hashes["amendment_1"]
    manifest = json.loads((HERE / "results/predictions_manifest.json").read_text())
    assert manifest["PLAN.md"] == plan
    # Shared packages leave out result files above 5 MB (here predictions.npz, 74 MB). A file that is absent is
    # reported as not checked instead of stopping the verification; the full workspace has it.
    absent = []
    for name in ("predictions.npz", "prediction_info.json", "reference_fit.npz"):
        if not (HERE / "results" / name).exists():
            absent.append(name)
            continue
        assert sha256(HERE / "results" / name) == manifest[name], name
    for name in ("seal.json", "hao_seal.json"):
        assert sha256(HERE / name) == manifest[name], name
    reference = json.loads((HERE / "results/reference_fit.json").read_text())
    assert reference["sha256"] == manifest["reference_fit.npz"]
    report["provenance"] = {"plan_sha256": plan, "predictions_sha256": manifest["predictions.npz"],
                            "absent_from_this_package_not_checked": absent}
    if args.data and absent:
        raise SystemExit(f"--data needs {', '.join(absent)}, which this package leaves out (full workspace only)")

    import analyze
    scores = json.loads((HERE / "results/scores.json").read_text())
    perm = json.loads((HERE / "results/permutation_scores.json").read_text())["E"]
    saved = json.loads((HERE / "results/summary.json").read_text())
    rng = np.random.default_rng(analyze.SEED)
    for cohort in ("hao", "colon"):
        n = len({k.split("/")[1] for k in scores["scores"] if k.startswith(cohort + "/")})
        idx = rng.integers(0, n, size=(analyze.BOOT, n))
        close(json.loads(json.dumps(analyze.summarize(scores["scores"], perm, cohort, idx))), saved[cohort],
              f"summary.{cohort}")
    report["summary_recomputed"] = True

    # The figure needs results/truth.npz (22 MB), which the lite package also leaves out.
    if (HERE / "results/truth.npz").exists():
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "make_assets.py", "--out", tmp, "--results-md",
                            str(Path(tmp) / "RESULTS.md")], cwd=HERE, env=env, check=True,
                           stdout=subprocess.DEVNULL)
            for name in ("cross_macros.tex", "cross_figure.tex", "cross_table.tex", "cross_rigor_table.tex"):
                assert (Path(tmp) / name).read_bytes() == (SOURCE / name).read_bytes(), name
            assert (Path(tmp) / "RESULTS.md").read_bytes() == (HERE / "results/RESULTS.md").read_bytes()
        report["manuscript_assets_identical"] = True
    else:
        report["manuscript_assets_identical"] = "not checked: results/truth.npz is absent from this package"

    if args.data:
        from threadpoolctl import threadpool_limits
        import estimators as est
        from data import load_recipient, load_training
        from evaluate import metrics, scoring_target
        from predict import ARMS, donor_predictions, reference
        with threadpool_limits(limits=1):
            train = load_training()
            patients = sorted(set(train["patient"]))
            raw = est.pf_raw_moments(train, patients)
            units = est.pf_units(raw, *est.pooled_sd(raw))
            fit = est.fit_pf_means(units, est.patient_folds(patients))
            stored = np.load(HERE / "results/reference_fit.npz")["pf_means_B"]
            assert np.max(np.abs(fit["B"] - stored)) <= 1e-8 * np.max(np.abs(stored))
            report["pf_means_refit"] = "matches"
            preds = np.load(HERE / "results/predictions.npz")
            ref = reference()
            for cohort, donor in (("hao", "P3"), ("colon", "XAUT1-HS10")):
                adapt, score = load_recipient(cohort, "adaptation"), load_recipient(cohort, "scoring")
                P, _ = donor_predictions(adapt, donor, "total", None, ref)
                T, _, _ = scoring_target(score, donor, None, None)
                for arm in ARMS:
                    stored_pred = preds[f"{cohort}/{donor}/total/{arm}"]
                    assert np.max(np.abs(P[arm] - stored_pred)) <= 1e-9, (cohort, donor, arm)
                    close(metrics(P[arm], T), scores["scores"][f"{cohort}/{donor}/total"]["arms"][arm],
                          f"{cohort}.{donor}.{arm}", atol=1e-9)
                report[f"rerun_{cohort}_{donor}"] = "matches"
    skipped = absent + ([] if (HERE / "results/truth.npz").exists() else ["asset regeneration (truth.npz)"])
    report["status"] = "passed" if not skipped else f"passed, partial: not checked here: {', '.join(skipped)}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
