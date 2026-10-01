#!/usr/bin/env python3
"""Light verification of the recoverability study without raw data.

Checks that the code and protocol match the latest freeze record, that every
result file matches the manifest written before unsealing, that the bootstrap
output matches its pre-unsealing record, that the prespecified verdicts follow
from the evaluation, and that the manuscript macros equal those regenerated from
the result files. Large result files omitted from the lite archive are reported as
not checked. With --full, also re-runs the evaluation (needs the bone-marrow
files and every result file) and compares its verdicts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
SOURCE = HERE.parent.parent / "revision" / "source"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_check():
    """freeze.check_freeze, except that files absent from a lite archive are listed instead of failing."""
    from freeze import FILES, latest
    record = latest()
    missing = []
    for f in FILES:
        if not (HERE / f).exists():
            missing.append(f)
            continue
        assert sha(HERE / f) == record["files"][f], f"{f} changed since the freeze"
    return record, missing


def main():
    from predict_bmmc import RESULT_FILES
    record, missing = freeze_check()
    manifest = json.loads((R / "bmmc_manifest.json").read_text())
    absent = [name for name in RESULT_FILES if not (R / name).exists()]
    for name in RESULT_FILES:
        if name not in absent:
            assert sha(R / name) == manifest[name], f"manifest mismatch: {name}"
    boot = json.loads((R / "bmmc_bootstrap_record.json").read_text())
    assert sha(R / "bmmc_bootstrap.json") == boot["bmmc_bootstrap.json"], "bootstrap record mismatch"
    assert sha(R / "bmmc_manifest.json") == boot["bmmc_manifest.json"], "manifest changed after bootstrap record"
    ev = json.loads((R / "bmmc_evaluation.json").read_text())
    c = ev["failure"]["comparisons"]
    verdict = {"H1": c["primary_vs_coverage"]["ci"][0] > 0,
               "H2": {k: c[f"primary_vs_{k}"]["ci"][0] > 0 for k in ("composition", "cells", "n_train", "shift")},
               "H3": ev["failure"]["high_coverage"]["primary"]["ci"][0] > 0.5,
               "H6": all(ev["reference"]["tests"][t]["ci"][1] < 0 and ev["reference"]["tests"][t]["lower_at_every_budget"]
                         for t in ("guided_vs_random", "guided_vs_balanced"))}
    stored = ev["verdict"]
    assert verdict["H1"] == stored["H1_primary_beats_coverage"]
    assert verdict["H2"] == stored["H2_primary_beats_controls"]
    assert verdict["H3"] == stored["H3_high_coverage_failures_recognized"]
    assert verdict["H6"] == (stored["H6_guided_beats_random"] and stored["H6_guided_beats_balanced"])
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(HERE / "make_assets.py"), "--out", tmp], check=True,
                       stdout=subprocess.DEVNULL, cwd=HERE)
        for name in ("recov_macros.tex", "recov_figure.tex", "recov_table.tex", "recov_auc_table.tex",
                     "recov_ref_table.tex", "recov_sim_table.tex", "recov_id_table.tex"):
            assert (Path(tmp) / name).read_bytes() == (SOURCE / name).read_bytes(), f"{name} differs from the manuscript"
    if "--full" in sys.argv:
        subprocess.run([sys.executable, str(HERE / "evaluate_bmmc.py")], check=True, cwd=HERE)
        again = json.loads((R / "bmmc_evaluation.json").read_text())
        assert again["verdict"] == stored, "re-evaluation changed the verdicts"
    print(json.dumps({"frozen_at": record["frozen_at"], "amendments": len(list(R.glob("freeze_amendment_*.json"))),
                      "manifest_files_checked": len(RESULT_FILES) - len(absent),
                      "absent_large_files_not_checked": sorted(set(absent) | {m.split("/")[-1] for m in missing}),
                      "verdict": verdict, "macros": "match"}, indent=1))


if __name__ == "__main__":
    main()
