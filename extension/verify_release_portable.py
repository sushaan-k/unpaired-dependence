#!/usr/bin/env python3
"""Run the original release verification with platform-tolerant comparisons.

The release's `analysis/assay_resolution/verify_reported_results.py` fails on
Linux x86-64 (Python 3.11, numpy 2.0.2) because it demands bitwise equality
between recomputed and stored floating-point losses: 9 of 380 values differ by
at most 3.5e-18.
This script leaves the checksummed release untouched. In a temporary copy it
replaces each bitwise comparison with an absolute tolerance of 1e-15 and the
exact adaptation-only dictionary comparison with the verifier's own 1e-12
recursive comparison, then runs both unit-test suites and both verifiers.

    python verify_release_portable.py
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"

PATCH_HELPER = '''
_MAX_DIFFERENCE = {}


def _equal(actual, desired, err_msg=""):
    """Portable replacement for bitwise equality (absolute tolerance 1e-15)."""
    actual = np.asarray(actual, dtype=float)
    desired = np.asarray(desired, dtype=float)
    key = len(_MAX_DIFFERENCE)
    _MAX_DIFFERENCE[key] = float(np.max(np.abs(actual - desired))) if actual.size else 0.0
    np.testing.assert_allclose(actual, desired, rtol=0, atol=1e-15, err_msg=err_msg)

'''


def patched(source: str) -> str:
    anchor = "RETAINED_GAIN_BOOTSTRAP_95 = {"
    assert source.count(anchor) == 1
    source = source.replace(anchor, PATCH_HELPER + anchor)
    assert source.count("np.testing.assert_array_equal(") == 7
    source = source.replace("np.testing.assert_array_equal(", "_equal(")
    exact = 'assert actual == expected, "strict adaptation-only evaluation is not an exact match"'
    assert source.count(exact) == 1
    source = source.replace(exact, 'assert_close(actual, expected, "adaptation_only")')
    final = "print(json.dumps(report, indent=2))"
    assert source.count(final) == 1
    return source.replace(final, 'report["max_differences_at_former_bitwise_checks"] = _MAX_DIFFERENCE\n    ' + final)


def main():
    entries = (RELEASE / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        expected, name = line.split(maxsplit=1)
        assert hashlib.sha256((RELEASE / name).read_bytes()).hexdigest() == expected, name
    print(f"Verified {len(entries)} release checksums.", flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", VECLIB_MAXIMUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="cbio-portable-") as temporary:
        work = Path(temporary) / "analysis"
        shutil.copytree(RELEASE / "analysis", work)
        target = work / "assay_resolution" / "verify_reported_results.py"
        target.write_text(patched(target.read_text()))
        jobs = [
            ("assay_resolution", ["-m", "unittest", "-v", "test_reconstruct.py"]),
            ("colon_generalization", ["-m", "unittest", "-v", "test_protocol.py"]),
            ("assay_resolution", ["verify_reported_results.py"]),
            ("colon_generalization", ["verify_results.py"]),
        ]
        for folder, arguments in jobs:
            print(f"Running {folder}: {' '.join(arguments)}", flush=True)
            subprocess.run([sys.executable, *arguments], cwd=work / folder, env=env, check=True)
    print(json.dumps({"status": "passed", "release_files": len(entries),
                      "verification_commands": len(jobs),
                      "modification": "bitwise float equality -> atol 1e-15, in a temporary copy only"}))


if __name__ == "__main__":
    main()
