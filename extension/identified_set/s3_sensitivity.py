#!/usr/bin/env python3
"""Sensitivity of S3, not prespecified in PLAN.md (added 2026-09-26 after S3).

(a) Restrict reversals to held-out tables with no empty cell, because a table
    with an empty cell has a sign fixed by its zero rather than by association.
(b) Held-out pair deviance of the two arms on reversal queries (descriptive).
Same bootstrap seed and draw count as S3.
"""

import json

import numpy as np

from analyze_identified import COHORTS, DRAWS, FAMILY_TAIL, SEED, pair_gain, reversal_summary
from run_identified import RESULTS, load_recipients


def compute(data):
    _, signs = reversal_summary(data)
    reversal = signs["s_freq"] != signs["s_full"]
    nonempty = (data["truth"] > 0).all(axis=(2, 3))
    usable = reversal & nonempty & (signs["s_emp"] != 0)
    agree = usable & (signs["s_emp"] == signs["s_full"])
    gain, deviance = pair_gain(data)
    out = {}
    for cohort in COHORTS:
        mask = data["cohorts"] == cohort
        a, u = agree[mask].sum(1), usable[mask].sum(1)
        draws = np.random.default_rng(SEED).integers(0, mask.sum(), (DRAWS, mask.sum()))
        tot = u[draws].sum(1)
        boot = a[draws].sum(1) / np.where(tot > 0, tot, np.nan)
        rev = reversal[mask]
        out[cohort] = {
            "nonempty_reversals": int(u.sum()),
            "nonempty_agreement": float(a.sum() / u.sum()),
            "nonempty_bootstrap_95": np.nanquantile(boot, (.025, .975)).tolist(),
            "nonempty_bootstrap_familywise": np.nanquantile(boot, (FAMILY_TAIL, 1 - FAMILY_TAIL)).tolist(),
            "reversal_mean_pair_deviance_frequency_only": float(deviance[mask, 0][rev].mean()),
            "reversal_mean_pair_deviance_full_patterns": float(deviance[mask, 3][rev].mean()),
            "nonreversal_mean_pair_deviance_frequency_only": float(deviance[mask, 0][~rev].mean()),
            "nonreversal_mean_pair_deviance_full_patterns": float(deviance[mask, 3][~rev].mean()),
            "share_of_total_gain_from_reversals": float(gain[mask][rev].sum() / gain[mask].sum()),
            "reversal_fraction": float(rev.mean()),
        }
    return out


if __name__ == "__main__":
    result = compute(load_recipients())
    (RESULTS / "s3_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
