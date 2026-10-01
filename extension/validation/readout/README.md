# Biological readout of the validation test

Does the saving of paired cells in the validation test (`../`) translate into a readout a laboratory would use:
calling within-cell-type gene-protein associations that replicate in independent held-out cells?

`PLAN.md` was written on 29 September 2026, after the validation had been scored and before any statistic of the
endpoints below had been computed; `freeze.py` hashed it with `brun.py`, every module they import (including the
validation's frozen `vrun.py`), the data files and the validation's hashed predictions at 07:06:12 EDT
(`results/freeze.json`). `brun.py run` checked that freeze and the validation's, regenerated the predictions of every
draw with the validation's frozen code path (the validation stored their means over draws), checked them against the
hashed means (largest absolute difference 1.2e-7 over 11,220 predictions; squared norms identical), added a budget of
150 paired cells, and scored.

## Endpoints

In each held-out sample and cell type with at least 20 held-out cells in each sub-half (170 units), a gene-protein
pair is a reproducible association if its correlation has the same sign in both sub-halves with two-sided P < 0.01 in
each (120,649 associations, 3.2% of pairs).

* E1, direction: fraction of reproducible associations whose sign the estimate gets right (zero counts one half).
* E2, nomination: fraction of the 100 largest absolute estimates per unit that are reproducible associations of the
  estimated sign (random 1.6%, perfect 97%).
* E3, cognate pairs: recovered fraction over the correlations of the 16 panel genes that encode a panel antibody
  with their proteins.

Paired-only estimation is the best of the eight paired-only methods for each endpoint and budget.

## Results (`results/readout.json`)

| Paired cells | 25 | 50 | 100 | 150 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|---|
| E1 atlas (%) | 70.7 | 80.2 | 86.1 | 88.6 | 89.8 | 92.2 | 93.9 |
| E1 paired only | 61.5 | 67.3 | 74.8 | 79.5 | 82.3 | 88.1 | 92.5 |
| E2 atlas (%) | 22.7 | 35.5 | 47.2 | 52.3 | 55.2 | 61.5 | 66.5 |
| E2 paired only | 6.9 | 15.3 | 27.9 | 35.5 | 40.0 | 51.6 | 60.9 |
| E3 atlas (%) | 12.6 | 24.3 | 38.8 | 46.9 | 52.9 | 64.4 | 73.4 |
| E3 paired only | 4.7 | 11.4 | 24.7 | 32.5 | 38.8 | 54.3 | 64.6 |

* **B1 met:** E1 differences 12.9 (12.1-13.7), 11.2 (10.3-12.0) and 9.1 (8.4-9.8) points at 50, 100 and 150 paired cells.
* **B2 met:** E2 differences 20.1 (18.9-21.3), 19.3 (18.0-20.5) and 16.8 (15.6-18.0) points.
* **B3 met:** E3 differences 14.2 (12.5-15.8) and 14.4 (12.9-15.3) points at 100 and 150.
* Paired cells to call the direction of 72.9% of associations: atlas 29, paired only 84 (2.85 times fewer,
  2.65-3.07); to make 33.7% of the top pairs reproducible: 45 and 136 (3.0 times fewer, 2.8-3.2).
* The own-sample and other-pool references gave the direction of more associations than the atlas at every budget;
  from 100 paired cells on the atlas nominated more reproducible pairs than the own-sample reference.
* By cell type, the gain lay in CD4 T, CD8 T and NK cells. In B cells the atlas gave the direction of slightly more
  associations up to 400 paired cells; in CD14 monocytes about as many below 400 and fewer from 400 on.
* Restricted to cognate pairs that were reproducible associations, paired-only estimation gave the direction slightly
  more often than the atlas at every budget (94.9% against 93.1% with 100 paired cells).

## Files and commands

```bash
python brun.py smoke   # synthetic values, real design; checked against the validation's smoke predictions
python freeze.py       # results/freeze.json
python brun.py run     # about 38 minutes on two cores; results/readout.json and run.log
cd ../.. && python verify_validation.py   # checks this freeze, the order of steps and the integrity record
```
