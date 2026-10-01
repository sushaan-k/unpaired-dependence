# Second-order and pairing-free transfer (extension, 2026-09-26)

This extension turns the paper's resolution theory into a method that scales
past the nine-marker panel. It also asks whether a paired reference is needed
at all. Every analysis here is post hoc relative to the released paper. `PLAN.md`
records the plan and three amendments; each amendment was made before the
results it affects were computed.

## What is new

1. **Resolution orders** (Supplementary Information, Proposition S3). Matching all
   within-assay pairwise statistics fixes the first-order transferred
   association. Individual frequencies leave it free (Proposition S2). So the
   remaining ambiguity is second order in the interaction strength.
2. **Closed-form transfer** (Proposition S4). For Gaussian assays, the paper's
   reconstruction (Eq. 1) has the closed form
   `C = Sx^1/2 U g(beta) V^T Sy^1/2`, where
   `g(b) = (sqrt(1+4b^2)-1)/(2b)`. Its linear term is the paper's response
   formula (Eq. 5), and the saturation keeps the joint law valid. The same
   formula appears in entropic optimal transport between Gaussians (Janati et
   al. 2020; Mallasto et al. 2022); here it is a transfer operator for a
   learned cross-assay interaction.
3. **Concave fitting** (Proposition S5). With paired reference people sharing
   B, the profile likelihood of B is concave. Its gradient is the observed
   minus the transferred cross-covariance.
4. **Pairing-free identification** (Proposition S6). Unpaired data cannot
   identify B under the paper's model with free marginals. They can when the
   protein-given-RNA channel is shared across samples, and then
   `B = W^T Psi^-1`.

## Evidence

`results/RESULTS.md` is generated from the result files and gives every number.
In brief:

- **Nine-marker binary panel** (released arrays, unchanged blood interaction).
  Exact reconstruction from recipient means and covariances retained 91-98% of
  the full-pattern gain in Cambridge, Newcastle, T-ALL and colon. The closed
  form retained 66-89%. The first-order formula alone was worse than
  individual frequencies in every cohort.
- **Colon panel**: 209 genes x 177 proteins, 15,953 paired cells from 12 donors.
  In leave-one-donor-out transfer, the closed form predicted held-out
  gene-protein covariances with relative error 0.341. The comparators scored:
  reference regression 0.380, reference correlation 0.477, reference
  covariance 0.669, and the closed form given only variances 0.936.
  The closed form was better in 12/12 donors on every prespecified contrast.
  It was also better than exact coupling of the recipient's cells (0.371) and
  than the recipient's own paired half (0.367).
  Healthy-to-colitis transfer replicated this in 8/8 patients.
- **Pairing-free (first version; superseded).** A corrected re-analysis in
  `../cross_study/colon_pf_rigor.py` fixes five weaknesses of this first version
  (shared cells between the two assays' moments, biopsy-level folds, a penalty
  at the grid edge, iteration-limited rank-8 fits, a control that allowed
  within-donor reassignment); Figure 3b of the revision shows the corrected
  values, and `../cross_study/results/RESULTS.md` lists them. The first
  version's numbers follow. With no paired cell, the interaction was learned
  from 19-20 unpaired biopsies. The primary estimator (ecological regression on biopsy
  means) recovered 70% of the paired reference's error reduction from
  independence (error 0.536; correlation 0.840). The secondary rank-8
  likelihood estimator recovered 82% (error 0.460; correlation 0.881).
  - The paired reference was still better in every donor.
  - For the primary estimator, a derangement control abolished the signal
    (p = 1/201 for each donor). The control was not run for the rank-8
    estimator.
  - The error fell with more training biopsies and had not levelled off.

## Reproduce

The colon analyses need the public count matrix, which is not redistributed:

```bash
curl -L -o colon_counts.h5ad https://ndownloader.figshare.com/files/54027674   # 3.8 GB
md5sum colon_counts.h5ad        # d0dbb0fc4c95fbd5ce8e72cf418aad21
pip install h5py
python extract_colon.py --h5ad colon_counts.h5ad --out colon_paired.npz
# set DATA in colon_scale.py to the npz path, then:
python colon_scale.py --genes 200 --out results/d1_lodo.json          # ~1.5 h on one core
python colon_scale.py --genes 200 --reference HC --out results/d2_hc_to_uc.json
python pairing_free.py --out results/d3_pairing_free.json            # ~6 min
python pairing_free_control.py && python pairing_free_curve.py
```

The binary analysis uses the released arrays only:

```bash
python binary_released.py                 # ~5 min
python analyze_scale.py
python make_scale_assets.py --out ../../revision/source
python -m unittest -v test_spectral.py
python verify_spectral.py [--full] [--colon-data colon_paired.npz]
```

## Files

| File | Content |
|---|---|
| `PLAN.md` | Plan and amendments with reasons |
| `gaussian_transfer.py` | Closed form, profile likelihood, fitting |
| `binary_released.py` | Nine-marker comparison: pairwise exact, closed form, first order |
| `extract_colon.py` | Extraction of paired cells from the public count matrix |
| `colon_scale.py` | D1 (leave one donor out) and D2 (healthy to colitis) |
| `pairing_free.py`, `pairing_free_control.py`, `pairing_free_curve.py` | D3, negative control, learning curve |
| `analyze_scale.py`, `make_scale_assets.py` | Prespecified summaries; manuscript assets; `results/RESULTS.md` |
| `test_spectral.py`, `verify_spectral.py` | Unit tests; end-to-end verification |

## Caveats

- One study, 12 donors, 21 biopsies. Transfers stay within that study, and
  all analyses are retrospective.
- The closed form is exact only for Gaussian assays. On binary data it
  retained less of the gain than exact reconstruction from the same
  covariances.
- For predicting protein from RNA in single cells, the closed form matched
  reference regression (R^2 0.285 vs 0.283). Its advantage is in the
  dependence endpoint.
- Exact cell coupling used the interaction fitted for the closed form, not
  one refitted by inverse optimal transport.
- Pairing-free learning assumes a shared channel across samples. In this first
  version, its penalty was chosen at the grid edge for 11 of 12 recipients, its
  rank-8 fit hit its iteration limit, and it stayed less accurate than the
  paired reference. See `../cross_study/` for the corrected re-analysis and a
  cross-study test on an untouched cohort.
- Gene selection used RNA of all paired cells, including recipients' scoring
  cells. It is unsupervised and uses no protein or pairing information, and it
  is the same for every arm.
