#!/bin/bash
# Every run of the analyses added in revision (PLAN.md), then the scoring. Needs the public data (../DATA.md) and
# results/freeze.json, which each run checks. About twelve hours on two cores; the runs were made in two lanes side
# by side (the validation review in one, everything else in the other), and each command below can be run alone.
cd "$(dirname "$0")"
L=results/logs
mkdir -p $L
export OPENBLAS_THREAD_TIMEOUT=4      # idle BLAS threads sleep, so two runs side by side do not spin against each other

# A. the noise estimate after contrasts
python3 noise_calibration.py run validation > $L/noise_validation.log 2>&1
python3 noise_calibration.py run bone_marrow > $L/noise_bone_marrow.log 2>&1
# B, C, D1, E, G. the validation test regenerated with the new arms and truths (even and odd folds)
python3 validation_review.py run --part 0 > $L/validation_part0.log 2>&1
python3 validation_review.py run --part 1 > $L/validation_part1.log 2>&1
# F. the association test under permutation
python3 readout_checks.py run > $L/readout_checks.log 2>&1
# H, I. the law with data sets as units, and a pilot that sets the budget
python3 law_review.py inference > $L/law_inference.log 2>&1
for n in sln111 sln206 pbmc10k malt10k bmcite fetal_cortex snare_cortex fafb_vpn banc_vpn human_gaba; do
    python3 law_review.py pilot $n > $L/law_pilot_$n.log 2>&1
done
# J. savings at fixed accuracy (stephenson.npz keeps its feature names as object arrays: read, as in the benchmark,
# through ../generality/with_object_arrays.py, the benchmark's Deviation 1; DEVIATIONS.md)
for n in external gouwens_visp scala_m1 malecns banc hao colon bmmc_cite bmmc_multiome; do
    python3 benchmark_review.py run $n > $L/benchmark_$n.log 2>&1
done
PYTHONPATH="$PWD" python3 ../generality/with_object_arrays.py benchmark_review.py run stephenson \
    > $L/benchmark_stephenson.log 2>&1
# D2. the atlas's patients resampled (replicate -1 takes every patient once); it needs about 4 GB of memory and, on
# the 8 GB machine used here, could not run beside both parts of the validation review (DEVIATIONS.md)
python3 atlas_bootstrap.py run -1 0 1 2 3 4 5 6 7 8 9 > $L/atlas_bootstrap.log 2>&1

# scoring (needs no data)
for s in noise_calibration validation_review atlas_bootstrap law_review benchmark_review; do
    python3 $s.py score > $L/score_$s.log 2>&1
done
python3 verify.py > $L/verify.log 2>&1
