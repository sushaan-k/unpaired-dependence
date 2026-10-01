#!/bin/bash
# Everything else, one job at a time, after the running noise calibration of the validation test.
cd "$(dirname "$0")"
L=results/logs
while kill -0 2378 2>/dev/null; do sleep 30; done
python3 noise_calibration.py run bone_marrow > $L/noise_bone_marrow.log 2>&1
python3 readout_checks.py run > $L/readout_checks.log 2>&1
python3 law_review.py inference > $L/law_inference.log 2>&1
for n in sln111 sln206 pbmc10k malt10k bmcite fetal_cortex snare_cortex fafb_vpn banc_vpn human_gaba; do
    python3 law_review.py pilot $n > $L/law_pilot_$n.log 2>&1
done
for n in external gouwens_visp scala_m1 malecns banc hao stephenson colon bmmc_cite bmmc_multiome; do
    python3 benchmark_review.py run $n > $L/benchmark_$n.log 2>&1
done
python3 atlas_bootstrap.py run -1 0 1 2 3 4 5 6 7 8 9 > $L/atlas_bootstrap.log 2>&1
cd ../checks
for n in sln111 sln206 pbmc10k malt10k bmcite fetal_cortex snare_cortex fafb_vpn banc_vpn human_gaba; do
    python3 draw_scoring.py law $n > results/log_law_$n.txt 2>&1
done
for n in gouwens_visp scala_m1 malecns banc banc_crossanimal hao stephenson colon bmmc_cite bmmc_multiome; do
    python3 draw_scoring.py benchmark $n --folds 0 > results/log_benchmark_$n.txt 2>&1
done
touch ../second_review/$L/lane_b.done
