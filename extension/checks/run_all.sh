#!/bin/bash
# Every regeneration of the draw-scoring check, one after another, then the report and the data-free verification.
# Needs the public data (../DATA.md); several hours on two cores. A step that fails is logged and the others still run.
# The validation test is checked from the per-draw scores of its regeneration in ../second_review (run
# `python3 draw_scoring.py validation` instead to regenerate it here, about 20 minutes per held-out sample).
cd "$(dirname "$0")"
L=results
mkdir -p $L
run() {
    local log=$1
    shift
    python3 draw_scoring.py "$@" > "$L/$log" 2>&1 || echo "failed: draw_scoring.py $* (see $L/$log)" | tee -a "$L/failures.txt"
}
rm -f $L/failures.txt
run log_identity.txt identity
run log_external.txt external
run log_deployment.txt deployment
run log_validation.txt validation --from-review
for n in sln111 sln206 pbmc10k malt10k bmcite fetal_cortex snare_cortex fafb_vpn banc_vpn human_gaba; do
    run log_law_$n.txt law $n
done
for n in gouwens_visp scala_m1 malecns banc banc_crossanimal hao colon bmmc_cite bmmc_multiome; do
    run log_benchmark_$n.txt benchmark $n --folds 0
done
# stephenson.npz keeps its feature names as object arrays; the benchmark reads it through with_object_arrays.py
python3 ../generality/with_object_arrays.py draw_scoring.py benchmark stephenson --folds 0 \
    > $L/log_benchmark_stephenson.txt 2>&1 \
    || echo "failed: draw_scoring.py benchmark stephenson (see $L/log_benchmark_stephenson.txt)" | tee -a "$L/failures.txt"
run log_report.txt report
run log_verify.txt verify
touch $L/run_all.done
